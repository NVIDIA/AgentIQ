# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from aiq.data_models.evaluate import EvalConfig
from aiq.eval.config import EvaluationRunConfig
from aiq.eval.config import EvaluationRunOutput
from aiq.eval.dataset_handler.dataset_handler import DatasetHandler
from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem
from aiq.eval.evaluator.evaluator_model import EvalOutput
from aiq.eval.utils.output_uploader import OutputUploader
from aiq.runtime.session import AIQSessionManager

logger = logging.getLogger(__name__)


class EvaluationRun:  # pylint: disable=too-many-public-methods
    """
    Instantiated for each evaluation run and used to store data for that single run.
    """

    def __init__(self, config: EvaluationRunConfig):
        """
        Initialize an EvaluationRun with configuration.
        """
        from aiq.eval.intermediate_step_adapter import IntermediateStepAdapter

        # Run-specific configuration
        self.config: EvaluationRunConfig = config
        self.eval_config: EvalConfig | None = None

        # Helpers
        self.intermediate_step_adapter: IntermediateStepAdapter = IntermediateStepAdapter()

        # Metadata
        self.eval_input: EvalInput | None = None
        self.workflow_interrupted: bool = False

        # evaluation_results is list of tuples (evaluator_name, EvalOutput)
        self.evaluation_results: list[tuple[str, EvalOutput]] = []

        # workflow output file
        self.workflow_output_file: Path | None = None

        # evaluation output files
        self.evaluator_output_files: list[Path] = []

    async def run_workflow_local(self, session_manager: AIQSessionManager):
        '''
        Launch the workflow with the specified questions and extract the output using the jsonpath
        '''
        # import function level dependencies
        from jsonpath_ng import parse

        from aiq.eval.runtime_event_subscriber import pull_intermediate

        # Run the workflow
        jsonpath_expr = parse(self.config.result_json_path)
        stop_event = asyncio.Event()

        async def run_one(item: EvalInputItem):
            if stop_event.is_set():
                return "", []

            async with session_manager.run(item.input_obj) as runner:
                try:
                    # Start usage stats and intermediate steps collection in parallel
                    intermediate_future = pull_intermediate()

                    if session_manager.workflow.has_single_output:
                        base_output = await runner.result()
                    else:
                        # raise an error if the workflow has multiple outputs
                        raise NotImplementedError("Multiple outputs are not supported")
                    intermediate_steps = await intermediate_future
                except NotImplementedError as e:
                    # raise original error
                    raise e
                except Exception as e:
                    logger.exception("Failed to run the workflow: %s", e, exc_info=True)
                    # stop processing if a workflow error occurs
                    self.workflow_interrupted = True
                    stop_event.set()
                    return

                try:
                    base_output = runner.convert(base_output, to_type=str)
                except ValueError:
                    pass

                # if base_output is a pydantic model dump it to json
                if isinstance(base_output, BaseModel):
                    output = base_output.model_dump_json(indent=2)
                else:
                    m = jsonpath_expr.find(base_output)
                    if (not m):
                        raise RuntimeError(f"Failed to extract output using jsonpath: {self.config.result_json_path}")
                    if (len(m) > 1):
                        logger.warning("Multiple matches found for jsonpath at row '%s'. Matches: %s. Using the first",
                                       base_output,
                                       m)
                    output = m[0].value

                item.output_obj = output
                item.trajectory = self.intermediate_step_adapter.validate_intermediate_steps(intermediate_steps)

        async def wrapped_run(item: EvalInputItem) -> None:
            await run_one(item)
            pbar.update(1)

        # if self.config.skip_complete is set skip eval_input_items with a non-empty output_obj
        if self.config.skip_completed_entries:
            eval_input_items = [item for item in self.eval_input.eval_input_items if not item.output_obj]
            if not eval_input_items:
                logger.warning("All items have a non-empty output. Skipping workflow pass altogether.")
                return
        else:
            eval_input_items = self.eval_input.eval_input_items
        pbar = tqdm(total=len(eval_input_items), desc="Running workflow")
        await asyncio.gather(*[wrapped_run(item) for item in eval_input_items])
        pbar.close()

    async def run_workflow_remote(self):
        from aiq.eval.remote_workflow import EvaluationRemoteWorkflowHandler
        handler = EvaluationRemoteWorkflowHandler(self.config, self.eval_config.general.max_concurrency)
        await handler.run_workflow_remote(self.eval_input)

    async def profile_workflow(self):
        """
        Profile a dataset
        """

        if not self.eval_config.general.profiler:
            logger.info("Profiler is not enabled. Skipping profiling.")
            return

        from aiq.profiler.profile_runner import ProfilerRunner

        all_stats = []
        for input_item in self.eval_input.eval_input_items:
            all_stats.append(input_item.trajectory)

        profiler_runner = ProfilerRunner(self.eval_config.general.profiler, self.eval_config.general.output_dir)

        await profiler_runner.run(all_stats)

    def cleanup_output_directory(self):
        '''Remove contents of the output directory if it exists'''
        if self.eval_config.general.output and self.eval_config.general.output.dir and \
                self.eval_config.general.output.dir.exists():
            logger.info("Cleaning up output directory %s", self.eval_config.general.output.dir)
            shutil.rmtree(self.eval_config.general.output.dir)

    def write_output(self, dataset_handler: DatasetHandler):
        workflow_output_file = self.eval_config.general.output_dir / "workflow_output.json"
        workflow_output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the workflow output to a file (this can be used for re-running the evaluation)
        workflow_output = dataset_handler.publish_eval_input(self.eval_input)
        with open(workflow_output_file, "w", encoding="utf-8") as f:
            # set indent to 2 for pretty printing
            f.write(workflow_output)
        self.workflow_output_file = workflow_output_file
        logger.info("Workflow output written to %s", workflow_output_file)

        # Write the output of each evaluator to a separate json file
        for evaluator_name, eval_output in self.evaluation_results:
            output_file = self.eval_config.general.output_dir / f"{evaluator_name}_output.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # create json content using the evaluation results
            output = eval_output.model_dump_json(indent=2)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            self.evaluator_output_files.append(output_file)
            logger.info("Evaluation results written to %s", output_file)

        if self.workflow_interrupted:
            # Issue a warning if the workflow was not completed on all datasets
            msg = ("Workflow execution was interrupted due to an error. The results may be incomplete. "
                   "You can re-execute evaluation for incomplete results by running "
                   "`eval` with the --skip_completed_entries flag.")
            logger.warning(msg)

    async def run_single_evaluator(self, evaluator_name: str, evaluator: Any):
        """Run a single evaluator and store its results."""
        try:
            eval_output = await evaluator.evaluate_fn(self.eval_input)
            self.evaluation_results.append((evaluator_name, eval_output))
        except Exception as e:
            logger.exception("An error occurred while running evaluator %s: %s", evaluator_name, e, exc_info=True)

    async def run_evaluators(self, evaluators: dict[str, Any]):
        """Run all configured evaluators asynchronously."""
        tasks = [self.run_single_evaluator(name, evaluator) for name, evaluator in evaluators.items() if evaluator]

        if not tasks:
            logger.warning("All evaluators were empty or invalid.")
            return

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.exception("An error occurred while running evaluators: %s", e, exc_info=True)
            raise

    async def run_and_evaluate(self,
                               session_manager: AIQSessionManager | None = None,
                               job_id: str | None = None) -> EvaluationRunOutput:
        """
        Run the workflow with the specified config file and evaluate the dataset
        """
        logger.info("Starting evaluation run with config file: %s", self.config.config_file)

        from aiq.builder.eval_builder import WorkflowEvalBuilder
        from aiq.runtime.loader import load_config

        # Load the config object
        config = load_config(self.config.config_file)
        self.eval_config = config.eval
        logger.debug("Loaded evaluation configuration: %s", self.eval_config)

        # Cleanup the output directory
        if self.eval_config.general.output and self.eval_config.general.output.cleanup:
            self.cleanup_output_directory()

        # If a job id is provided keep the data per-job
        if job_id:
            self.eval_config.general.output_dir = self.eval_config.general.output_dir / f"jobs/{job_id}"
            if self.eval_config.general.output:
                self.eval_config.general.output.dir = self.eval_config.general.output_dir

        # Load the input dataset
        # For multiple datasets, one handler per dataset can be created
        dataset_config = self.eval_config.general.dataset  # Currently only one dataset is supported
        if not dataset_config:
            logger.info("No dataset found, nothing to evaluate")
            return EvaluationRunOutput(
                workflow_output_file=self.workflow_output_file,
                evaluator_output_files=self.evaluator_output_files,
                workflow_interrupted=self.workflow_interrupted,
            )

        dataset_handler = DatasetHandler(dataset_config=dataset_config, reps=self.config.reps)
        self.eval_input = dataset_handler.get_eval_input_from_dataset(self.config.dataset)
        if not self.eval_input.eval_input_items:
            logger.info("Dataset is empty. Nothing to evaluate.")
            return EvaluationRunOutput(
                workflow_output_file=self.workflow_output_file,
                evaluator_output_files=self.evaluator_output_files,
                workflow_interrupted=self.workflow_interrupted,
            )

        # Run workflow and evaluate
        async with WorkflowEvalBuilder.from_config(config=config) as eval_workflow:
            if self.config.endpoint:
                await self.run_workflow_remote()
            else:
                if not self.config.skip_workflow:
                    if session_manager is None:
                        session_manager = AIQSessionManager(eval_workflow.build(),
                                                            max_concurrency=self.eval_config.general.max_concurrency)
                    await self.run_workflow_local(session_manager)

            # Evaluate
            evaluators = {name: eval_workflow.get_evaluator(name) for name in self.eval_config.evaluators}
            await self.run_evaluators(evaluators)

        # Profile the workflow
        await self.profile_workflow()

        # Write the results to the output directory
        self.write_output(dataset_handler)

        # Run custom scripts and upload evaluation outputs to S3
        if self.eval_config.general.output:
            output_uploader = OutputUploader(self.eval_config.general.output, job_id=job_id)
            output_uploader.run_custom_scripts()
            await output_uploader.upload_directory()

        return EvaluationRunOutput(
            workflow_output_file=self.workflow_output_file,
            evaluator_output_files=self.evaluator_output_files,
            workflow_interrupted=self.workflow_interrupted,
        )
