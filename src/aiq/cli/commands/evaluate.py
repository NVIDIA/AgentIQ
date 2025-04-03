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
from pathlib import Path

import click

from aiq.eval.evaluate import EvaluationRun
from aiq.eval.evaluate import EvaluationRunConfig

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=True, help="Evaluate a workflow with the specified dataset.")
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="A JSON/YAML file that sets the parameters for the workflow and evaluation.",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="A json file with questions and ground truth answers. This will override the dataset path in the config file.",
)
@click.option(
    "--result_json_path",
    type=str,
    default="$",
    help=("A JSON path to extract the result from the workflow. Use this when the workflow returns "
          "multiple objects or a dictionary. For example, '$.output' will extract the 'output' field "
          "from the result."),
)
@click.option(
    "--skip_workflow",
    is_flag=True,
    default=False,
    help="Skip the workflow execution and use the provided dataset for evaluation. "
    "In this case the dataset should have the 'generated_' columns.",
)
@click.option(
    "--skip_completed_entries",
    is_flag=True,
    default=False,
    help="Skip the dataset entries that have a generated answer.",
)
@click.option(
    "--endpoint",
    type=str,
    default=None,
    help="Use endpoint for running the workflow. Example: http://localhost:8000/generate",
)
@click.option(
    "--endpoint_timeout",
    type=int,
    default=300,
    help="HTTP response timeout in seconds. Only relevant if endpoint is specified.",
)
@click.option(
    "--reps",
    type=int,
    default=1,
    help="Number of repetitions for the evaluation.",
)
@click.option(
    "--generate_ground_truth",
    is_flag=True,
    default=False,
    help="Generate ground truth dataset based on evaluation results. \
        Can be used with --reps greater than 1 to get the best answer.",
)
@click.pass_context
def eval_command(ctx, **kwargs) -> None:
    """ Evaluate datasets with the specified mechanism"""
    pass


async def run_and_evaluate(config: EvaluationRunConfig):
    # Run evaluation
    eval_runner = EvaluationRun(config=config)
    await eval_runner.run_and_evaluate()


@eval_command.result_callback(replace=True)
def process_aiq_eval(
    processors,  # pylint: disable=unused-argument
    *,
    config_file: Path,
    dataset: Path,
    result_json_path: str,
    skip_workflow: bool,
    skip_completed_entries: bool,
    endpoint: str,
    endpoint_timeout: int,
    reps: int,
    generate_ground_truth: bool,
):
    """
    Process the eval command and execute the evaluation. Here the config_file, if provided, is checked for its existence
    on disk.
    """
    # Cannot skip_workflow if endpoint is specified
    if skip_workflow and endpoint:
        raise click.UsageError("The options '--skip_workflow' and '--endpoint' are mutually exclusive. "
                               "Please use only one of them.")

    # You cannot run multiple repetitions if you are skipping the workflow or skipping completed entries
    if reps > 1 and (skip_workflow or skip_completed_entries):
        raise click.UsageError("The options '--reps' and '--skip_workflow' or '--skip_completed_entries' are mutually "
                               "exclusive. You cannot run multiple repetitions if you are skipping the workflow or "
                               "have a partially completed dataset.")

    # Create the configuration object
    config = EvaluationRunConfig(
        config_file=config_file,
        dataset=str(dataset) if dataset else None,
        result_json_path=result_json_path,
        skip_workflow=skip_workflow,
        skip_completed_entries=skip_completed_entries,
        endpoint=endpoint,
        endpoint_timeout=endpoint_timeout,
        reps=reps,
        generate_ground_truth=generate_ground_truth,
    )
    asyncio.run(run_and_evaluate(config))
