# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import pytest

from aiq.data_models.config import AIQConfig
from aiq.data_models.dataset_handler import EvalDatasetJsonConfig
from aiq.data_models.evaluate import EvalConfig
from aiq.data_models.evaluate import EvalOutputConfig
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.eval.evaluate import EvaluationRun
from aiq.eval.evaluate import EvaluationRunConfig
from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem
from aiq.eval.evaluator.evaluator_model import EvalOutput
from aiq.eval.evaluator.evaluator_model import EvalOutputItem
from aiq.runtime.session import AIQSessionManager

# pylint: disable=redefined-outer-name


@pytest.fixture
def default_eval_run_config():
    """Fixture for default evaluation run configuration."""
    return EvaluationRunConfig(config_file="config.yml",
                               dataset=None,
                               result_json_path="$",
                               skip_workflow=False,
                               skip_completed_entries=False,
                               endpoint=None,
                               endpoint_timeout=300,
                               reps=1)


@pytest.fixture
def eval_input():
    """Fixture to provide a mock EvalInput with a single item."""
    eval_item = EvalInputItem(id=1,
                              input_obj="User input",
                              expected_output_obj="Golden answer",
                              output_obj=None,
                              expected_trajectory=[],
                              trajectory=[])
    return EvalInput(eval_input_items=[eval_item])


@pytest.fixture
def evaluation_run(default_eval_run_config, eval_input, default_eval_config):
    """Fixture for creating an EvaluationRun instance with defaults and one eval input item."""
    eval_run = EvaluationRun(default_eval_run_config)
    eval_run.eval_input = eval_input
    eval_run.eval_config = default_eval_config
    return eval_run


@pytest.fixture
def generated_answer():
    """Fixture to provide a generated answer."""
    return "Generated answer"


@pytest.fixture
def tool_end_intermediate_step():
    """Fixture to create a valid TOOL_END IntermediateStep."""
    return IntermediateStep(payload=IntermediateStepPayload(
        event_type=IntermediateStepType.TOOL_END, data=StreamEventData(input="Tool input", output="Tool output")))


@pytest.fixture
def llm_end_intermediate_step(generated_answer):
    """Fixture to create a valid LLM_END IntermediateStep."""
    return IntermediateStep(payload=IntermediateStepPayload(
        event_type=IntermediateStepType.LLM_END, data=StreamEventData(input="User input", output=generated_answer)))


@pytest.fixture
def average_score():
    return 0.9


@pytest.fixture
def eval_output(average_score):
    """Fixture to provide a mock EvalOutput with a single item."""
    return EvalOutput(average_score=average_score,
                      eval_output_items=[EvalOutputItem(id=1, score=average_score, reasoning="All is well")])


@pytest.fixture
def mock_evaluator(eval_output):
    """Fixture to create a mock evaluator."""

    async def mock_evaluate_fn(_eval_input):
        return eval_output

    # Create a mock evaluator
    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_fn = AsyncMock(side_effect=mock_evaluate_fn)

    return mock_evaluator


@pytest.fixture
def default_eval_config(mock_evaluator):
    """Fixture for default evaluation configuration."""
    eval_config = EvalConfig()
    eval_config.general.dataset = EvalDatasetJsonConfig()
    eval_config.general.output = EvalOutputConfig()
    eval_config.general.max_concurrency = 1
    eval_config.general.output.dir = Path(".tmp/aiq/examples/mock/")
    eval_config.evaluators = {"MockEvaluator": mock_evaluator}

    return eval_config


# Simple mock workflow class defined to the extent needed for eval testing
class MockWorkflow:

    def __init__(self):
        self.has_single_output = True


@pytest.fixture
def mock_pull_intermediate(tool_end_intermediate_step, llm_end_intermediate_step, generated_answer):
    """Fixture to mock pull_intermediate as a simple async function returning TOOL_END and LLM_END steps."""
    with patch("aiq.eval.runtime_event_subscriber.pull_intermediate",
               AsyncMock(return_value=[tool_end_intermediate_step, llm_end_intermediate_step])) as mock:
        yield mock


@pytest.fixture
def session_manager(generated_answer, mock_pull_intermediate):
    """
    Fixture to provide a mocked AIQSessionManager instance.

    DONT REMOVE mock_pull_intermediate arg. Although it is not used in this function,
    it is needed to ensure that pull_intermediate is mocked for all tests that use session_manager.
    """
    session_manager = MagicMock(spec=AIQSessionManager)

    # Create a mock runner that behaves like an async context manager
    mock_runner = AsyncMock()

    mock_workflow = MockWorkflow()

    session_manager.workflow = mock_workflow

    async def mock_result():
        return generated_answer

    mock_runner.result = AsyncMock(side_effect=mock_result)
    mock_runner.convert = MagicMock(return_value=generated_answer)

    # Define an async context manager for runner
    @asynccontextmanager
    async def mock_run(_message, _fn):
        """Mock async context manager for runner."""
        yield mock_runner

    session_manager.run = mock_run
    return session_manager


# Batch-1: Tests for running workflow to evaluate
async def test_run_workflow_local_success(evaluation_run, session_manager, generated_answer):
    """Test successful workflow execution with local runner."""

    # Run the actual function
    await evaluation_run.run_workflow_local(session_manager)

    # Ensure output is correctly set
    final_output = evaluation_run.eval_input.eval_input_items[0].output_obj
    assert final_output == generated_answer, f"Expected {generated_answer}, but got {final_output}"

    # Ensure workflow was not interrupted
    assert not evaluation_run.workflow_interrupted


async def test_run_workflow_local_errors(evaluation_run, session_manager):
    """Test workflow with no 'single output' fails gracefully."""

    session_manager.workflow.has_single_output = False

    with pytest.raises(NotImplementedError):
        # Run the actual function
        await evaluation_run.run_workflow_local(session_manager)


async def test_run_workflow_local_skip_completed(evaluation_run, session_manager, generated_answer):
    """Test that 'skip_completed_entries=True' skips completed items and processes only unfinished ones."""

    old_answer = "Can't touch this"
    # Create two eval input items:
    # - One completed (should be skipped)
    # - One pending (should be processed)
    completed_item = EvalInputItem(id=1,
                                   input_obj="Completed Question",
                                   expected_output_obj="Golden Answer",
                                   output_obj=old_answer,
                                   expected_trajectory=[],
                                   trajectory=[])
    pending_item = EvalInputItem(id=2,
                                 input_obj="Pending Question",
                                 expected_output_obj="Golden Answer",
                                 output_obj=None,
                                 expected_trajectory=[],
                                 trajectory=[])

    # Assign mock eval input items to the evaluation run
    evaluation_run.eval_input = EvalInput(eval_input_items=[completed_item, pending_item])

    # Enable skipping completed entries
    evaluation_run.config.skip_completed_entries = True

    # Run the actual function
    await evaluation_run.run_workflow_local(session_manager)

    # Ensure the completed item was NOT processed
    assert completed_item.output_obj == old_answer, "Completed item should be skipped"

    # Ensure the pending item was processed
    assert pending_item.output_obj == generated_answer, "Pending item output should have been processed"


async def test_run_workflow_local_workflow_interrupted(evaluation_run, eval_input, session_manager):
    """Test that workflow_interrupted is set to True when an exception occurs during workflow execution."""

    # Assign the mock eval input to the evaluation run
    evaluation_run.eval_input = eval_input

    # Create a mock runner that will raise an exception when awaited
    mock_error_runner = AsyncMock()

    # Mock result to raise an exception when awaited
    async def mock_result():
        raise RuntimeError("Simulated workflow failure")

    mock_error_runner.result = AsyncMock(side_effect=mock_result)

    @asynccontextmanager
    async def mock_error_run(_message, _entry_fn):
        """Mock async context manager for runner."""
        yield mock_error_runner

    session_manager.run = mock_error_run
    # Run the actual function
    # Check if workflow_interrupted is set to True
    await evaluation_run.run_workflow_local(session_manager)
    assert evaluation_run.workflow_interrupted, "Expected workflow_interrupted to be True after failure"


async def test_run_workflow_remote_success(evaluation_run, generated_answer):
    """
    Mock RemoteWorkflowHandler and test evaluation with a remote workflow.
    """
    # Patch the remote handler
    with patch("aiq.eval.remote_workflow.EvaluationRemoteWorkflowHandler") as MockHandler:
        mock_handler = MockHandler.return_value

        async def fake_run_workflow_remote(eval_input):
            """
            Mock the run_workflow_remote method to update the output field of the item.
            """
            for item in eval_input.eval_input_items:
                item.output_obj = generated_answer
            return eval_input

        mock_handler.run_workflow_remote = AsyncMock(side_effect=fake_run_workflow_remote)

        # Run the remote evaluation (this calls the mocked handler)
        await evaluation_run.run_workflow_remote()

        # Assert that each item was updated with the generated output
        for item in evaluation_run.eval_input.eval_input_items:
            assert item.output_obj == generated_answer, f"Expected {generated_answer}, got {item.output_obj}"


# Batch-2: Tests for running evaluators
async def test_run_single_evaluator_success(evaluation_run, mock_evaluator, eval_output, average_score):
    """Test for running a single evaluator."""
    # Run the evaluator (actual function)
    await evaluation_run.run_single_evaluator("MockEvaluator", mock_evaluator)

    # Ensure at least one result is stored
    assert evaluation_run.evaluation_results, "Evaluation results should not be empty"

    # Get the last and only result
    evaluator_name, result = evaluation_run.evaluation_results[-1]
    # Validate stored values
    assert evaluator_name == "MockEvaluator", "Evaluator name should match"
    assert isinstance(result, EvalOutput), "Stored result should be an instance of EvalOutput"
    assert result == eval_output, "Stored result should match the expected eval_output"
    assert result.average_score == average_score, f"Expected average score to be {average_score}"


async def test_run_evaluators_success(evaluation_run, mock_evaluator, eval_output, average_score):
    """Test for running multiple evaluators successfully."""

    # Create multiple evaluators
    evaluators = {
        "MockEvaluator1": mock_evaluator,
        "MockEvaluator2": mock_evaluator,  # Reusing the same mock for simplicity
    }

    # Run the evaluators (actual function)
    await evaluation_run.run_evaluators(evaluators)

    # Ensure the results are stored correctly
    assert len(evaluation_run.evaluation_results) == len(evaluators), "All evaluators should store results"

    for evaluator_name, result in evaluation_run.evaluation_results:
        assert evaluator_name in evaluators, f"Evaluator name {evaluator_name} should match one of the evaluators"
        assert result == eval_output, f"Stored result for {evaluator_name} should match the provided eval_output"
        assert result.average_score == average_score, f"Expected average score to be {average_score}"


async def test_run_evaluators_partial_failure(evaluation_run, mock_evaluator, eval_output, average_score):
    """
    Test run_evaluators where one evaluator fails but others succeed.
    When one fails we still want to complete others while logging exception on the failing evaluator.
    """

    # Define evaluators (one failing, one successful)
    good_evaluator_name = "GoodEvaluator"
    bad_evaluator_name = "BadEvaluator"

    # Create a failing evaluator
    mock_failing_evaluator = AsyncMock()
    mock_failing_evaluator.evaluate_fn.side_effect = RuntimeError("Evaluator failed")

    evaluators = {good_evaluator_name: mock_evaluator, bad_evaluator_name: mock_failing_evaluator}

    # Patch logger to check error logging
    with patch("aiq.eval.evaluate.logger.exception") as mock_logger:
        # Run the evaluators (actual function)
        await evaluation_run.run_evaluators(evaluators)

    # Ensure successful evaluator result is stored
    assert len(evaluation_run.evaluation_results) == 1, "Only successful evaluators should store results"
    # Get the last and only result
    evaluator_name, result = evaluation_run.evaluation_results[-1]
    # Validate stored values
    assert evaluator_name == good_evaluator_name, "Evaluator name should match"
    assert result == eval_output, "Stored result should match the expected eval_output"
    assert result.average_score == average_score, f"Expected average score to be {average_score}"

    # Ensure the failure is logged
    mock_logger.assert_called()
    logged_message = mock_logger.call_args[0][0]  # Extract the actual log message
    assert "An error occurred while running evaluator" in logged_message, \
        "Error message should indicate evaluator failure"


# Batch-3: Tests for running eval and writing results
def test_write_output(evaluation_run, default_eval_config, eval_input, eval_output, generated_answer):
    """Test writing the workflow and evaluation results."""
    # Mock dataset handler to get the formatted workflow results
    for eval_input_item in eval_input.eval_input_items:
        eval_input_item.output_obj = generated_answer

    mock_dataset_handler = MagicMock()
    workflow_output = json.dumps([item.dict() for item in eval_input.eval_input_items])
    mock_dataset_handler.publish_eval_input.return_value = workflow_output

    # Mock evaluation results
    evaluator_name = "MockEvaluator"
    evaluation_run.evaluation_results = [(evaluator_name, eval_output)]

    # Mock eval_config output directory
    evaluation_run.eval_config = default_eval_config
    output_dir = default_eval_config.general.output_dir

    # Workflow output must be written to workflow_output.json
    workflow_output_path = output_dir / "workflow_output.json"

    # Evaluator results must be written to {evaluator_name}_output.json
    evaluator_output_path = output_dir / f"{evaluator_name}_output.json"

    # Patch file operations and logging. It is important to keep logs frozen to match user expectations.
    with patch("builtins.open", mock_open()) as mock_file, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("aiq.eval.evaluate.logger.info") as mock_logger:

        # Run the actual function
        evaluation_run.write_output(mock_dataset_handler)

        # Ensure directories are created
        mock_mkdir.assert_called()

        # Ensure the workflow output is written
        mock_file.assert_any_call(workflow_output_path, "w", encoding="utf-8")
        mock_file().write.assert_any_call(workflow_output)

        # Ensure the evaluator output is written
        mock_file.assert_any_call(evaluator_output_path, "w", encoding="utf-8")
        eval_output_dict = eval_output.model_dump_json(indent=2)
        mock_file().write.assert_any_call(eval_output_dict)

        # Ensure log format has not changed
        mock_logger.assert_any_call("Workflow output written to %s", workflow_output_path)
        mock_logger.assert_any_call("Evaluation results written to %s", evaluator_output_path)


@pytest.mark.parametrize("skip_workflow", [True, False])
async def test_run_and_evaluate(evaluation_run, default_eval_config, session_manager, mock_evaluator, skip_workflow):
    """
    Test that run_and_evaluate
    1. correctly loads config
    2. runs workflow
    3. evaluates
    4. profiles
    5. writes output.
    """
    evaluation_run.config.skip_workflow = skip_workflow
    # Patch load_config to return an AIQConfig instance with eval_config set
    mock_aiq_config = AIQConfig()
    mock_aiq_config.eval = default_eval_config
    mock_load_config = MagicMock(return_value=mock_aiq_config)

    # Mock dataset handler
    mock_dataset_handler = MagicMock()
    mock_dataset_handler.get_eval_input_from_dataset.return_value = evaluation_run.eval_input

    # Mock evaluator
    mock_eval_workflow = MagicMock()
    mock_eval_workflow.build.return_value = MagicMock()
    mock_eval_workflow.get_evaluator.return_value = mock_evaluator

    # Mock WorkflowEvalBuilder
    @asynccontextmanager
    async def mock_eval_builder(config):
        yield mock_eval_workflow

    # Mock OutputUploader and its methods
    mock_uploader = MagicMock()
    mock_uploader.run_custom_scripts = MagicMock()
    mock_uploader.upload_directory = AsyncMock()

    # check if run_custom_scripts and upload_directory are called
    # Patch functions and classes. Goal here is simply to ensure calls are made to the right functions.
    with patch("aiq.runtime.loader.load_config", mock_load_config), \
         patch("aiq.builder.eval_builder.WorkflowEvalBuilder.from_config", side_effect=mock_eval_builder), \
         patch("aiq.runtime.session.AIQSessionManager", return_value=session_manager), \
         patch("aiq.eval.evaluate.DatasetHandler", return_value=mock_dataset_handler), \
         patch("aiq.eval.evaluate.OutputUploader", return_value=mock_uploader), \
         patch.object(evaluation_run, "run_workflow_local",
                      wraps=evaluation_run.run_workflow_local) as mock_run_workflow, \
         patch.object(evaluation_run, "run_evaluators", AsyncMock()) as mock_run_evaluators, \
         patch.object(evaluation_run, "profile_workflow", AsyncMock()) as mock_profile_workflow, \
         patch.object(evaluation_run, "write_output", MagicMock()) as mock_write_output:

        # Run the function
        await evaluation_run.run_and_evaluate()

        # Ensure config is loaded
        assert evaluation_run.eval_config == default_eval_config, "Evaluation config should be set correctly"

        # Ensure dataset is loaded
        assert mock_dataset_handler.get_eval_input_from_dataset.call_count == 1, \
            "get_eval_input_from_dataset should be called once"

        # Ensure workflow runs only if skip_workflow is False
        if not evaluation_run.config.skip_workflow:
            assert mock_run_workflow.call_count == 1, "run_workflow should be called once"
        else:
            mock_run_workflow.assert_not_called()

        # Ensure evaluators run
        mock_run_evaluators.assert_called_once_with({"MockEvaluator": mock_evaluator})

        # Ensure profiling is executed
        mock_profile_workflow.assert_called_once()

        # Ensure output is written
        mock_write_output.assert_called_once_with(mock_dataset_handler)

        # Ensure custom scripts are run and directory is uploaded
        mock_uploader.run_custom_scripts.assert_called_once()
        mock_uploader.upload_directory.assert_awaited_once()
