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

import typing

import pytest

if typing.TYPE_CHECKING:
    from aiq.eval.evaluator.evaluator_model import EvalInput
    from aiq.eval.intermediate_step_adapter import IntermediateStepAdapter


@pytest.fixture(name="rag_expected_outputs")
def rag_expected_outputs_fixture() -> list[str]:
    """Fixture providing expected outputs corresponding to user inputs."""
    return ["Machine Learning", "Natural Language Processing"]


@pytest.fixture(name="intermediate_step_adapter")
def intermediate_step_adapter_fixture() -> "IntermediateStepAdapter":
    from aiq.eval.intermediate_step_adapter import IntermediateStepAdapter
    return IntermediateStepAdapter()


@pytest.fixture
def rag_eval_input(rag_user_inputs, rag_expected_outputs, rag_generated_outputs, rag_intermediate_steps) -> "EvalInput":
    """Fixture to create a mock EvalInput with multiple items."""

    from aiq.eval.evaluator.evaluator_model import EvalInput
    from aiq.eval.evaluator.evaluator_model import EvalInputItem

    # Unpack intermediate steps
    steps_1, steps_2 = rag_intermediate_steps
    intermediate_steps_map = [steps_1, steps_2]

    eval_items = [
        EvalInputItem(
            id=index + 1,  # Ensure unique IDs (1, 2, ...)
            input_obj=user_input,
            expected_output_obj=expected_output,
            output_obj=generated_output,
            expected_trajectory=[],  # Modify if needed
            trajectory=intermediate_steps_map[index]  # Ensure correct step assignment
        ) for index, (user_input, expected_output,
                      generated_output) in enumerate(zip(rag_user_inputs, rag_expected_outputs, rag_generated_outputs))
    ]

    return EvalInput(eval_input_items=eval_items)
