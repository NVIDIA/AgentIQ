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

from pathlib import Path

from pydantic import BaseModel


class EvaluationRunConfig(BaseModel):
    """
    Parameters used for a single evaluation run.
    """
    config_file: Path
    dataset: str | None  # dataset file path can be specified in the config file
    result_json_path: str = "$"
    skip_workflow: bool = False
    skip_completed_entries: bool = False
    endpoint: str | None = None  # only used when running the workflow remotely
    endpoint_timeout: int = 300
    reps: int = 1


class EvaluationRunOutput(BaseModel):
    """
    Output of a single evaluation run.
    """
    workflow_output_file: Path | None
    evaluator_output_files: list[Path]
    workflow_interrupted: bool
