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

from pydantic import Field

from aiq.data_models.front_end import FrontEndBaseConfig


class ConsoleFrontEndConfig(FrontEndBaseConfig, name="console"):
    """
    A front end that allows an AgentIQ workflow to be run from the console.
    """

    input_query: list[str] | None = Field(default=None,
                                          alias="input",
                                          description="A single input to submit the the workflow.")

    input_file: Path | None = Field(default=None,
                                    description="Path to a json file of inputs to submit to the workflow.")

    entry_fn: str | None = Field(default=None,
                                 description="Custom function for run. If none, Workflow function will be used")
