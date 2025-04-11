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

import importlib
import importlib.resources
import inspect
import logging
from pathlib import Path

import pytest
from aiq_simple_calculator.register import DivisionToolConfig
from aiq_simple_calculator.register import InequalityToolConfig
from aiq_simple_calculator.register import MultiplyToolConfig

from aiq.runtime.loader import load_workflow

logger = logging.getLogger(__name__)


@pytest.mark.e2e
async def test_inequality_tool_workflow():

    package_name = inspect.getmodule(InequalityToolConfig).__package__

    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "config.yml").absolute()

    async with load_workflow(config_file) as workflow:

        async with workflow.run("Is 8 greater than 15?") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert "no" in result


@pytest.mark.e2e
async def test_multiply_tool_workflow():

    package_name = inspect.getmodule(MultiplyToolConfig).__package__

    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "config.yml").absolute()

    async with load_workflow(config_file) as workflow:

        async with workflow.run("What is the product of 2 * 4?") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert "8" in result


@pytest.mark.e2e
async def test_division_tool_workflow():

    package_name = inspect.getmodule(DivisionToolConfig).__package__

    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "config.yml").absolute()

    async with load_workflow(config_file) as workflow:

        async with workflow.run("What is 8 divided by 2?") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert "4" in result


@pytest.mark.e2e
async def test_custom_entry_points():
    """
    Test the custom entry points for the simple calculator workflow.
    """

    package_name = inspect.getmodule(DivisionToolConfig).__package__

    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "config.yml").absolute()

    async with load_workflow(config_file) as workflow:

        async with workflow.run("4 and 5", "calculator_inequality") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert result == "first number 4 is less than the second number 5"

        async with workflow.run("4 and 5", "calculator_multiply") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert result == "the product of 4 * 5 is 20"


@pytest.mark.e2e
async def test_custom_entry_point_invalid():
    """
    Tests that provided a function name that does not exist in the workflow
    raises a ValueError.
    """

    package_name = inspect.getmodule(DivisionToolConfig).__package__

    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "config.yml").absolute()

    async with load_workflow(config_file) as workflow:

        with pytest.raises(ValueError, match="Entry function foo not found in functions"):
            async with workflow.run("4 and 5", "foo") as runner:
                _ = await runner.result(to_type=str)
