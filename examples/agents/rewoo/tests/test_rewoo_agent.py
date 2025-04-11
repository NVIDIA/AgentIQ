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

import logging
import os

import pytest

from aiq.runtime.loader import load_workflow

logger = logging.getLogger(__name__)


async def _test_workflow(config_file: str, question: str, answer: str):
    async with load_workflow(config_file) as workflow:

        async with workflow.run(question) as runner:
            result = await runner.result(to_type=str)

        result = result.lower()
        assert answer in result


@pytest.mark.e2e
async def test_full_workflow():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_file = os.path.join(current_dir, "../configs", "config.yml")

    test_cases = [
        {
            "question":
                "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?",
            "answer":
                "athens"
        },
        {
            "question": "Which U.S. historical event occurred in the year obtained by multiplying 48 and 37?",
            "answer": "declaration of independence"
        },
        {
            "question": "Which country hosted the FIFA World Cup in the year obtained by dividing 6054 by 3?",
            "answer": "russia"
        },
        {
            "question": "Which renowned physicist was born in the year resulting from subtracting 21 from 1900?",
            "answer": "albert einstein"
        },
        {
            "question":
                "Which city hosted the Summer Olympics in the year obtained by subtracting 4 from the larger number"
                "between 2008 and 2012?",
            "answer": "beijing"
        }
    ]

    for test_case in test_cases:
        await _test_workflow(config_file, test_case["question"], test_case["answer"])
