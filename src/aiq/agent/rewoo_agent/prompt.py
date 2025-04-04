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

# flake8: noqa
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder

SYSTEM_PROMPT = """
For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

You may ask the human to the following tools:

{tools}

The tools should be one of the following: [{tool_names}]

Please not that you don't need to use all the tools. You can use any of the tools you want.
Make sure to follow the pattern: #E = tool_name[tool_input]

For example,
Task: Determine which is greater: the result of subtracting 13 from 25, or the result of dividing 132 by 12.

Plan: Calculate the result of 25 minus 13.
#E1 = calculator_subtract[25, 13]

Plan: Calculate the result of 132 divided by 12.
#E2 = calculator_divide[132, 12]

Plan: Compare the results from steps 1 and 2 to determine which is greater.
#E3 = calculator_inequality[#E1, ">", #E2]

Begin!
Describe your plans with rich details. Each Plan should be followed by only one #E.
"""
USER_PROMPT = """
task: {task}
"""

rewoo_plan_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"

solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
               retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
               contain irrelevant information.

               {plan}

                Now solve the question or task according to provided Evidence above. Respond with the answer
                directly with no extra words.

                Task: {task}
                Response:"""

# This is the prompt - (ReWOO Agent prompt)
rewoo_agent_prompt = ChatPromptTemplate([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])
