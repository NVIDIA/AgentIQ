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

PLANNER_SYSTEM_PROMPT = """
For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

You may ask the human to the following tools:

{tools}

The tools should be one of the following: [{tool_names}]

You are not required to use all the tools listed. Choose only the ones that best fit the needs of each plan step.

Your output must be a JSON array where each element represents one planning step. Each step must be an object with exactly two keys:

1. "plan": A string that describes in detail the action or reasoning for that step.

2. "evidence": An object representing the external tool call associated with that plan step. This object must have the following keys:

   -"variable": A string that identifies the evidence variable (e.g., "#E1", "#E2", etc.). The numbering should be sequential based on the order of steps.

   -"tool": A string specifying the name of the external tool used.

   -"tool_input": The input to the tool. This can be a string, array, or object, depending on the requirements of the tool.

Do not include any additional keys or characters in your output, and do not wrap your response with markdown formatting. Your output must be strictly valid JSON.

Important instructions:

Do not output any additional text, comments, or markdown formatting.

Do not include any explanation or reasoning text outside of the JSON array.

The output must be a valid JSON array that can be parsed directly.

Here is an example of how a valid JSON output should look:

[
  \'{{
    "plan": "Calculate the result of 2023 minus 25.",
    "evidence": \'{{
      "variable": "#E1",
      "tool": "calculator_subtract",
      "tool_input": [2023, 25]
    }}\'
  }}\',
  \'{{
    "plan": "Retrieve the year represented by the result stored in #E1.",
    "evidence": \'{{
      "variable": "#E2",
      "tool": "haystack_chitchat_agent",
      "tool_input": "Response with the result number contained in #E1"
    }}\'
  }}\',
  \'{{
    "plan": "Search for the CEO of Golden State Warriors in the year stored in #E2.",
    "evidence": \'{{
      "variable": "#E3",
      "tool": "internet_search",
      "tool_input": "Who was the CEO of Golden State Warriors in the year #E2?"
    }}\'
  }}\'
]

Begin!
"""

PLANNER_USER_PROMPT = """
task: {task}
"""

rewoo_planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])

SOLVER_SYSTEM_PROMPT = """
Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

"""
SOLVER_USER_PROMPT = """
plan: {plan}
task: {task}

Response:
"""

rewoo_solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])
