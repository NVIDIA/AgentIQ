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
from textwrap import dedent

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AgnoPersonalFinanceFunctionConfig(FunctionBaseConfig, name="agno_personal_finance"):
    llm_name: LLMRef
    serp_api_tool: FunctionRef
    api_key: str | None = None


@register_function(config_type=AgnoPersonalFinanceFunctionConfig, framework_wrappers=[LLMFrameworkEnum.AGNO])
async def agno_personal_finance_function(config: AgnoPersonalFinanceFunctionConfig, builder: Builder):
    """
    Create a financial planning function that uses a researcher and planner to generate
    personalized financial plans.

    Parameters
    ----------
    config : AgnoPersonalFinanceFunctionConfig
        Configuration for the financial planning function
    builder : Builder
        The AgentIQ builder instance

    Returns
    -------
    A FunctionInfo object that can generate personalized financial plans
    """

    from agno.agent import Agent
    if (not config.api_key):
        config.api_key = os.getenv("NVIDIA_API_KEY")

    if not config.api_key:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    # Get the language model
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AGNO)

    # Get the search tool
    search_tool = builder.get_tool(fn_name=config.serp_api_tool, wrapper_type=LLMFrameworkEnum.AGNO)

    # Create researcher agent
    researcher = Agent(
        name="Researcher",
        role="Searches for financial advice, investment opportunities, and savings strategies "
        "based on user preferences",
        model=llm,
        description=dedent("""\
        You are a world-class financial researcher. Given a user's financial goals and current financial situation,
        generate a list of search terms for finding relevant financial advice, investment opportunities, and savings
        strategies. Then search the web for each term, analyze the results, and return the 10 most relevant results.
        """),
        instructions=[
            "Given a user's financial goals and current financial situation, first generate a list of 3 search terms "
            "related to those goals.",
            "For each search term, use search_google function to search the web. Always use exactly 5 as the "
            "num_results parameter.",
            "The search_google function requires a specific format: search_google(query='your query', num_results=5). "
            "Use this format precisely.",
            "From the results of all searches, return the 10 most relevant results to the user's preferences.",
            "Remember: the quality of the results is important.",
        ],
        tools=[search_tool],
        add_datetime_to_instructions=True,
    )

    # Create planner agent
    planner = Agent(
        name="Planner",
        role="Generates a personalized financial plan based on user preferences and research results",
        model=llm,
        description=dedent("""\
        You are a senior financial planner. Given a user's financial goals, current financial situation, and a list of
        research results, your goal is to generate a personalized financial plan that meets the user's needs and
        preferences.
        """),
        instructions=[
            "Given a user's financial goals, current financial situation, and a list of research results, ",
            "generate a personalized financial plan that includes suggested budgets, investment plans, ",
            "and savings strategies. Ensure the plan is well-structured, informative, and engaging.",
            "Ensure you provide a nuanced and balanced plan, quoting facts where possible.",
            "Remember: the quality of the plan is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
            "Do not use any search functions directly; use only the information provided to create your plan.",
        ],
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=3,
    )

    # Create a function that uses the researcher and planner to generate a personalized financial plan
    async def _arun(inputs: str) -> str:
        """
        State your financial goals and current situation, and the planner will generate a personalized financial plan.
        Args:
            inputs : user query
        """
        try:
            # First, use the researcher to gather relevant financial information
            researcher_response = await researcher.arun(inputs, stream=False)
            logger.debug("Research results: \n %s", researcher_response)

            # Combine the original input with the research results for the planner
            planner_input = f"""
                User query: {inputs}

                Research results:
                {researcher_response}

                Based on the above information, please create a personalized financial plan.
                """

            # Now run the planner with the research results
            planner_response = await planner.arun(planner_input, stream=False)
            logger.info("response from agno_personal_finance: \n %s", planner_response)

            # Extract content from RunResponse
            planner_content = (planner_response.content
                               if hasattr(planner_response, 'content') else str(planner_response))

            # Return the content as a string
            return planner_content
        except Exception as e:
            logger.error(f"Error in agno_personal_finance function: {str(e)}")
            return f"Sorry, I encountered an error while generating your financial plan: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="extract relevant personal finance data per user input query")
