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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class ReWOOAgentWorkflowConfig(FunctionBaseConfig, name="rewoo_agent"):
    """
    Defines an AgentIQ function that uses a ReWOO Agent performs reasoning inbetween tool calls, and utilizes the tool
    names and descriptions to select the optimal tool.
    """

    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="The list of tools to provide to the rewoo agent.")
    llm_name: LLMRef = Field(description="The LLM model to use with the rewoo agent.")
    verbose: bool = Field(default=False, description="Set the verbosity of the rewoo agent's logging.")
    include_tool_input_schema_in_tool_description: bool = Field(
        default=True, description="Specify inclusion of tool input schemas in the prompt.")
    max_iterations: int = Field(default=15, description="Number of tool calls before stoping the rewoo agent.")
    description: str = Field(default="ReWOO Agent Workflow", description="The description of this functions use.")
    planner_prompt: str | None = Field(
        default=None,
        description="Provides the PLANNER_PROMPT to use with the agent")  # defaults to PLANNER_PROMPT in prompt.py
    solver_prompt: str | None = Field(
        default=None,
        description="Provides the SOLVER_PROMPT to use with the agent")  # defaults to SOLVER_PROMPT in prompt.py
    max_history: int = Field(default=15, description="Maximum number of messages to keep in the conversation history.")
    use_openai_api: bool = Field(default=False,
                                 description=("Use OpenAI API for the input/output types to the function. "
                                              "If False, strings will be used."))
    additional_instructions: str | None = Field(
        default=None, description="Additional instructions to provide to the agent in addition to the base prompt.")


@register_function(config_type=ReWOOAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def ReWOO_agent_workflow(config: ReWOOAgentWorkflowConfig, builder: Builder):
    from langchain.schema import BaseMessage
    from langchain_core.messages import trim_messages
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph.graph import CompiledGraph

    from aiq.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from aiq.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    from .agent import ReWOOAgentGraph
    from .agent import ReWOOGraphState
    from .prompt import rewoo_planner_prompt
    from .prompt import rewoo_solver_prompt

    # the ReWOO Agent prompt comes from prompt.py, and can be customized there or via config option planner_prompt and
    # solver_prompt.
    if config.planner_prompt:
        planner_prompt = config.planner_prompt
        if config.additional_instructions:
            planner_prompt += f"{config.additional_instructions}"
        valid = ReWOOAgentGraph.validate_planner_prompt(config.planner_prompt)
        if not valid:
            logger.exception("Invalid planner_prompt")
            raise ValueError("Invalid planner_prompt")
        planner_prompt = ChatPromptTemplate([("system", config.planner_prompt), ("user", PLANNER_USER_PROMPT)])
    else:
        planner_prompt = rewoo_planner_prompt

    if config.solver_prompt:
        solver_prompt = config.solver_prompt
        if config.additional_instructions:
            solver_prompt += f"{config.additional_instructions}"
        valid = ReWOOAgentGraph.validate_solver_prompt(config.solver_prompt)
        if not valid:
            logger.exception("Invalid solver_prompt")
            raise ValueError("Invalid solver_prompt")
        solver_prompt = ChatPromptTemplate([("system", config.solver_prompt), ("user", SOLVER_USER_PROMPT)])
    else:
        solver_prompt = rewoo_solver_prompt

    # we can choose an LLM for the ReWOO agent in the config file
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # the agent can run any installed tool, simply install the tool and add it to the config file
    # the sample tool provided can easily be copied or changed
    tools = builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # construct the ReWOO Agent Graph from the configured llm, prompt, and tools
    graph: CompiledGraph = await ReWOOAgentGraph(llm=llm,
                                                 planner_prompt=planner_prompt,
                                                 solver_prompt=solver_prompt,
                                                 tools=tools,
                                                 use_tool_schema=config.include_tool_input_schema_in_tool_description,
                                                 detailed_logs=config.verbose).build_graph()

    async def _response_fn(input_message: AIQChatRequest) -> AIQChatResponse:
        try:
            # initialize the starting state with the user query
            messages: list[BaseMessage] = trim_messages(messages=[m.model_dump() for m in input_message.messages],
                                                        max_tokens=config.max_history,
                                                        strategy="last",
                                                        token_counter=len,
                                                        start_on="human",
                                                        include_system=True)
            task = messages[0].content
            state = ReWOOGraphState(task=task)

            # run the ReWOO Agent Graph
            state = await graph.ainvoke(state)

            # get and return the output from the state
            state = ReWOOGraphState(**state)
            output_message = state.result  # pylint: disable=E1136
            return AIQChatResponse.from_string(output_message)

        except Exception as ex:
            logger.exception("ReWOO Agent failed with exception: %s", ex, exc_info=ex)
            # here, we can implement custom error messages
            if config.verbose:
                return AIQChatResponse.from_string(str(ex))
            return AIQChatResponse.from_string("I seem to be having a problem.")

    if (config.use_openai_api):
        yield FunctionInfo.from_fn(_response_fn, description=config.description)

    else:

        async def _str_api_fn(input_message: str) -> str:
            oai_input = GlobalTypeConverter.get().convert(input_message, to_type=AIQChatRequest)
            oai_output = await _response_fn(oai_input)

            return GlobalTypeConverter.get().convert(oai_output, to_type=str)

        yield FunctionInfo.from_fn(_str_api_fn, description=config.description)
