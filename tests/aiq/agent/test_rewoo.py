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

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from langgraph.graph.graph import CompiledGraph

from aiq.agent.base import AgentDecision
from aiq.agent.rewoo_agent.agent import NO_INPUT_ERROR_MESSAGE
from aiq.agent.rewoo_agent.agent import TOOL_NOT_FOUND_ERROR_MESSAGE
from aiq.agent.rewoo_agent.agent import ReWOOAgentGraph
from aiq.agent.rewoo_agent.agent import ReWOOGraphState
from aiq.agent.rewoo_agent.prompt import rewoo_planner_prompt
from aiq.agent.rewoo_agent.prompt import rewoo_solver_prompt
from aiq.agent.rewoo_agent.register import ReWOOAgentWorkflowConfig


async def test_state_schema():
    state = ReWOOGraphState()

    assert isinstance(state.task, str)
    assert isinstance(state.plan, str)
    assert isinstance(state.steps, list)
    assert isinstance(state.intermediate_results, dict)
    assert isinstance(state.result, str)


@pytest.fixture(name='mock_config_rewoo_agent', scope="module")
def mock_config():
    return ReWOOAgentWorkflowConfig(tool_names=["tool_1", "tool_2", "tool_3"], llm_name="llm", verbose=True)


@pytest.fixture(scope="module")
def mock_tool():

    def create_mock_tool(name: str):
        # Create a mock object
        tool = AsyncMock()
        tool.name = name  # Set the name attribute

        # Configure the mock to return its name when called
        tool.__call__ = AsyncMock(return_value=name)

        return tool

    return create_mock_tool


@pytest.fixture(scope="module")
def mock_llm():
    # Create an AsyncMock instance
    llm = AsyncMock()

    # Configure the mock to return the input directly when called
    async def mock_astream(input_data, config=None):
        # Simulate the behavior of returning the input directly
        for key, value in input_data.items():
            yield value

    llm.astream = mock_astream
    return llm


def test_rewoo_init(mock_config_rewoo_agent, mock_llm, mock_tool):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    planner_prompt = rewoo_planner_prompt
    solver_prompt = rewoo_solver_prompt
    agent = ReWOOAgentGraph(llm=mock_llm,
                            planner_prompt=planner_prompt,
                            solver_prompt=solver_prompt,
                            tools=tools,
                            detailed_logs=mock_config_rewoo_agent.verbose)
    assert isinstance(agent, ReWOOAgentGraph)
    assert agent.llm == mock_llm
    assert agent.solver_prompt == solver_prompt
    assert agent.tools == tools
    assert agent.detailed_logs == mock_config_rewoo_agent.verbose


@pytest.fixture(name='mock_rewoo_agent', scope="module")
def mock_agent(mock_config_rewoo_agent, mock_llm, mock_tool):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    planner_prompt = rewoo_planner_prompt
    solver_prompt = rewoo_solver_prompt
    agent = ReWOOAgentGraph(llm=mock_llm,
                            planner_prompt=planner_prompt,
                            solver_prompt=solver_prompt,
                            tools=tools,
                            detailed_logs=mock_config_rewoo_agent.verbose)
    return agent


async def test_build_graph(mock_rewoo_agent):
    graph = await mock_rewoo_agent.build_graph()
    assert isinstance(graph, CompiledGraph)
    assert list(graph.nodes.keys()) == ['__start__', 'planner', 'executor', 'solver']
    assert graph.builder.edges == {('planner', 'executor'), ('__start__', 'planner'), ('solver', '__end__')}
    assert set(graph.builder.branches.get('executor').get('conditional_edge').ends.keys()) == {
        AgentDecision.TOOL, AgentDecision.END
    }


async def test_planner_node_no_input(mock_rewoo_agent):
    state = await mock_rewoo_agent.planner_node(ReWOOGraphState())
    assert state["result"] == NO_INPUT_ERROR_MESSAGE


async def test_conditional_edge_no_input(mock_rewoo_agent):
    # if the state.steps is empty, the conditional_edge should return END
    decision = await mock_rewoo_agent.conditional_edge(ReWOOGraphState())
    assert decision == AgentDecision.END


async def test_conditional_edge_decisions(mock_rewoo_agent):
    mock_state = ReWOOGraphState(task="This is a task",
                                 plan="This is the plan",
                                 steps=[('step1', '#E1', 'Tool A', 'arg1, arg2'),
                                        ('step2', '#E2', 'Tool B', 'arg3, arg4'),
                                        ('step3', '#E3', 'Tool A', 'arg5, arg6')])
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL

    mock_state.intermediate_results = {'#E1': 'result1'}
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL

    # Now all the steps have been executed and generated intermediate results
    mock_state.intermediate_results = {'#E1': 'result1', '#E2': 'result2', '#E3': 'result3'}
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.END


async def test_executor_node_with_not_configured_tool(mock_rewoo_agent):
    tool_not_configured = 'Tool not configured'
    mock_state = ReWOOGraphState(task="This is a task",
                                 plan="This is the plan",
                                 steps=[('step1', '#E1', 'Tool A', 'arg1, arg2'),
                                        ('step2', '#E2', tool_not_configured, 'arg3, arg4')],
                                 intermediate_results={'#E1': 'result1'})
    state = await mock_rewoo_agent.executor_node(mock_state)
    assert isinstance(state, dict)
    configured_tool_names = ['Tool A', 'Tool B']
    assert state["intermediate_results"]["#E2"] == TOOL_NOT_FOUND_ERROR_MESSAGE.format(tool_name=tool_not_configured,
                                                                                       tools=configured_tool_names)


async def test_executor_node_parse_input(mock_rewoo_agent):
    with patch('aiq.agent.rewoo_agent.agent.logger.info') as mock_logger_info:
        # Test with valid JSON as tool input
        mock_state = ReWOOGraphState(
            task="This is a task",
            plan="This is the plan",
            steps=[('step1',
                    '#E1',
                    'Tool A',
                    '{"query": "What is the capital of France?", "input_metadata": {"entities": ["France", "Paris"]}}')
                   ],
            intermediate_results={})
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_info.assert_any_call("Successfully parsed structured tool input")

        # Test with string with single quote as tool input
        mock_state.steps = [('step1', '#E1', 'Tool A', "{'arg1': 'arg_1', 'arg2': 'arg_2'}")]
        mock_state.intermediate_results = {}
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_info.assert_any_call(
            "Successfully parsed structured tool input after replacing single quotes with double quotes")

        # Test with string that cannot be parsed as a JSON as tool input
        mock_state.steps = [('step1', '#E1', 'Tool A', "arg1, arg2")]
        mock_state.intermediate_results = {}
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_info.assert_any_call("Unable to parse structured tool input. Using raw tool input as is.")


async def test_executor_node_should_not_be_invoked_after_all_steps_executed(mock_rewoo_agent):
    mock_state = ReWOOGraphState(task="This is a task",
                                 plan="This is the plan",
                                 steps=[('step1', '#E1', 'Tool A', '{"arg1": "arg_1", "arg2": "arg_2"}'),
                                        ('step2', '#E2', 'Tool B', '{"arg3": "arg_3", "arg4": "arg_4"}'),
                                        ('step3', '#E3', 'Tool A', '{"arg1": "arg_1", "arg2": "arg_2"}')],
                                 intermediate_results={
                                     '#E1': 'result1', '#E2': 'result2', '#E3': 'result3'
                                 })
    # After executing all the steps, the executor_node should not be invoked
    with pytest.raises(RuntimeError):
        await mock_rewoo_agent.executor_node(mock_state)


def test_validate_planner_prompt_no_input():
    mock_prompt = ''
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_planner_prompt(mock_prompt)


def test_validate_planner_prompt_no_tools():
    mock_prompt = '{tools}'
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_planner_prompt(mock_prompt)


def test_validate_planner_prompt_no_tool_names():
    mock_prompt = '{tool_names}'
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_planner_prompt(mock_prompt)


def test_validate_planner_prompt():
    mock_prompt = '{tools} {tool_names}'
    assert ReWOOAgentGraph.validate_planner_prompt(mock_prompt)


def test_validate_solver_prompt_no_input():
    mock_prompt = ''
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_solver_prompt(mock_prompt)


def test_validate_solver_prompt():
    mock_prompt = 'solve the problem'
    assert ReWOOAgentGraph.validate_solver_prompt(mock_prompt)
