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
from abc import abstractmethod

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

from .base import AgentDecision
from .base import BaseAgent

log = logging.getLogger(__name__)


class DualNodeAgent(BaseAgent):

    def __init__(self,
                 llm: BaseChatModel,
                 tools: list[BaseTool],
                 callbacks: list[AsyncCallbackHandler] = None,
                 detailed_logs: bool = False):
        super().__init__(llm=llm, tools=tools, callbacks=callbacks, detailed_logs=detailed_logs)

    @abstractmethod
    async def agent_node(self, state: BaseModel) -> BaseModel:
        pass

    @abstractmethod
    async def tool_node(self, state: BaseModel) -> BaseModel:
        pass

    @abstractmethod
    async def conditional_edge(self, state: BaseModel) -> str:
        pass

    async def _build_graph(self, state_schema) -> CompiledGraph:
        log.debug("Building and compiling the Agent Graph")

        graph = StateGraph(state_schema)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tool", self.tool_node)
        graph.add_edge("tool", "agent")

        conditional_edge_possible_outputs = {AgentDecision.TOOL: "tool", AgentDecision.END: "__end__"}
        graph.add_conditional_edges("agent", self.conditional_edge, conditional_edge_possible_outputs)

        graph.set_entry_point("agent")
        self.graph = graph.compile()

        return self.graph
