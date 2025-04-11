# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry.sdk.trace.export import SpanExporter

from aiq.builder.context import AIQContextState
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.builder.function import Function
from aiq.builder.function_base import FunctionBase
from aiq.builder.function_base import InputT
from aiq.builder.function_base import SingleOutputT
from aiq.builder.function_base import StreamingOutputT
from aiq.builder.llm import LLMProviderInfo
from aiq.builder.retriever import RetrieverProviderInfo
from aiq.data_models.config import AIQConfig
from aiq.memory.interfaces import MemoryEditor
from aiq.runtime.runner import AIQRunner

callback_handler_var: ContextVar[Any | None] = ContextVar("callback_handler_var", default=None)


class Workflow(FunctionBase[InputT, StreamingOutputT, SingleOutputT]):

    def __init__(self,
                 *,
                 config: AIQConfig,
                 entry_fn: Function[InputT, StreamingOutputT, SingleOutputT],
                 functions: dict[str, Function] | None = None,
                 llms: dict[str, LLMProviderInfo] | None = None,
                 embeddings: dict[str, EmbedderProviderInfo] | None = None,
                 memory: dict[str, MemoryEditor] | None = None,
                 exporters: dict[str, SpanExporter] | None = None,
                 retrievers: dict[str | None, RetrieverProviderInfo] | None = None,
                 context_state: AIQContextState):

        super().__init__(input_schema=entry_fn.input_schema,
                         streaming_output_schema=entry_fn.streaming_output_schema,
                         single_output_schema=entry_fn.single_output_schema)

        self.config = config
        self.functions = functions or {}
        self.llms = llms or {}
        self.embeddings = embeddings or {}
        self.memory = memory or {}
        self.retrievers = retrievers or {}

        self._entry_fn = entry_fn

        self._context_state = context_state

        self._exporters = exporters or {}

    @property
    def has_streaming_output(self) -> bool:

        return self._entry_fn.has_streaming_output

    @property
    def has_single_output(self) -> bool:

        return self._entry_fn.has_single_output

    @asynccontextmanager
    async def run(self, message: InputT, entry_fn: str | None = None):
        """
        Called each time we start a new workflow run. We'll create
        a new top-level workflow span here.
        """

        if entry_fn:
            # check if entry_fn is in the keys of self.functions
            if entry_fn not in self.functions:
                raise ValueError(f"Entry function {entry_fn} not found in functions")
            entry_fn = self.functions[entry_fn]
        else:
            entry_fn = self._entry_fn

        async with AIQRunner(input_message=message, entry_fn=entry_fn, context_state=self._context_state) as runner:

            # The caller can `yield runner` so they can do `runner.result()` or `runner.result_stream()`
            yield runner

    async def result_with_steps(self, message: InputT, to_type: type | None = None):

        async with self.run(message) as runner:

            from aiq.eval.runtime_event_subscriber import pull_intermediate

            # Start the intermediate stream
            pull_done, intermediate_steps = pull_intermediate()

            # Wait on the result
            result = await runner.result(to_type=to_type)

            await pull_done.wait()

            return result, intermediate_steps

    @staticmethod
    def from_entry_fn(*,
                      config: AIQConfig,
                      entry_fn: Function[InputT, StreamingOutputT, SingleOutputT],
                      functions: dict[str, Function] | None = None,
                      llms: dict[str, LLMProviderInfo] | None = None,
                      embeddings: dict[str, EmbedderProviderInfo] | None = None,
                      memory: dict[str, MemoryEditor] | None = None,
                      exporters: dict[str, SpanExporter] | None = None,
                      retrievers: dict[str | None, RetrieverProviderInfo] | None = None,
                      context_state: AIQContextState) -> 'Workflow[InputT, StreamingOutputT, SingleOutputT]':

        input_type: type = entry_fn.input_type
        streaming_output_type = entry_fn.streaming_output_type
        single_output_type = entry_fn.single_output_type

        class WorkflowImpl(Workflow[input_type, streaming_output_type, single_output_type]):
            pass

        return WorkflowImpl(config=config,
                            entry_fn=entry_fn,
                            functions=functions,
                            llms=llms,
                            embeddings=embeddings,
                            memory=memory,
                            exporters=exporters,
                            retrievers=retrievers,
                            context_state=context_state)
