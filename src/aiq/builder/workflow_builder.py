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

import dataclasses
import inspect
import logging
import typing
import warnings
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter

from aiq.builder.builder import Builder
from aiq.builder.builder import UserManagerHolder
from aiq.builder.component_utils import build_dependency_sequence
from aiq.builder.context import AIQContext
from aiq.builder.context import AIQContextState
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.builder.function import LambdaFunction
from aiq.builder.function_info import FunctionInfo
from aiq.builder.llm import LLMProviderInfo
from aiq.builder.retriever import RetrieverProviderInfo
from aiq.builder.workflow import Workflow
from aiq.cli.type_registry import GlobalTypeRegistry
from aiq.cli.type_registry import TypeRegistry
from aiq.data_models.component import ComponentGroup
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.component_ref import MemoryRef
from aiq.data_models.component_ref import RetrieverRef
from aiq.data_models.config import AIQConfig
from aiq.data_models.config import GeneralConfig
from aiq.data_models.embedder import EmbedderBaseConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.function_dependencies import FunctionDependencies
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.memory import MemoryBaseConfig
from aiq.data_models.retriever import RetrieverBaseConfig
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from aiq.memory.interfaces import MemoryEditor
from aiq.profiler.decroators.framework_wrapper import chain_wrapped_build_fn
from aiq.profiler.utils import detect_llm_frameworks_in_build_fn

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConfiguredExporter:
    config: TelemetryExporterBaseConfig
    instance: SpanExporter


@dataclasses.dataclass
class ConfiguredFunction:
    config: FunctionBaseConfig
    instance: Function


@dataclasses.dataclass
class ConfiguredLLM:
    config: LLMBaseConfig
    instance: LLMProviderInfo


@dataclasses.dataclass
class ConfiguredEmbedder:
    config: EmbedderBaseConfig
    instance: EmbedderProviderInfo


@dataclasses.dataclass
class ConfiguredMemory:
    config: MemoryBaseConfig
    instance: MemoryEditor


@dataclasses.dataclass
class ConfiguredRetriever:
    config: RetrieverBaseConfig
    instance: RetrieverProviderInfo


# pylint: disable=too-many-public-methods
class WorkflowBuilder(Builder, AbstractAsyncContextManager):

    def __init__(self, *, general_config: GeneralConfig | None = None, registry: TypeRegistry | None = None):

        if general_config is None:
            general_config = GeneralConfig()

        if registry is None:
            registry = GlobalTypeRegistry.get()

        self.general_config = general_config

        self._registry = registry

        self._logging_handlers: dict[str, logging.Handler] = {}
        self._exporters: dict[str, ConfiguredExporter] = {}

        self._functions: dict[str, ConfiguredFunction] = {}
        self._workflow: ConfiguredFunction | None = None

        self._llms: dict[str, ConfiguredLLM] = {}
        self._embedders: dict[str, ConfiguredEmbedder] = {}
        self._memory_clients: dict[str, ConfiguredMemory] = {}
        self._retrievers: dict[str, ConfiguredRetriever] = {}

        self._context_state = AIQContextState.get()

        self._exit_stack: AsyncExitStack | None = None

        # Create a mapping to track function name -> other function names it depends on
        self.function_dependencies: dict[str, FunctionDependencies] = {}
        self.current_function_building: str | None = None

    async def __aenter__(self):

        self._exit_stack = AsyncExitStack()

        # Get the exporter info from the config
        telemetry_config = self.general_config.telemetry

        for key, logging_config in telemetry_config.logging.items():
            # Use the same pattern as tracing, but for logging
            logging_info = self._registry.get_logging_method(type(logging_config))
            handler = await self._exit_stack.enter_async_context(logging_info.build_fn(logging_config, self))

            # Type check
            if not isinstance(handler, logging.Handler):
                raise TypeError(f"Expected a logging.Handler from {key}, got {type(handler)}")

            # Store them in a dict so we can un-register them if needed
            self._logging_handlers[key] = handler

            # Now attach to AgentIQ's root logger
            logging.getLogger().addHandler(handler)

        provider = TracerProvider()
        trace.set_tracer_provider(provider)

        for key, trace_exporter_config in telemetry_config.tracing.items():

            exporter_info = self._registry.get_telemetry_exporter(type(trace_exporter_config))

            instance = await self._exit_stack.enter_async_context(exporter_info.build_fn(trace_exporter_config, self))

            span_processor_instance = BatchSpanProcessor(instance)
            provider.add_span_processor(span_processor_instance)

            self._exporters[key] = ConfiguredExporter(config=trace_exporter_config, instance=instance)

        return self

    async def __aexit__(self, *exc_details):

        assert self._exit_stack is not None, "Exit stack not initialized"

        for _, handler in self._logging_handlers.items():
            logging.getLogger().removeHandler(handler)

        await self._exit_stack.__aexit__(*exc_details)

    def build(self, entry_function: str | None = None) -> Workflow:
        """
        Creates an instance of a workflow object using the added components and the desired entry function.

        Parameters
        ----------
        entry_function : str | None, optional
            The function name to use as the entry point for the created workflow. If None, the entry point will be the
            specified workflow function. By default None

        Returns
        -------
        Workflow
            A created workflow.

        Raises
        ------
        ValueError
            If the workflow has not been set before building.
        """

        if (self._workflow is None):
            raise ValueError("Must set a workflow before building")

        # Build the config from the added objects
        config = AIQConfig(general=self.general_config,
                           functions={
                               k: v.config
                               for k, v in self._functions.items()
                           },
                           workflow=self._workflow.config,
                           llms={
                               k: v.config
                               for k, v in self._llms.items()
                           },
                           embedders={
                               k: v.config
                               for k, v in self._embedders.items()
                           },
                           memory={
                               k: v.config
                               for k, v in self._memory_clients.items()
                           },
                           retrievers={
                               k: v.config
                               for k, v in self._retrievers.items()
                           })

        if (entry_function is None):
            entry_fn_obj = self.get_workflow()
        else:
            entry_fn_obj = self.get_function(entry_function)

        workflow = Workflow.from_entry_fn(config=config,
                                          entry_fn=entry_fn_obj,
                                          functions={
                                              k: v.instance
                                              for k, v in self._functions.items()
                                          },
                                          llms={
                                              k: v.instance
                                              for k, v in self._llms.items()
                                          },
                                          embeddings={
                                              k: v.instance
                                              for k, v in self._embedders.items()
                                          },
                                          memory={
                                              k: v.instance
                                              for k, v in self._memory_clients.items()
                                          },
                                          exporters={
                                              k: v.instance
                                              for k, v in self._exporters.items()
                                          },
                                          retrievers={
                                              k: v.instance
                                              for k, v in self._retrievers.items()
                                          },
                                          context_state=self._context_state)

        return workflow

    def _get_exit_stack(self) -> AsyncExitStack:

        if self._exit_stack is None:
            raise ValueError(
                "Exit stack not initialized. Did you forget to call `async with WorkflowBuilder() as builder`?")

        return self._exit_stack

    async def _build_function(self, name: str, config: FunctionBaseConfig) -> ConfiguredFunction:
        registration = self._registry.get_function(type(config))

        inner_builder = ChildBuilder(self)

        # We need to do this for every function because we don't know
        # Where LLama Index Agents are Instantiated and Settings need to
        # be set before the function is built
        # It's only slower the first time because of the import
        # So we can afford to do this for every function

        llms = {k: v.instance for k, v in self._llms.items()}
        function_frameworks = detect_llm_frameworks_in_build_fn(registration)

        build_fn = chain_wrapped_build_fn(registration.build_fn, llms, function_frameworks)

        # Set the currently building function so the ChildBuilder can track dependencies
        self.current_function_building = config.type
        # Empty set of dependencies for the current function
        self.function_dependencies[config.type] = FunctionDependencies()

        build_result = await self._get_exit_stack().enter_async_context(build_fn(config, inner_builder))

        self.function_dependencies[name] = inner_builder.dependencies

        # If the build result is a function, wrap it in a FunctionInfo
        if inspect.isfunction(build_result):

            build_result = FunctionInfo.from_fn(build_result)

        if (isinstance(build_result, FunctionInfo)):
            # Create the function object
            build_result = LambdaFunction.from_info(config=config, info=build_result)

        if (not isinstance(build_result, Function)):
            raise ValueError("Expected a function, FunctionInfo object, or FunctionBase object to be "
                             f"returned from the function builder. Got {type(build_result)}")

        return ConfiguredFunction(config=config, instance=build_result)

    @typing.override
    async def add_function(self, name: str | FunctionRef, config: FunctionBaseConfig) -> Function:

        if (name in self._functions):
            raise ValueError(f"Function `{name}` already exists in the list of functions")

        build_result = await self._build_function(name=name, config=config)

        self._functions[name] = build_result

        return build_result.instance

    @typing.override
    def get_function(self, name: str | FunctionRef) -> Function:

        if name not in self._functions:
            raise ValueError(f"Function `{name}` not found")

        return self._functions[name].instance

    @typing.override
    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        if name not in self._functions:
            raise ValueError(f"Function `{name}` not found")

        return self._functions[name].config

    @typing.override
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:

        if self._workflow is not None:
            warnings.warn("Overwriting existing workflow")

        build_result = await self._build_function(name="<workflow>", config=config)

        self._workflow = build_result

        return build_result.instance

    @typing.override
    def get_workflow(self) -> Function:

        if self._workflow is None:
            raise ValueError("No workflow set")

        return self._workflow.instance

    @typing.override
    def get_workflow_config(self) -> FunctionBaseConfig:
        if self._workflow is None:
            raise ValueError("No workflow set")

        return self._workflow.config

    @typing.override
    def get_function_dependencies(self, fn_name: str | FunctionRef) -> FunctionDependencies:
        return self.function_dependencies[fn_name]

    @typing.override
    def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str):

        if fn_name not in self._functions:
            raise ValueError(f"Function `{fn_name}` not found in list of functions")

        fn = self._functions[fn_name]

        try:
            # Using the registry, get the tool wrapper for the requested framework
            tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)

            # Wrap in the correct wrapper
            return tool_wrapper_reg.build_fn(fn_name, fn.instance, self)
        except Exception as e:
            logger.error("Error fetching tool `%s`", fn_name, exc_info=True)
            raise e

    @typing.override
    async def add_llm(self, name: str | LLMRef, config: LLMBaseConfig):

        if (name in self._llms):
            raise ValueError(f"LLM `{name}` already exists in the list of LLMs")

        try:
            llm_info = self._registry.get_llm_provider(type(config))

            info_obj = await self._get_exit_stack().enter_async_context(llm_info.build_fn(config, self))

            self._llms[name] = ConfiguredLLM(config=config, instance=info_obj)
        except Exception as e:
            logger.error("Error adding llm `%s` with config `%s`", name, config, exc_info=True)
            raise e

    @typing.override
    async def get_llm(self, llm_name: str | LLMRef, wrapper_type: LLMFrameworkEnum | str):

        if (llm_name not in self._llms):
            raise ValueError(f"LLM `{llm_name}` not found")

        try:
            # Get llm info
            llm_info = self._llms[llm_name]

            # Generate wrapped client from registered client info
            client_info = self._registry.get_llm_client(config_type=type(llm_info.config), wrapper_type=wrapper_type)

            client = await self._get_exit_stack().enter_async_context(client_info.build_fn(llm_info.config, self))

            # Return a frameworks specific client
            return client
        except Exception as e:
            logger.error("Error getting llm `%s` with wrapper `%s`", llm_name, wrapper_type, exc_info=True)
            raise e

    @typing.override
    def get_llm_config(self, llm_name: str | LLMRef) -> LLMBaseConfig:

        if llm_name not in self._llms:
            raise ValueError(f"LLM `{llm_name}` not found")

        # Return the tool configuration object
        return self._llms[llm_name].config

    @typing.override
    async def add_embedder(self, name: str | EmbedderRef, config: EmbedderBaseConfig):

        if (name in self._embedders):
            raise ValueError(f"Embedder `{name}` already exists in the list of embedders")

        try:
            embedder_info = self._registry.get_embedder_provider(type(config))

            info_obj = await self._get_exit_stack().enter_async_context(embedder_info.build_fn(config, self))

            self._embedders[name] = ConfiguredEmbedder(config=config, instance=info_obj)
        except Exception as e:
            logger.error("Error adding embedder `%s` with config `%s`", name, config, exc_info=True)

            raise e

    @typing.override
    async def get_embedder(self, embedder_name: str | EmbedderRef, wrapper_type: LLMFrameworkEnum | str):

        if (embedder_name not in self._embedders):
            raise ValueError(f"Embedder `{embedder_name}` not found")

        try:
            # Get embedder info
            embedder_info = self._embedders[embedder_name]

            # Generate wrapped client from registered client info
            client_info = self._registry.get_embedder_client(config_type=type(embedder_info.config),
                                                             wrapper_type=wrapper_type)
            client = await self._get_exit_stack().enter_async_context(client_info.build_fn(embedder_info.config, self))

            # Return a frameworks specific client
            return client
        except Exception as e:
            logger.error("Error getting embedder `%s` with wrapper `%s`", embedder_name, wrapper_type, exc_info=True)
            raise e

    @typing.override
    def get_embedder_config(self, embedder_name: str | EmbedderRef) -> EmbedderBaseConfig:

        if embedder_name not in self._embedders:
            raise ValueError(f"Tool `{embedder_name}` not found")

        # Return the tool configuration object
        return self._embedders[embedder_name].config

    @typing.override
    async def add_memory_client(self, name: str | MemoryRef, config: MemoryBaseConfig) -> MemoryEditor:

        if (name in self._memory_clients):
            raise ValueError(f"Memory `{name}` already exists in the list of memories")

        memory_info = self._registry.get_memory(type(config))

        info_obj = await self._get_exit_stack().enter_async_context(memory_info.build_fn(config, self))

        self._memory_clients[name] = ConfiguredMemory(config=config, instance=info_obj)

        return info_obj

    @typing.override
    def get_memory_client(self, memory_name: str | MemoryRef) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        if memory_name not in self._memory_clients:
            raise ValueError(f"Memory `{memory_name}` not found")

        return self._memory_clients[memory_name].instance

    @typing.override
    def get_memory_client_config(self, memory_name: str | MemoryRef) -> MemoryBaseConfig:

        if memory_name not in self._memory_clients:
            raise ValueError(f"Memory `{memory_name}` not found")

        # Return the tool configuration object
        return self._memory_clients[memory_name].config

    @typing.override
    async def add_retriever(self, name: str | RetrieverRef, config: RetrieverBaseConfig):

        if (name in self._retrievers):
            raise ValueError(f"Retriever '{name}' already exists in the list of retrievers")

        try:
            retriever_info = self._registry.get_retriever_provider(type(config))

            info_obj = await self._get_exit_stack().enter_async_context(retriever_info.build_fn(config, self))

            self._retrievers[name] = ConfiguredRetriever(config=config, instance=info_obj)

        except Exception as e:
            logger.error("Error adding retriever `%s` with config `%s`", name, config, exc_info=True)

            raise e

        # return info_obj

    @typing.override
    async def get_retriever(self,
                            retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str | None = None):

        if retriever_name not in self._retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")

        try:
            # Get retriever info
            retriever_info = self._retrievers[retriever_name]

            # Generate wrapped client from registered client info
            client_info = self._registry.get_retriever_client(config_type=type(retriever_info.config),
                                                              wrapper_type=wrapper_type)

            client = await self._get_exit_stack().enter_async_context(client_info.build_fn(retriever_info.config, self))

            # Return a frameworks specific client
            return client
        except Exception as e:
            logger.error("Error getting retriever `%s` with wrapper `%s`", retriever_name, wrapper_type, exc_info=True)
            raise e

    @typing.override
    async def get_retriever_config(self, retriever_name: str | RetrieverRef) -> RetrieverBaseConfig:

        if retriever_name not in self._retrievers:
            raise ValueError(f"Retriever `{retriever_name}` not found")

        return self._retrievers[retriever_name].config

    @typing.override
    def get_user_manager(self):
        return UserManagerHolder(context=AIQContext(self._context_state))

    async def populate_builder(self, config: AIQConfig):

        # Generate the build sequence
        build_sequence = build_dependency_sequence(config)

        # Loop over all objects and add to the workflow builder
        for component_instance in build_sequence:
            # Instantiate a the llm
            if component_instance.component_group == ComponentGroup.LLMS:
                await self.add_llm(component_instance.name, component_instance.config)
            # Instantiate a the embedder
            elif component_instance.component_group == ComponentGroup.EMBEDDERS:
                await self.add_embedder(component_instance.name, component_instance.config)
            # Instantiate a memory client
            elif component_instance.component_group == ComponentGroup.MEMORY:
                await self.add_memory_client(component_instance.name, component_instance.config)
            # Instantiate a retriever client
            elif component_instance.component_group == ComponentGroup.RETRIEVERS:
                await self.add_retriever(component_instance.name, component_instance.config)
            # Instantiate a function
            elif component_instance.component_group == ComponentGroup.FUNCTIONS:
                # If the function is the root, set it as the workflow later
                if (not component_instance.is_root):
                    await self.add_function(component_instance.name, component_instance.config)
            else:
                raise ValueError(f"Unknown component group {component_instance.component_group}")

        # Instantiate the workflow
        await self.set_workflow(config.workflow)

    @classmethod
    @asynccontextmanager
    async def from_config(cls, config: AIQConfig):

        async with cls(general_config=config.general) as builder:
            await builder.populate_builder(config)
            yield builder


class ChildBuilder(Builder):

    def __init__(self, workflow_builder: WorkflowBuilder) -> None:

        self._workflow_builder = workflow_builder

        self._dependencies = FunctionDependencies()

    @property
    def dependencies(self) -> FunctionDependencies:
        return self._dependencies

    @typing.override
    async def add_function(self, name: str, config: FunctionBaseConfig) -> Function:
        return await self._workflow_builder.add_function(name, config)

    @typing.override
    def get_function(self, name: str) -> Function:
        # If a function tries to get another function, we assume it uses it
        fn = self._workflow_builder.get_function(name)

        self._dependencies.add_function(name)

        return fn

    @typing.override
    def get_function_config(self, name: str) -> FunctionBaseConfig:
        return self._workflow_builder.get_function_config(name)

    @typing.override
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        return await self._workflow_builder.set_workflow(config)

    @typing.override
    def get_workflow(self) -> Function:
        return self._workflow_builder.get_workflow()

    @typing.override
    def get_workflow_config(self) -> FunctionBaseConfig:
        return self._workflow_builder.get_workflow_config()

    @typing.override
    def get_tool(self, fn_name: str, wrapper_type: LLMFrameworkEnum | str):
        # If a function tries to get another function as a tool, we assume it uses it
        fn = self._workflow_builder.get_tool(fn_name, wrapper_type)

        self._dependencies.add_function(fn_name)

        return fn

    @typing.override
    async def add_llm(self, name: str, config: LLMBaseConfig):
        return await self._workflow_builder.add_llm(name, config)

    @typing.override
    async def get_llm(self, llm_name: str, wrapper_type: LLMFrameworkEnum | str):
        llm = await self._workflow_builder.get_llm(llm_name, wrapper_type)

        self._dependencies.add_llm(llm_name)

        return llm

    @typing.override
    def get_llm_config(self, llm_name: str) -> LLMBaseConfig:
        return self._workflow_builder.get_llm_config(llm_name)

    @typing.override
    async def add_embedder(self, name: str, config: EmbedderBaseConfig):
        return await self._workflow_builder.add_embedder(name, config)

    @typing.override
    async def get_embedder(self, embedder_name: str, wrapper_type: LLMFrameworkEnum | str):
        embedder = await self._workflow_builder.get_embedder(embedder_name, wrapper_type)

        self._dependencies.add_embedder(embedder_name)

        return embedder

    @typing.override
    def get_embedder_config(self, embedder_name: str) -> EmbedderBaseConfig:
        return self._workflow_builder.get_embedder_config(embedder_name)

    @typing.override
    async def add_memory_client(self, name: str, config: MemoryBaseConfig) -> MemoryEditor:
        return await self._workflow_builder.add_memory_client(name, config)

    @typing.override
    def get_memory_client(self, memory_name: str) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        memory_client = self._workflow_builder.get_memory_client(memory_name)

        self._dependencies.add_memory_client(memory_name)

        return memory_client

    @typing.override
    def get_memory_client_config(self, memory_name: str) -> MemoryBaseConfig:
        return self._workflow_builder.get_memory_client_config(memory_name=memory_name)

    @typing.override
    async def add_retriever(self, name: str, config: RetrieverBaseConfig):
        return await self._workflow_builder.add_retriever(name, config)

    @typing.override
    async def get_retriever(self, retriever_name: str, wrapper_type: LLMFrameworkEnum | str | None = None):
        if not wrapper_type:
            return await self._workflow_builder.get_retriever(retriever_name=retriever_name)
        return await self._workflow_builder.get_retriever(retriever_name=retriever_name, wrapper_type=wrapper_type)

    @typing.override
    async def get_retriever_config(self, retriever_name: str) -> RetrieverBaseConfig:
        return await self._workflow_builder.get_retriever_config(retriever_name=retriever_name)

    @typing.override
    def get_user_manager(self) -> UserManagerHolder:
        return self._workflow_builder.get_user_manager()

    @typing.override
    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        return self._workflow_builder.get_function_dependencies(fn_name)
