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

# pylint disable=ungrouped-imports

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager as AsyncContextManager
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)

_library_instrumented = {
    "langchain": False,
    "crewai": False,
    "semantic_kernel": False,
    "agno": False,
}

callback_handler_var: ContextVar[Any | None] = ContextVar("callback_handler_var", default=None)


def set_framework_profiler_handler(
    workflow_llms: dict = None,
    frameworks: list[LLMFrameworkEnum] = None,
) -> Callable[[Callable[..., AsyncContextManager[Any]]], Callable[..., AsyncContextManager[Any]]]:
    """
    Decorator that wraps an async context manager function to set up framework-specific profiling.
    """

    def decorator(func: Callable[..., AsyncContextManager[Any]]) -> Callable[..., AsyncContextManager[Any]]:

        @functools.wraps(func)
        @asynccontextmanager
        async def wrapper(workflow_config, builder):

            if LLMFrameworkEnum.LANGCHAIN in frameworks and not _library_instrumented["langchain"]:
                from langchain_core.tracers.context import register_configure_hook

                from aiq.profiler.callbacks.langchain_callback_handler import LangchainProfilerHandler

                handler = LangchainProfilerHandler()
                callback_handler_var.set(handler)
                register_configure_hook(callback_handler_var, inheritable=True)
                _library_instrumented["langchain"] = True
                logger.info("Langchain callback handler registered")

            if LLMFrameworkEnum.LLAMA_INDEX in frameworks:
                from llama_index.core import Settings
                from llama_index.core.callbacks import CallbackManager

                from aiq.profiler.callbacks.llama_index_callback_handler import LlamaIndexProfilerHandler

                handler = LlamaIndexProfilerHandler()
                Settings.callback_manager = CallbackManager([handler])
                logger.info("LlamaIndex callback handler registered")

            if LLMFrameworkEnum.CREWAI in frameworks and not _library_instrumented["crewai"]:
                from aiq.plugins.crewai.crewai_callback_handler import \
                    CrewAIProfilerHandler  # pylint: disable=ungrouped-imports,line-too-long  # noqa E501

                handler = CrewAIProfilerHandler()
                handler.instrument()
                _library_instrumented["crewai"] = True
                logger.info("CrewAI callback handler registered")

            if LLMFrameworkEnum.SEMANTIC_KERNEL in frameworks and not _library_instrumented["semantic_kernel"]:
                from aiq.profiler.callbacks.semantic_kernel_callback_handler import SemanticKernelProfilerHandler

                handler = SemanticKernelProfilerHandler(workflow_llms=workflow_llms)
                handler.instrument()
                _library_instrumented["semantic_kernel"] = True
                logger.info("SemanticKernel callback handler registered")

            if LLMFrameworkEnum.AGNO in frameworks and not _library_instrumented["agno"]:
                from aiq.profiler.callbacks.agno_callback_handler import AgnoProfilerHandler

                handler = AgnoProfilerHandler()
                handler.instrument()
                _library_instrumented["agno"] = True
                logger.info("Agno callback handler registered")

            # IMPORTANT: actually call the wrapped function as an async context manager
            async with func(workflow_config, builder) as result:
                yield result

        return wrapper

    return decorator


def chain_wrapped_build_fn(
    original_build_fn: Callable[..., AsyncContextManager],
    workflow_llms: dict,
    function_frameworks: list[LLMFrameworkEnum],
) -> Callable[..., AsyncContextManager]:
    """
    Convert an original build function into an async context manager that
    wraps it with a single call to set_framework_profiler_handler, passing
    all frameworks at once.
    """

    # Define a base async context manager that simply calls the original build function.
    @asynccontextmanager
    async def base_fn(*args, **kwargs):
        async with original_build_fn(*args, **kwargs) as w:
            yield w

    # Instead of wrapping iteratively, we now call the decorator once,
    # passing the entire list of frameworks along with the workflow_llms.
    wrapped_fn = set_framework_profiler_handler(workflow_llms, function_frameworks)(base_fn)
    return wrapped_fn
