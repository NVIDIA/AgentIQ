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

import functools
import inspect
import uuid
from typing import Any

from pydantic import BaseModel

from aiq.builder.context import AIQContext
from aiq.builder.intermediate_step_manager import IntermediateStepManager
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import TraceMetadata


# --- Helper function to recursively serialize any object into JSON-friendly data ---
def _serialize_data(obj: Any) -> Any:
    """Convert `obj` into a structure that can be passed to `json.dumps(...)`."""
    if isinstance(obj, BaseModel):
        # Convert Pydantic model to dict
        return obj.model_dump()

    if isinstance(obj, dict):
        return {str(k): _serialize_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialize_data(item) for item in obj]

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Fallback
    return str(obj)


def _prepare_serialized_args_kwargs(*args, **kwargs) -> tuple[list[Any], dict[str, Any]]:
    """Serialize args and kwargs before calling the wrapped function."""
    serialized_args = [_serialize_data(a) for a in args]
    serialized_kwargs = {k: _serialize_data(v) for k, v in kwargs.items()}
    return serialized_args, serialized_kwargs


def push_intermediate_step(step_manager: IntermediateStepManager,
                           identifier: str,
                           function_name: str,
                           event_type: IntermediateStepType,
                           args: Any = None,
                           kwargs: Any = None,
                           output: Any = None,
                           metadata: dict[str, Any] | None = None) -> None:
    """Push an intermediate step to the AgentIQ Event Stream."""

    payload = IntermediateStepPayload(UUID=identifier,
                                      event_type=event_type,
                                      name=function_name,
                                      metadata=TraceMetadata(
                                          span_inputs=[args, kwargs],
                                          span_outputs=output,
                                          provided_metadata=metadata,
                                      ))

    step_manager.push_intermediate_step(payload)


def track_function(func: Any = None, *, metadata: dict[str, Any] | None = None):
    """
    Decorator that can wrap any type of function (sync, async, generator,
    async generator) and executes "tracking logic" around it.

    - If the function is async, it will be wrapped in an async function.
    - If the function is a generator, it will be wrapped in a generator function.
    - If the function is an async generator, it will be wrapped in an async generator function.
    - If the function is sync, it will be wrapped in a sync function.
    """

    step_manager: IntermediateStepManager = AIQContext.get().intermediate_step_manager
    function_name: str = func.__name__ if func else "<unknown_function>"

    # If called as @track_function(...) but not immediately passed a function
    if func is None:

        def decorator_wrapper(actual_func):
            return track_function(actual_func, metadata=metadata)

        return decorator_wrapper

    # --- Validate metadata ---
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict[str, Any].")
        if any(not isinstance(k, str) for k in metadata.keys()):
            raise TypeError("All metadata keys must be strings.")

    # --- Now detect the function type and wrap accordingly ---
    if inspect.isasyncgenfunction(func):
        # ---------------------
        # ASYNC GENERATOR
        # ---------------------

        @functools.wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            # 1) Serialize input
            serialized_args, serialized_kwargs = _prepare_serialized_args_kwargs(*args, **kwargs)

            invocation_id = str(uuid.uuid4())
            push_intermediate_step(step_manager,
                                   invocation_id,
                                   function_name,
                                   IntermediateStepType.SPAN_START,
                                   args=serialized_args,
                                   kwargs=serialized_kwargs,
                                   metadata=metadata)

            # 2) Call the original async generator
            async for item in func(*args, **kwargs):
                # 3) Serialize the yielded item before yielding it
                serialized_item = _serialize_data(item)
                push_intermediate_step(step_manager,
                                       invocation_id,
                                       function_name,
                                       IntermediateStepType.SPAN_CHUNK,
                                       args=serialized_args,
                                       kwargs=serialized_kwargs,
                                       output=serialized_item,
                                       metadata=metadata)
                yield item  # yield the original item

            push_intermediate_step(step_manager,
                                   invocation_id,
                                   function_name,
                                   IntermediateStepType.SPAN_END,
                                   args=serialized_args,
                                   kwargs=serialized_kwargs,
                                   output=None,
                                   metadata=metadata)

            # 4) Post-yield logic if any

        return async_gen_wrapper

    if inspect.iscoroutinefunction(func):
        # ---------------------
        # ASYNC FUNCTION
        # ---------------------
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            serialized_args, serialized_kwargs = _prepare_serialized_args_kwargs(*args, **kwargs)
            invocation_id = str(uuid.uuid4())
            push_intermediate_step(step_manager,
                                   invocation_id,
                                   function_name,
                                   IntermediateStepType.SPAN_START,
                                   args=serialized_args,
                                   kwargs=serialized_kwargs,
                                   metadata=metadata)

            result = await func(*args, **kwargs)

            serialized_result = _serialize_data(result)
            push_intermediate_step(step_manager,
                                   invocation_id,
                                   function_name,
                                   IntermediateStepType.SPAN_END,
                                   args=serialized_args,
                                   kwargs=serialized_kwargs,
                                   output=serialized_result,
                                   metadata=metadata)

            return result

        return async_wrapper

    if inspect.isgeneratorfunction(func):
        # ---------------------
        # SYNC GENERATOR
        # ---------------------
        @functools.wraps(func)
        def sync_gen_wrapper(*args, **kwargs):
            serialized_args, serialized_kwargs = _prepare_serialized_args_kwargs(*args, **kwargs)
            invocation_id = str(uuid.uuid4())
            push_intermediate_step(step_manager,
                                   invocation_id,
                                   function_name,
                                   IntermediateStepType.SPAN_START,
                                   args=serialized_args,
                                   kwargs=serialized_kwargs,
                                   metadata=metadata)

            for item in func(*args, **kwargs):
                serialized_item = _serialize_data(item)
                push_intermediate_step(step_manager,
                                       invocation_id,
                                       function_name,
                                       IntermediateStepType.SPAN_CHUNK,
                                       args=serialized_args,
                                       kwargs=serialized_kwargs,
                                       output=serialized_item,
                                       metadata=metadata)

                yield item  # yield the original item

            push_intermediate_step(step_manager,
                                   invocation_id,
                                   function_name,
                                   IntermediateStepType.SPAN_END,
                                   args=serialized_args,
                                   kwargs=serialized_kwargs,
                                   output=None,
                                   metadata=metadata)

        return sync_gen_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        serialized_args, serialized_kwargs = _prepare_serialized_args_kwargs(*args, **kwargs)
        invocation_id = str(uuid.uuid4())
        push_intermediate_step(step_manager,
                               invocation_id,
                               function_name,
                               IntermediateStepType.SPAN_START,
                               args=serialized_args,
                               kwargs=serialized_kwargs,
                               metadata=metadata)

        result = func(*args, **kwargs)

        serialized_result = _serialize_data(result)
        push_intermediate_step(step_manager,
                               invocation_id,
                               function_name,
                               IntermediateStepType.SPAN_END,
                               args=serialized_args,
                               kwargs=serialized_kwargs,
                               output=serialized_result,
                               metadata=metadata)

        return result

    return sync_wrapper
