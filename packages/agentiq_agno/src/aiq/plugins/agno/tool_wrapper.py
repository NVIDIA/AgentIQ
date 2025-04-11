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

import asyncio
import json
import logging
import textwrap
import traceback
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import List

from agno.tools import tool

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.cli.register_workflow import register_tool_wrapper

logger = logging.getLogger(__name__)

# Add a module-level dictionary to track tool call counts for each tool
_tool_call_counters = {}
_MAX_EMPTY_CALLS = 1  # Maximum number of empty/metadata-only calls before signaling a problem
# For better UX, stop after just 1 empty call for search tools

# Dictionary to track which tools have already handled an initialization call
_tool_initialization_done = {}


async def process_result(result: Any, name: str) -> str:
    """
    Process the result from a function to ensure it's in the expected format.
    This function guarantees that the output will be a properly formatted string,
    suitable for consumption by language models like OpenAI's API.

    Parameters
    ----------
    result : Any
        The result to process
    name : str
        The name of the tool (for logging)

    Returns
    -------
    str: The processed result as a properly formatted string
    """
    logger.debug(f"{name} processing result of type {type(result)}")

    # Handle None or empty results
    if result is None:
        logger.warning(f"{name} returned None, converting to empty string")
        return ""

    # If the result is already a string, validate and return it
    if isinstance(result, str):
        logger.debug(f"{name} returning string result directly")
        # Ensure result is not empty
        if not result.strip():
            return f"The {name} tool completed successfully but returned an empty result."
        return result

    # Handle Agno Agent.arun response objects
    if hasattr(result, 'content'):
        logger.debug(f"{name} returning result.content")
        content = result.content
        # Make sure content is a string
        if not isinstance(content, str):
            logger.debug(f"{name} result.content is not a string, converting")
            content = str(content)
        return content

    # Handle OpenAI style responses
    if hasattr(result, 'choices') and len(result.choices) > 0:
        if hasattr(result.choices[0], 'message') and hasattr(result.choices[0].message, 'content'):
            logger.debug(f"{name} returning result.choices[0].message.content")
            return str(result.choices[0].message.content)
        elif hasattr(result.choices[0], 'text'):
            logger.debug(f"{name} returning result.choices[0].text")
            return str(result.choices[0].text)

    # Handle list of dictionaries by converting to a formatted string
    if isinstance(result, list):
        logger.debug(f"{name} converting list to string")
        if len(result) == 0:
            return f"The {name} tool returned an empty list."

        if all(isinstance(item, dict) for item in result):
            logger.debug(f"{name} converting list of dictionaries to string")
            formatted_result = ""
            for i, item in enumerate(result, 1):
                formatted_result += f"Result {i}:\n"
                for k, v in item.items():
                    formatted_result += f"  {k}: {v}\n"
                formatted_result += "\n"
            return formatted_result
        else:
            # For other lists, convert to a simple list format
            formatted_result = "Results:\n\n"
            for i, item in enumerate(result, 1):
                formatted_result += f"{i}. {str(item)}\n"
            return formatted_result

    # Handle dictionaries
    if isinstance(result, dict):
        logger.debug(f"{name} converting dictionary to string")
        try:
            # Try to format as JSON for readability
            return json.dumps(result, indent=2)
        except (TypeError, OverflowError):
            # Fallback to manual formatting if JSON fails
            formatted_result = "Result:\n\n"
            for k, v in result.items():
                formatted_result += f"{k}: {v}\n"
            return formatted_result

    # For all other types, convert to string
    logger.debug(f"{name} converting {type(result)} to string")
    return str(result)


def execute_agno_tool(name: str,
                      coroutine_fn: Callable[..., Awaitable[Any]],
                      required_fields: List[str],
                      loop: asyncio.AbstractEventLoop,
                      **kwargs: Any) -> Any:
    """
    Execute an Agno tool with the given parameters.

    Parameters
    ----------
    name : str
        The name of the tool
    coroutine_fn : Callable
        The async function to invoke
    required_fields : List[str]
        List of required fields for validation
    loop : asyncio.AbstractEventLoop
        The event loop to use for async execution
    **kwargs : Any
        The arguments to pass to the function

    Returns
    -------
    The result of the function execution as a string
    """
    global _tool_call_counters, _tool_initialization_done

    try:
        logger.debug(f"Running {name} with kwargs: {kwargs}")

        # Initialize counter for this tool if it doesn't exist
        if name not in _tool_call_counters:
            _tool_call_counters[name] = 0

        # Track if this tool has already been initialized
        if name not in _tool_initialization_done:
            _tool_initialization_done[name] = False

        # Filter out any known reserved keywords or metadata fields that might cause issues
        # These are typically added by frameworks and not meant for the function itself
        reserved_keywords = {'type', '_type', 'model_config', 'model_fields', 'model_dump', 'model_dump_json'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in reserved_keywords}

        # Check if we're only receiving metadata fields (potential infinite loop indicator)
        only_metadata = len(filtered_kwargs) == 0 and len(kwargs) > 0

        # Check if this is a search api tool with empty query
        is_search_api = name.lower().endswith("_api_tool")
        has_empty_query = "query" in filtered_kwargs and (not filtered_kwargs["query"]
                                                          or filtered_kwargs["query"].strip() == "")

        # Log if we filtered anything
        filtered_keys = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_keys:
            logger.debug(f"Filtered reserved keywords from kwargs: {filtered_keys}")

        # IMPORTANT: Special handling for SerpApi and other search API calls
        if is_search_api and (only_metadata or has_empty_query):
            # If this is the first time this tool is called with empty query, allow it for initialization
            if not _tool_initialization_done[name]:
                logger.info(f"First-time initialization call for {name}")
                _tool_initialization_done[name] = True
            else:
                # If we've already initialized this tool, prevent repeated empty calls
                logger.error(f"Tool {name} called with empty query after initialization. Blocking repeated calls.")
                return f"ERROR: Tool {name} requires a valid query. Provide a specific search term to continue."

        # IMPORTANT: Safeguard for infinite loops
        # If we're only getting metadata fields and no actual parameters repeatedly
        if only_metadata:
            _tool_call_counters[name] += 1
            logger.warning(
                f"Tool {name} called with only metadata fields (call {_tool_call_counters[name]}/{_MAX_EMPTY_CALLS})")

            # Break potential infinite loops after too many metadata-only calls
            if _tool_call_counters[name] >= _MAX_EMPTY_CALLS:
                logger.error(
                    f"Detected potential infinite loop for tool {name} - received {_tool_call_counters[name]} calls")
                _tool_call_counters[name] = 0  # Reset counter
                return f"ERROR: Tool {name} appears to be in a loop. Provide parameters when calling this tool."
        else:
            # Reset counter when we get actual parameters
            _tool_call_counters[name] = 0

        # Fix for the 'kwargs' wrapper issue - unwrap if needed
        if len(filtered_kwargs) == 1 and 'kwargs' in filtered_kwargs and isinstance(filtered_kwargs['kwargs'], dict):
            logger.debug("Detected wrapped kwargs, unwrapping")
            # If input is {'kwargs': {'actual': 'params'}}, we need to unwrap it
            unwrapped_kwargs = filtered_kwargs['kwargs']

            # Also filter the unwrapped kwargs
            unwrapped_kwargs = {k: v for k, v in unwrapped_kwargs.items() if k not in reserved_keywords}

            # Check if we're missing required fields and try to recover
            for field in required_fields:
                if field not in unwrapped_kwargs:
                    logger.warning(f"Missing required field '{field}' in unwrapped kwargs: {unwrapped_kwargs}")
                    # Try to build a query from all the provided values if query is required
                    if field == 'query' and len(unwrapped_kwargs) > 0:
                        # Simple fallback for search tools - cobble together a query string
                        query_parts = []
                        for k, v in unwrapped_kwargs.items():
                            query_parts.append(f"{k}: {v}")
                        unwrapped_kwargs['query'] = " ".join(query_parts)
                        logger.info(f"Built fallback query: {unwrapped_kwargs['query']}")

            filtered_kwargs = unwrapped_kwargs

        # Special handling for initialization calls - these are often empty or partial
        is_initialization = len(filtered_kwargs) == 0

        # Further validation to ensure all required fields are present
        # If this looks like an initialization call, we'll be more lenient
        missing_fields = []
        for field in required_fields:
            if field not in filtered_kwargs:
                missing_fields.append(field)
                logger.warning(f"Missing field '{field}' in kwargs: {filtered_kwargs}")

        # Special handling for search tools - query can be optional during initialization
        if not is_initialization and missing_fields and "query" in missing_fields and name.lower().endswith(
                "_api_tool"):
            logger.info(f"Tool {name} was called without a 'query' parameter, treating as initialization")
            is_initialization = True

        # Only enforce required fields for non-initialization calls
        if not is_initialization and missing_fields:
            if "query" in missing_fields:
                # Add a specific message for missing query
                raise ValueError(f"Missing required parameter 'query'. The tool {name} requires a search query.")
            else:
                missing_fields_str = ", ".join([f"'{f}'" for f in missing_fields])
                raise ValueError(f"Missing required parameters: {missing_fields_str} for {name}.")

        logger.debug(f"Invoking function with parameters: {filtered_kwargs}")

        # Try different calling styles to handle both positional and keyword arguments
        try:
            # First try calling with kwargs directly - this works for functions that use **kwargs
            future = asyncio.run_coroutine_threadsafe(coroutine_fn(**filtered_kwargs), loop)
            result = future.result(timeout=120)  # 2-minute timeout
        except TypeError as e:
            if "missing 1 required positional argument: 'input_obj'" in str(e):
                # If we get a specific error about missing positional arg, try passing as positional
                logger.debug(f"Retrying with positional argument style for {name}")
                future = asyncio.run_coroutine_threadsafe(coroutine_fn(filtered_kwargs), loop)
                result = future.result(timeout=120)  # 2-minute timeout
            else:
                # For other TypeError errors, reraise
                raise

        # Always process the result to ensure proper formatting, regardless of type
        process_future = asyncio.run_coroutine_threadsafe(process_result(result, name), loop)
        return process_future.result(timeout=30)  # 30-second timeout for processing

    except Exception as e:
        logger.exception(f"Error executing Agno tool {name}: {e}")
        error_traceback = traceback.format_exc()
        logger.error(f"Exception traceback: {error_traceback}")
        raise


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.AGNO)
def agno_tool_wrapper(name: str, fn: Function, builder: Builder):
    """
    Wraps an AgentIQ Function to be usable as an Agno tool.

    This wrapper handles the conversion of async AgentIQ functions to
    the format expected by Agno tools. It properly handles input schema,
    descriptions, and async invocation.

    Parameters
    ----------
    name : str
        The name of the tool
    fn : Function
        The AgentIQ Function to wrap
    builder : Builder
        The builder instance

    Returns
    -------
    A callable that can be used as an Agno tool
    """
    # Ensure input schema is present
    assert fn.input_schema is not None, "Tool must have input schema"

    # Get the event loop for running async functions
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If there's no running event loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Get the async function to invoke
    coroutine_fn = fn.acall_invoke

    # Extract metadata for the tool
    description = fn.description or ""
    if description:
        description = textwrap.dedent(description).strip()

    # Input schema handling from LangChain-style
    required_fields = []
    if fn.input_schema is not None:
        try:
            schema_json = fn.input_schema.model_json_schema()
            required_fields = schema_json.get("required", [])
            # Add schema description to the tool description if available
            schema_desc = schema_json.get("description")
            if schema_desc and schema_desc not in description:
                description = f"{description}\n\nArguments: {schema_desc}"
        except Exception as e:
            logger.warning(f"Error extracting JSON schema from input_schema: {e}")

    # Create a function specific to this tool with proper closure variables
    def tool_sync_wrapper(**kwargs: Any) -> Any:
        """Synchronous implementation of the tool function."""
        return execute_agno_tool(name, coroutine_fn, required_fields, loop, **kwargs)

    # Prepare the documentation for the tool
    if description:
        tool_sync_wrapper.__doc__ = description

    # Set the function name
    tool_sync_wrapper.__name__ = name

    # Apply the tool decorator and return it
    decorated_tool = tool(name=name, description=description)(tool_sync_wrapper)

    return decorated_tool
