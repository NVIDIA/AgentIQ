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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
# Import the module under test with the correct import path
from aiq.plugins.agno.tool_wrapper import agno_tool_wrapper
from aiq.plugins.agno.tool_wrapper import execute_agno_tool
from aiq.plugins.agno.tool_wrapper import process_result


class TestToolWrapper:
    """Tests for the agno_tool_wrapper function."""

    @pytest.fixture
    def mock_event_loop(self):
        """Create a mock event loop for testing."""
        loop = MagicMock()
        return loop

    @pytest.fixture
    def mock_function(self):
        """Create a mock Function object."""
        mock_fn = MagicMock(spec=Function)
        mock_fn.description = "Test function description"
        mock_fn.input_schema = {"type": "object", "properties": {"input": {"type": "string"}}}

        # Set up the acall_invoke coroutine
        async def mock_acall_invoke(*args, **kwargs):
            return "test_result"

        mock_fn.acall_invoke = mock_acall_invoke
        return mock_fn

    @pytest.fixture
    def mock_model_schema_function(self):
        """Create a mock Function object with a model_json_schema method."""
        mock_fn = MagicMock(spec=Function)
        mock_fn.description = "Test function with schema description"

        # Create a mock schema with model_json_schema method
        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {
            "properties": {
                "query": {
                    "type": "string"
                }
            },
            "required": ["query"],
            "description": "This is a schema description"
        }
        mock_fn.input_schema = schema_mock

        # Set up the acall_invoke coroutine
        async def mock_acall_invoke(*args, **kwargs):
            return "test_result"

        mock_fn.acall_invoke = mock_acall_invoke
        return mock_fn

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @patch("aiq.plugins.agno.tool_wrapper.tool")
    def test_agno_tool_wrapper(self, mock_tool, mock_function, mock_builder):
        """Test that agno_tool_wrapper creates an Agno Tool with the correct parameters."""
        # Mock the tool decorator to return a function that returns its input
        mock_tool.return_value = lambda x: x

        # Call the function under test
        result = agno_tool_wrapper("test_tool", mock_function, mock_builder)

        # Verify that tool was called with the correct parameters
        mock_tool.assert_called_once_with(name="test_tool", description="Test function description")

        # Verify the wrapper function attributes
        assert result.__name__ == "test_tool"
        assert result.__doc__ == "Test function description"

    @patch("aiq.plugins.agno.tool_wrapper.tool")
    def test_agno_tool_wrapper_with_schema_description(self, mock_tool, mock_model_schema_function, mock_builder):
        """Test that agno_tool_wrapper correctly incorporates schema description."""
        # Mock the tool decorator to return a function that returns its input
        mock_tool.return_value = lambda x: x

        # Call the function under test
        result = agno_tool_wrapper("test_tool", mock_model_schema_function, mock_builder)

        # Verify that tool was called with the correct parameters including schema description
        expected_description = "Test function with schema description\n\nArguments: This is a schema description"
        mock_tool.assert_called_once_with(name="test_tool", description=expected_description)

        # Verify the wrapper function attributes
        assert result.__name__ == "test_tool"
        assert result.__doc__ == expected_description

    @patch("aiq.plugins.agno.tool_wrapper.execute_agno_tool")
    @patch("aiq.plugins.agno.tool_wrapper.tool")
    def test_wrapper_function(self, mock_tool, mock_execute_agno_tool, mock_function, mock_builder):
        """Test that the wrapper function correctly calls execute_agno_tool."""
        # Mock the tool decorator to return a function that returns its input
        mock_tool.return_value = lambda x: x

        # Set up the mock for execute_agno_tool
        mock_execute_agno_tool.return_value = "test_result"

        # Call the function under test
        wrapper_func = agno_tool_wrapper("test_tool", mock_function, mock_builder)

        # Call the wrapper function
        result = wrapper_func(kwarg1="value1")

        # Verify that execute_agno_tool was called with the correct arguments
        mock_execute_agno_tool.assert_called_once()

        # Verify the result
        assert result == "test_result"

    @patch("aiq.plugins.agno.tool_wrapper.asyncio.get_running_loop")
    def test_get_event_loop_called(self, mock_get_running_loop, mock_function, mock_builder):
        """Test that get_running_loop is called when agno_tool_wrapper is executed."""
        # Set up the mock event loop
        mock_loop = MagicMock()
        mock_get_running_loop.return_value = mock_loop

        # Call the function under test
        agno_tool_wrapper("test_tool", mock_function, mock_builder)

        # Verify that get_running_loop was called
        mock_get_running_loop.assert_called_once()

    @patch("aiq.plugins.agno.tool_wrapper.asyncio.new_event_loop")
    @patch("aiq.plugins.agno.tool_wrapper.asyncio.set_event_loop")
    @patch("aiq.plugins.agno.tool_wrapper.asyncio.get_running_loop")
    def test_create_event_loop_if_none_available(self,
                                                 mock_get_running_loop,
                                                 mock_set_event_loop,
                                                 mock_new_event_loop,
                                                 mock_function,
                                                 mock_builder):
        """Test that a new event loop is created if none is available."""
        # Make get_running_loop raise a RuntimeError
        mock_get_running_loop.side_effect = RuntimeError("No running event loop")

        # Set up a mock loop to be returned by new_event_loop
        mock_loop = MagicMock()
        mock_new_event_loop.return_value = mock_loop

        # Call the function under test
        agno_tool_wrapper("test_tool", mock_function, mock_builder)

        # Verify that a new event loop was created and set
        mock_new_event_loop.assert_called_once()
        mock_set_event_loop.assert_called_once_with(mock_loop)

    def test_registration_decorator(self):
        """Test that the register_tool_wrapper decorator correctly registers the agno_tool_wrapper function."""
        # Get the global type registry to access registered tool wrappers
        from aiq.cli.type_registry import GlobalTypeRegistry

        # Get the registered tool wrappers
        registry = GlobalTypeRegistry.get()

        # Check that agno_tool_wrapper is registered for LLMFrameworkEnum.AGNO
        agno_wrapper = registry.get_tool_wrapper(LLMFrameworkEnum.AGNO)
        assert agno_wrapper.build_fn == agno_tool_wrapper

    def test_input_schema_validation(self, mock_builder):
        """Test that agno_tool_wrapper raises an assertion error when input_schema is None."""
        # Create a mock function with no input_schema
        mock_fn = MagicMock(spec=Function)
        mock_fn.description = "Test function description"
        mock_fn.input_schema = None

        # Set up the acall_invoke coroutine
        async def mock_acall_invoke(*args, **kwargs):
            return "test_result"

        mock_fn.acall_invoke = mock_acall_invoke

        # Check that an assertion error is raised
        with pytest.raises(AssertionError, match="Tool must have input schema"):
            agno_tool_wrapper("test_tool", mock_fn, mock_builder)

    @patch("aiq.plugins.agno.tool_wrapper._tool_call_counters", {})
    @patch("aiq.plugins.agno.tool_wrapper._tool_initialization_done", {})
    @patch("aiq.plugins.agno.tool_wrapper.asyncio.run_coroutine_threadsafe")
    def test_execute_agno_tool_initialization(self, mock_run_coroutine_threadsafe, mock_event_loop):
        """Test that execute_agno_tool correctly handles tool initialization."""
        # Set up the mock future
        mock_future = MagicMock()
        mock_future.result.return_value = "initialization_result"
        mock_run_coroutine_threadsafe.return_value = mock_future

        # Create a mock coroutine function
        mock_coroutine_fn = AsyncMock()

        # Call the function under test for a tool with an empty kwargs dict (initialization)
        result = execute_agno_tool("test_tool", mock_coroutine_fn, ["query"], mock_event_loop)

        # Verify that the counters and initialization flags were set correctly
        from aiq.plugins.agno.tool_wrapper import _tool_call_counters
        from aiq.plugins.agno.tool_wrapper import _tool_initialization_done
        assert "test_tool" in _tool_call_counters
        assert "test_tool" in _tool_initialization_done

        # Verify that run_coroutine_threadsafe was called with the coroutine function
        mock_run_coroutine_threadsafe.assert_called()

        # Verify the result
        assert result == "initialization_result"

    @patch("aiq.plugins.agno.tool_wrapper._tool_call_counters", {"search_api_tool": 0})
    @patch("aiq.plugins.agno.tool_wrapper._tool_initialization_done", {"search_api_tool": True})
    @patch("aiq.plugins.agno.tool_wrapper.asyncio.run_coroutine_threadsafe")
    def test_execute_agno_tool_search_api_empty_query(self, mock_run_coroutine_threadsafe, mock_event_loop):
        """Test that execute_agno_tool correctly handles search API tools with empty queries."""
        # Create a mock coroutine function
        mock_coroutine_fn = AsyncMock()

        # Call the function under test for a search tool with an empty query
        result = execute_agno_tool("search_api_tool", mock_coroutine_fn, ["query"], mock_event_loop, query="")

        # Verify that an error message is returned for empty query after initialization
        assert "ERROR" in result
        assert "requires a valid query" in result

        # Verify that run_coroutine_threadsafe was not called since we blocked the empty query
        mock_run_coroutine_threadsafe.assert_not_called()

    @patch("aiq.plugins.agno.tool_wrapper._tool_call_counters", {"test_tool": 0})
    @patch("aiq.plugins.agno.tool_wrapper._tool_initialization_done", {"test_tool": False})
    @patch("aiq.plugins.agno.tool_wrapper.asyncio.run_coroutine_threadsafe")
    def test_execute_agno_tool_filtered_kwargs(self, mock_run_coroutine_threadsafe, mock_event_loop):
        """Test that execute_agno_tool correctly filters reserved keywords."""
        # Set up the mock future
        mock_future = MagicMock()
        mock_future.result.return_value = "filtered_result"
        mock_run_coroutine_threadsafe.return_value = mock_future

        # Create a mock process_result future
        process_future = MagicMock()
        process_future.result.return_value = "processed_result"
        mock_run_coroutine_threadsafe.side_effect = [mock_future, process_future]

        # Create a mock coroutine function
        mock_coroutine_fn = AsyncMock()

        # Call the function under test with kwargs containing reserved keywords
        result = execute_agno_tool("test_tool",
                                   mock_coroutine_fn, ["query"],
                                   mock_event_loop,
                                   query="test query",
                                   model_config="should be filtered",
                                   _type="should be filtered")

        # Verify that run_coroutine_threadsafe was called with filtered kwargs
        args, kwargs = mock_coroutine_fn.call_args
        assert "query" in kwargs
        assert "model_config" not in kwargs
        assert "_type" not in kwargs

        # Verify the result
        assert result == "processed_result"

    @patch("aiq.plugins.agno.tool_wrapper._tool_call_counters", {"test_tool": 0})
    @patch("aiq.plugins.agno.tool_wrapper._tool_initialization_done", {"test_tool": False})
    @patch("aiq.plugins.agno.tool_wrapper.asyncio.run_coroutine_threadsafe")
    def test_execute_agno_tool_wrapped_kwargs(self, mock_run_coroutine_threadsafe, mock_event_loop):
        """Test that execute_agno_tool correctly unwraps nested kwargs."""
        # Set up the mock future
        mock_future = MagicMock()
        mock_future.result.return_value = "unwrapped_result"

        # Create a mock process_result future
        process_future = MagicMock()
        process_future.result.return_value = "processed_result"
        mock_run_coroutine_threadsafe.side_effect = [mock_future, process_future]

        # Create a mock coroutine function
        mock_coroutine_fn = AsyncMock()

        # Call the function under test with wrapped kwargs
        result = execute_agno_tool("test_tool",
                                   mock_coroutine_fn, ["query"],
                                   mock_event_loop,
                                   kwargs={
                                       "query": "test query", "other_param": "value"
                                   })

        # Verify that run_coroutine_threadsafe was called with unwrapped kwargs
        args, kwargs = mock_coroutine_fn.call_args
        assert "query" in kwargs
        assert kwargs["query"] == "test query"
        assert "other_param" in kwargs
        assert kwargs["other_param"] == "value"

        # Verify the result
        assert result == "processed_result"

    @patch("aiq.plugins.agno.tool_wrapper._tool_call_counters", {"test_tool": 0})
    @patch("aiq.plugins.agno.tool_wrapper._MAX_EMPTY_CALLS", 2)
    @patch("aiq.plugins.agno.tool_wrapper.asyncio.run_coroutine_threadsafe")
    def test_execute_agno_tool_infinite_loop_detection(self, mock_run_coroutine_threadsafe, mock_event_loop):
        """Test that execute_agno_tool detects and prevents infinite loops."""
        # Create a mock coroutine function
        mock_coroutine_fn = AsyncMock()

        # First call with only metadata should increment counter but proceed
        execute_agno_tool("test_tool", mock_coroutine_fn, ["query"], mock_event_loop, model_config="metadata only")

        # Second call with only metadata should detect potential infinite loop
        result2 = execute_agno_tool("test_tool",
                                    mock_coroutine_fn, ["query"],
                                    mock_event_loop,
                                    model_config="metadata only")

        # Verify that the second call returned an error about infinite loops
        assert "ERROR" in result2
        assert "appears to be in a loop" in result2

        # Verify that coroutine_fn was called only once (for the first call)
        assert mock_coroutine_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_process_result_string(self):
        """Test process_result with string input."""
        result = await process_result("test string result", "test_tool")
        assert result == "test string result"

    @pytest.mark.asyncio
    async def test_process_result_none(self):
        """Test process_result with None input."""
        result = await process_result(None, "test_tool")
        assert result == ""

    @pytest.mark.asyncio
    async def test_process_result_dict(self):
        """Test process_result with dictionary input."""
        dict_result = {"key1": "value1", "key2": "value2"}
        result = await process_result(dict_result, "test_tool")
        assert "key1" in result
        assert "value1" in result
        assert "key2" in result
        assert "value2" in result

    @pytest.mark.asyncio
    async def test_process_result_list_of_dicts(self):
        """Test process_result with a list of dictionaries."""
        list_result = [{"name": "item1", "value": 100}, {"name": "item2", "value": 200}]
        result = await process_result(list_result, "test_tool")
        assert "Result 1" in result
        assert "item1" in result
        assert "Result 2" in result
        assert "item2" in result

    @pytest.mark.asyncio
    async def test_process_result_object_with_content(self):
        """Test process_result with an object that has a content attribute."""
        # Create a mock object with a content attribute
        mock_obj = MagicMock()
        mock_obj.content = "content attribute value"

        result = await process_result(mock_obj, "test_tool")
        assert result == "content attribute value"

    @pytest.mark.asyncio
    async def test_process_result_openai_style_response(self):
        """Test process_result with an OpenAI-style response object."""

        # Create a simple class-based structure to simulate an OpenAI response
        class Message:

            def __init__(self, content):
                self.content = content

        class Choice:

            def __init__(self, message):
                self.message = message

        class OpenAIResponse:

            def __init__(self, choices):
                self.choices = choices

        # Create an actual object hierarchy instead of mocks
        mock_response = OpenAIResponse([Choice(Message("OpenAI response content"))])

        result = await process_result(mock_response, "test_tool")
        assert result == "OpenAI response content"

    @patch("aiq.plugins.agno.tool_wrapper.tool")
    @patch("aiq.plugins.agno.tool_wrapper.asyncio.run_coroutine_threadsafe")
    def test_different_calling_styles(self, mock_run_coroutine_threadsafe, mock_tool, mock_function, mock_builder):
        """Test that execute_agno_tool handles different function calling styles."""
        # Mock the tool decorator to return a function that returns its input
        mock_tool.return_value = lambda x: x

        # Set up the mock futures
        future1 = MagicMock()
        future1.result.side_effect = TypeError("missing 1 required positional argument: 'input_obj'")

        future2 = MagicMock()
        future2.result.return_value = "positional_arg_result"

        process_future = MagicMock()
        process_future.result.return_value = "processed_result"

        mock_run_coroutine_threadsafe.side_effect = [future1, future2, process_future]

        # Create a mock coroutine function
        AsyncMock()

        # Call the function under test
        wrapper_func = agno_tool_wrapper("test_tool", mock_function, mock_builder)

        # Patch execute_agno_tool to use our mock
        with patch("aiq.plugins.agno.tool_wrapper.execute_agno_tool") as mock_execute:
            mock_execute.return_value = "test_result"
            result = wrapper_func(kwarg1="value1")

            # Verify that execute_agno_tool was called
            mock_execute.assert_called_once()
            assert result == "test_result"
