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

# limitations under the License.
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
import typing
import uuid
from collections.abc import AsyncGenerator
from collections.abc import Callable
from unittest import mock

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel

TESTS_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
EXAMPLES_DIR = os.path.join(PROJECT_DIR, "examples")
sys.path.append(SRC_DIR)

if typing.TYPE_CHECKING:
    from aiq.data_models.intermediate_step import IntermediateStep
    from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor


@pytest.fixture(name="test_data_dir")
def test_data_dir_fixture():
    return os.path.join(TESTS_DIR, "test_data")


@pytest.fixture(name="config_file")
def config_file_fixture(test_data_dir: str):
    return os.path.join(test_data_dir, "config.yaml")


@pytest.fixture(name="mock_aiohttp_session")
def mock_aiohttp_session_fixture():
    with mock.patch("aiohttp.ClientSession") as mock_aiohttp_session:
        mock_aiohttp_session.return_value = mock_aiohttp_session
        mock_aiohttp_session.__aenter__.return_value = mock_aiohttp_session

        mock_get = mock.AsyncMock()
        mock_get.return_value = mock_get
        mock_get.__aenter__.return_value = mock_get
        mock_get.text.return_value = '<td data-testid="vuln-CWEs-link-0">test_output</td>'
        mock_get.json.return_value = {"test": "output"}
        mock_aiohttp_session.request.return_value = mock_get

        yield mock_aiohttp_session


@pytest.fixture(name="restore_environ")
def restore_environ_fixture():
    orig_vars = os.environ.copy()
    yield os.environ

    # Iterating over a copy of the keys as we will potentially be deleting keys in the loop
    for key in list(os.environ.keys()):
        orig_val = orig_vars.get(key)
        if orig_val is not None:
            os.environ[key] = orig_val
        else:
            del (os.environ[key])


@pytest.fixture(name="set_test_api_keys")
def set_test_api_keys_fixture(restore_environ):
    # restore_environ fixture is used implicitly, do not remove
    for key in ("NGC_API_KEY", "NVD_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY", "SERPAPI_API_KEY"):
        os.environ[key] = "test_key"


@pytest.fixture(name="rapids_repo_names")
def rapids_repo_names_fixture() -> list[str]:
    return ["cugraph", "cuvs", "rmm", "raft", "cuspatial", "cuxfilter", "cucim"]


@pytest.fixture(name="rapids_repo_urls")
def rapids_repo_urls_fixture(rapids_repo_names: list[str]) -> dict[str, str]:
    return {repo: f"https://github.com/rapidsai/{repo}.git" for repo in rapids_repo_names}


@pytest.fixture(name="workflow_config")
def workflow_config_fixture():
    from _utils.configs import WorkflowTestConfig
    return WorkflowTestConfig(llm_name='test_llm', functions=['test_function'], prompt='Are you a unittest?')


@pytest.fixture(name="tools_config")
def tools_config_fixture() -> dict[str, typing.Any]:
    return {
        "test_function": {
            "_type": "test_function"
        },
        "test_tool_2": {
            "_type": "test_function"
        },
        "test_tool_3": {
            "_type": "test_function"
        },
    }


@pytest.fixture(name="llms_config")
def llms_config_fixture() -> dict[str, typing.Any]:
    return {"test_llm": {"_type": "test_llm"}, "test_llm_2": {"_type": "test_llm"}, "test_llm_3": {"_type": "test_llm"}}


class StreamingOutputModel(BaseModel):
    result: str


class SingleOutputModel(BaseModel):
    summary: str


@pytest.fixture(name="test_workflow_fn")
def test_workflow_fn_fixture():

    async def workflow_fn(param: BaseModel) -> SingleOutputModel:
        return SingleOutputModel(summary="This is a coroutine function")

    return workflow_fn


@pytest.fixture(name="test_streaming_fn")
def test_streaming_fn_fixture():

    async def streaming_fn(param: BaseModel) -> typing.Annotated[AsyncGenerator[StreamingOutputModel], ...]:
        yield StreamingOutputModel(result="this is an async generator")

    return streaming_fn


@pytest.fixture(name="register_test_workflow")
def register_test_workflow_fixture(test_workflow_fn) -> Callable[[], Callable]:

    def register_test_workflow():
        from _utils.configs import WorkflowTestConfig
        from aiq.builder.builder import Builder
        from aiq.cli.register_workflow import register_function

        @register_function(config_type=WorkflowTestConfig)
        async def build_fn(_: WorkflowTestConfig, __: Builder):
            yield test_workflow_fn

        return build_fn

    return register_test_workflow


@pytest.fixture(name="reactive_stream")
def reactive_stream_fixture():
    """
    A fixture that sets up a fresh usage_stats queue in the context var
    for each test, then resets it afterward.
    """
    from aiq.builder.context import AIQContextState
    from aiq.utils.reactive.subject import Subject

    token = None
    original_queue = AIQContextState.get().event_stream.get()

    try:
        new_queue = Subject()
        token = AIQContextState.get().event_stream.set(new_queue)
        yield new_queue
    finally:
        if token is not None:
            # Reset to the original queue after the test
            AIQContextState.get().event_stream.reset(token)
            AIQContextState.get().event_stream.set(original_queue)


@pytest.fixture(name="global_settings", scope="function", autouse=False)
def function_settings_fixture():
    """
    Resets and returns the global settings for testing.

    This gets automatically used at the function level to ensure no state is leaked between functions.
    """

    from aiq.settings.global_settings import GlobalSettings

    with GlobalSettings.push() as settings:
        yield settings


@pytest.fixture(name="pypi_registry_channel")
def pypi_registry_channel_fixture():
    """
    Returns a pypi registry channel configuration.
    """
    return {
        "channels": {
            "pypi_channel": {
                "_type": "pypi",
                "endpoint": "http://localhost:1234",
                "publish_route": "",
                "pull_route": "",
                "search_route": "simple",
                "token": "test-token"
            }
        }
    }


@pytest.fixture(name="rest_registry_channel")
def rest_registry_channel_fixture():
    """
    Returns a rest registry channel configuration.
    """
    return {
        "channels": {
            "rest_channel": {
                "_type": "rest",
                "endpoint": "http://localhost:1234",
                "publish_route": "publish",
                "pull_route": "pull",
                "search_route": "search",
                "remove_route": "remove",
                "token": "test-token"
            }
        }
    }


@pytest.fixture(name="local_registry_channel")
def local_registry_channel_fixture():
    """
    Returns a local registry channel configuration.
    """
    return {"channels": {"local_channel": {"_type": "local"}}}


@pytest.fixture(scope="session")
def httpserver_listen_address():

    return "127.0.0.1", 0


@pytest.fixture(scope="module")
async def mock_llm():

    class MockLLM(BaseChatModel):

        async def _agenerate(self,
                             messages: list[BaseMessage],
                             stop: list[str] | None = None,
                             run_manager: AsyncCallbackManagerForLLMRun | None = None,
                             **kwargs: typing.Any) -> ChatResult:
            # mock behavior to test agent features
            if len(messages) == 1:
                if 'mock tool call' in messages[0].content:
                    message = AIMessage(content='mock tool call',
                                        response_metadata={"mock_llm_response": True},
                                        tool_calls=[{
                                            "name": "Tool A",
                                            "args": {
                                                "query": "mock query"
                                            },
                                            "id": "Tool A",
                                            "type": "tool_call",
                                        }])
                    generation = ChatGeneration(message=message)
                    return ChatResult(generations=[generation], llm_output={'mock_llm_response': True})
            if len(messages) == 4:
                if 'fix the input on retry' in messages[2].content:
                    response = 'Thought: not many\nAction: Tool A\nAction Input: give me final answer!\nObservation:'
                    message = AIMessage(content=response, response_metadata={"mock_llm_response": True})
                    generation = ChatGeneration(message=message)
                    return ChatResult(generations=[generation], llm_output={'mock_llm_response': True})
                if 'give me final answer' in messages[3].content:
                    response = 'Final Answer: hello, world!'
                    message = AIMessage(content=response, response_metadata={"mock_llm_response": True})
                    generation = ChatGeneration(message=message)
                    return ChatResult(generations=[generation], llm_output={'mock_llm_response': True})
            message = AIMessage(content=messages[-1].content, response_metadata={"mock_llm_response": True})
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation], llm_output={'mock_llm_response': True})

        def _generate(self,
                      messages: list[BaseMessage],
                      stop: list[str] | None = None,
                      run_manager: CallbackManagerForLLMRun | None = None,
                      **kwargs: typing.Any) -> ChatResult:
            message = AIMessage(content=messages[-1].content, response_metadata={"mock_llm_response": True})
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation], llm_output={'mock_llm_response': True})

        @property
        def _llm_type(self) -> str:
            return 'mock-llm'

    return MockLLM()


@pytest.fixture(scope="module")
def mock_tool():

    def _create_mock_tool(tool_name: str):

        class MockTool(BaseTool):
            name: str = tool_name
            description: str = 'test tool:' + tool_name

            async def _arun(self, query: str = 'test', run_manager: AsyncCallbackManagerForToolRun | None = None):  # noqa: E501  # pylint: disable=arguments-differ
                return query

            def _run(self, query: str = 'test', run_manager: CallbackManagerForToolRun | None = None):  # noqa: E501  # pylint: disable=arguments-differ
                return query

        return MockTool()

    return _create_mock_tool


@pytest.fixture(scope="function", autouse=True)
def patched_async_memory_client(monkeypatch):

    from mem0.client.main import MemoryClient

    mock_method = mock.MagicMock(return_value=None)
    monkeypatch.setattr(MemoryClient, "_validate_api_key", mock_method)
    return MemoryClient


@pytest.fixture(name="rag_user_inputs")
def rag_user_inputs_fixture() -> list[str]:
    """Fixture providing multiple user inputs."""
    return ["What is ML?", "What is NLP?"]


@pytest.fixture(name="rag_generated_outputs")
def rag_generated_outputs_fixture() -> list[str]:
    """Fixture providing workflow generated outputs corresponding to user inputs."""
    return ["ML is the abbreviation for Machine Learning", "NLP stands for Natural Language Processing"]


@pytest.fixture(name="rag_intermediate_steps")
def rag_intermediate_steps_fixture(rag_user_inputs, rag_generated_outputs) -> list[list["IntermediateStep"]]:
    """
    Fixture to generate separate lists of IntermediateStep objects for each user input.

    Each list includes:
    1. LLM_START, LLM_NEW_TOKENs, LLM_END
    2. TOOL_START, and TOOL_END.

    Returns:
        (list for user_input_1, list for user_input_2)
    """
    from aiq.builder.framework_enum import LLMFrameworkEnum
    from aiq.data_models.intermediate_step import IntermediateStep
    from aiq.data_models.intermediate_step import IntermediateStepPayload
    from aiq.data_models.intermediate_step import IntermediateStepType
    from aiq.data_models.intermediate_step import StreamEventData

    framework = LLMFrameworkEnum.LANGCHAIN
    token_cnt = 10
    llm_name = "mock_llm"
    tool_name = "mock_tool"

    def create_step(event_type,
                    name=llm_name,
                    input_data=None,
                    output_data=None,
                    chunk=None,
                    step_uuid: str | None = None):
        if step_uuid is None:
            step_uuid = str(uuid.uuid4())
        """Helper to create an `IntermediateStep`."""
        return IntermediateStep(
            payload=IntermediateStepPayload(UUID=step_uuid,
                                            event_type=event_type,
                                            framework=framework,
                                            name=name,
                                            data=StreamEventData(input=input_data, output=output_data, chunk=chunk)))

    step_lists = []  # Store separate lists

    for user_input, generated_ouput in zip(rag_user_inputs, rag_generated_outputs):
        tool_input = f"Get me the documents for {user_input}"
        tool_output = f"Here is information I have on {user_input}"
        generated_output = generated_ouput

        llm_start_step = create_step(IntermediateStepType.LLM_START, input_data=user_input)

        steps = [
            llm_start_step,
            *[
                create_step(IntermediateStepType.LLM_NEW_TOKEN, chunk=f"Token {i} for {user_input}")
                for i in range(token_cnt)
            ],
            create_step(IntermediateStepType.LLM_END,
                        input_data=user_input,
                        output_data=generated_output,
                        step_uuid=llm_start_step.UUID)
        ]

        tool_start_step = create_step(IntermediateStepType.TOOL_START, name=tool_name, input_data=tool_input)
        steps.append(tool_start_step)

        steps.append(
            create_step(IntermediateStepType.TOOL_END,
                        name=tool_name,
                        input_data=tool_input,
                        output_data=tool_output,
                        step_uuid=tool_start_step.UUID))

        step_lists.append(steps)  # Append separate list for each user input

    return step_lists


@pytest.fixture(name="rag_intermediate_property_adaptor")
def rag_intermediate_property_adaptor_fixture(rag_intermediate_steps) -> list[list["IntermediatePropertyAdaptor"]]:
    """
    Fixture to transform the rag_intermediate_steps fixture data into IntermediatePropertyAdaptor objects.
    """
    from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor

    return [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps]
            for steps in rag_intermediate_steps]
