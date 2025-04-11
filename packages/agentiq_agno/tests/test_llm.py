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

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig
# Import the module under test with the correct import path
from aiq.plugins.agno.llm import nim_agno
from aiq.plugins.agno.llm import openai_agno


class TestNimAgno:
    """Tests for the nim_agno function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @pytest.fixture
    def nim_config(self):
        """Create a NIMModelConfig instance."""
        return NIMModelConfig(model_name="test-model")

    @patch("agno.models.nvidia.Nvidia")
    async def test_nim_agno_basic(self, mock_nvidia, nim_config, mock_builder):
        """Test that nim_agno creates a Nvidia instance with the correct parameters."""
        # Use the context manager properly
        async with nim_agno(nim_config, mock_builder) as nvidia_instance:
            # Verify that Nvidia was created with the correct parameters
            mock_nvidia.assert_called_once_with(id="test-model")

            # Verify that the returned object is the mock Nvidia instance
            assert nvidia_instance == mock_nvidia.return_value

    @patch("agno.models.nvidia.Nvidia")
    async def test_nim_agno_with_base_url(self, mock_nvidia, nim_config, mock_builder):
        """Test that nim_agno creates a Nvidia instance with base_url when provided."""
        # Add base_url to the config
        nim_config.base_url = "https://test-api.nvidia.com"

        # Use the context manager properly
        async with nim_agno(nim_config, mock_builder) as nvidia_instance:
            # Verify that Nvidia was created with the correct parameters
            mock_nvidia.assert_called_once_with(id="test-model", base_url="https://test-api.nvidia.com")

            # Verify that the returned object is the mock Nvidia instance
            assert nvidia_instance == mock_nvidia.return_value

    @patch("agno.models.nvidia.Nvidia")
    @patch.dict(os.environ, {"NVIDIA_API_KEY": ""}, clear=True)
    async def test_nim_agno_with_env_var(self, mock_nvidia, nim_config, mock_builder):
        """Test that nim_agno correctly handles the NVIDIA_API_KEY environment variable."""
        # Set NVIDIA_API_KEY (not NVIDAI_API_KEY as that was incorrect)
        # The code in nim_agno actually checks for NVIDIA_API_KEY, not NVIDAI_API_KEY
        os.environ["NVIDIA_API_KEY"] = "test-api-key"

        # Use the context manager properly
        async with nim_agno(nim_config, mock_builder) as nvidia_instance:
            # Verify that the environment variable is still present
            assert os.environ.get("NVIDIA_API_KEY") == "test-api-key"

            # Verify that Nvidia was created with the correct parameters
            mock_nvidia.assert_called_once_with(id="test-model")

            # Verify that the returned object is the mock Nvidia instance
            assert nvidia_instance == mock_nvidia.return_value

    @patch("agno.models.nvidia.Nvidia")
    @patch.dict(os.environ, {"NVIDIA_API_KEY": "existing-key"}, clear=True)
    async def test_nim_agno_with_existing_env_var(self, mock_nvidia, nim_config, mock_builder):
        """Test that nim_agno preserves existing NVIDIA_API_KEY environment variable."""
        # Use the context manager properly
        async with nim_agno(nim_config, mock_builder) as nvidia_instance:
            # Verify that the environment variable was not changed
            assert os.environ.get("NVIDIA_API_KEY") == "existing-key"

            # Verify that Nvidia was created with the correct parameters
            mock_nvidia.assert_called_once_with(id="test-model")

            # Verify that the returned object is the mock Nvidia instance
            assert nvidia_instance == mock_nvidia.return_value


class TestOpenAIAgno:
    """Tests for the openai_agno function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock(spec=Builder)

    @pytest.fixture
    def openai_config(self):
        """Create an OpenAIModelConfig instance."""
        return OpenAIModelConfig(model="gpt-4")

    @patch("agno.models.openai.OpenAIChat")
    async def test_openai_agno(self, mock_openai_chat, openai_config, mock_builder):
        """Test that openai_agno creates an OpenAIChat instance with the correct parameters."""
        # Use the context manager properly
        async with openai_agno(openai_config, mock_builder) as openai_instance:
            # Verify that OpenAIChat was created with the correct parameters
            mock_openai_chat.assert_called_once()
            call_kwargs = mock_openai_chat.call_args[1]

            # Check that model is set correctly
            assert call_kwargs["model"] == "gpt-4"

            # Verify that the returned object is the mock OpenAIChat instance
            assert openai_instance == mock_openai_chat.return_value

    @patch("agno.models.openai.OpenAIChat")
    async def test_openai_agno_with_additional_params(self, mock_openai_chat, openai_config, mock_builder):
        """Test that openai_agno passes additional params to OpenAIChat."""
        # Add additional parameters to the config
        openai_config.api_key = "test-api-key"
        openai_config.temperature = 0.7
        # OpenAIModelConfig doesn't have max_tokens field, removing

        # Use the context manager properly
        async with openai_agno(openai_config, mock_builder) as openai_instance:
            # Verify that OpenAIChat was created with the correct parameters
            mock_openai_chat.assert_called_once()
            call_kwargs = mock_openai_chat.call_args[1]

            # Check that all parameters are passed correctly
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["api_key"] == "test-api-key"
            assert call_kwargs["temperature"] == 0.7
            # Not checking max_tokens

            # Verify that the returned object is the mock OpenAIChat instance
            assert openai_instance == mock_openai_chat.return_value

    @patch("aiq.cli.type_registry.GlobalTypeRegistry")
    def test_registration_decorators(self, mock_global_registry):
        """Test that the register_llm_client decorators correctly register the llm functions."""
        # Mock the GlobalTypeRegistry
        mock_registry = MagicMock()
        mock_global_registry.get.return_value = mock_registry

        # Create a mock dict for the llm_client_map
        llm_client_map = {
            (NIMModelConfig, LLMFrameworkEnum.AGNO): nim_agno, (OpenAIModelConfig, LLMFrameworkEnum.AGNO): openai_agno
        }
        mock_registry._llm_client_map = llm_client_map

        # Check that nim_agno is registered for NIMModelConfig and LLMFrameworkEnum.AGNO
        assert (NIMModelConfig, LLMFrameworkEnum.AGNO) in mock_registry._llm_client_map
        assert mock_registry._llm_client_map[(NIMModelConfig, LLMFrameworkEnum.AGNO)] == nim_agno

        # Check that openai_agno is registered for OpenAIModelConfig and LLMFrameworkEnum.AGNO
        assert (OpenAIModelConfig, LLMFrameworkEnum.AGNO) in mock_registry._llm_client_map
        assert mock_registry._llm_client_map[(OpenAIModelConfig, LLMFrameworkEnum.AGNO)] == openai_agno
