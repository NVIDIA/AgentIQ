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

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.cli.register_workflow import register_embedder_provider
from aiq.data_models.embedder import EmbedderBaseConfig


class OpenAIEmbedderModelConfig(EmbedderBaseConfig, name="openai"):
    """An OpenAI LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=())

    api_key: str | None = Field(default=None, description="OpenAI API key to interact with hosted model.")
    base_url: str | None = Field(default=None, description="Base url to the hosted model.")
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The OpenAI hosted model name.")
    max_retries: int = Field(default=2, description="The max number of retries for the request.")


@register_embedder_provider(config_type=OpenAIEmbedderModelConfig)
async def openai_llm(config: OpenAIEmbedderModelConfig, builder: Builder):

    yield EmbedderProviderInfo(config=config, description="An OpenAI model for use with an Embedder client.")
