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

import typing

from pydantic import AfterValidator
from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.cli.register_workflow import register_embedder_provider
from aiq.data_models.embedder import EmbedderBaseConfig

allowed_truncate_values = ["NONE", "START", "END"]


def option_in_allowed_values(v):
    """Ensures option is allowed"""
    assert v in allowed_truncate_values
    return v


TruncationOption = typing.Annotated[str, AfterValidator(option_in_allowed_values)]


class NIMEmbedderModelConfig(EmbedderBaseConfig, name="nim"):
    """A NVIDIA Inference Microservice (NIM) embedder provider to be used with an embedder client."""

    api_key: str | None = Field(default=None, description="NVIDIA API key to interact with hosted NIM.")
    base_url: str | None = Field(default=None, description="Base url to the hosted NIM.")
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The model name for the hosted NIM.")
    truncate: TruncationOption = Field(default="NONE",
                                       description=("The truncation strategy if the input on the "
                                                    "server side if it's too large."))

    model_config = ConfigDict(protected_namespaces=())


@register_embedder_provider(config_type=NIMEmbedderModelConfig)
async def nim_embedder_model(embedder_config: NIMEmbedderModelConfig, builder: Builder):

    yield EmbedderProviderInfo(config=embedder_config, description="A NIM model for use with an Embedder client.")
