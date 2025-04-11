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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.AGNO)
async def nim_agno(llm_config: NIMModelConfig, builder: Builder):

    from agno.models.nvidia import Nvidia

    config_obj = {
        **llm_config.model_dump(exclude={"type", "model_name"}, by_alias=True),
        "id": f"{llm_config.model_name}",
    }

    # Because Agno uses a different environment variable for the API key, we need to set it here manually
    if ("api_key" not in config_obj or config_obj["api_key"] is None):

        if ("NVIDIA_API_KEY" in os.environ):
            # Dont need to do anything. User has already set the correct key
            pass
        else:
            nvidai_api_key = os.getenv("NVIDIA_API_KEY")

            if (nvidai_api_key is not None):
                # Transfer the key to the correct environment variable
                os.environ["NVIDIA_API_KEY"] = nvidai_api_key

    # Create Nvidia instance with conditional base_url
    nvidia_args = {"id": config_obj.get("id")}
    if "base_url" in config_obj and config_obj.get("base_url") is not None:
        nvidia_args["base_url"] = config_obj.get("base_url")
    yield Nvidia(**nvidia_args)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.AGNO)
async def openai_agno(llm_config: OpenAIModelConfig, builder: Builder):

    from agno.models.openai import OpenAIChat

    config_obj = {
        **llm_config.model_dump(exclude={"type"}, by_alias=True),
    }

    yield OpenAIChat(**config_obj)
