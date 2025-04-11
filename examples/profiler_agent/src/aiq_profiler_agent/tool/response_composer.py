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
import base64
import logging

from aiq_profiler_agent.data_models import TraceInfo
from pydantic import BaseModel
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ResponseComposerConfig(FunctionBaseConfig, name="response_composer"):
    """Configuration for the ResponseComposer tool."""

    pass


class ResponseComposerInput(BaseModel):
    """Input parameters model for the ResponseComposer tool."""

    trace_infos: dict[str, TraceInfo] = Field(
        ...,
        description="The trace infos for the traces that user want to analyze.",
    )


@register_function(config_type=ResponseComposerConfig)
async def response_composer(config: ResponseComposerConfig, builder: Builder):
    """
    Compose a response from the given LLM traces.
    """

    async def _compose_response(input_obj: ResponseComposerInput) -> str:
        """
        Compose a response from the given LLM traces.
        """
        ret = ""

        # Add individual trace details
        for trace_id in input_obj.trace_infos:
            ret += f"Trace id: {trace_id}:<br><br>"
            ret += f"User query: {input_obj.trace_infos[trace_id].user_query}<br><br>"
            if input_obj.trace_infos[trace_id].flow_info and input_obj.trace_infos[trace_id].flow_info.flow_chart_path:
                with open(input_obj.trace_infos[trace_id].flow_info.flow_chart_path, "rb") as image_file:
                    image_data = image_file.read()
                    image_data_base64 = base64.b64encode(image_data).decode("utf-8")
                ret += f"Start time: {input_obj.trace_infos[trace_id].flow_info.start_time}<br><br>"
                ret += f"End time: {input_obj.trace_infos[trace_id].flow_info.end_time}<br><br>"
                duration = (input_obj.trace_infos[trace_id].flow_info.end_time -
                            input_obj.trace_infos[trace_id].flow_info.start_time)
                ret += f"Duration: {duration}<br><br>"
                ret += f"Flow chart: <br><br>![image.jpeg](data:image/jpeg;base64,{image_data_base64})<br><br>"
            if (input_obj.trace_infos[trace_id].token_usage_info
                    and input_obj.trace_infos[trace_id].token_usage_info.token_usage_detail_chart_path):
                with open(input_obj.trace_infos[trace_id].token_usage_info.token_usage_detail_chart_path,
                          "rb") as image_file:
                    image_data = image_file.read()
                    image_data_base64 = base64.b64encode(image_data).decode("utf-8")
                ret += "Total prompt tokens: "
                ret += f"{input_obj.trace_infos[trace_id].token_usage_info.total_prompt_tokens}<br><br>"
                ret += "Total completion tokens: "
                ret += f"{input_obj.trace_infos[trace_id].token_usage_info.total_completion_tokens}<br><br>"
                ret += "Total tokens: "
                ret += f"{input_obj.trace_infos[trace_id].token_usage_info.total_tokens}<br><br>"
                ret += "Token usage: "
                ret += f"<br><br>![image.jpeg](data:image/jpeg;base64,{image_data_base64})<br><br>"

        # Clean up the temporary directory
        return ret

    yield FunctionInfo.create(
        single_fn=_compose_response,
        description="Compose a response from the given LLM traces.",
        input_schema=ResponseComposerInput,
    )
