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

import logging
import os
import tempfile
import uuid
from datetime import datetime

from aiq_profiler_agent.tool.utils import first_valid_query
from pydantic import BaseModel
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PxQueryConfig(FunctionBaseConfig, name="px_query"):
    """Configuration for the PxQuery tool."""

    phoenix_url: str = Field(
        ...,
        description="The URL of the Phoenix server.",
    )
    time_window_seconds: int = Field(
        600,
        description="The time window in seconds for each trace, used for last-n queries",
    )


class PxQueryOutput(BaseModel):
    """Output parameters model for the Phoenix query."""

    df_path: str | None = Field(None, description="Path to the CSV file containing the dataframe")
    row_count: int = Field(..., description="Number of rows in the dataframe")
    user_queries: dict[str, str] = Field(
        default_factory=dict,
        description="A mapping from trace id to the user query",
    )


class PxQueryInput(BaseModel):
    """Input parameters model for Phoenix server queries."""

    # not used for now, we will filter the dataframe in the tool
    # filter_conditions: list[str] | None = Field(
    #     None,
    #     description="The filter conditions to apply to the dataframe.",
    # )
    start_time: datetime | None = Field(
        None,
        description="The start time to apply to the dataframe. leave blank for last-n queries",
    )
    end_time: datetime = Field(
        ...,
        description="The end time to apply to the dataframe. for last-n queries, use the current datetime",
    )
    last_n: int | None = Field(
        None,
        description="The number of traces to return. for queries of a specific time range, use None. "
        "If user ask about the last run, then use 1",
    )
    project_name: str = Field(
        "default",
        description="The project name to apply to the dataframe, if not provided, use 'default'",
    )


@register_function(config_type=PxQueryConfig)
async def px_query(config: PxQueryConfig, builder: Builder):
    """Query the Phoenix server for a dataframe of LLM traces"""
    import phoenix as px

    px_client = px.Client(endpoint=config.phoenix_url)

    # Create a temporary directory for storing CSV files
    temp_dir = tempfile.mkdtemp(prefix="px_query_")
    logger.info("Created temporary directory for px_query: %s", temp_dir)

    async def _query_phoenix(px_api_input: PxQueryInput) -> PxQueryOutput:
        """Takes the user query and calls the Phoenix server to get the dataframe of LLM traces.

        Args:
            query (PxQueryInput): The user query to query the Phoenix server
            user can provide the project name, otherwise it will use the default project name.
            user can also provide the time range of the traces, or last n traces.

        Returns:
            PxQueryOutput: Contains the path to the CSV file with the dataframe.
        """
        logger.info(
            "Querying Phoenix server at %s, with user query: %s",
            config.phoenix_url,
            px_api_input,
        )

        logger.info("Phoenix query px_api_input: %s", px_api_input)
        logger.info("Querying Phoenix server for traces between %s and %s",
                    px_api_input.start_time,
                    px_api_input.end_time)
        df = px_client.get_spans_dataframe(
            project_name=px_api_input.project_name,
            start_time=px_api_input.start_time,
            end_time=px_api_input.end_time,
        )
        # filter out last n traces based, sorted by start time
        if px_api_input.last_n:
            trace_latest_times = (df.groupby("context.trace_id")["start_time"].min().sort_values(
                ascending=False).reset_index())
            if len(trace_latest_times) < px_api_input.last_n:
                logger.warning(
                    "Not enough traces found in the time range %s to %s",
                    px_api_input.start_time,
                    px_api_input.end_time,
                )
                raise ValueError(
                    f"Not enough traces found in the time range {px_api_input.start_time} to {px_api_input.end_time}")
            last_n_trace_ids = trace_latest_times.head(px_api_input.last_n)["context.trace_id"].tolist()
            df = df[df["context.trace_id"].isin(last_n_trace_ids)]
            logger.info("Filtered dataframe to last %d traces", px_api_input.last_n)

        if len(df) == 0:
            logger.warning("No traces found in the time range %s to %s", px_api_input.start_time, px_api_input.end_time)
            raise ValueError(f"No traces found in the time range {px_api_input.start_time} to {px_api_input.end_time}")

        # Save the dataframe to a CSV file
        df_path = os.path.join(temp_dir, f"px_query_{uuid.uuid4().hex}.csv")
        # First, extract user query from each span
        # Then, apply first_valid_query for each trace ID to fill in None values
        valid_queries = df.groupby("context.trace_id")["attributes.input.value"].apply(first_valid_query).to_dict()
        # Apply the valid queries back to the dataframe
        df["user_query"] = df["context.trace_id"].map(valid_queries)

        df.to_csv(df_path, index=False)
        logger.info("Saved dataframe to %s with %d rows", df_path, len(df))

        return PxQueryOutput(df_path=df_path, row_count=len(df), user_queries=valid_queries)

    yield FunctionInfo.create(
        single_fn=_query_phoenix,
        description="Query the Phoenix server for a dataframe of LLM traces, run this tool before other tools",
        input_schema=PxQueryInput,
        single_output_schema=PxQueryOutput,
    )
    # clear the temporary directory
    import shutil

    shutil.rmtree(temp_dir)
