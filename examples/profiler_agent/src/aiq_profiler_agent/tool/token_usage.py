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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aiq_profiler_agent.data_models import TokenUsageInfo
from pydantic import BaseModel
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class TokenUsageConfig(FunctionBaseConfig, name="token_usage"):
    """Configuration for the TokenUsage tool."""

    pass


class TokenUsageOutput(BaseModel):
    """Output parameters model for the token usage."""

    trace_id_to_token_usage: dict[str, TokenUsageInfo] = Field(
        ...,
        description="A mapping from trace id to the token usage info",
    )


class TokenUsageInput(BaseModel):
    """Input parameters model for the token usage tool."""

    df_path: str = Field(..., description="Path to the CSV file containing the dataframe")


@register_function(config_type=TokenUsageConfig)
async def token_usage(config: TokenUsageConfig, builder: Builder):
    """Generate a token usage chart for the given LLM traces."""
    temp_dir = tempfile.mkdtemp(prefix="token_usage_")

    async def _create_token_usage(input_obj: TokenUsageInput) -> TokenUsageOutput:
        """
        Create a token usage chart for the given LLM traces.

        Args:
            input_obj: Contains the path to the CSV file with the dataframe
        Returns:
            A dictionary containing a mapping from trace id to the token usage info.
        """
        # Load dataframe from CSV file
        if not os.path.exists(input_obj.df_path):
            raise ValueError(f"Dataframe file not found: {input_obj.df_path}")

        try:
            df = pd.read_csv(input_obj.df_path)

            # Convert string timestamp columns to datetime
            timestamp_columns = ["start_time", "end_time"]
            for col in timestamp_columns:
                if col in df.columns and df[col].dtype == "object":
                    df[col] = pd.to_datetime(df[col])

            logger.info("Successfully loaded dataframe from %s with %d rows", input_obj.df_path, len(df))
        except Exception as e:
            logger.error("Error loading dataframe from CSV: %s", e)
            raise ValueError(f"Failed to load dataframe from {input_obj.df_path}: {e}") from e

        # Group by trace_id
        grouped_traces = df.groupby("context.trace_id")
        logger.info("Found %d different traces", len(grouped_traces))
        token_usage_results = {}

        for trace_id, trace_df in grouped_traces:
            token_usage_results[trace_id] = create_token_usage_chart(trace_df, temp_dir)

        return TokenUsageOutput(trace_id_to_token_usage=token_usage_results)

    try:
        yield FunctionInfo.create(
            single_fn=_create_token_usage,
            description="Create a token usage chart for the given LLM traces.",
            input_schema=TokenUsageInput,
            single_output_schema=TokenUsageOutput,
        )
    except Exception as e:
        logger.error("Error in token_usage tool: %s", e, exc_info=True)
        raise e
    finally:
        # We won't clean up temp_dir here since the image paths will be needed later
        pass


def create_token_usage_chart(df: pd.DataFrame, temp_dir: str) -> TokenUsageInfo:
    """
    Create a bar chart showing token usage for LLM calls.

    Args:
        df: Dataframe containing the trace data
        temp_dir: Directory to save the chart image

    Returns:
        TokenUsageInfo object with chart path and metrics
    """
    # Filter for only LLM spans
    llm_df = df[df["span_kind"] == "LLM"].copy()

    # Generate a unique filename for the chart
    chart_filename = f"token_usage_{uuid.uuid4().hex}.png"
    chart_path = os.path.join(temp_dir, chart_filename)

    if llm_df.empty:
        logger.warning("No LLM calls found in the trace")
        # Create an empty chart if no LLM calls found

        return TokenUsageInfo(
            total_tokens=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            token_usage_detail_chart_path=None,
            average_token_per_second=0.0,
        )

    # Sort by start time
    llm_df = llm_df.sort_values("start_time")

    # Prepare data for plotting
    llm_names = llm_df["name"].tolist()

    # Handle missing token count columns
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []

    for _, row in llm_df.iterrows():
        prompt = row.get("attributes.llm.token_count.prompt", 0)
        completion = row.get("attributes.llm.token_count.completion", 0)
        total = row.get("attributes.llm.token_count.total", 0)

        # If any are NaN or None, handle accordingly
        prompt = 0 if pd.isna(prompt) else int(prompt)
        completion = 0 if pd.isna(completion) else int(completion)
        total = 0 if pd.isna(total) else int(total)

        # If total is missing but we have prompt and completion, calculate it
        if total == 0 and (prompt > 0 or completion > 0):
            total = prompt + completion

        # If we have total but missing prompt or completion, estimate
        if total > 0:
            if prompt == 0 and completion > 0:
                prompt = total - completion
            elif completion == 0 and prompt > 0:
                completion = total - prompt

        prompt_tokens.append(prompt)
        completion_tokens.append(completion)
        total_tokens.append(total)

    # Calculate token generation speed
    durations = []
    tokens_per_second = []

    for i, row in llm_df.iterrows():
        duration = (row["end_time"] - row["start_time"]).total_seconds()
        durations.append(duration)

        # Only calculate for completion tokens
        completion = completion_tokens[llm_df.index.get_loc(i)]
        if duration > 0 and completion > 0:
            tokens_per_second.append(completion / duration)
        else:
            tokens_per_second.append(0)

    # Set up the figure and axes for token counts
    fig, ax = plt.subplots(figsize=(14, 8))

    # Bar positions
    x = np.arange(len(llm_names))
    width = 0.2  # Make bars even narrower to fit four bars

    # Scale tokens per second to be visible alongside token counts
    # Find an appropriate scaling factor
    max_token_count = max(max(prompt_tokens or [0]), max(completion_tokens or [0]), max(total_tokens or [0]))
    max_tps = max(tokens_per_second or [0])
    scaling_factor = max_token_count / max(max_tps, 1) * 0.8  # Scale tokens/sec to be visible but not dominate

    scaled_tokens_per_second = [tps * scaling_factor for tps in tokens_per_second]

    # Add name index to each LLM call for clarity
    labeled_names = [f"{i + 1}. {name}" for i, name in enumerate(llm_names)]

    # Plot bars
    bars1 = ax.bar(x - width * 1.5, prompt_tokens, width, label="Prompt Tokens", color="skyblue")
    bars2 = ax.bar(x - width * 0.5, completion_tokens, width, label="Completion Tokens", color="lightgreen")
    bars3 = ax.bar(x + width * 0.5, total_tokens, width, label="Total Tokens", color="orange", alpha=0.8)
    bars4 = ax.bar(x + width * 1.5, scaled_tokens_per_second, width, label="Tokens/Sec", color="red", alpha=0.7)

    # Add token count labels on bars
    def add_labels(bars, values=None, format_str="{:.0f}"):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                # Use original values for tokens/sec display
                display_value = values[i] if values is not None else height
                ax.annotate(
                    format_str.format(display_value),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    # For tokens/sec, use original unscaled values with 2 decimal places
    add_labels(bars4, tokens_per_second, "{:.2f}")

    # Configure axes
    ax.set_ylabel("Token Count")
    ax.set_title("LLM Token Usage and Generation Speed by Call")
    ax.set_xticks(x)
    ax.set_xticklabels(labeled_names, rotation=45, ha="right")
    ax.legend()

    # Add a second y-axis for tokens/sec for reference
    ax2 = ax.twinx()
    ax2.set_ylabel("Tokens per Second")
    # Set y-limits based on the scaling
    ax2.set_ylim(0, ax.get_ylim()[1] / scaling_factor)

    # Add grid and adjust layout
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Calculate summary metrics
    total_prompt_sum = sum(prompt_tokens)
    total_completion_sum = sum(completion_tokens)
    total_tokens_sum = sum(total_tokens)

    # Calculate average tokens per second
    total_duration = sum(durations)
    avg_tokens_per_second = sum(completion_tokens) / total_duration if total_duration > 0 else 0

    # Save the chart to disk
    plt.savefig(chart_path)
    plt.close(fig)

    logger.info("Saved token usage chart to %s", chart_path)

    # Create and return TokenUsageInfo
    return TokenUsageInfo(
        total_tokens=total_tokens_sum,
        total_prompt_tokens=total_prompt_sum,
        total_completion_tokens=total_completion_sum,
        token_usage_detail_chart_path=chart_path,
        average_token_per_second=avg_tokens_per_second,
    )
