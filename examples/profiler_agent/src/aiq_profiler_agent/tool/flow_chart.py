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

import io
import logging
import os
import tempfile
import uuid

import matplotlib.pyplot as plt
import pandas as pd
from aiq_profiler_agent.data_models import TraceFlowInfo
from matplotlib.patches import Patch
from pydantic import BaseModel
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class FlowChartConfig(FunctionBaseConfig, name="flow_chart"):
    """Configuration for the FlowChart tool."""

    pass


class FlowChartOutput(BaseModel):
    """Output parameters model for the flow chart."""

    trace_id_to_flow_info: dict[str, TraceFlowInfo] = Field(
        ...,
        description="A mapping from trace id to the flow chart info",
    )


# Add a new input model that works with a file path
class FlowChartInput(BaseModel):
    """Input parameters model for the flow chart tool."""

    df_path: str = Field(..., description="Path to the CSV file containing the dataframe")


@register_function(config_type=FlowChartConfig)
async def flow_chart(config: FlowChartConfig, builder: Builder):
    """Generate a flow chart for the given LLM traces."""
    temp_dir = tempfile.mkdtemp(prefix="flow_chart_")

    async def _create_flow_chart(input_obj: FlowChartInput) -> FlowChartOutput:
        """Create a flow chart for the given LLM traces.

        Args:
            input_obj: Contains the path to the CSV file with the dataframe
        Returns:
            A dictionary containing a mapping from trace id to the flow chart info.
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
        charts = {}
        for trace_id, trace_df in grouped_traces:
            charts[trace_id] = create_trace_flow_diagram(trace_df, temp_dir)
        return FlowChartOutput(trace_id_to_flow_info=charts)

    try:
        yield FunctionInfo.create(
            single_fn=_create_flow_chart,
            description="Create a flow chart for the given LLM traces.",
            input_schema=FlowChartInput,
            single_output_schema=FlowChartOutput,
        )
    except Exception as e:
        logger.error("Error in flow_chart tool", error=e, exec_info=True)
        raise e
    finally:
        # clean up temp directory
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def create_trace_flow_diagram(df: pd.DataFrame, temp_dir: str) -> TraceFlowInfo:
    """
    Create a simple flow diagram visualization
    Each span is plotted separately regardless of name.
    """
    # Sort by start_time
    df = df.sort_values("start_time")

    # Calculate seconds from start
    min_time = df["start_time"].min()
    df["start_seconds"] = (df["start_time"] - min_time).dt.total_seconds()
    df["end_seconds"] = (df["end_time"] - min_time).dt.total_seconds()
    df["duration"] = df["end_seconds"] - df["start_seconds"]
    # Filter out spans with very short durations (less than 0.02 seconds)

    df = df[df["span_kind"] != "UNKNOWN"]
    # Create figure
    _, ax = plt.subplots(figsize=(14, 10))

    # Define colors for different span kinds
    colors = {"LLM": "skyblue", "CHAIN": "lightgreen", "TOOL": "orange"}

    # Create a vertical timeline with connections
    y_positions = {}
    max_x = 0

    # First pass: determine y positions for each span
    for i, (_, row) in enumerate(df.iterrows()):
        span_id = row["context.span_id"]
        y_positions[span_id] = i

    # Second pass: draw boxes and connections
    for i, (_, row) in enumerate(df.iterrows()):
        span_id = row["context.span_id"]
        span_kind = row["span_kind"]
        name = row["name"]
        x_start = row["start_seconds"]
        x_end = row["end_seconds"]
        max_x = max(max_x, x_end)

        # Draw span box
        color = colors.get(span_kind, "lightgray")
        rect = plt.Rectangle((x_start, 0.5 * i - 0.2), x_end - x_start, 0.4, color=color, alpha=0.8, edgecolor="black")

        ax.add_patch(rect)

        # Add label
        duration_text = f"{row['duration']:.2f}s"
        if i == 0:
            txt = f"{name}: {row['user_query']}\n{row['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n{duration_text}"
        else:
            txt = f"{name}\n{duration_text}"
        ax.text(x_start + (x_end - x_start) / 2, 0.5 * i - 0.05, txt, ha="center", va="center", fontsize=12)

    # Set y-axis ticks (no labels to avoid clutter)
    ax.set_yticks([])

    # Set x-axis limits and labels
    ax.set_xlim(-1, max_x + 1)
    ax.set_ylim(-1, len(df))
    ax.set_xlabel("Time (seconds from start)")

    # Add grid for better readability
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # Add title
    ax.set_title("Trace Flow Diagram")

    # Create legend handles (these won't be drawn on the plot)
    legend_handles = [Patch(color=color, label=kind) for kind, color in colors.items()]

    # Add the legend using handles
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    # Convert the plot to a base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_path = os.path.join(temp_dir, f"flow_chart_{uuid.uuid4().hex}.png")
    with open(image_path, "wb") as f:
        f.write(buf.getvalue())
    return TraceFlowInfo(
        start_time=df["start_time"].iloc[0],
        end_time=df["end_time"].iloc[-1],
        flow_chart_path=image_path,
    )
