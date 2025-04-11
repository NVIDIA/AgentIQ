from collections import deque
from datetime import datetime

import pandas as pd
from pydantic import BaseModel
from pydantic import Field


class ExecPlan(BaseModel):
    """The execution plan of the trace."""

    tools: deque[str] = Field(
        ...,
        description="The tools to be used to answer the user query,"
        "you should only include the exact names of each tool",
    )

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


class TraceFlowInfo(BaseModel):
    """Information about a trace."""

    start_time: datetime = Field(..., description="The start time of the trace")
    end_time: datetime = Field(..., description="The end time of the trace")
    flow_chart_path: str = Field("", description="The path to the flow chart of the trace")


class TokenUsageInfo(BaseModel):
    """Information about the token usage of a trace."""

    total_tokens: int = Field(..., description="The total number of tokens in the trace")
    total_prompt_tokens: int = Field(..., description="The total number of prompt tokens in the trace")
    total_completion_tokens: int = Field(..., description="The total number of completion tokens in the trace")
    token_usage_detail_chart_path: str | None = Field(
        ...,
        description="The path to the token usage detail chart of the trace, "
        "if the trace doesn't call LLM, this field will be None",
    )
    average_token_per_second: float = Field(..., description="The average token price per second in the trace")


class PxDataFrame(BaseModel):
    """Output parameters model for Phoenix server queries."""

    data: list[dict] = Field(
        ...,
        description="The dataframe recordsof LLM traces.",
    )
    index: list[str] = Field(
        ...,
        description="The index of the dataframe.",
    )
    columns: list[str] = Field(
        ...,
        description="The columns of the dataframe.",
    )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, index=self.index, columns=self.columns)


class TraceInfo(BaseModel):
    """Information about a trace."""

    flow_info: TraceFlowInfo | None = Field(None, description="The flow chart of the trace.")
    token_usage_info: TokenUsageInfo | None = Field(None, description="The token usage of the trace.")
    user_query: str | None = Field(None, description="The user query to the agent")
