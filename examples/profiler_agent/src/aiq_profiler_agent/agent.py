import logging
import uuid
from typing import Any
from typing import TypedDict

from aiq_profiler_agent.data_models import ExecPlan
from aiq_profiler_agent.data_models import TraceInfo
from aiq_profiler_agent.tool.flow_chart import FlowChartOutput
from aiq_profiler_agent.tool.px_query import PxQueryOutput
from aiq_profiler_agent.tool.token_usage import TokenUsageOutput
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)


class ProfilerAgentState(TypedDict):
    """State for the ProfilerAgent."""

    exec_plan: ExecPlan
    messages: list[BaseMessage]
    df_path: str | None = None
    trace_infos: dict[str, TraceInfo] | None = None
    end_condition: bool = False
    retry_count: int = 0
    user_query: str | None = None


class ProfilerAgent:
    """Agent for profiling LLM traces."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: dict[str, BaseTool],
        response_composer_tool: BaseTool,
        detailed_logs: bool = False,
        max_retries: int = 3,
        retry_prompt: str = "",
    ):
        self.llm = llm
        self.detailed_logs = detailed_logs
        self.tools = tools
        self.callbacks = []
        self.max_retries = max_retries
        self.retry_prompt = retry_prompt
        # pydantic parser
        self.output_parser = PydanticOutputParser(pydantic_object=ExecPlan)
        self.response_composer = response_composer_tool
        self.graph = None
        logger.info("ProfilerAgent initialized")

    async def conditional_edge(self, state: ProfilerAgentState):
        try:
            logger.debug("Starting the Tool Calling Conditional Edge")
            if "exec_plan" in state and len(state["exec_plan"].tools) > 0:
                return "executor"
            else:
                return "response_composer"
        except Exception as ex:
            if "retry_count" in state and state["retry_count"] >= self.max_retries:
                logger.warning("Max retries reached, returning without meaningful output")
                state["messages"].append(AIMessage(content="No meaningful output, please try again with another query"))
                return "__end__"
            else:
                state.setdefault("retry_count", 1)
                logger.warning(
                    "Error in the conditional edge: %s, retrying %d times out of %d",
                    ex,
                    state["retry_count"],
                    self.max_retries,
                )
                return "agent"

    async def build_graph(self):
        try:
            logger.debug("Building and compiling the Agent Graph")
            graph = StateGraph(ProfilerAgentState)
            graph.add_node("agent", self.agent_node)
            graph.add_node("response_composer", self.response_composer_node)
            graph.add_node("executor", self.executor_node)
            graph.add_conditional_edges(
                "agent",
                self.conditional_edge,
            )
            graph.add_conditional_edges(
                "executor",
                self.conditional_edge,
            )
            graph.set_entry_point("agent")
            graph.set_finish_point("response_composer")
            self.graph = graph.compile()
            logger.info("ProfilerAgent Graph built and compiled successfully")
            return self.graph
        except Exception as ex:
            logger.exception("Failed to build ProfilerAgent Graph: %s", ex, exc_info=ex)
            raise ex

    async def agent_node(self, state: ProfilerAgentState):
        try:
            logger.debug("Starting Agent Node")
            logger.info("Calling agent to plan the execution")
            if len(state["messages"]) == 0:
                raise RuntimeError('No input received in state: "messages"')

            response = await self.llm.ainvoke(state["messages"], config=RunnableConfig(callbacks=self.callbacks))
            if self.detailed_logs:
                logger.debug("The agent's input was:\n%s", state["messages"])
                logger.debug("The agent's output is:\n%s", response)
            # parse the response to get the exec_plan
            try:
                exec_plan = self.output_parser.parse(response.content)
                logger.info("Agent planned the execution: %s", exec_plan)
                state["exec_plan"] = exec_plan
            except Exception as ex:
                logger.warning("Failed to parse the agent's output: %s", response.content)
                state.setdefault("retry_count", 0)
                message = self.retry_prompt.format(error=ex, output_parser=self.output_parser.get_format_instructions())
                state["messages"].append(HumanMessage(content=message))
            return state
        except Exception as ex:
            logger.exception("Failed to call agent_node: %s", ex, exc_info=True)
            raise ex

    async def executor_node(self, state: ProfilerAgentState):
        # check if the tool is px_query
        try:
            if state["exec_plan"].tools[0] == "px_query":
                query_result = await self.tools["px_query"].ainvoke(input={**state["exec_plan"].model_dump()})
                self.update_state(state, query_result)
                state["exec_plan"].tools.popleft()
            else:
                tool_name = state["exec_plan"].tools.popleft()
                tool_result = await self.tools[tool_name].ainvoke(input={"df_path": state["df_path"]})
                self.update_state(state, tool_result)
        except Exception as ex:
            logger.exception("Failed to call executor_node: %s", ex, exc_info=True)
            raise ex
        return state

    async def response_composer_node(self, state: ProfilerAgentState):
        try:
            if len(state["trace_infos"]) == 0:
                state["messages"].append(HumanMessage(content="No traces retrieved. Exiting..."))
            else:
                tool_response = await self.response_composer.ainvoke(input={"trace_infos": state["trace_infos"]})
                self.update_state(state, tool_response)
            return state
        except Exception as ex:
            logger.exception("Failed to call response_composer_node: %s", ex, exc_info=True)
            raise ex

    def update_state(self, state: ProfilerAgentState, tool_response: Any) -> ProfilerAgentState:
        """Update the state with the tool response."""
        match tool_response:
            case PxQueryOutput():
                state["df_path"] = tool_response.df_path
                for trace_id, user_query in tool_response.user_queries.items():
                    state["trace_infos"].setdefault(trace_id, TraceInfo()).user_query = user_query
                state["messages"].append(
                    HumanMessage(content=f"PxQuery returned a PxDataFrame with {tool_response.row_count} rows"
                                 f"you can use it to analyze the traces by calling tools, you can omit the "
                                 f"dataframe parameter in tool calls, it will be automatically added. "
                                 f"Don't call px_query tool again! "
                                 f"You should call all analysis tools unless user specifies otherwise."))

            case FlowChartOutput():
                # update the trace_infos with the flow chart
                for trace_id, flow_info in tool_response.trace_id_to_flow_info.items():
                    state["trace_infos"].setdefault(trace_id, TraceInfo()).flow_info = flow_info
                num_traces = len(tool_response.trace_id_to_flow_info)
                state["messages"].append(
                    HumanMessage(content=f"FlowChartOutput returned a FlowChartOutput with {num_traces} traces"))
            case TokenUsageOutput():
                # update the trace_infos with the token usage
                for trace_id, token_usage in tool_response.trace_id_to_token_usage.items():
                    state["trace_infos"].setdefault(trace_id, TraceInfo()).token_usage_info = token_usage
                state["messages"].append(
                    HumanMessage(content=f"TokenUsageOutput returned a TokenUsageOutput with "
                                 f"{len(tool_response.trace_id_to_token_usage)} traces"))
            case str():
                state["messages"].append(ToolMessage(content=tool_response, tool_call_id=uuid.uuid4()))
            case _:
                raise ValueError(f"Unsupported tool response type: {type(tool_response)}")

        return state
