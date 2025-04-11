import logging
from datetime import datetime

from aiq_profiler_agent.agent import ProfilerAgent
from aiq_profiler_agent.agent import ProfilerAgentState
from aiq_profiler_agent.data_models import ExecPlan
from aiq_profiler_agent.tool import flow_chart  # noqa: F401
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph.graph import CompiledGraph
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ProfilerAgentConfig(FunctionBaseConfig, name="profiler_agent"):
    """
    Profiler agent config
    """

    llm_name: LLMRef = Field(..., description="The LLM to use for the profiler agent")
    max_iterations: int = Field(..., description="The maximum number of iterations for the profiler agent")
    tools: list[str] = Field(..., description="The tools to use for the profiler agent")

    sys_prompt: str = Field(
        """You are a helpful assistant that analyzes LLM traces from Phoenix server.

IMPORTANT: You MUST ONLY return a valid JSON object matching the format below.
Do not include any explanations or text outside the JSON.

Based on the user query, create an execution plan:
1. First determine which tools to use from: {tools}
2. Create a list of these tools in the exact order to execute them

Your response MUST follow these strict requirements:
- You MUST use px_query tool FIRST
- You SHOULD use each tool at most once
- For queries not specifying tools, use all available tools
- Start time and end time should be in ISO format (YYYY-MM-DD HH:MM:SS)

RESPONSE FORMAT:
{output_parser}

USER QUERY: {query}
CURRENT TIME: {current_time}
""",
        description="The prompt to use for the PxQuery tool.",
    )

    retry_prompt: str = Field(
        """The output you provided wasn't in the expected format. Please fix the issues below and try again:

{error}

IMPORTANT REMINDER:
1. Your response must ONLY contain a valid JSON object
2. DO NOT include any explanation text before or after the JSON
3. Make sure the 'tools' field is a list of strings in the exact order they should be executed
4. Include px_query first.

EXPECTED FORMAT:
{output_parser}
""",
        description="Prompt to use when retrying after parser failure",
    )

    max_retries: int = Field(
        ...,
        description="The maximum number of retries for the profiler agent",
    )


@register_function(config_type=ProfilerAgentConfig)
async def profiler_agent(config: ProfilerAgentConfig, builder: Builder):
    """
    Profiler agent that uses Phoenix to analyze LLM telemetry data
    This agent retrieves LLM telemetry data using Phoenix's Client API
    and analyzes the data to provide insights about LLM usage, performance,
    and issues.
    """

    # Create the agent executor
    # llm = builder.get_llm(config.llm_name)  # type: ignore
    tools = builder.get_tools(tool_names=config.tools, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    output_parser = PydanticOutputParser(pydantic_object=ExecPlan)
    tools_dict = {tool.name: tool for tool in tools}
    graph: CompiledGraph = await ProfilerAgent(
        llm=llm,
        tools=tools_dict,
        response_composer_tool=builder.get_tool("response_composer", wrapper_type=LLMFrameworkEnum.LANGCHAIN),
        detailed_logs=True,
        max_retries=config.max_retries,
        retry_prompt=config.retry_prompt,
    ).build_graph()

    async def _profiler_agent(input_message: str) -> str:
        """
        Profiler agent that uses Phoenix to analyze LLM telemetry data
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = PromptTemplate(
            template=config.sys_prompt,
            input_variables=["query"],
            partial_variables={
                "current_time": current_time,
                "output_parser": output_parser.get_format_instructions(),
                "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
            },
        )

        state = ProfilerAgentState(messages=[SystemMessage(content=prompt.format(query=input_message))], trace_infos={})
        state = await graph.ainvoke(state, config={"recursion_limit": (config.max_iterations + 1) * 2})
        return state["messages"][-1].content

    try:
        yield FunctionInfo.create(single_fn=_profiler_agent)
    except Exception as e:
        logger.error("Error in profiler agent, exit early", exc_info=True)
        raise e
    finally:
        logger.info("Profiler agent finished")
