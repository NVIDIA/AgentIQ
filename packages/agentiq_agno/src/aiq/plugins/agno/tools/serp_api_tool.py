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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# Module level variable to track empty query handling
_empty_query_handled = False


class SerpApiToolConfig(FunctionBaseConfig, name="serp_api_tool"):
    """
    Tool that retrieves search results from the web using SerpAPI.
    Requires a SERP_API_KEY.
    """
    _type: str = "serp_api_tool"  # Flat type name without namespacing
    api_key: str | None = None
    max_results: int = 5


@register_function(config_type=SerpApiToolConfig, framework_wrappers=[LLMFrameworkEnum.AGNO])
async def serp_api_tool(tool_config: SerpApiToolConfig, builder: Builder):
    """
    Create a SerpAPI search tool for use with Agno.

    This creates a search function that uses SerpAPI to search the web.

    Parameters
    ----------
    tool_config : SerpApiToolConfig
        Configuration for the SerpAPI tool
    builder : Builder
        The AgentIQ builder instance

    Returns
    -------
    A FunctionInfo object wrapping the SerpAPI search functionality
    """
    from agno.tools.serpapi import SerpApiTools

    if (not tool_config.api_key):
        tool_config.api_key = os.getenv("SERP_API_KEY")

    if not tool_config.api_key:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `SERP_API_KEY`")

    # Create the SerpAPI tools instance
    search_tool = SerpApiTools(api_key=tool_config.api_key)

    # Simple search function with a single string parameter
    async def _serp_api_search(query: str = "") -> str:
        """
        Search the web using SerpAPI.

        Args:
            query: The search query to perform. If empty, returns initialization message.

        Returns:
            Formatted search results or initialization message
        """
        # Use the module-level variable for tracking
        global _empty_query_handled

        # Handle the case where no query is provided
        if not query or query.strip() == "":
            # Only provide initialization message once, then provide a more direct error
            if not _empty_query_handled:
                _empty_query_handled = True
                logger.info("Empty query provided, returning initialization message (first time)")
                return "SerpAPI Tool is initialized and ready for use. Please provide a search query."
            else:
                logger.warning("Empty query provided again, returning error message to stop looping")
                return "ERROR: Search query cannot be empty. Please provide a specific search term to continue."
        else:
            # Reset the empty query flag when we get a valid query
            _empty_query_handled = False

        logger.info(f"Searching SerpAPI with query: '{query}', max_results: {tool_config.max_results}")

        try:
            # Perform the search
            results = await search_tool.search_google(query=query, num_results=tool_config.max_results)
            logger.info(f"SerpAPI returned {len(results)} results")

            # Format the results as a string
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No Title')
                link = result.get('link', 'No Link')
                snippet = result.get('snippet', 'No Snippet')

                formatted_result = f'<Document href="{link}"/>\n'
                formatted_result += f'# {title}\n\n'
                formatted_result += f'{snippet}\n'
                formatted_result += '</Document>'
                formatted_results.append(formatted_result)

            return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            logger.exception(f"Error searching with SerpAPI: {e}")
            return f"Error performing search: {str(e)}"

    # Create a FunctionInfo object with simple string parameter
    fn_info = FunctionInfo.from_fn(
        _serp_api_search,
        description="""
            This tool searches the web using SerpAPI and returns relevant results for the given query.

            Args:
                query (str, optional): The search query to perform. A valid search query is required.

            Returns:
                str: Formatted search results or error message if query is empty.
        """,
    )

    yield fn_info
