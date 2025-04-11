<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Personal Finance

<!-- Note: "Agno" is the official product name despite Vale spelling checker warnings -->
Built on [`Agno`](https://github.com/agno-agi/agno) and AgentIQ, this workflow is a personal financial planner that generates personalized financial plans using NVIDIA NIM (can be customized to use OpenAI models). It automates the process of researching, planning, and creating tailored budgets, investment strategies, and savings goals, empowering you to take control of your financial future with ease.

This personal financial planner was revised based on the [Awesome-LLM-App](https://github.com/Shubhamsaboo/awesome-llm-apps) GitHub repo's [AI Personal Finance Planner](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/ai_agent_tutorials/ai_personal_finance_agent) sample.


## Table of Contents

- [Personal Finance](#personal-finance)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation and Setup](#installation-and-setup)
    - [Setup Virtual Environment and Install AgentIQ](#setup-virtual-environment-and-install-agentiq)
    - [Install this Workflow:](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
  - [Example Usage](#example-usage)
    - [Run the Workflow](#run-the-workflow)
  - [Deployment-Oriented Setup](#deployment-oriented-setup)
    - [Build the Docker Image](#build-the-docker-image)
    - [Run the Docker Container](#run-the-docker-container)
    - [Test the API](#test-the-api)
    - [Expected API Output](#expected-api-output)


## Key Features

### AgentIQ

- **Pre-built Tools:** Leverages core AgentIQ library agent and tools.
- **ReAct Agent:** Performs reasoning between tool call; utilizes tool names and descriptions to appropriately route to the correct tool
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

### `Agno`

`Agno` is a lightweight library for building multimodal agents. Some of the key features of `Agno` include lightning fast, model agnostic, multimodal, multi agent, etc.  See `Agno` README [here](https://github.com/agno-agi/agno/blob/main/README.md) for more information about the library.


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e examples/agno_personal_finance
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export SERP_API_KEY=<SERP_API_KEY>
```

## Example Usage

### Run the Workflow

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/agno_personal_finance/src/agno_personal_finance/configs/config.yml --input "My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA."
```

**Expected Output**
```console
$ aiq run --config_file examples/agno_personal_finance/src/agno_personal_finance/configs/config.yml --input "My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA."
2025-03-25 15:30:44,745 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (906.506062 ms). Ensure all imports are inside your registered functions.
2025-03-25 15:30:44,911 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples\agno_personal_finance\src\agno_personal_finance\configs\config.yml'
2025-03-25 15:30:44,912 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
{'api_key': None, 'base_url': None, 'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 300, 'id': 'meta/llama-3.3-70b-instruct'}
2025-03-25 15:30:45,585 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-03-25 15:30:45,688 - aiq.agent.tool_calling_agent.agent - INFO - Initialized Tool Calling Agent Graph
2025-03-25 15:30:45,698 - aiq.agent.tool_calling_agent.agent - INFO - Tool Calling Agent Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: tool_calling_agent
Number of Functions: 1
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-03-25 15:30:45,715 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA.',)
2025-03-25 15:30:45,723 - aiq.agent.tool_calling_agent.agent - INFO - Calling agent
2025-03-25 15:30:47,585 - aiq.agent.tool_calling_agent.agent - INFO - Calling tools: ['agno_personal_finance']
2025-03-25 15:30:52,484 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-03-25 15:30:53,479 - httpx - INFO - HTTP Request: POST https://api.agno.com/v1/telemetry/agent/run/create "HTTP/1.1 200 OK"
2025-03-25 15:30:53,485 - agno_personal_finance.agno_personal_finance_function - INFO - Research results:
  RunResponse(content='Based on the user\'s financial goals and current financial situation, I would generate the following list of 3 search terms:\n1. "retirement planning for 40-year-old tech professionals"\n2. "investments for long-term wealth creation for NVIDIA employees"\n3. "savings strategies for early retirement at 60"\n\nNow, let\'s search for each term:\n\n<function(search_google){"query": "retirement planning for 40-year-old tech professionals", "num_results": 10}</function>', content_type='str', thinking=None, event='RunResponse', messages=[Message(role='system', content="You are a world-class financial researcher. Given a user's financial goals and current financial situation,\ngenerate a list of search terms for finding relevant financial advice, investment opportunities, and savings strategies.\nThen search the web for each term, analyze the results, and return the 10 most relevant results.\n\n\n<your_role>\nSearches for financial advice, investment opportunities, and savings strategies based on user preferences\n</your_role>\n\n<instructions>\n- Given a user's financial goals and current financial situation, first generate a list of 3 search terms related to those goals.\n- For each search term, `search_google` and analyze the results.\n- From the results of all searches, return the 10 most relevant results to the user's preferences.\n- Remember: the quality of the results is important.\n</instructions>\n\n<additional_information>\n- The current time is 2025-03-25 15:30:47.596153\n</additional_information>", name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1742931047), Message(role='user', content='retirement planning for a 40-year-old Machine Learning engineer at NVIDIA with a goal to retire at 60', name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1742931047), Message(role='assistant', content='Based on the user\'s financial goals and current financial situation, I would generate the following list of 3 search terms:\n1. "retirement planning for 40-year-old tech professionals"\n2. "investments for long-term wealth creation for NVIDIA employees"\n3. "savings strategies for early retirement at 60"\n\nNow, let\'s search for each term:\n\n<function(search_google){"query": "retirement planning for 40-year-old tech professionals", "num_results": 10}</function>', name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=515, output_tokens=104, total_tokens=619, prompt_tokens=515, completion_tokens=104, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=4.902170599991223, time_to_first_token=None, timer=<agno.utils.timer.Timer object at 0x0000020E3D133980>), references=None, created_at=1742931047)], metrics={'input_tokens': [515], 'output_tokens': [104], 'total_tokens': [619], 'prompt_tokens': [515], 'completion_tokens': [104], 'time': [4.902170599991223]}, model='meta/llama-3.3-70b-instruct', run_id='d3ee2021-003e-4f11-99ed-582ffae9ec67', agent_id='5db3856d-c0d2-4ac4-aee4-ceff5b4dcf12', session_id='7e2df956-664d-49c3-97f2-9762d330a2b9', workflow_id=None, tools=[], formatted_tool_calls=None, images=None, videos=None, audio=None, response_audio=None, citations=None, extra_data=None, created_at=1742931045)
2025-03-25 15:31:16,444 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-03-25 15:31:17,278 - httpx - INFO - HTTP Request: POST https://api.agno.com/v1/telemetry/agent/run/create "HTTP/1.1 200 OK"
2025-03-25 15:31:17,278 - agno_personal_finance.agno_personal_finance_function - INFO - response from agno_personal_finance:
  RunResponse(content='To create a personalized financial plan for a 40-year-old Machine Learning engineer at NVIDIA with a goal to retire at 60, we\'ll need to consider several factors, including their current income, expenses, savings, and investment goals.\n\nAssuming the engineer has a stable income and a decent savings rate, here\'s a possible plan:\n\n1. **Retirement Savings**: Contribute at least 10% to 15% of their income towards retirement accounts such as 401(k) or IRA. NVIDIA may offer a matching program, so it\'s essential to contribute enough to maximize the match.\n2. **Investment Strategy**: Allocate a significant portion of their portfolio towards low-cost index funds or ETFs, which provide broad diversification and tend to be less volatile. A possible allocation could be:\n\t* 60% Stocks (40% US, 20% International)\n\t* 30% Bonds (20% Government, 10% Corporate)\n\t* 10% Alternatives (Real Estate, Commodities, etc.)\n3. **Savings Rate**: Aim to save at least 20% to 25% of their income towards short-term and long-term goals, including retirement, emergencies, and large purchases.\n4. **Expense Management**: Create a budget that accounts for all necessary expenses, including housing, food, transportation, and insurance. Aim to keep discretionary spending in check and allocate any excess funds towards savings and investments.\n5. **Tax Optimization**: Consider contributing to tax-advantaged accounts such as a Roth IRA or a tax-loss harvesting strategy to minimize tax liabilities.\n6. **Insurance and Risk Management**: Review and adjust insurance coverage, including life, disability, and health insurance, to ensure adequate protection against unexpected events.\n7. **Estate Planning**: Establish a will, consider setting up a trust, and designate beneficiaries for retirement accounts and life insurance policies.\n8. **Regular Review and Rebalancing**: Schedule annual reviews to assess progress, rebalance the portfolio, and make adjustments as needed to stay on track with retirement goals.\n\nBy following this plan, the Machine Learning engineer at NVIDIA can work towards achieving their goal of retiring at 60, while also ensuring a comfortable lifestyle and financial security.\n\nNext, let\'s search for investments that can help with long-term wealth creation.\n\n<function(search_google){"query": "investments for long-term wealth creation for NVIDIA employees", "num_results": 10}</function>', content_type='str', thinking=None, event='RunResponse', messages=[Message(role='system', content="You are a senior financial planner. Given a user's financial goals, current financial situation, and a list of research results,\nyour goal is to generate a personalized financial plan that meets the user's needs and preferences.\n\n\n<your_role>\nGenerates a personalized financial plan based on user preferences and research results\n</your_role>\n\n<instructions>\n- Given a user's financial goals, current financial situation, and a list of research results, generate a personalized financial plan that includes suggested budgets, investment plans, and savings strategies.\n- Ensure the plan is well-structured, informative, and engaging.\n- Ensure you provide a nuanced and balanced plan, quoting facts where possible.\n- Remember: the quality of the plan is important.\n- Focus on clarity, coherence, and overall quality.\n- Never make up facts or plagiarize. Always provide proper attribution.\n</instructions>\n\n<additional_information>\n- The current time is 2025-03-25 15:30:53.486565\n</additional_information>", name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1742931053), Message(role='user', content='\n            User query: retirement planning for a 40-year-old Machine Learning engineer at NVIDIA with a goal to retire at 60\n\n            Research results:\n            RunResponse(content=\'Based on the user\\\'s financial goals and current financial situation, I would generate the following list of 3 search terms:\\n1. "retirement planning for 40-year-old tech professionals"\\n2. "investments for long-term wealth creation for NVIDIA employees"\\n3. "savings strategies for early retirement at 60"\\n\\nNow, let\\\'s search for each term:\\n\\n<function(search_google){"query": "retirement planning for 40-year-old tech professionals", "num_results": 10}</function>\', content_type=\'str\', thinking=None, event=\'RunResponse\', messages=[Message(role=\'system\', content="You are a world-class financial researcher. Given a user\'s financial goals and current financial situation,\\ngenerate a list of search terms for finding relevant financial advice, investment opportunities, and savings strategies.\\nThen search the web for each term, analyze the results, and return the 10 most relevant results.\\n\\n\\n<your_role>\\nSearches for financial advice, investment opportunities, and savings strategies based on user preferences\\n</your_role>\\n\\n<instructions>\\n- Given a user\'s financial goals and current financial situation, first generate a list of 3 search terms related to those goals.\\n- For each search term, `search_google` and analyze the results.\\n- From the results of all searches, return the 10 most relevant results to the user\'s preferences.\\n- Remember: the quality of the results is important.\\n</instructions>\\n\\n<additional_information>\\n- The current time is 2025-03-25 15:30:47.596153\\n</additional_information>", name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1742931047), Message(role=\'user\', content=\'retirement planning for a 40-year-old Machine Learning engineer at NVIDIA with a goal to retire at 60\', name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1742931047), Message(role=\'assistant\', content=\'Based on the user\\\'s financial goals and current financial situation, I would generate the following list of 3 search terms:\\n1. "retirement planning for 40-year-old tech professionals"\\n2. "investments for long-term wealth creation for NVIDIA employees"\\n3. "savings strategies for early retirement at 60"\\n\\nNow, let\\\'s search for each term:\\n\\n<function(search_google){"query": "retirement planning for 40-year-old tech professionals", "num_results": 10}</function>\', name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=515, output_tokens=104, total_tokens=619, prompt_tokens=515, completion_tokens=104, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=4.902170599991223, time_to_first_token=None, timer=<agno.utils.timer.Timer object at 0x0000020E3D133980>), references=None, created_at=1742931047)], metrics={\'input_tokens\': [515], \'output_tokens\': [104], \'total_tokens\': [619], \'prompt_tokens\': [515], \'completion_tokens\': [104], \'time\': [4.902170599991223]}, model=\'meta/llama-3.3-70b-instruct\', run_id=\'d3ee2021-003e-4f11-99ed-582ffae9ec67\', agent_id=\'5db3856d-c0d2-4ac4-aee4-ceff5b4dcf12\', session_id=\'7e2df956-664d-49c3-97f2-9762d330a2b9\', workflow_id=None, tools=[], formatted_tool_calls=None, images=None, videos=None, audio=None, response_audio=None, citations=None, extra_data=None, created_at=1742931045)\n\n            Based on the above information, please create a personalized financial plan.\n            ', name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1742931053), Message(role='assistant', content='To create a personalized financial plan for a 40-year-old Machine Learning engineer at NVIDIA with a goal to retire at 60, we\'ll need to consider several factors, including their current income, expenses, savings, and investment goals.\n\nAssuming the engineer has a stable income and a decent savings rate, here\'s a possible plan:\n\n1. **Retirement Savings**: Contribute at least 10% to 15% of their income towards retirement accounts such as 401(k) or IRA. NVIDIA may offer a matching program, so it\'s essential to contribute enough to maximize the match.\n2. **Investment Strategy**: Allocate a significant portion of their portfolio towards low-cost index funds or ETFs, which provide broad diversification and tend to be less volatile. A possible allocation could be:\n\t* 60% Stocks (40% US, 20% International)\n\t* 30% Bonds (20% Government, 10% Corporate)\n\t* 10% Alternatives (Real Estate, Commodities, etc.)\n3. **Savings Rate**: Aim to save at least 20% to 25% of their income towards short-term and long-term goals, including retirement, emergencies, and large purchases.\n4. **Expense Management**: Create a budget that accounts for all necessary expenses, including housing, food, transportation, and insurance. Aim to keep discretionary spending in check and allocate any excess funds towards savings and investments.\n5. **Tax Optimization**: Consider contributing to tax-advantaged accounts such as a Roth IRA or a tax-loss harvesting strategy to minimize tax liabilities.\n6. **Insurance and Risk Management**: Review and adjust insurance coverage, including life, disability, and health insurance, to ensure adequate protection against unexpected events.\n7. **Estate Planning**: Establish a will, consider setting up a trust, and designate beneficiaries for retirement accounts and life insurance policies.\n8. **Regular Review and Rebalancing**: Schedule annual reviews to assess progress, rebalance the portfolio, and make adjustments as needed to stay on track with retirement goals.\n\nBy following this plan, the Machine Learning engineer at NVIDIA can work towards achieving their goal of retiring at 60, while also ensuring a comfortable lifestyle and financial security.\n\nNext, let\'s search for investments that can help with long-term wealth creation.\n\n<function(search_google){"query": "investments for long-term wealth creation for NVIDIA employees", "num_results": 10}</function>', name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=1683, output_tokens=492, total_tokens=2175, prompt_tokens=1683, completion_tokens=492, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=22.959491099987645, time_to_first_token=None, timer=<agno.utils.timer.Timer object at 0x0000020E3D1B4C20>), references=None, created_at=1742931053)], metrics={'input_tokens': [1683], 'output_tokens': [492], 'total_tokens': [2175], 'prompt_tokens': [1683], 'completion_tokens': [492], 'time': [22.959491099987645]}, model='meta/llama-3.3-70b-instruct', run_id='72096751-31f3-43ed-9e1b-38af0cd51e57', agent_id='52c09ed1-bce9-4909-9844-1e77c5f1a54f', session_id='8e1e1072-53f2-4eb1-91d7-6d2cccd413a0', workflow_id=None, tools=[], formatted_tool_calls=None, images=None, videos=None, audio=None, response_audio=None, citations=None, extra_data=None, created_at=1742931045)
2025-03-25 15:31:17,299 - aiq.agent.tool_calling_agent.agent - INFO - Calling agent
2025-03-25 15:31:36,945 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-25 15:31:36,945 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
["To create a personalized financial plan for a 40-year-old Machine Learning engineer at NVIDIA with a goal to retire at 60, we'll need to consider several factors, including their current income, expenses, savings, and investment goals.\n\nAssuming the engineer has a stable income and a decent savings rate, here's a possible plan:\n\n1. **Retirement Savings**: Contribute at least 10% to 15% of their income towards retirement accounts such as 401(k) or IRA. NVIDIA may offer a matching program, so it's essential to contribute enough to maximize the match.\n2. **Investment Strategy**: Allocate a significant portion of their portfolio towards low-cost index funds or ETFs, which provide broad diversification and tend to be less volatile. A possible allocation could be:\n\t* 60% Stocks (40% US, 20% International)\n\t* 30% Bonds (20% Government, 10% Corporate)\n\t* 10% Alternatives (Real Estate, Commodities, etc.)\n3. **Savings Rate**: Aim to save at least 20% to 25% of their income towards short-term and long-term goals, including retirement, emergencies, and large purchases.\n4. **Expense Management**: Create a budget that accounts for all necessary expenses, including housing, food, transportation, and insurance. Aim to keep discretionary spending in check and allocate any excess funds towards savings and investments.\n5. **Tax Optimization**: Consider contributing to tax"]
--------------------------------------------------
```
---

## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the AgentIQ virtual environment.

From the root directory of the `agentiq` repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t agno_personal_finance -f examples/agno_personal_finance/Dockerfile .
```

### Run the Docker Container
Deploy the container:

```bash
docker run -p 8000:8000 -e NVIDIA_API_KEY -e SERP_API_KEY agno_personal_finance
```

### Test the API
Use the following curl command to test the deployed API:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"inputs": "My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA."}'
  ```

### Expected API Output
The API response should look like this:

```bash
{"value":"Based on the research results, I've created a personalized financial plan for you to achieve your goal of retiring at age 60.\n\n1. **Invest in a balanced portfolio**: Invest in a mix of low-cost index funds, stocks, and bonds to achieve long-term growth. Consider consulting with a financial advisor to create a personalized portfolio.\n2. **Consider real estate**: Invest in real estate to not only allow for early retirement but also to sustain an early retirement lifestyle. You can invest in rental properties, real estate investment trusts (REITs), or real estate crowdfunding platforms.\n3. **Invest more conservatively as you get older**: As you approach retirement, consider investing more conservatively by putting more money into bonds and less into stocks. This will help reduce risk and ensure a steady income stream during retirement.\n4. **Know all your income sources**: Make sure you have a clear understanding of all your income sources, including your salary, investments, and any side hustles. This will help you create a comprehensive retirement plan.\n5. **Leave retirement savings alone**: Avoid withdrawing from your retirement accounts, such as your 401(k) or IRA, before age 59 to avoid penalties and ensure you have enough savings for retirement.\n6. **Consider alternative account types**: Look into other account types, such as a taxable brokerage account or a Roth IRA, that can provide more flexibility for early retirement.\n7. **Consult with a financial advisor**: Consider consulting with a financial advisor to create a personalized retirement plan that takes into account your specific financial situation and goals.\n8. **Research and understand tax implications**: Research and understand the tax implications of different investment strategies and account types to minimize taxes and maximize your retirement savings.\n9. **Diversify your portfolio**: Consider investing in a diversified portfolio that includes a mix of stocks, bonds, and other assets to reduce risk and increase potential returns.\n10. **Start saving and investing early**: Start saving and investing as early as possible to take advantage of compound interest and maximize your retirement savings.\n\nAdditionally, consider the following:\n\n* **Maximize your 401(k) contributions**: Contribute as much as possible to your 401(k) account, especially if your employer matches contributions.\n* **Consider a Roth IRA**: Invest in a Roth IRA, which allows you to contribute after-tax dollars and potentially reduce your taxable income in retirement.\n* **Invest in a tax-efficient manner**: Consider investing in tax-efficient manner, such as investing in index funds or ETFs, to minimize taxes and maximize your returns.\n\nRemember, this is just a general plan, and it's essential to consult with a financial advisor to create a personalized plan tailored to your specific needs and goals."}
```
