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

<!--
  SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# ReWOO Agent

A configurable ReWOO Agent. This agent leverages the AgentIQ plugin system and `WorkflowBuilder` to integrate pre-built and custom tools into the workflow. Key elements are summarized below:

## Key Features

- **Pre-built Tools:** Leverages core AgentIQ library agent and tools.
- **ReAct Agent:** Performs reasoning between tool call; utilizes tool names and descriptions to appropriately route to the correct tool
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e .
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

Prior to using the `tavily_internet_search` tool, create an account at [`tavily.com``](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```
---

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run  --config_file=examples/agents/rewoo/configs/config.yml --input "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"
```

**Expected Output**

```console
$ aiq run  --config_file=examples/agents/rewoo/configs/config.yml --input "Wh
ich city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"
2025-04-07 15:27:19,264 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (508.254051 ms). Ensure all imports are inside your registered functions.
2025-04-07 15:27:19,439 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/agents/rewoo/configs/config.yml'
2025-04-07 15:27:19,444 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-07 15:27:19,707 - haystack.tracing.tracer - INFO - Auto-enabled tracing for 'OpenTelemetryTracer'
2025-04-07 15:27:19,713 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-04-07 15:27:19,743 - aiq.agent.rewoo_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-04-07 15:27:19,743 - aiq.agent.rewoo_agent.agent - INFO - Adding the tools' input schema to the tools' description
2025-04-07 15:27:19,743 - aiq.agent.rewoo_agent.agent - INFO - Initialized ReWOO Agent Graph
2025-04-07 15:27:19,750 - aiq.agent.rewoo_agent.agent - INFO - ReWOO Graph built and compiled successfully
2025-04-07 15:27:19,750 - aiq.agent.rewoo_agent.agent - INFO - ReAct Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: rewoo_agent
Number of Functions: 6
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-07 15:27:19,752 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?',)
2025-04-07 15:27:21,916 - aiq.agent.rewoo_agent.agent - INFO - The task was: Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?
2025-04-07 15:27:21,916 - aiq.agent.rewoo_agent.agent - INFO - The planner's thoughts are:
Here are the plans to solve the task:

Plan: Compare 1996 and 2004 to determine the bigger number.
#E1 = calculator_inequality[{'text': '2004 > 1996'}]

Plan: Since #E1 will return True, we know 2004 is the bigger number. Now, search for the city that held the Olympic game in 2004.
#E2 = internet_search["Which city held the Olympic game in 2004?"]

Plan: Extract the relevant information from the search result #E2 to get the final answer.
#E3 = haystack_chitchat_agent[{'inputs': #E2}]
2025-04-07 15:27:21,919 - aiq.agent.rewoo_agent.agent - INFO - Calling tool calculator_inequality with input: {'text': '2004 > 1996'}
2025-04-07 15:27:21,919 - aiq.agent.rewoo_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-04-07 15:27:21,924 - aiq.agent.rewoo_agent.agent - INFO - Calling tool internet_search with input: "Which city held the Olympic game in 2004?"
2025-04-07 15:27:21,924 - aiq.agent.rewoo_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-04-07 15:27:24,122 - aiq.agent.rewoo_agent.agent - INFO - Calling tool haystack_chitchat_agent with input: {'inputs': <Document href="https://en.wikipedia.org/wiki/List_of_Olympic_Games_host_cities"/>
1896:  Athens
1900:  Paris
1904:  St. Louis
1908:  London
1912:  Stockholm
1916: None[c1]
1920:  Antwerp
1924:  Paris
1928:  Amsterdam
1932:  Los Angeles
1936:  Berlin
1940: None[c2]
1944: None[c2]
1948:  London
1952:  Helsinki
1956:  Melbourne
1960:  Rome
1964:  Tokyo
1968:  Mexico City
1972:  Munich
1976:  Montreal
1980:  Moscow
1984:  Los Angeles
1988:  Seoul
1992:  Barcelona
1996:  Atlanta
2000:  Sydney
2004:  Athens
2008:  Beijing
2012:  London
2016:  Rio de Janeiro
2020:  Tokyo[c3] [...] 1988 Seoul
1992 Barcelona
1996 Atlanta
2000 Sydney
2004 Athens
2008 Beijing
2012 London
2016 Rio
2020 Tokyo[c]
2024 Paris
2028 Los Angeles
2032 Brisbane
2036 TBD
Winter
1924 Chamonix
1928 St. Moritz
1932 Lake Placid
1936 Garmisch-Partenkirchen
1940 Sapporo[b]
1944 Cortina d'Ampezzo[b]
1948 St. Moritz
1952 Oslo
1956 Cortina d'Ampezzo
1960 Squaw Valley
1964 Innsbruck
1968 Grenoble
1972 Sapporo
1976 Innsbruck
1980 Lake Placid
1984 Sarajevo
1988 Calgary
1992 Albertville
1994 Lillehammer
1998 Nagano [...] Atlanta  United States  1996    North America   XXVI        19 July 1996    4 August 1996
    Nagano   Japan  1998    Asia        XVIII   7 February 1998 22 February 1998
    Sydney   Australia  2000    Oceania XXVII       15 September 2000   1 October 2000
    Salt Lake City   United States  2002    North America       XIX 8 February 2002 24 February 2002
    Athens   Greece 2004    Europe  XXVIII      13 August 2004  29 August 2004
</Document>

---

<Document href="https://www.britannica.com/sports/Olympic-Games-host-cities-2228216"/>
| 2000 | Sydney, Austl. | *** |
| 2002 | *** | Salt Lake City, Utah, U.S. |
| 2004 | Athens | *** |
| 2006 | *** | Turin, Italy |
| 2008 | Beijing | *** |
| 2010 | *** | Vancouver, B.C., Can. |
| 2012 | London | *** |
| 2014 | *** | Sochi, Russia |
| 2016 | Rio de Janeiro | *** |
| 2018 | *** | P'yŏngch'ang, S.Kor. |
| 2020 | Tokyo | *** |
| 2022 | *** | Beijing |
| 2024 | Paris | *** |
| 2026 | *** | Milan and Cortina d'Ampezzo, Italy |
| 2028 | Los Angeles | *** |
| 2030 | *** | French Alps |
</Document>

---

<Document href="https://www.quora.com/What-field-event-took-place-in-the-Ancient-Olympic-Stadium-in-2004"/>
In the 2004 Summer Olympics held in Athens, Greece, the Ancient Olympic Stadium in Olympia was specially used to host the men's and women's
</Document>}
2025-04-07 15:27:24,122 - aiq.agent.rewoo_agent.agent - INFO - Unable to parse structured tool input from Action Input. Using Action Input as is.
2025-04-07 15:27:45,153 - aiq_multi_frameworks.haystack_agent - INFO - output from langchain_research_tool: It appears that you have provided a collection of documents related to the Olympic Games, specifically the host cities of the Summer and Winter Olympics. Here's a summary of the information:

**Summer Olympics Host Cities:**

1. 1896: Athens
2. 1900: Paris
3. 1904: St. Louis
4. 1908: London
5. 1912: Stockholm
6. 1920: Antwerp
7. 1924: Paris
8. 1928: Amsterdam
9. 1932: Los Angeles
10. 1936: Berlin
11. 1948: London
12. 1952: Helsinki
13. 1956: Melbourne
14. 1960: Rome
15. 1964: Tokyo
16. 1968: Mexico City
17. 1972: Munich
18. 1976: Montreal
19. 1980: Moscow
20. 1984: Los Angeles
21. 1988: Seoul
22. 1992: Barcelona
23. 1996: Atlanta
24. 2000: Sydney
25. 2004: Athens
26. 2008: Beijing
27. 2012: London
28. 2016: Rio de Janeiro
29. 2020: Tokyo
30. 2024: Paris
31. 2028: Los Angeles
32. 2032: Brisbane

**Additional Information:**

* The Ancient Olympic Stadium in Olympia was used to host the men's and women's shot put events during the 2004 Summer Olympics in Athens, Greece.

Please let me know if you have any specific questions or if there's anything else I can help you with!
2025-04-07 15:27:45,549 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-04-07 15:27:45,549 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['Athens']
--------------------------------------------------
```
---

### Starting the AgentIQ Server

You can start the AgentIQ server using the `aiq serve` command with the appropriate configuration file.

**Starting the ReAct Agent Example Workflow**

```bash
aiq serve --config_file=examples/agents/react/configs/config.yml
```

### Making Requests to the AgentIQ Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the ReAct Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"}'
```

#### Streaming Requests

**Streaming Request to the ReAct Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"}'
```
---

### Evaluating the ReWOO Agent Workflow
**Run and evaluate the `rewoo_agent` example Workflow**

```bash
aiq eval --config_file=examples/agents/rewoo/configs/config.yml
```
