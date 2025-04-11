# AIQ Profiler Agent

An agent-based system for analyzing and profiling LLM applications.


## Installation

0. Start the pheonix server locally or use a remote phoenix server
```
docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest 
```

1. Clone the repository and submodules:
   ```
   uv pip install examples/profiler_agent  
   ```
3. Configuration
    To use a remote phoenix server, modify the config/config.yml to point to the URL

## Usage

1. Start the Phoenix server if not already running.

2. Run the profiler agent:
   ```
   aiq serve --config_file=configs/config.yml  --host 0.0.0.0 --port 8088 
   ```

3. Query the agent with natural language:
   ```
   "Show me flowchart of last 3 runs"
   "Show me the token usage of last run"
   "Analyze the last 2 runs"
   
   ```

## Features

- Query Phoenix traces with natural language
- Analyze LLM application performance metrics
- Generate trace visualizations
- Extract user queries across trace spans



