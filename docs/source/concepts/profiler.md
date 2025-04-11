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

# Profiling and Performance Monitoring of NVIDIA AgentIQ Workflows

The AgentIQ Profiler Module provides profiling and forecasting capabilities for AgentIQ workflows. The profiler instruments the workflow execution by:
- Collecting usage statistics in real time (via callbacks).
- Recording the usage statistics on a per-invocation basis (e.g., tokens used, time between calls, LLM calls).
- Storing the data for offline analysis.
- Forecasting usage metrics using time-series style models (linear, random forest, etc.).
- Computing workflow specific metrics for performance analysis (e.g., latency, throughput, etc.).
- Analyzing workflow performance measures such as bottlenecks, latency, and concurrency spikes.

These functionalities will allow AgentIQ developers to dynamically stress test their workflows in pre-production phases to receive workflow-specific sizing guidance based on observed latency and throughput of their specific workflows
At any or every stage in a workflow execution, the AgentIQ profiler generates predictions/forecasts about future token and tool usage.  Client side forecasting allows for workflow-specific predictions which can be difficult, if not impossible, to achieve server side in order to facilitate inference planning.
Will allow for features such as offline-replay or simulation of workflow runs without the need for deployed infrastructure such as tooling/vector DBs, etc. Will also allow for AgentIQ native observability and workflow fingerprinting.

## Current Profiler Architecture
The AgentIQ Profiler can be broken into the following components:

### Profiler Decorators and Callbacks
- **profiler/decorators.py** defines decorators that can wrap each workflow or LLM framework context manager to inject usage-collection callbacks.
- **profiler/callbacks** directory implements callback handlers. These handlers track usage statistics (tokens, time, inputs/outputs) and push them to the AgentIQ usage stats queue. We currently support callback handlers for LangChain,
LLama Index, CrewAI, and Semantic Kernel.

### Profiler Runner

- **profiler/profile_runner.py** is the main orchestration class. It collects workflow run statistics from the AgentIQ Eval module, computed workflow-specific metrics, and optionally forecasts usage metrics using the AgentIQ Profiler module.

- **Under profiler/forecasting**, the code trains scikit-learn style models on the usage data.
model_trainer.py can train a LinearModel or a RandomForestModel on the aggregated usage data (the raw statistics collected).
base_model.py, linear_model.py, and random_forest_regressor.py define the abstract base and specific scikit-learn wrappers.

- **Under profiler/inference_optimization** we have several metrics that can be computed out evaluation traces of your workflow including workflow latency, commonly used prompt prefixes for caching, identifying workflow bottlenecks, and concurrency analysis.

### CLI Integrations
Native integrations with `aiq eval` to allow for running of the profiler through a unified evaluation interface. Configurability is exposed through a workflow YAML configuration file consistent with evaluation configurations.


## Using the Profiler

### Step 1: Enabling Instrumentation on a Workflow [Optional]
**NOTE:** If you don't set it, AgentIQ will inspect your code to infer frameworks used. We recommend you set it explicitly.
To enable profiling on a workflow, you need to wrap the workflow with the profiler decorators. The decorators can be applied to any workflow using the `framework_wrappers` argument of the `register_function` decorator.
Simply specify which AgentIQ supported frameworks you will be using anywhere in your workflow (including tools) upon registration and AgentIQ will automatically apply the appropriate profiling decorators at build time.
For example:

```python
@register_function(config_type=WebQueryToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def webquery_tool(config: WebQueryToolConfig, builder: Builder):
```

Once workflows are instrumented, the profiler will collect usage statistics in real time and store them for offline analysis for any LLM invocations or tool calls your workflow makes during execution. Runtime telemetry
is stored in a `intermediate_steps_stream` context variable during runtime. AgentIQ has a subscriber that will read intermediate steps through eval.

Even if a function isn’t one of the built-in AgentIQ “Functions”, you can still profile it with our simple decorator. The `@track_function` decorator helps you capture details such as when a function starts and ends, its input arguments, and its output—even if the function is asynchronous, a generator, or a class method.

#### How It Works

The decorator automatically logs key events in three stages:
- **`SPAN_START`:** Logged when the function begins executing. It records the serialized inputs.
- **`SPAN_CHUNK`:** For generator functions, each yielded value is captured as it’s produced.
- **`SPAN_END`:** Logged when the function finishes executing. It records the serialized output.

It supports all kinds of functions:
- **Synchronous functions & methods**
- **Asynchronous functions**
- **Generators (both `sync` and `async`)**

#### Key Benefits

- **Broad Compatibility:**  
  Use this decorator on any Python function, regardless of its type.

- **Simple Metadata:**  
  Optionally pass a dictionary of metadata to add extra context about the function call.

- **Automatic Data Serialization:**  
  The decorator converts input arguments and outputs into a `JSON`-friendly format (with special handling for Pydantic models), making the data easier to analyze.

- **Reactive Event Streaming:**  
  All profiling events are pushed to the `AgentIQ` intermediate step stream, so you can subscribe and monitor events in real time.

#### How to Use

Just decorate your custom function with `@track_function` and provide any optional metadata if needed:

```python
from aiq.profiler.decorators.function_tracking import track_function

@track_function(metadata={"action": "compute", "source": "custom_function"})
def my_custom_function(a, b):
    # Your function logic here
    return a + b
```

### Step 2: Configuring the Profiler with Eval
The profiler can be run through the `aiq eval` command. The profiler can be configured through the `profiler` section of the workflow configuration file. The following is an example `eval` configuration section from the `simple` workflow which shows how to enable the profiler:

```yaml
eval:
  general:
    output_dir: ./.tmp/aiq/examples/simple/
    dataset:
      _type: json
      file_path: examples/simple/data/langsmith.json
    profiler:
      # Compute inter query token uniqueness
      token_uniqueness_forecast: true
      # Compute expected workflow runtime
      workflow_runtime_forecast: true
      # Compute inference optimization metrics
      compute_llm_metrics: true
      # Avoid dumping large text into the output CSV (helpful to not break structure)
      csv_exclude_io_text: true
      # Idenitfy common prompt prefixes
      prompt_caching_prefixes:
        enable: true
        min_frequency: 0.1
      bottleneck_analysis:
        # Can also be simple_stack
        enable_nested_stack: true
      concurrency_spike_analysis:
        enable: true
        spike_threshold: 7

  evaluators:
    rag_accuracy:
      _type: ragas
      metric: AnswerAccuracy
      llm_name: nim_rag_eval_llm
    rag_groundedness:
      _type: ragas
      metric: ResponseGroundedness
      llm_name: nim_rag_eval_llm
    rag_relevance:
      _type: ragas
      metric: ContextRelevance
      llm_name: nim_rag_eval_llm
    trajectory_accuracy:
      _type: trajectory
      llm_name: nim_trajectory_eval_llm
```

Please also note the `output_dir` parameter which specifies the directory where the profiler output will be stored. Let us explore the profiler configuration options:
- `token_uniqueness_forecast`: Compute the inter-query token uniqueness forecast. This computes the expected number of unique tokens in the next query based on the tokens used in the previous queries.
- `workflow_runtime_forecast`: Compute the expected workflow runtime forecast. This computes the expected runtime of the workflow based on the runtime of the previous queries.
- `compute_llm_metrics`: Compute inference optimization metrics. This computes workflow-specific metrics for performance analysis (e.g., latency, throughput, etc.).
- `csv_exclude_io_text`: Avoid dumping large text into the output CSV. This is helpful to not break the structure of the CSV output.
- `prompt_caching_prefixes`: Identify common prompt prefixes. This is helpful for identifying if you have commonly repeated prompts that can be pre-populated in KV caches
- `bottleneck_analysis`: Analyze workflow performance measures such as bottlenecks, latency, and concurrency spikes. This can be set to `simple_stack` for a simpler analysis. Nested stack will provide a more detailed analysis identifying nested bottlenecks like tool calls inside other tools calls.
- `concurrency_spike_analysis`: Analyze concurrency spikes. This will identify if there are any spikes in the number of concurrent tool calls. At a `spike_threshold` of 7, the profiler will identify any spikes where the number of concurrent running functions is greater than or equal to 7. Those are surfaced to the user in a dedicated section of the workflow profiling report.

### Step 3: Running the Profiler

To run the profiler, simply run the `aiq eval` command with the workflow configuration file. The profiler will collect usage statistics and store them in the output directory specified in the configuration file.

```bash
aiq eval --config_file examples/simple/configs/eval_config.yml
```

This will, based on the above configuration, produce the following files in the `output_dir` specified in the configuration file:

- `all_requests_profiler_traces.json` : This file contains the raw usage statistics collected by the profiler. Includes raw traces of LLM and tool input, runtimes, and other metadata.
- `inference_optimization.json`: This file contains the computed workflow-specific metrics. This includes 90%, 95%, and 99% confidence intervals for latency, throughput, and workflow runtime.
- `standardized_data_all.csv`: This file contains the standardized usage data including prompt tokens, completion tokens, LLM input, framework, and other metadata.
- You'll also find a JSON file and text report of any advanced or experimental techniques you ran including concurrency analysis, bottleneck analysis, or PrefixSpan.

## Providing Feedback

We welcome feedback on the AgentIQ Profiler module. Please provide feedback by creating an issue on the AgentIQ Git repository.

If you're filing a bug report, please also include a reproducer workflow and the profiler output files.
