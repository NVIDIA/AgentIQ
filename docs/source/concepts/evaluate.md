<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Evaluating NVIDIA AgentIQ Workflows Details

:::{note}
It is recommended that the [Evaluating AgentIQ Workflows](../guides/evaluate.md) guide be read before proceeding with this detailed documentation.
:::

AgentIQ provides a set of evaluators to run and evaluate the AgentIQ workflows. In addition to the built-in evaluators, AgentIQ provides a plugin system to add custom evaluators.

Example:
```bash
aiq eval --config_file=examples/simple/configs/eval_config.yml
```

## Using Datasets
Run and evaluate the workflow on a specified dataset. The dataset files types are `json`, `jsonl`, `csv`, `xls`, or `parquet`.

Download and use datasets provided by AgentIQ examples by running the following.

```bash
git lfs fetch
git lfs pull
```
 The dataset used for evaluation is specified in the configuration file  via `eval.general.dataset`. For example, to use the `langsmith.json` dataset, the configuration is as follows:
```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: examples/simple/data/langsmith.json
```

### Dataset Format
Each dataset file contains a list of records. Each record is a dictionary with keys as the column names and values as the data. For example, a sample record in a `json` dataset file is as follows:
```json
{
"id": "q_1",
"question": "What is langsmith",
"answer": "LangSmith is a platform for LLM application development, monitoring, and testing"
},
```

A dataset entry are either structured or unstructured.
For structured entries, the default names of the columns are `id`, `question`, and `answer`,
where the libraries know that `question` is an input and `answer` is the output. You can
change the column names and their configurations in the `config.yml` file
with `eval.general.dataset.structure`.
```yaml
eval:
  general:
    dataset:
      structure:
        id: "my_id"
        question: "my_question"
        answer: "my_answer"
```

For unstructured entries, the entire dictionary is the input to the workflow
but the libraries don't know the individual columns. The input and the workflow output
goes through evaluation, where evaluators, such as swe-bench evaluator, handle
unstructured entries. The following is an example configuration for
the swe-bench evaluator:
```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: examples/swe_bench/data/test_dataset_lite.json
      id_key: instance_id
      structure: # For swe-bench the entire row is the input
        disable: true
```

### Filtering Datasets
While evaluating large datasets, you can filter the dataset to a
smaller subset by allowing or denying entries with the `eval.general.dataset.filter`
in the `config.yml` file. The filter is a dictionary with keys as the column names and
values as the filter values.

The following is an example configuration, where evaluation
runs on a subset of the swe-bench-verified dataset, which has 500 entries. The configuration runs the
evaluation on two entries with instance identifications (`instance_id`), `sympy__sympy-20590`
and `sympy__sympy-21055`. The evaluation iteratively develops and debugs the workflows.
```yaml
eval:
    dataset:
      _type: json
      file_path: examples/swe_bench/data/test_dataset_verified.json
      id_key: instance_id
      structure:
        disable: true
      filter:
        allowlist:
          field:
            instance_id:
              - sympy__sympy-20590
              - sympy__sympy-21055
```
The swe-bench verified dataset has 500 entries but above configuration runs the workflow and evaluation on only two entries with `instance_id` `sympy__sympy-20590` and `sympy__sympy-21055`. This is useful for iterative development and troubleshooting of the workflow.

You can also skip entries from the dataset. Here is an example configuration to skip entries with `instance_id` `sympy__sympy-20590` and `sympy__sympy-21055`:
```yaml
eval:
    dataset:
      _type: json
      file_path: examples/swe_bench/data/test_dataset_verified.json
      id_key: instance_id
      structure:
        disable: true
      filter:
        denylist:
          field:
            instance_id:
              - sympy__sympy-20590
              - sympy__sympy-21055
```

## AgentIQ Built-in Evaluators
AgentIQ provides the following built-in evaluator:
- `ragas` - An evaluator to run and evaluate RAG-like workflows using the public RAGAS API.
- `trajectory` - An evaluator to run and evaluate the LangChain agent trajectory.
- `swe_bench` - An evaluator to run and evaluate the workflow on the SWE-Bench dataset.

### RAGAS Evaluator
[RAGAS](https://docs.ragas.io/) is an OSS evaluation framework that enables end-to-end
evaluation of RAG workflows. AgentIQ provides an interface to RAGAS to evaluate the performance
of RAG-like AgentIQ workflows.

RAGAS provides a set of evaluation metrics to configure in the `config.yml` file
by adding an evaluator section with type`ragas`.

**Example:**
```yaml
eval:
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
    rag_factual_correctness:
      _type: ragas
      metric:
        FactualCorrectness:
          kwargs:
            mode: precision
      llm_name: nim_rag_eval_large_llm # requires more tokens
```

In the example four `ragas` evaluators are configured to evaluate various `ragas`metrics. The metric can be a string or a dictionary. If the metric is a dictionary, the `kwargs` provided are passed to the metric function.

The following `ragas` metrics are recommended for RAG like workflows -
`AnswerAccuracy`: Evaluates the accuracy of the answer generated by the workflow against the expected answer or ground truth.
`ContextRelevance`: Evaluates the relevance of the context retrieved by the workflow against the question.
`ResponseGroundedness`: Evaluates the `groundedness` of the response generated by the workflow based on the context retrieved by the workflow.

### Agent Trajectory Evaluator
The `trajectory` evaluator uses LangChain agent trajectory evaluation to evaluate the workflow. To use the `trajectory` evaluator, add the following configuration to the `config.yml` file.
```yaml
eval:
  evaluators:
    trajectory:
      _type: trajectory
      llm_name: nim_trajectory_eval_llm
```

### Swe-benchmark Evaluator
Workflows can use the swe-bench evaluator to solve swe-bench problems. To evaluate the patch, generated by the workflow, install the repository and run the `PASS_TO_PASS` and `FAIL_TO_PASS` tests.

**Example:**
```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: examples/swe_bench/data/test_dataset_lite.json
      id_key: instance_id
      structure: # For swe-bench the entire row is the input
        disable: true

  evaluators:
    swe_bench:
      _type: swe_bench
      run_id: aiq_1
```
The swe-bench evaluator uses unstructured dataset entries. The entire row is provided as input to the workflow.

### Tunable RAG Evaluator
The tunable RAG evaluator is a customizable LLM evaluator that allows for flexible evaluation of RAG workflows. 
It includes a default scoring mechanism based on an expected answer description rather than a ground truth answer. 

The judge LLM prompt is tunable and can be provided in the `config.yml` file.

A default scoring method is provided as follows:
- Coverage: Evaluates if the answer covers all mandatory elements of the expected answer.
- Correctness: Evaluates if the answer is correct compared to the expected answer.
- Relevance: Evaluates if the answer is relevant to the question.

These weights can be optionally tuned by setting the `default_score_weights` parameter in the `config.yml` file. If not set, each score will be equally weighted.

The default scoring can be overridden by setting the config boolean `default_scoring` to false and providing your own scoring mechanism which you describe in your custom judge LLM prompt.
Note: if you do choose to use the default scoring method, you are still able to tune the judge LLM prompt.

**Example:**
```yaml
eval:
  evaluators:
    custom_rag_evaluation:
      _type: tunable_rag_evaluator
      llm_name: nim_rag_eval_llm
      default_scoring: false
      default_score_weights:
        coverage: 0.5
        correctness: 0.3
        relevance: 0.2
      judge_llm_prompt: >
        You are an intelligent evaluator that scores the generated answer based on the description of the expected answer.
        The score is a measure of how well the generated answer matches the description of the expected answer based on the question.
        Take into account the question, the relevance of the answer to the question and the quality compared to the description of the expected answer.

        Rules:
        - The score must be a float of any value between 0.0 and 1.0 on a sliding scale.
        - The reasoning string must be concise and to the point. It should be 1 sentence and 2 only if extra description is needed. It must explain why the score was given and what is different between the generated answer and the expected answer.
```

Note: In your evaluation dataset, make sure that the `answer` field is a description of the expected answer with details on what is expected from the generated answer.

**Example:**
```json
{
  "id": 1,
  "question": "What is the product of 3 and 7, and is it greater than the current hour?",
  "answer": "Answer must have the answer of product of 3 and 7 and whether it is greater than the current hour"
}
```

## Adding Custom Evaluators
You can add custom evaluators to evaluate the workflow output. To add a custom evaluator, you need to implement the evaluator and register it with the AgentIQ evaluator system. See the [Custom Evaluator](../guides/custom-evaluator.md) documentation for more information.


## Running multiple repetitions
You can run multiple repetitions of the evaluation by running a command line option `--reps`. For example, to run the evaluation 5 times, run the following command:
```bash
aiq eval --config_file=examples/simple/configs/eval_config.yml --reps=5
```
This will allow you to get an average score across multiple runs and analyze the variation in the generated outputs.

## Running evaluation on large datasets
Similar to how evaluators are run in parallel, entries in the dataset are also processed in parallel. Concurrency is configurable using the `eval.general.concurrency` parameter in the `config.yml` file. The default value is 8. Increase or decrease the value based on the available resources.
```yaml
eval:
  general:
    concurrency: 4
```

## Pickup where you left off
When running the evaluation on a large dataset, it is recommended to resume the evaluation from where it was left off. This is particularly useful while using overloaded services that may timeout while running the workflow. When that happens a workflow interrupted warning is issued and workflow output is saved to a file.

You can then re-run evaluation on that output file along with `--skip_completed_entries` options.

Pass-1:
```
aiq eval --config_file=examples/simple/configs/eval_config.yml
```
This pass results in workflow interrupted warning. You can then do another pass.

Pass-2:
```bash
aiq eval --config_file=examples/simple/configs/eval_config.yml --skip_completed_entries --dataset=.tmp/aiq/examples/simple/workflow_output.json
```

## Running evaluation offline
You can evaluate a dataset with previously generated answers via the `--skip_workflow` option. In this case the dataset has both the expected `answer` and the `generated_answer`.
```bash
aiq eval --config_file=examples/simple/configs/eval_config.yml --skip_workflow --dataset=.tmp/aiq/examples/simple/workflow_output.json
```
This assumes that the workflow output is previously generated and stored in the `.tmp/aiq/examples/simple/workflow_output.json` file.

## Running the workflow over a dataset without evaluation
You can do this by running `aiq eval` with a workflow configuration file that includes an `eval` section with no `evaluators`.
```yaml
eval:
  general:
    output_dir: ./.tmp/aiq/examples/simple/
    dataset:
      _type: json
      file_path: examples/simple/data/langsmith.json
```

## Evaluation output
The output of the workflow is stored as `workflow_output.json` in the `output_dir` provided in the config.yml -
```yaml
eval:
  general:
    output_dir: ./.tmp/aiq/examples/simple/
```
Here is a sample workflow output snipped generated by running evaluation on the `simple` example workflow -
```
  {
    "id": "1",
    "question": "What is langsmith",
    "answer": "LangSmith is a platform for LLM application development, monitoring, and testing",
    "generated_answer": "LangSmith is a platform for LLM (Large Language Model) application development, monitoring, and testing. It provides features such as automations, threads, annotating traces, adding runs to a dataset, prototyping, and debugging to support the development lifecycle of LLM applications.",
    "intermediate_steps": [
      {
        >>>>>>>>>>>>>>> SNIPPED >>>>>>>>>>>>>>>>>>>>>>
      }
    ],
    "expected_intermediate_steps": []
  },
```

The output of the evaluators are stored in distinct files in the same `output_dir` as `<evaluator_name>_output.json`. An evaluator typically provides an average score and a score per-entry. Here is a sample `rag_accuracy` output -
```bash
{
  "average_score": 0.6666666666666666,
  "eval_output_items": [
    {
      "id": 1,
      "score": 0.5,
      "reasoning": {
        "user_input": "What is langsmith"
      }
    },
    {
      "id": 2,
      "score": 0.75,
      "reasoning": {
        "user_input": "How do I proptotype with langsmith"
      }
    },
    {
      "id": 3,
      "score": 0.75,
      "reasoning": {
        "user_input": "What are langsmith automations?"
      }
    }
  ]
}
```

## Customizing the output
You can customize the output of the pipeline by providing custom scripts. One or more Python scripts can be provided in the `eval.general.output_scripts` section of the `config.yml` file.

The custom scripts are executed after the evaluation is complete. They are executed as Python scripts with the `kwargs` provided in the `eval.general.output.custom_scripts.<script_name>.kwargs` section.

The `kwargs` typically include the file or directory to operate on. To avoid overwriting contents it is recommended to provide a unique output file or directory name for the customization. It is also recommended that changes be limited to the contents of the output directory to avoid unintended side effects.

**Example:**
```yaml
eval:
  general:
    output:
      dir: ./.tmp/aiq/examples/simple_output/
      custom_scripts:
        convert_workflow_to_csv:
          script: examples/simple/src/aiq_simple/scripts/workflow_to_csv.py
          kwargs:
            input: ./.tmp/aiq/examples/simple_output/workflow_output.json
            output: ./.tmp/aiq/examples/simple_output/workflow.csv
```

## Remote Storage
### Evaluating remote datasets
You can evaluate a remote dataset by provide the information needed to download the dataset in the `eval.general.dataset` section of the `config.yml` file. The following is an example configuration to evaluate a remote dataset.
```yaml
eval:
  general:
    dataset:
      _type: json
      # Download dataset from remote storage using S3 credentials
      remote_file_path: input/langsmith.json
      file_path: ./.tmp/aiq/examples/simple_input/langsmith.json
      s3:
        endpoint_url: http://10.185.X.X:9000
        bucket: aiq-simple-bucket
        access_key: fake_access_key
        secret_key: fake_secret_key
```
The `remote_file_path` is the path to the dataset in the remote storage. The `file_path` is the local path where the dataset will be downloaded. The `s3` section contains the information needed to access the remote storage.

### Uploading output directory to remote storage
You can upload the contents of the entire output directory to remote storage by providing the information needed to upload the output directory in the `eval.general.output` section of the `config.yml` file. The following is an example configuration to upload the output directory to remote storage.
```yaml
eval:
  general:
    output:
      # Upload contents of output directory to remote storage using S3 credentials
      remote_dir: output
      s3:
        endpoint_url: http://10.185.X.X:9000
        bucket: aiq-simple-bucket
        access_key: fake-access-key
        secret_key: fake-secret-key
```
### Cleanup output directory
The contents of the output directory can be deleted before running the evaluation pipeline by specifying the `eval.general.output.cleanup` section in the `config.yml` file. The following is an example configuration to clean up the output directory before running the evaluation pipeline.
```yaml
eval:
  general:
    output:
      dir: ./.tmp/aiq/examples/simple_output/
      cleanup: true
```
Output directory cleanup is disabled by default for easy troubleshooting.

## Profiling and Performance Monitoring of AgentIQ Workflows
You can profile workflows via the AgentIQ evaluation system. For more information, see the [Profiler](profiler.md) documentation.
