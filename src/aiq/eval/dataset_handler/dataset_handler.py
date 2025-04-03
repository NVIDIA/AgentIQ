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

import json
from collections import defaultdict

import pandas as pd

from aiq.data_models.dataset_handler import EvalDatasetConfig
from aiq.data_models.dataset_handler import EvalDatasetJsonConfig
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.eval.dataset_handler.dataset_downloader import DatasetDownloader
from aiq.eval.dataset_handler.dataset_filter import DatasetFilter
from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem
from aiq.eval.evaluator.evaluator_model import EvalOutput


class DatasetHandler:
    """
    Read the datasets and pre-process (apply filters, deduplicate etc.) before turning them into EvalInput objects.
    One DatasetHandler object is needed for each dataset to be evaluated.
    """

    def __init__(self, dataset_config: EvalDatasetConfig, reps: int):
        from aiq.eval.intermediate_step_adapter import IntermediateStepAdapter

        self.dataset_config = dataset_config
        self.dataset_filter = DatasetFilter(dataset_config.filter)
        self.reps = reps
        # Helpers
        self.intermediate_step_adapter = IntermediateStepAdapter()

    def is_structured_input(self) -> bool:
        '''Check if the input is structured or unstructured'''
        return not self.dataset_config.structure.disable

    @property
    def id_key(self) -> str:
        return self.dataset_config.id_key

    @property
    def question_key(self) -> str:
        return self.dataset_config.structure.question_key

    @property
    def answer_key(self) -> str:
        return self.dataset_config.structure.answer_key

    @property
    def generated_answer_key(self) -> str:
        return self.dataset_config.structure.generated_answer_key

    @property
    def trajectory_key(self) -> str:
        return self.dataset_config.structure.trajectory_key

    @property
    def expected_trajectory_key(self) -> str:
        return self.dataset_config.structure.expected_trajectory_key

    def get_eval_input_from_df(self, input_df: pd.DataFrame) -> EvalInput:

        def create_eval_item(row: pd.Series, structured: bool) -> EvalInputItem:
            """Helper function to create EvalInputItem."""
            return EvalInputItem(
                id=row.get(self.id_key, ""),
                input_obj=row.to_json() if not structured else row.get(self.question_key, ""),
                expected_output_obj=row.get(self.answer_key, "") if structured else "",
                output_obj=row.get(self.generated_answer_key, "") if structured else "",
                trajectory=row.get(self.trajectory_key, []) if structured else [],
                expected_trajectory=row.get(self.expected_trajectory_key, []) if structured else [],
            )

        # if input dataframe is empty return an empty list
        if input_df.empty:
            return EvalInput(eval_input_items=[])

        structured = self.is_structured_input()
        if structured:
            # Fill missing ground truth answers with empty string
            input_df[self.answer_key] = input_df[self.answer_key].fillna("")
            # For structured input, question is mandatory. Ignore rows with missing or empty questions
            input_df = input_df[input_df[self.question_key].notnull() & input_df[self.question_key].str.strip().ne("")]
        eval_input_items = [create_eval_item(row, structured) for _, row in input_df.iterrows()]

        return EvalInput(eval_input_items=eval_input_items)

    def setup_reps(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """replicate the rows and update the id to id_key + "_rep" + rep_number"""
        # Replicate the rows
        input_df = pd.concat([input_df] * self.reps, ignore_index=True)
        # Compute repetition index
        rep_index = input_df.groupby(self.dataset_config.id_key).cumcount().astype(str)
        # Convert id_key to string (id can be integer) if needed and update IDs
        input_df[self.dataset_config.id_key] = input_df[self.dataset_config.id_key].astype(str) + "_rep" + rep_index
        # Ensure unique ID values after modification
        input_df.drop_duplicates(subset=[self.dataset_config.id_key], inplace=True)

        return input_df

    def get_eval_input_from_dataset(self, dataset: str) -> EvalInput:
        # read the dataset and convert it to EvalInput

        # if a dataset file has been provided in the command line, use that
        dataset_config = EvalDatasetJsonConfig(file_path=dataset) if dataset else self.dataset_config

        # Download the dataset if it is remote
        downloader = DatasetDownloader(dataset_config=dataset_config)
        downloader.download_dataset()

        parser, kwargs = dataset_config.parser()
        # Parse the dataset into a DataFrame
        input_df = parser(dataset_config.file_path, **kwargs)

        # Apply filters and deduplicate
        input_df = self.dataset_filter.apply_filters(input_df)
        input_df.drop_duplicates(subset=[self.dataset_config.id_key], inplace=True)

        # If more than one repetition is needed, replicate the rows
        if self.reps > 1:
            input_df = self.setup_reps(input_df)

        # Convert the DataFrame to a list of EvalInput objects
        return self.get_eval_input_from_df(input_df)

    def filter_intermediate_steps(self, intermediate_steps: list[IntermediateStep]) -> list[dict]:
        """
        Filter out the intermediate steps that are not relevant for evaluation.
        The output is written with with the intention of re-running the evaluation using the original config file.
        """
        filtered_steps = self.intermediate_step_adapter.filter_intermediate_steps(
            intermediate_steps, self.intermediate_step_adapter.DEFAULT_EVENT_FILTER)
        return self.intermediate_step_adapter.serialize_intermediate_steps(filtered_steps)

    def publish_eval_input(self, eval_input) -> str:
        """
        Convert the EvalInput object to a JSON output for storing in a file. Use the orginal keys to
        allow re-running evaluation using the orignal config file and '--skip_workflow' option.
        """
        indent = 2
        if self.is_structured_input():
            # Extract structured data from EvalInputItems
            data = [{
                self.id_key: item.id,
                self.question_key: item.input_obj,
                self.answer_key: item.expected_output_obj,
                self.generated_answer_key: item.output_obj,
                self.trajectory_key: self.filter_intermediate_steps(item.trajectory),
                self.expected_trajectory_key: self.filter_intermediate_steps(item.expected_trajectory),
            } for item in eval_input.eval_input_items]
        else:
            # Unstructured case: return only raw output objects as a JSON array
            data = [json.loads(item.output_obj) for item in eval_input.eval_input_items]

        return json.dumps(data, indent=indent, ensure_ascii=False)

    def publish_ground_truth(self, eval_input) -> str | None:
        """
        Convert the EvalInput object to a JSON output for storing in a file. Use the orginal keys to
        allow re-running evaluation using the orignal config file and '--skip_workflow' option.
        """
        indent = 2
        if not self.is_structured_input():
            None

        # Extract structured data from EvalInputItems
        data = [{
            self.id_key: item.id,
            self.question_key: item.input_obj,
            self.answer_key: item.expected_output_obj,
        } for item in eval_input.eval_input_items]

        return json.dumps(data, indent=indent, ensure_ascii=False)

    def generate_ground_truth(self, original_eval_input: EvalInput,
                              evaluation_results: list[tuple[str, EvalOutput]]) -> EvalInput:
        """
        Generate a ground truth EvalInput based on the evaluation results from multiple evaluators.

        1. Each evaluator's result can have multiple repetitions for the same `id_key`.
           Reps are stored with `id_key = id_key + "_rep" + rep_number`.
        2. The average score is computed per-repetition across all evaluators.
        3. The repetition with the highest average score is selected.
        4. The best repetitions replace the original input items, and a new EvalInput is returned.
        """
        if not self.is_structured_input():
            return None

        if self.reps == 1:
            # Directly use original_eval_input since there's only one repetition
            best_eval_items = [
                EvalInputItem(
                    id=item.id,
                    input_obj=item.input_obj,
                    expected_output_obj=item.expected_output_obj,  # Comes from original input
                    output_obj=None,
                    expected_trajectory=[],
                    trajectory=[],
                ) for item in original_eval_input.eval_input_items  # No dictionary needed
            ]
            return EvalInput(eval_input_items=best_eval_items)

        # Aggregate scores for each rep across all evaluators
        rep_scores = defaultdict(list)
        for _, eval_output in evaluation_results:
            for item in eval_output.eval_output_items:
                rep_scores[item.id].append(item.score)

        # Compute the average score per repetition
        avg_scores_per_rep = {rep_id: sum(scores) / len(scores) for rep_id, scores in rep_scores.items()}

        # Select the best repetition per original ID; if single rep, it is the best by default
        best_reps = {}
        for rep_id, avg_score in avg_scores_per_rep.items():
            original_id = rep_id.rsplit("_rep", 1)[0] if "_rep" in rep_id else rep_id
            if original_id not in best_reps or avg_score > best_reps[original_id][1]:
                best_reps[original_id] = (rep_id, avg_score)

        # Create a new EvalInput with original IDs using the best repetitions
        best_eval_items = []
        for id, (rep_id, _) in best_reps.items():
            # Find the original item using the rep_id
            for item in original_eval_input.eval_input_items:
                if item.id == rep_id:  # Match the best rep_id
                    # Use the generated answer as the ground truth/expected output
                    best_eval_items.append(
                        EvalInputItem(
                            id=id,  # Assign the base ID, without the rep suffix
                            input_obj=item.input_obj,
                            expected_output_obj=item.output_obj,
                            output_obj=None,
                            expected_trajectory=[],
                            trajectory=[],
                        ))
                    break
        return EvalInput(eval_input_items=best_eval_items)
