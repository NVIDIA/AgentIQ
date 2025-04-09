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

import numpy as np

from aiq.profiler.forecasting.models.forecasting_base_model import ForecastingBaseModel
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor

logger = logging.getLogger(__name__)


class LinearModel(ForecastingBaseModel):
    """
    A linear regression model that conforms to the BaseModel interface.
    """

    def __init__(self):
        super().__init__()

        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            logger.error("scikit-learn is not installed. Please install scikit-learn to use the LinearModel "
                         "profiling model or install `aiq[profiler]` to install all necessary profiling packages.")

            raise

        self.model = LinearRegression()
        self.matrix_length = None

    def fit(self, raw_stats: list[list[IntermediatePropertyAdaptor]]):
        """
        X: shape (N, M)  # M = matrix_length * 4
        y: shape (N, 4)
        """
        x_flat, y_flat = self._prep_for_model_training(raw_stats)

        logger.info("Training dataset size: X=%s, y=%s", x_flat.shape, y_flat.shape)

        # 3) Fit
        self.model.fit(x_flat, y_flat)

    def predict(self, raw_stats: list[list[IntermediatePropertyAdaptor]]) -> np.ndarray:
        """
        Predict using the fitted linear model.
        Returns shape (N, 4)
        """
        X = self._prep_single(raw_stats)
        return self.model.predict(X)

    def _prep_single(self, raw_stats: list[list[IntermediatePropertyAdaptor]]) -> np.ndarray:
        arr, _ = self._extract_token_usage_meta(raw_stats)
        arr = arr[0]
        n_rows = arr.shape[0]

        matrix_length = self.matrix_length

        assert matrix_length is not None, "matrix_length must be set before calling _prep_single"

        if n_rows >= matrix_length:
            # Keep the latest matrix_length rows
            x_mat = arr[-matrix_length:, :]
        else:
            # Pad with zeros at the top
            pad_size = matrix_length - n_rows
            pad_block = np.zeros((pad_size, arr.shape[1]), dtype=arr.dtype)
            x_mat = np.vstack([pad_block, arr])

        return x_mat

    def _prep_for_model_training(self, raw_stats: list[list[IntermediatePropertyAdaptor]]):
        raw_matrices, matrix_length = self._extract_token_usage_meta(raw_stats)

        self.matrix_length = matrix_length

        x_list = []
        y_list = []
        for arr in raw_matrices:
            samples = self._preprocess_for_forecasting(arr, matrix_length)
            for (x_mat, y_mat) in samples:
                x_list.append(x_mat)
                y_list.append(y_mat)

        # 2) Flatten features
        x_flat, y_flat = self._flatten_features(x_list, y_list)

        return x_flat, y_flat

    def _extract_token_usage_meta(self, all_requests_data: list[list[IntermediatePropertyAdaptor]]):
        import math

        all_run_data = []
        call_stack_sizes = []

        for prompt in all_requests_data:
            run_data = []
            seconds_between_call_map = {}

            for stat in prompt:
                if stat.event_type.value == "LLM_START":
                    seconds_between_call_map[stat.UUID] = stat.seconds_between_calls

                if stat.event_type.value == "LLM_END":
                    step_data = [
                        seconds_between_call_map[stat.UUID],
                        stat.token_usage.prompt_tokens,
                        stat.token_usage.completion_tokens
                    ]

                    run_data.append(step_data)

            all_run_data.append(run_data)
            call_stack_sizes.append(len(run_data))

        all_run_data = [np.array(run) for run in all_run_data]
        recommended_matrix_length = math.ceil(sum(call_stack_sizes) / len(call_stack_sizes))

        return all_run_data, recommended_matrix_length

    def _preprocess_for_forecasting(self, arr: np.ndarray, matrix_length: int):
        """
        Given a 2D NumPy array `arr` of shape (n_rows, 4), generate a list of
        (input_array, output_array) pairs for forecasting, each of shape:

        - input_array: (matrix_length, 4) after padding/trimming
        - output_array: (1, 4)
        """
        n_rows = arr.shape[0]

        # partial_sums[i] = sum of arr[i:] per column
        partial_sums = np.flip(np.cumsum(np.flip(arr, axis=0), axis=0), axis=0)

        samples = []
        for i in range(n_rows):
            x_untrimmed = arr[:i + 1, :]
            # Trim or pad
            current_len = x_untrimmed.shape[0]
            if current_len > matrix_length:
                x_mat = x_untrimmed[-matrix_length:, :]
            elif current_len < matrix_length:
                pad_size = matrix_length - current_len
                pad_block = np.zeros((pad_size, x_untrimmed.shape[1]), dtype=arr.dtype)
                x_mat = np.vstack([pad_block, x_untrimmed])
            else:
                x_mat = x_untrimmed

            # Compute output
            if i == n_rows - 1:
                y_vec = np.array([0, 0, 0, 0], dtype=arr.dtype)
            else:
                n_below = n_rows - (i + 1)
                sum_below = partial_sums[i + 1]
                avg_col0 = sum_below[0] / n_below
                sum_rest = sum_below[1:]
                y_vec = np.concatenate(([avg_col0], sum_rest))

            samples.append((x_mat, y_vec.reshape(1, 4)))

        return samples

    def _flatten_features(self, x_list, y_list):
        """
        x_list: list of arrays, each of shape (matrix_length, 4)
        y_list: list of arrays, each of shape (1, 4)

        Returns:
            x_flat: np.array of shape (N, matrix_length*4)
            y_flat: np.array of shape (N, 4)
        """
        flattened_x = []
        flattened_y = []

        for x_mat, y_mat in zip(x_list, y_list):
            x_1d = x_mat.flatten()  # shape -> (matrix_length*4,)
            y_1d = y_mat.flatten()  # shape -> (4,)
            flattened_x.append(x_1d)
            flattened_y.append(y_1d)

        x_flat = np.array(flattened_x)
        y_flat = np.array(flattened_y)
        logger.debug("Flattened features to shapes: %s (X), %s (y).", x_flat.shape, y_flat.shape)
        return x_flat, y_flat
