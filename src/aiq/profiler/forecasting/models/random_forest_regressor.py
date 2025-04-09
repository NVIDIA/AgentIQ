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


class RandomForestModel(ForecastingBaseModel):
    """
    A random forest regressor that predicts n_step token usage and call latency.
    """

    def __init__(self):
        super().__init__()

        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            logger.error("scikit-learn is not installed. Please install scikit-learn to use the RandomForest "
                         "profiling model or install `aiq[profiler]` to install all necessary profiling packages.")

            raise

        self.model = RandomForestRegressor(n_estimators=3, max_depth=2)
        self.matrix_length = None

    def fit(self, raw_stats: list[list[IntermediatePropertyAdaptor]]):
        """
        X: shape (N, M)  # M = matrix_length * 4
        y: shape (N, 4)
        """

        x_flat, y_flat = self._prep_for_model_training(raw_stats)

        # 3) Fit
        self.model.fit(x_flat, y_flat)

    def predict(self, raw_stats: list[list[IntermediatePropertyAdaptor]]) -> np.ndarray:
        """
        Predict using the fitted linear model.
        Returns shape (N, 4)
        """
        x = self._prep_single(raw_stats)
        return self.model.predict(x)

    def _prep_single(self, raw_stats: list[list[IntermediatePropertyAdaptor]]) -> np.ndarray:

        arr, _ = self._extract_token_usage_meta(raw_stats)
        arr = arr[0]

        assert self.matrix_length is not None, "Model has not been trained yet."

        n = self.matrix_length

        if arr.shape[1] != 3:
            raise ValueError("The input array must have exactly 3 columns.")

        t = arr.shape[0]

        # 1) Slice or pad to get the latest n rows
        if t >= n:
            x_mat = arr[-n:].copy()
        else:
            pad_size = n - t
            pad_block = np.zeros((pad_size, 3), dtype=arr.dtype)
            x_mat = np.vstack([pad_block, arr])

        # 2) Zero out the output_prompt_tokens in the last row (index 2)
        x_mat[-1, 2] = 0

        return x_mat

    def _prep_for_model_training(self, raw_stats: list[list[IntermediatePropertyAdaptor]]):

        raw_matrices, matrix_length = self._extract_token_usage_meta(raw_stats)

        self.matrix_length = matrix_length

        samples = self._preprocess_for_forecasting(raw_matrices, matrix_length, matrix_length)

        x_list = []
        y_list = []
        for (x_mat, y_mat) in samples:
            x_list.append(x_mat)
            y_list.append(y_mat)

        # 2) Flatten features
        x_flat, y_flat = self._flatten_features(x_list, y_list)

        return x_flat, y_flat

    def _preprocess_for_forecasting(
        self,
        arrays: list[np.ndarray],
        n: int = 3,
        k: int = 3,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Preprocess a list of arrays where each array has shape (T, 3),
        with columns:

        0: seconds_since_last_llm_call
        1: input_prompt_tokens
        2: output_prompt_tokens

        For each row 'i' in each array, produce:
          X: shape (n, 3)
             -> The previous n calls up to row i (padded if needed).
                For row i itself, set output_prompt_tokens=0
                (simulate unknown current output).
          Y: shape (k, 3)
             -> The next k calls after row i (padded if needed).

        Parameters
        ----------
        arrays : list of np.ndarray
            Each array is shape (T, 3).
        n : int
            Number of past calls to include for the input context (window size).
        k : int
            Number of future calls to include in the label (forecast horizon).

        Returns
        -------
        samples : list of (X, Y) tuples
            Each X has shape (n, 3), each Y has shape (k, 3).
        """

        samples = []

        for arr in arrays:
            t = arr.shape[0]

            # Safety check (optional)
            if arr.shape[1] != 3:
                raise ValueError("Each array must have exactly 3 columns.")

            for i in range(t):
                # --- 1) Build X: the context window for rows [i-n+1 .. i] ---

                # The 'start_idx' is the first row in the n-window
                start_idx = i - n + 1
                if start_idx < 0:
                    # we need padding at the top
                    pad_size = -start_idx
                    # create a zero block for that portion
                    pad_block = np.zeros((pad_size, 3), dtype=arr.dtype)
                    # portion of the real data we actually have
                    real_block = arr[:i + 1, :].copy()  # up to row i inclusive

                    # Concatenate
                    x_mat = np.vstack([pad_block, real_block])
                else:
                    # we have enough rows, just slice
                    x_mat = arr[start_idx:i + 1, :].copy()

                # Now X_mat is shape (<= n, 3). If it's < n, we've padded.
                # If it's exactly n, fine. If it's bigger (shouldn't be), we slice again:
                if x_mat.shape[0] > n:
                    x_mat = x_mat[-n:, :]

                # For the "current" row in X_mat (the last row in that slice),
                # we zero-out the output_prompt_tokens column:
                # This simulates "unknown" output for the current call.
                x_mat[-1, 2] = 0

                # If it's still shorter than n, do final padding from the top:
                if x_mat.shape[0] < n:
                    missing = n - x_mat.shape[0]
                    pad_block2 = np.zeros((missing, 3), dtype=arr.dtype)
                    x_mat = np.vstack([pad_block2, x_mat])

                # Ensure shape is exactly (n, 3)
                assert x_mat.shape == (n, 3), f"Expected (n,3), got {x_mat.shape}"

                # --- 2) Build Y: the next k calls i+1 .. i+k ---
                end_idx = i + k
                if end_idx > t - 1:
                    # if we go beyond the last row, we pad
                    real_portion = arr[i + 1:t, :].copy()  # might be empty if i == T-1
                    pad_needed = k - real_portion.shape[0]
                    if pad_needed > 0:
                        pad_block = np.zeros((pad_needed, 3), dtype=arr.dtype)
                        y_mat = np.vstack([real_portion, pad_block])
                    else:
                        y_mat = real_portion
                else:
                    # we have enough future rows
                    y_mat = arr[i + 1:i + 1 + k, :].copy()

                # Ensure shape is exactly (k, 3)
                assert y_mat.shape == (k, 3), f"Expected (k,3), got {y_mat.shape}"

                # 3) Collect the (X, Y) pair
                samples.append((x_mat, y_mat))

        return samples

    def _extract_token_usage_meta(self, all_requests_data: list[list[IntermediatePropertyAdaptor]]):

        import math

        all_run_data = []
        call_stack_sizes = []
        seconds_between_call_map = {}

        for usage_stats in all_requests_data:
            run_data = []
            for stat in usage_stats:
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

    def _flatten_features(self, x_list, y_list):
        """
        X_list: list of arrays, each of shape (matrix_length, 4)
        y_list: list of arrays, each of shape (1, 4)

        Returns:
            X_flat: np.array of shape (N, matrix_length*4)
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
        return x_flat, y_flat
