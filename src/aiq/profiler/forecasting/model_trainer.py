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

# forecasting/model_trainer.py

import logging

from aiq.profiler.forecasting.config import DEFAULT_MODEL_TYPE
from aiq.profiler.forecasting.models import ForecastingBaseModel
from aiq.profiler.forecasting.models import LinearModel
from aiq.profiler.forecasting.models import RandomForestModel
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor

logger = logging.getLogger(__name__)


def create_model(model_type: str) -> ForecastingBaseModel:
    """
    A simple factory method that returns a model instance
    based on the input string. Extend this with more model
    classes (e.g., PolynomialModel, RandomForestModel, etc.).
    """
    if model_type == "linear":
        return LinearModel()
    if model_type == "randomforest":
        return RandomForestModel()

    raise ValueError(f"Unsupported model_type: {model_type}")


class ModelTrainer:
    """
    Orchestrates data preprocessing, training, and returning
    a fitted model.

    Parameters
    ----------
    model_type: str, default = "randomforest"
        The type of model to train. Options include "linear" and "randomforest".
    """

    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE):
        self.model_type = model_type
        self._model = create_model(self.model_type)

    def train(self, raw_stats: list[list[IntermediatePropertyAdaptor]]) -> ForecastingBaseModel:
        """
        Train the model using the `raw_stats` training data.

        Parameters
        ----------
        raw_stats: list[list[IntermediatePropertyAdaptor]]
            Stats collected by the profiler.

        Returns
        -------
        ForecastingBaseModel
            A fitted model.
        """

        self._model.fit(raw_stats)

        return self._model
