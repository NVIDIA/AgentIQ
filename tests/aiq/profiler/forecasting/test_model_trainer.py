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

import pytest

from aiq.profiler.forecasting.model_trainer import ModelTrainer
from aiq.profiler.forecasting.model_trainer import create_model
from aiq.profiler.forecasting.models import ForecastingBaseModel
from aiq.profiler.forecasting.models import LinearModel
from aiq.profiler.forecasting.models import RandomForestModel
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor


@pytest.mark.parametrize("model_type, expected_model_class", [
    ("linear", LinearModel),
    ("randomforest", RandomForestModel),
],
                         ids=["linear", "randomforest"])
def test_create_model(model_type: str, expected_model_class: type[ForecastingBaseModel]):
    assert isinstance(create_model(model_type), expected_model_class)


def test_create_model_invalid_type():
    with pytest.raises(ValueError, match="Unsupported model_type: unsupported_model"):
        create_model("unsupported_model")


@pytest.mark.parametrize("model_trainer_kwargs", [
    {},
    {
        "model_type": "linear"
    },
    {
        "model_type": "randomforest"
    },
])
def test_model_trainer_initialization(model_trainer_kwargs: dict):
    mt = ModelTrainer(**model_trainer_kwargs)
    if "model_type" in model_trainer_kwargs:
        assert mt.model_type == model_trainer_kwargs["model_type"]


@pytest.mark.parametrize("model_type, expected_model_class", [("linear", LinearModel),
                                                              ("randomforest", RandomForestModel)],
                         ids=["linear", "randomforest"])
def test_model_trainer_train(model_type: str,
                             expected_model_class: type[ForecastingBaseModel],
                             rag_intermediate_property_adaptor: list[list[IntermediatePropertyAdaptor]]):
    mt = ModelTrainer(model_type=model_type)
    model = mt.train(rag_intermediate_property_adaptor)
    assert isinstance(model, expected_model_class)
