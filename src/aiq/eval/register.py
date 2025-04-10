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

# flake8: noqa
# pylint: disable=unused-import

# Import evaluators which need to be automatically registered here
from .rag_evaluator.register import register_ragas_evaluator
from .swe_bench_evaluator.register import register_swe_bench_evaluator
from .trajectory_evaluator.register import register_trajectory_evaluator
from .custom_rag_evaluator.register import register_custom_rag_evaluator
