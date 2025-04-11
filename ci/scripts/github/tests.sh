#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

GITHUB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${GITHUB_SCRIPT_DIR}/common.sh
get_lfs_files

create_env group:dev extra:all
rapids-logger "Git Version: $(git describe)"

rapids-logger "Running tests"
set +e

pytest --junit-xml=${WORKSPACE_TMP}/report_pytest.xml \
       --cov=aiq --cov-report term-missing \
       --cov-report=xml:${WORKSPACE_TMP}/report_pytest_coverage.xml
PYTEST_RESULTS=$?

exit ${PYTEST_RESULTS}