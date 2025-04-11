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
"""
Sample output custom script to convert the workflow output to a CSV file.
"""

import argparse
import csv
import json
from pathlib import Path


def customize_workflow_json(output_dir: Path, input_path: Path, output_path: Path):

    # input and output paths are relative to the output_dir
    input_path = output_dir / input_path
    output_path = output_dir / output_path

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    with input_path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a list of objects in the JSON file")

    cleaned = []
    for item in data:
        item.pop("intermediate_steps", None)
        cleaned.append(item)

    # Determine all field names across all rows
    fieldnames = sorted({key for row in cleaned for key in row})

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)

    print(f"Converted {input_path.name} to {output_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert workflow_output.json to workflow.csv")
    # output_dir is a mandatory first argument
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--input", type=Path, required=True, help="Path to workflow_output.json")
    parser.add_argument("--output", type=Path, required=True, help="Path to output CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    customize_workflow_json(args.output_dir, args.input, args.output)
