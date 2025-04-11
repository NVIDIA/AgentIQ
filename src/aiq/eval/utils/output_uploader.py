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

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import aioboto3
from botocore.exceptions import NoCredentialsError
from tqdm import tqdm

from aiq.data_models.evaluate import EvalOutputConfig

logger = logging.getLogger(__name__)


class OutputUploader:
    """
    Run custom scripts and upload evaluation outputs using the configured s3
    credentials.
    """

    def __init__(self, output_config: EvalOutputConfig, job_id: str | None = None):
        self.output_config = output_config
        self._s3_client = None
        self.job_id = job_id

    @property
    def s3_config(self):
        return self.output_config.s3

    async def _upload_file(self, s3_client, bucket, s3_key, local_path, pbar):
        try:
            await s3_client.upload_file(str(local_path), bucket, s3_key)
            logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, s3_key)
            pbar.update(1)
        except Exception as e:
            logger.error("Failed to upload %s to s3://%s/%s: %s", local_path, bucket, s3_key, e)
            raise

    async def upload_directory(self):
        """
        Upload the contents of the local output directory to the remote S3 bucket in parallel.
        """
        if not self.output_config.s3:
            logger.info("No S3 config provided; skipping upload.")
            return

        local_dir = self.output_config.dir
        bucket = self.s3_config.bucket
        remote_prefix = self.output_config.remote_dir or ""
        if self.job_id:
            remote_prefix = str(Path(remote_prefix) / f"jobs/{self.job_id}")

        file_entries = []
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = Path(root) / file
                relative_path = local_path.relative_to(local_dir)
                s3_path = Path(remote_prefix) / relative_path
                s3_key = str(s3_path).replace("\\", "/")  # Normalize for S3
                file_entries.append((local_path, s3_key))

        session = aioboto3.Session()
        try:
            async with session.client(
                    "s3",
                    endpoint_url=self.s3_config.endpoint_url,
                    aws_access_key_id=self.s3_config.access_key,
                    aws_secret_access_key=self.s3_config.secret_key,
            ) as s3_client:
                with tqdm(total=len(file_entries), desc="Uploading files to S3") as pbar:
                    upload_tasks = [
                        self._upload_file(s3_client, bucket, s3_key, local_path, pbar)
                        for local_path, s3_key in file_entries
                    ]
                    await asyncio.gather(*upload_tasks)

        except NoCredentialsError as e:
            logger.error("AWS credentials not available: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to upload files to S3: %s", e)
            raise

    def run_custom_scripts(self):
        """
        Run custom Python scripts defined in the EvalOutputConfig.
        Each script is run with its kwargs passed as command-line arguments.
        The output directory is passed as the first argument.
        """
        for script_name, script_config in self.output_config.custom_scripts.items():
            script_path = script_config.script
            if not script_path.exists():
                logger.error("Custom script %s does not exist.", script_path)
                continue

            # use python interpreter
            args = [sys.executable, str(script_path)]
            # add output directory as first keyword argument
            args.append("--output_dir")
            args.append(str(self.output_config.dir))
            if script_config.kwargs:
                for key, value in script_config.kwargs.items():
                    args.append(f"--{key}")
                    args.append(str(value))

            display_args = " ".join(f'"{arg}"' if " " in arg else arg for arg in args[1:])

            try:
                logger.info("Running custom script: %s %s", script_path, display_args)
                subprocess.run(args, check=True, text=True)
                logger.info("Custom script %s completed successfully.", script_path)
            except subprocess.CalledProcessError as e:
                logger.error("Custom script %s failed with return code %s", script_path, e.returncode)
                raise
