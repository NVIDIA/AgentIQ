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
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"


# pydantic model for the job status
class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    config_file: str
    error: str | None
    output_path: str | None
    created_at: datetime
    updated_at: datetime
    expiry_seconds: int


class JobStore:

    MIN_EXPIRY = 600  # 10 minutes
    MAX_EXPIRY = 86400  # 24 hours
    DEFAULT_EXPIRY = 3600  # 1 hour

    # active jobs are exempt from expiry
    ACTIVE_STATUS = {"running", "submitted"}

    def __init__(self):
        self._jobs = {}

    def create_job(self, config_file: str, job_id: str | None = None, expiry_seconds: int = DEFAULT_EXPIRY) -> str:
        if job_id is None:
            job_id = str(uuid4())

        clamped_expiry = max(self.MIN_EXPIRY, min(expiry_seconds, self.MAX_EXPIRY))
        if expiry_seconds != clamped_expiry:
            logger.info(f"Clamped expiry_seconds from {expiry_seconds} to {clamped_expiry} for job {job_id}")

        job = JobInfo(job_id=job_id,
                      status=JobStatus.SUBMITTED,
                      config_file=config_file,
                      created_at=datetime.now(UTC),
                      updated_at=datetime.now(UTC),
                      error=None,
                      output_path=None,
                      expiry_seconds=clamped_expiry)
        self._jobs[job_id] = job
        logger.info(f"Created new job {job_id} with config {config_file}")
        return job_id

    def update_status(self, job_id: str, status: str, error: str | None = None, output_path: str | None = None):
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs[job_id]
        job.status = status
        job.error = error
        job.output_path = output_path
        job.updated_at = datetime.now(UTC)

    def get_status(self, job_id: str) -> JobInfo | None:
        return self._jobs.get(job_id)

    def list_jobs(self):
        return self._jobs

    def get_job(self, job_id: str) -> JobInfo | None:
        """Get a job by its ID."""
        return self._jobs.get(job_id)

    def get_last_job(self) -> JobInfo | None:
        """Get the last created job."""
        if not self._jobs:
            logger.info("No jobs found in job store")
            return None
        last_job = max(self._jobs.values(), key=lambda job: job.created_at)
        logger.info(f"Retrieved last job {last_job.job_id} created at {last_job.created_at}")
        return last_job

    def get_jobs_by_status(self, status: str) -> list[JobInfo]:
        """Get all jobs with the specified status."""
        return [job for job in self._jobs.values() if job.status == status]

    def get_all_jobs(self) -> list[JobInfo]:
        """Get all jobs in the store."""
        return list(self._jobs.values())

    def get_expires_at(self, job: JobInfo) -> datetime | None:
        """Get the time for a job to expire."""
        if job.status in self.ACTIVE_STATUS:
            return None
        return job.created_at + timedelta(seconds=job.expiry_seconds)

    def cleanup_expired_jobs(self):
        """Cleanup expired jobs that are not active."""
        now = datetime.now(UTC)

        finished_jobs = {job_id: job for job_id, job in self._jobs.items() if job.status not in self.ACTIVE_STATUS}

        # Sort finished jobs by created_at descending
        sorted_finished = sorted(finished_jobs.items(), key=lambda item: item[1]["created_at"], reverse=True)

        # Always keep the most recent finished job
        jobs_to_check = sorted_finished[1:]

        expired_ids = [
            job_id for job_id, job in jobs_to_check
            if (now - job["created_at"]).total_seconds() > job.get("expiry_seconds", 3600)
        ]

        for job_id in expired_ids:
            del self.jobs[job_id]
