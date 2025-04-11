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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiq.data_models.config import AIQConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker


@pytest.fixture
def test_config():
    config = AIQConfig()
    config.general.front_end = FastApiFrontEndConfig(evaluate=FastApiFrontEndConfig.EndpointBase(
        path="/evaluate", method="POST", description="Test evaluate endpoint"))
    return config


@pytest.fixture(autouse=True)
def patch_evaluation_run():
    with patch("aiq.front_ends.fastapi.fastapi_front_end_plugin_worker.EvaluationRun") as MockEvaluationRun:
        mock_eval_instance = MagicMock()
        mock_eval_instance.run_and_evaluate = AsyncMock(
            return_value=MagicMock(workflow_interrupted=False, workflow_output_file="/fake/output/path.json"))
        MockEvaluationRun.return_value = mock_eval_instance
        yield MockEvaluationRun


@pytest.fixture
def test_client(test_config):
    worker = FastApiFrontEndPluginWorker(test_config)
    app = FastAPI()
    worker.set_cors_config(app)

    with patch("aiq.front_ends.fastapi.fastapi_front_end_plugin_worker.AIQSessionManager") as MockSessionManager:

        # Mock session manager
        mock_session = MagicMock()
        MockSessionManager.return_value = mock_session

        async def setup():
            await worker.add_evaluate_route(app, session_manager=mock_session)

        asyncio.run(setup())

    return TestClient(app)


def create_job(test_client, config_file="/path/to/config.yml"):
    """Helper to create an evaluation job."""
    return test_client.post("/evaluate", json={"config_file": config_file})


def test_create_job(test_client):
    """Test creating a new evaluation job."""
    response = create_job(test_client)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "submitted"


def test_get_job_status(test_client):
    """Test getting the status of a specific job."""
    create_response = create_job(test_client)
    job_id = create_response.json()["job_id"]

    status_response = test_client.get(f"/evaluate/job/{job_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "success"
    assert data["config_file"] == "/path/to/config.yml"


def test_get_job_status_not_found(test_client):
    """Test getting status of a non-existent job."""
    response = test_client.get("/evaluate/job/non-existent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job non-existent-id not found"


def test_get_last_job(test_client):
    """Test getting the last created job."""
    for i in range(3):
        create_job(test_client, config_file=f"/path/to/config_{i}.yml")

    response = test_client.get("/evaluate/job/last")
    assert response.status_code == 200
    data = response.json()
    assert data["config_file"] == "/path/to/config_2.yml"


def test_get_last_job_not_found(test_client):
    """Test getting last job when no jobs exist."""
    response = test_client.get("/evaluate/job/last")
    assert response.status_code == 404
    assert response.json()["detail"] == "No jobs found"


def test_get_all_jobs(test_client):
    """Test retrieving all jobs."""
    for i in range(3):
        create_job(test_client, config_file=f"/path/to/config_{i}.yml")

    response = test_client.get("/evaluate/jobs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


@pytest.mark.parametrize("status,expected_count", [
    ("success", 3),
    ("interrupted", 0),
])
def test_get_jobs_by_status(test_client, status, expected_count):
    """Test getting jobs filtered by status."""
    for i in range(3):
        create_job(test_client, config_file=f"/path/to/config_{i}.yml")

    response = test_client.get(f"/evaluate/jobs?status={status}")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == expected_count
    if status == "submitted":
        assert all(job["status"] == "submitted" for job in data)


def test_create_job_with_reps(test_client):
    """Test creating a new evaluation job with custom repetitions."""
    response = test_client.post("/evaluate", json={"config_file": "/path/to/config.yml", "reps": 3})
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "submitted"


def test_create_job_with_expiry(test_client):
    """Test creating a new evaluation job with custom expiry time."""
    response = test_client.post(
        "/evaluate",
        json={
            "config_file": "/path/to/config.yml",
            "expiry_seconds": 1800  # 30 minutes
        })
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "submitted"


def test_create_job_with_job_id(test_client):
    """Test creating a new evaluation job with a specific job ID."""
    job_id = "test-job-123"
    response = test_client.post("/evaluate", json={"config_file": "/path/to/config.yml", "job_id": job_id})
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "submitted"
