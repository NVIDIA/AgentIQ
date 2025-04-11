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
import typing
from abc import ABC
from abc import abstractmethod
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import BackgroundTasks
from fastapi import Body
from fastapi import FastAPI
from fastapi import Response
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import AIQResponseIntermediateStep
from aiq.data_models.config import AIQConfig
from aiq.eval.config import EvaluationRunOutput
from aiq.eval.evaluate import EvaluationRun
from aiq.eval.evaluate import EvaluationRunConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import AIQEvaluateRequest
from aiq.front_ends.fastapi.fastapi_front_end_config import AIQEvaluateResponse
from aiq.front_ends.fastapi.fastapi_front_end_config import AIQEvaluateStatusResponse
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.job_store import JobInfo
from aiq.front_ends.fastapi.job_store import JobStore
from aiq.front_ends.fastapi.response_helpers import generate_single_response
from aiq.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from aiq.front_ends.fastapi.response_helpers import generate_streaming_response_raw_as_str
from aiq.front_ends.fastapi.step_adaptor import StepAdaptor
from aiq.front_ends.fastapi.websocket import AIQWebSocket
from aiq.runtime.session import AIQSessionManager

logger = logging.getLogger(__name__)


class FastApiFrontEndPluginWorkerBase(ABC):

    def __init__(self, config: AIQConfig):
        self._config = config

        assert isinstance(config.general.front_end,
                          FastApiFrontEndConfig), ("Front end config is not FastApiFrontEndConfig")

        self._front_end_config = config.general.front_end

    @property
    def config(self) -> AIQConfig:
        return self._config

    @property
    def front_end_config(self) -> FastApiFrontEndConfig:

        return self._front_end_config

    def build_app(self) -> FastAPI:

        # Create the FastAPI app and configure it
        @asynccontextmanager
        async def lifespan(starting_app: FastAPI):

            logger.debug("Starting AgentIQ server from process %s", os.getpid())

            async with WorkflowBuilder.from_config(self.config) as builder:

                await self.configure(starting_app, builder)

                yield

                # If a cleanup task is running, cancel it
                cleanup_task = getattr(starting_app.state, "cleanup_task", None)
                if cleanup_task:
                    logger.info("Cancelling cleanup task")
                    cleanup_task.cancel()

            logger.debug("Closing AgentIQ server from process %s", os.getpid())

        aiq_app = FastAPI(lifespan=lifespan)

        self.set_cors_config(aiq_app)

        return aiq_app

    def set_cors_config(self, aiq_app: FastAPI) -> None:
        """
        Set the cross origin resource sharing configuration.
        """
        cors_kwargs = {}

        if self.front_end_config.cors.allow_origins is not None:
            cors_kwargs["allow_origins"] = self.front_end_config.cors.allow_origins

        if self.front_end_config.cors.allow_origin_regex is not None:
            cors_kwargs["allow_origin_regex"] = self.front_end_config.cors.allow_origin_regex

        if self.front_end_config.cors.allow_methods is not None:
            cors_kwargs["allow_methods"] = self.front_end_config.cors.allow_methods

        if self.front_end_config.cors.allow_headers is not None:
            cors_kwargs["allow_headers"] = self.front_end_config.cors.allow_headers

        if self.front_end_config.cors.allow_credentials is not None:
            cors_kwargs["allow_credentials"] = self.front_end_config.cors.allow_credentials

        if self.front_end_config.cors.expose_headers is not None:
            cors_kwargs["expose_headers"] = self.front_end_config.cors.expose_headers

        if self.front_end_config.cors.max_age is not None:
            cors_kwargs["max_age"] = self.front_end_config.cors.max_age

        aiq_app.add_middleware(
            CORSMiddleware,
            **cors_kwargs,
        )

    @abstractmethod
    async def configure(self, app: FastAPI, builder: WorkflowBuilder):
        pass

    @abstractmethod
    def get_step_adaptor(self) -> StepAdaptor:
        pass


class RouteInfo(BaseModel):

    function_name: str | None


class FastApiFrontEndPluginWorker(FastApiFrontEndPluginWorkerBase):

    def get_step_adaptor(self) -> StepAdaptor:

        return StepAdaptor(self.front_end_config.step_adaptor)

    async def configure(self, app: FastAPI, builder: WorkflowBuilder):

        # Do things like setting the base URL and global configuration options
        app.root_path = self.front_end_config.root_path

        await self.add_routes(app, builder)

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):

        await self.add_default_route(app, AIQSessionManager(builder.build()))
        await self.add_evaluate_route(app, AIQSessionManager(builder.build()))

        for ep in self.front_end_config.endpoints:

            entry_workflow = builder.build(entry_function=ep.function_name)

            await self.add_route(app, endpoint=ep, session_manager=AIQSessionManager(entry_workflow))

    async def add_default_route(self, app: FastAPI, session_manager: AIQSessionManager):

        await self.add_route(app, self.front_end_config.workflow, session_manager)

    async def add_evaluate_route(self, app: FastAPI, session_manager: AIQSessionManager):
        """Add the evaluate endpoint to the FastAPI app."""

        response_500 = {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }

        # Create job store for tracking evaluation jobs
        job_store = JobStore()
        # Don't run multiple evaluations at the same time
        evaluation_lock = asyncio.Lock()

        async def periodic_cleanup(job_store: JobStore):
            while True:
                try:
                    job_store.cleanup_expired_jobs()
                    logger.debug("Expired jobs cleaned up")
                except Exception as e:
                    logger.error("Error during job cleanup: %s", str(e))
                await asyncio.sleep(300)  # every 5 minutes

        def create_cleanup_task():
            # Schedule periodic cleanup of expired jobs on first job creation
            if not hasattr(app.state, "cleanup_task"):
                logger.info("Starting periodic cleanup task")
                app.state.cleanup_task = asyncio.create_task(periodic_cleanup(job_store))

        async def run_evaluation(job_id: str, config_file: str, reps: int, session_manager: AIQSessionManager):
            """Background task to run the evaluation."""
            async with evaluation_lock:
                try:
                    # Create EvaluationRunConfig using the CLI defaults
                    eval_config = EvaluationRunConfig(config_file=Path(config_file), dataset=None, reps=reps)

                    # Create a new EvaluationRun with the evaluation-specific config
                    job_store.update_status(job_id, "running")
                    eval_runner = EvaluationRun(eval_config)
                    output: EvaluationRunOutput = await eval_runner.run_and_evaluate(session_manager=session_manager,
                                                                                     job_id=job_id)
                    if output.workflow_interrupted:
                        job_store.update_status(job_id, "interrupted")
                    else:
                        parent_dir = os.path.dirname(
                            output.workflow_output_file) if output.workflow_output_file else None

                        job_store.update_status(job_id, "success", output_path=str(parent_dir))
                except Exception as e:
                    logger.error("Error in evaluation job %s: %s", job_id, str(e))
                    job_store.update_status(job_id, "failure", error=str(e))

        async def start_evaluation(request: AIQEvaluateRequest, background_tasks: BackgroundTasks):
            """Handle evaluation requests."""
            # if job_id is present and already exists return the job info
            if request.job_id:
                job = job_store.get_job(request.job_id)
                if job:
                    return AIQEvaluateResponse(job_id=job.job_id, status=job.status)

            job_id = job_store.create_job(request.config_file, request.job_id, request.expiry_seconds)
            create_cleanup_task()
            background_tasks.add_task(run_evaluation, job_id, request.config_file, request.reps, session_manager)
            return AIQEvaluateResponse(job_id=job_id, status="submitted")

        def translate_job_to_response(job: JobInfo) -> AIQEvaluateStatusResponse:
            """Translate a JobInfo object to an AIQEvaluateStatusResponse."""
            return AIQEvaluateStatusResponse(job_id=job.job_id,
                                             status=job.status,
                                             config_file=str(job.config_file),
                                             error=job.error,
                                             output_path=str(job.output_path),
                                             created_at=job.created_at,
                                             updated_at=job.updated_at,
                                             expires_at=job_store.get_expires_at(job))

        def get_job_status(job_id: str) -> AIQEvaluateStatusResponse:
            """Get the status of an evaluation job."""
            logger.info("Getting status for job %s", job_id)
            job = job_store.get_job(job_id)
            if not job:
                logger.warning("Job %s not found", job_id)
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            logger.info(f"Found job {job_id} with status {job.status}")
            return translate_job_to_response(job)

        def get_last_job_status() -> AIQEvaluateStatusResponse:
            """Get the status of the last created evaluation job."""
            logger.info("Getting last job status")
            job = job_store.get_last_job()
            if not job:
                logger.warning("No jobs found when requesting last job status")
                raise HTTPException(status_code=404, detail="No jobs found")
            logger.info("Found last job %s with status %s", job.job_id, job.status)
            return translate_job_to_response(job)

        def get_jobs(status: str | None = None) -> list[AIQEvaluateStatusResponse]:
            """Get all jobs, optionally filtered by status."""
            if status is None:
                logger.info("Getting all jobs")
                jobs = job_store.get_all_jobs()
            else:
                logger.info("Getting jobs with status %s", status)
                jobs = job_store.get_jobs_by_status(status)
            logger.info("Found %d jobs", len(jobs))
            return [translate_job_to_response(job) for job in jobs]

        if self.front_end_config.evaluate.path:
            # Add last job endpoint first (most specific)
            app.add_api_route(
                path=f"{self.front_end_config.evaluate.path}/job/last",
                endpoint=get_last_job_status,
                methods=["GET"],
                response_model=AIQEvaluateStatusResponse,
                description="Get the status of the last created evaluation job",
                responses={
                    404: {
                        "description": "No jobs found"
                    }, 500: response_500
                },
            )

            # Add jobs endpoint with optional status query parameter
            app.add_api_route(
                path=f"{self.front_end_config.evaluate.path}/jobs",
                endpoint=get_jobs,
                methods=["GET"],
                response_model=list[AIQEvaluateStatusResponse],
                description="Get all jobs, optionally filtered by status",
                responses={500: response_500},
            )

            # Add specific job endpoint (least specific)
            app.add_api_route(
                path=f"{self.front_end_config.evaluate.path}/job/{{job_id}}",
                endpoint=get_job_status,
                methods=["GET"],
                response_model=AIQEvaluateStatusResponse,
                description="Get the status of an evaluation job",
                responses={
                    404: {
                        "description": "Job not found"
                    }, 500: response_500
                },
            )

            # Add HTTP endpoint for evaluation
            app.add_api_route(
                path=self.front_end_config.evaluate.path,
                endpoint=start_evaluation,
                methods=[self.front_end_config.evaluate.method],
                response_model=AIQEvaluateResponse,
                description=self.front_end_config.evaluate.description,
                responses={500: response_500},
            )

    async def add_route(self,
                        app: FastAPI,
                        endpoint: FastApiFrontEndConfig.EndpointBase,
                        session_manager: AIQSessionManager):

        workflow = session_manager.workflow

        if (endpoint.websocket_path):
            app.add_websocket_route(endpoint.websocket_path,
                                    partial(AIQWebSocket, session_manager, self.get_step_adaptor()))

        GenerateBodyType = workflow.input_schema  # pylint: disable=invalid-name
        GenerateStreamResponseType = workflow.streaming_output_schema  # pylint: disable=invalid-name
        GenerateSingleResponseType = workflow.single_output_schema  # pylint: disable=invalid-name

        # Ensure that the input is in the body. POD types are treated as query parameters
        if (not issubclass(GenerateBodyType, BaseModel)):
            GenerateBodyType = typing.Annotated[GenerateBodyType, Body()]

        response_500 = {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }

        def get_single_endpoint(result_type: type | None):

            async def get_single(response: Response):

                response.headers["Content-Type"] = "application/json"

                return await generate_single_response(None, session_manager, result_type=result_type)

            return get_single

        def get_streaming_endpoint(streaming: bool, result_type: type | None, output_type: type | None):

            async def get_stream():

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_as_str(
                                             None,
                                             session_manager=session_manager,
                                             streaming=streaming,
                                             step_adaptor=self.get_step_adaptor(),
                                             result_type=result_type,
                                             output_type=output_type))

            return get_stream

        def get_streaming_raw_endpoint(streaming: bool, result_type: type | None, output_type: type | None):

            async def get_stream():

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_raw_as_str(None,
                                                                                        session_manager=session_manager,
                                                                                        streaming=streaming,
                                                                                        result_type=result_type,
                                                                                        output_type=output_type))

            return get_stream

        def post_single_endpoint(request_type: type, result_type: type | None):

            async def post_single(response: Response, payload: request_type):

                response.headers["Content-Type"] = "application/json"

                return await generate_single_response(payload, session_manager, result_type=result_type)

            return post_single

        def post_streaming_endpoint(request_type: type,
                                    streaming: bool,
                                    result_type: type | None,
                                    output_type: type | None):

            async def post_stream(payload: request_type):

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_as_str(
                                             payload,
                                             session_manager=session_manager,
                                             streaming=streaming,
                                             step_adaptor=self.get_step_adaptor(),
                                             result_type=result_type,
                                             output_type=output_type))

            return post_stream

        def post_streaming_raw_endpoint(request_type: type,
                                        streaming: bool,
                                        result_type: type | None,
                                        output_type: type | None):
            """
            Stream raw intermediate steps without any step adaptor translations.
            """

            async def post_stream(payload: request_type):

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_raw_as_str(payload,
                                                                                        session_manager=session_manager,
                                                                                        streaming=streaming,
                                                                                        result_type=result_type,
                                                                                        output_type=output_type))

            return post_stream

        if (endpoint.path):
            if (endpoint.method == "GET"):

                app.add_api_route(
                    path=endpoint.path,
                    endpoint=get_single_endpoint(result_type=GenerateSingleResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateSingleResponseType,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.path}/stream",
                    endpoint=get_streaming_endpoint(streaming=True,
                                                    result_type=GenerateStreamResponseType,
                                                    output_type=GenerateStreamResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateStreamResponseType,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.path}/stream/full",
                    endpoint=get_streaming_raw_endpoint(streaming=True,
                                                        result_type=GenerateStreamResponseType,
                                                        output_type=GenerateStreamResponseType),
                    methods=[endpoint.method],
                )

            elif (endpoint.method == "POST"):

                app.add_api_route(
                    path=endpoint.path,
                    endpoint=post_single_endpoint(request_type=GenerateBodyType,
                                                  result_type=GenerateSingleResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateSingleResponseType,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.path}/stream",
                    endpoint=post_streaming_endpoint(request_type=GenerateBodyType,
                                                     streaming=True,
                                                     result_type=GenerateStreamResponseType,
                                                     output_type=GenerateStreamResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateStreamResponseType,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.path}/stream/full",
                    endpoint=post_streaming_raw_endpoint(request_type=GenerateBodyType,
                                                         streaming=True,
                                                         result_type=GenerateStreamResponseType,
                                                         output_type=GenerateStreamResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateStreamResponseType,
                    description="Stream raw intermediate steps without any step adaptor translations",
                    responses={500: response_500},
                )

            else:
                raise ValueError(f"Unsupported method {endpoint.method}")

        if (endpoint.openai_api_path):
            if (endpoint.method == "GET"):

                app.add_api_route(
                    path=endpoint.openai_api_path,
                    endpoint=get_single_endpoint(result_type=AIQChatResponse),
                    methods=[endpoint.method],
                    response_model=AIQChatResponse,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.openai_api_path}/stream",
                    endpoint=get_streaming_endpoint(streaming=True,
                                                    result_type=AIQChatResponseChunk,
                                                    output_type=AIQChatResponseChunk),
                    methods=[endpoint.method],
                    response_model=AIQChatResponseChunk,
                    description=endpoint.description,
                    responses={500: response_500},
                )

            elif (endpoint.method == "POST"):

                app.add_api_route(
                    path=endpoint.openai_api_path,
                    endpoint=post_single_endpoint(request_type=AIQChatRequest, result_type=AIQChatResponse),
                    methods=[endpoint.method],
                    response_model=AIQChatResponse,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.openai_api_path}/stream",
                    endpoint=post_streaming_endpoint(request_type=AIQChatRequest,
                                                     streaming=True,
                                                     result_type=AIQChatResponseChunk,
                                                     output_type=AIQChatResponseChunk),
                    methods=[endpoint.method],
                    response_model=AIQChatResponseChunk | AIQResponseIntermediateStep,
                    description=endpoint.description,
                    responses={500: response_500},
                )

            else:
                raise ValueError(f"Unsupported method {endpoint.method}")
