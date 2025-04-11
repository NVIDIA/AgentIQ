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
import contextvars
import typing
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextlib import nullcontext

from aiq.builder.context import AIQContext
from aiq.builder.context import AIQContextState
from aiq.builder.workflow import Workflow
from aiq.data_models.config import AIQConfig
from aiq.data_models.interactive import HumanResponse
from aiq.data_models.interactive import InteractionPrompt

_T = typing.TypeVar("_T")


class UserManagerBase:
    pass


class AIQSessionManager:

    def __init__(self, workflow: Workflow, max_concurrency: int = 8):
        """
        The AIQSessionManager class is used to run and manage a user workflow session. It runs and manages the context,
        and configuration of a workflow with the specified concurrency.

        Parameters
        ----------
        workflow : Workflow
            The workflow to run
        max_concurrency : int, optional
            The maximum number of simultaneous workflow invocations, by default 8
        """

        if (workflow is None):
            raise ValueError("Workflow cannot be None")

        self._workflow: Workflow = workflow

        self._max_concurrency = max_concurrency
        self._context_state = AIQContextState.get()
        self._context = AIQContext(self._context_state)

        # We save the context because Uvicorn spawns a new process
        # for each request, and we need to restore the context vars
        self._saved_context = contextvars.copy_context()

        if (max_concurrency > 0):
            self._semaphore = asyncio.Semaphore(max_concurrency)
        else:
            # If max_concurrency is 0, then we don't need to limit the concurrency but we still need a context
            self._semaphore = nullcontext()

    @property
    def config(self) -> AIQConfig:
        return self._workflow.config

    @property
    def workflow(self) -> Workflow:
        return self._workflow

    @property
    def context(self) -> AIQContext:
        return self._context

    @asynccontextmanager
    async def session(self,
                      user_manager=None,
                      user_input_callback: Callable[[InteractionPrompt], Awaitable[HumanResponse]] = None):

        token_user_input = None
        if user_input_callback is not None:
            token_user_input = self._context_state.user_input_callback.set(user_input_callback)

        token_user_manager = None
        if user_manager is not None:
            token_user_manager = self._context_state.user_manager.set(user_manager)

        try:
            yield self
        finally:
            if token_user_manager is not None:
                self._context_state.user_manager.reset(token_user_manager)
            if token_user_input is not None:
                self._context_state.user_input_callback.reset(token_user_input)

    @asynccontextmanager
    async def run(self, message, entry_fn: str | None = None):
        """
        Start a workflow run
        """
        async with self._semaphore:
            # Apply the saved context
            for k, v in self._saved_context.items():
                k.set(v)

            async with self._workflow.run(message, entry_fn) as runner:
                yield runner
