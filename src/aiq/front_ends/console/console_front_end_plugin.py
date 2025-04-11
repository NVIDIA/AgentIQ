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
from io import StringIO

import click

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.interactive import HumanPromptModelType
from aiq.data_models.interactive import HumanResponse
from aiq.data_models.interactive import HumanResponseText
from aiq.data_models.interactive import InteractionPrompt
from aiq.front_ends.console.console_front_end_config import ConsoleFrontEndConfig
from aiq.front_ends.simple_base.simple_front_end_plugin_base import SimpleFrontEndPluginBase
from aiq.runtime.session import AIQSessionManager

logger = logging.getLogger(__name__)


async def prompt_for_input_cli(question: InteractionPrompt) -> HumanResponse:
    """
    A simple CLI-based callback.
    Takes question as str, returns the typed line as str.
    """

    if question.content.input_type == HumanPromptModelType.TEXT:
        user_response = click.prompt(text=question.content.text)

        return HumanResponseText(text=user_response)

    raise ValueError("Unsupported human propmt input type. The run command only supports the 'HumanPromptText' "
                     "input type. Please use the 'serve' command to ensure full support for all input types.")


class ConsoleFrontEndPlugin(SimpleFrontEndPluginBase[ConsoleFrontEndConfig]):

    async def pre_run(self):

        if (not self.front_end_config.input_query and not self.front_end_config.input_file):
            raise click.UsageError("Must specify either --input_query or --input_file")

    async def run(self):

        # Must yield the workflow function otherwise it cleans up
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:

            session_manager: AIQSessionManager = None

            if logger.isEnabledFor(logging.INFO):
                stream = StringIO()

                self.full_config.print_summary(stream=stream)

                click.echo(stream.getvalue())

                workflow = builder.build()
                session_manager = AIQSessionManager(workflow)

            await self.run_workflow(session_manager)

    async def run_workflow(self, session_manager: AIQSessionManager = None):

        runner_outputs = None

        if (self.front_end_config.input_query):

            async def run_single_query(query):

                async with session_manager.session(user_input_callback=prompt_for_input_cli) as session:
                    async with session.run(query, self.front_end_config.entry_fn) as runner:
                        base_output = await runner.result(to_type=str)

                        return base_output

            # Convert to a list
            input_list = list(self.front_end_config.input_query)
            logger.info("Processing input: %s", self.front_end_config.input_query)

            runner_outputs = await asyncio.gather(*[run_single_query(query) for query in input_list])

        elif (self.front_end_config.input_file):

            # Run the workflow
            with open(self.front_end_config.input_file, "r", encoding="utf-8") as f:

                async with session_manager.workflow.run(f, self.front_end_config.entry_fn) as runner:
                    runner_outputs = await runner.result(to_type=str)
        else:
            assert False, "Should not reach here. Should have been caught by pre_run"

        # Print result
        logger.info(f"{'-' * 50}\nWorkflow Result:\n%s\n{'-' * 50}", runner_outputs)
