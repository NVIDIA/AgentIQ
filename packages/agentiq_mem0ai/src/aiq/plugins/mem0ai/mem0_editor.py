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

import asyncio

from mem0 import AsyncMemoryClient

from aiq.memory.interfaces import MemoryEditor
from aiq.memory.models import MemoryItem


class Mem0Editor(MemoryEditor):
    """
    Wrapper class that implements AgentIQ Interfaces for Mem0 Integrations Async.
    """

    def __init__(self, mem0_client: AsyncMemoryClient):
        """
        Initialize class with Predefined Mem0 Client.

        Args:
        mem0_client (AsyncMemoryClient): Preinstantiated
        AsyncMemoryClient object for Mem0.
        """
        self._client = mem0_client

    async def add_items(self, items: list[MemoryItem]) -> None:
        """
        Insert Multiple MemoryItems into the memory.
        Each MemoryItem is translated and uploaded.
        """

        coroutines = []

        # Iteratively insert memories into Mem0
        for memory_item in items:
            item_meta = memory_item.metadata
            content = memory_item.conversation

            user_id = memory_item.user_id  # This must be specified
            run_id = item_meta.pop("run_id", None)
            tags = memory_item.tags

            coroutines.append(
                self._client.add(content,
                                 user_id=user_id,
                                 run_id=run_id,
                                 tags=tags,
                                 metadata=item_meta,
                                 output_format="v1.1"))

        await asyncio.gather(*coroutines)

    async def search(self, query: str, top_k: int = 5, **kwargs) \
            -> list[MemoryItem]:
        """
        Retrieve items relevant to the given query.

        Args:
            query (str): The query string to match.
            top_k (int): Maximum number of items to return.
            **kwargs: Other keyword arguments for search.

        Returns:
            list[MemoryItem]: The most relevant
            MemoryItems for the given query.
        """

        user_id = kwargs.pop("user_id")  # Ensure user ID is in keyword arguments

        search_result = await self._client.search(query, user_id=user_id, top_k=top_k, output_format="v1.1", **kwargs)

        # Construct MemoryItem instances
        memories = []

        for res in search_result["results"]:
            item_meta = res.pop("metadata", {})

            memories.append(
                MemoryItem(conversation=res.pop("input", []),
                           user_id=user_id,
                           memory=res["memory"],
                           tags=res.pop("categories", []) or [],
                           metadata=item_meta))

        return memories

    async def remove_items(self, **kwargs):

        if "memory_id" in kwargs:

            memory_id = kwargs.pop("memory_id")
            await self._client.delete(memory_id)

        elif "user_id" in kwargs:

            user_id = kwargs.pop("user_id")
            await self._client.delete_all(user_id=user_id)

        return
