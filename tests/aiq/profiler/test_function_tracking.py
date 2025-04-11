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

from pydantic import BaseModel

from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.profiler.decroators.function_tracking import track_function
from aiq.utils.reactive.subject import Subject


async def test_sync_function_no_metadata(reactive_stream: Subject):
    """Test a simple synchronous function with no metadata."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function
    def add(a, b):
        return a + b

    out = add(2, 3)
    assert out == 5

    # We expect exactly 2 events for a normal (non-generator) function: SPAN_START and SPAN_END
    assert len(published_events) == 2

    # Check SPAN_START
    start_event: IntermediateStepPayload = published_events[0].payload
    assert start_event.event_type == IntermediateStepType.SPAN_START
    assert start_event.metadata.span_inputs[0] == [2, 3]
    assert start_event.metadata.span_inputs[1] == {}

    # Check SPAN_END
    end_event: IntermediateStepPayload = published_events[1].payload
    assert end_event.event_type == IntermediateStepType.SPAN_END
    assert end_event.metadata.span_outputs == 5


async def test_sync_function_with_metadata(reactive_stream: Subject):
    """Test a synchronous function with metadata."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function(metadata={"purpose": "test_sync"})
    def multiply(x, y):
        return x * y

    result = multiply(4, 5)
    assert result == 20

    assert len(published_events) == 2
    start_event: IntermediateStepPayload = published_events[0].payload
    end_event: IntermediateStepPayload = published_events[1].payload

    assert start_event.event_type == IntermediateStepType.SPAN_START
    assert end_event.event_type == IntermediateStepType.SPAN_END

    assert end_event.metadata.span_outputs == 20
    assert start_event.metadata.provided_metadata == {"purpose": "test_sync"}


async def test_sync_generator(reactive_stream: Subject):
    """Test a synchronous generator with three yields."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function
    def number_generator(n):
        for i in range(n):  # pylint: disable=use-yield-from
            yield i

    nums = list(number_generator(3))
    assert nums == [0, 1, 2]

    # For a generator: SPAN_START, SPAN_CHUNK (for each yield), SPAN_END
    # We yield 3 items => 1 start, 3 chunk, 1 end => total 5 events
    assert len(published_events) == 5

    assert published_events[0].payload.event_type == IntermediateStepType.SPAN_START
    for i in range(1, 4):
        assert published_events[i].payload.event_type == IntermediateStepType.SPAN_CHUNK
        assert published_events[i].payload.metadata.span_outputs == i - 1  # i-th event has output i-1
    assert published_events[4].payload.event_type == IntermediateStepType.SPAN_END


async def test_class_method(reactive_stream: Subject):
    """Test decorating a class method."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    class Calculator:

        @track_function(metadata={"class_method": True})
        def subtract(self, x, y):
            return x - y

    calc = Calculator()
    result = calc.subtract(10, 4)
    assert result == 6

    assert len(published_events) == 2
    start_event: IntermediateStepPayload = published_events[0].payload
    end_event: IntermediateStepPayload = published_events[1].payload

    assert start_event.event_type == IntermediateStepType.SPAN_START
    assert start_event.metadata.span_inputs[0][1:] == [10, 4]
    assert end_event.metadata.span_outputs == 6


async def test_async_function(reactive_stream: Subject):
    """Test an async function decorated with track_function."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function
    async def async_add(a, b):
        await asyncio.sleep(0.1)
        return a + b

    result = await async_add(7, 3)
    assert result == 10

    # For an async, non-generator function => SPAN_START and SPAN_END
    assert len(published_events) == 2
    assert published_events[0].payload.event_type == IntermediateStepType.SPAN_START
    assert published_events[0].payload.metadata.span_inputs[0] == [7, 3]
    assert published_events[1].payload.event_type == IntermediateStepType.SPAN_END
    assert published_events[1].payload.metadata.span_outputs == 10


async def test_async_generator(reactive_stream: Subject):
    """Test an async generator function with multiple yields."""
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function(metadata={"test": "async_gen"})
    async def countdown(n):
        while n > 0:
            yield n
            n -= 1

    collected = []
    async for val in countdown(3):
        collected.append(val)

    assert collected == [3, 2, 1]

    # For an async generator with 3 yields => 1 SPAN_START, 3 SPAN_CHUNK, 1 SPAN_END => total 5
    assert len(published_events) == 5
    assert published_events[0].payload.event_type == IntermediateStepType.SPAN_START
    assert published_events[0].payload.metadata.span_inputs[0] == [3]
    for i in range(1, 4):
        assert published_events[i].payload.event_type == IntermediateStepType.SPAN_CHUNK
        # The output is 3, 2, 1 respectively
        assert published_events[i].payload.metadata.span_outputs == 4 - i
    assert published_events[4].payload.event_type == IntermediateStepType.SPAN_END


class MyModel(BaseModel):
    """Simple Pydantic model for testing serialization."""
    name: str
    value: int


async def test_sync_function_pydantic(reactive_stream: Subject):
    """
    Test that a synchronous function with a Pydantic model input
    properly serializes the model via model_dump().
    """
    published_events = []
    reactive_stream.subscribe(published_events.append)

    @track_function
    def process_model(m: MyModel):
        return f"Model is {m.name} with value {m.value}"

    my_obj = MyModel(name="test", value=42)
    output = process_model(my_obj)

    assert output == "Model is test with value 42"
    assert len(published_events) == 2

    start_event: IntermediateStepPayload = published_events[0].payload
    end_event: IntermediateStepPayload = published_events[1].payload

    # Check SPAN_START has the model fully serialized
    assert start_event.event_type == IntermediateStepType.SPAN_START
    # Should see something like [{"name": "test", "value": 42}] for the args
    assert start_event.metadata.span_inputs[0] == [{"name": "test", "value": 42}]
    assert start_event.metadata.span_inputs[1] == {}

    # Check SPAN_END output
    assert end_event.event_type == IntermediateStepType.SPAN_END
    assert end_event.metadata.span_outputs == "Model is test with value 42"
