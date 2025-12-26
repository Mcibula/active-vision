"""
Structures and utilities supporting
multithreaded vision pipeline construction
"""

import asyncio
import threading
import time
from collections import deque
from typing import Generic, Iterable, TypeVar

T = TypeVar('T')


class StageBuffer(Generic[T]):
    def __len__(self) -> int:
        return 0

    def write(self, item: T | list[T]) -> None:
        raise NotImplementedError

    def write_batch(self, batch: Iterable[T]) -> None:
        for item in batch:
            self.write(item)

    def read(self) -> T | None:
        raise NotImplementedError


class FrameBufferMetrics:
    def __init__(self, name: str) -> None:
        self.name = name
        self.lock = threading.Lock()

        self.writes: int = 0
        self.reads: int = 0
        self.size: int = 0
        self.lat_sum: float = 0.0
        self.lat_n: int = 0

    def on_write(self, n: int = 1) -> None:
        with self.lock:
            self.writes += n

    def on_read(self, n: int = 1) -> None:
        with self.lock:
            self.reads += n

    def on_latency(self, seconds: float) -> None:
        with self.lock:
            self.lat_sum += seconds
            self.lat_n += 1

    def on_size(self, size: int) -> None:
        with self.lock:
            self.size = size

    def report(self) -> dict[str, int | float]:
        with self.lock:
            return {
                'writes': self.writes,
                'reads': self.reads,
                'size': self.size,
                'lat_ms': (
                    (self.lat_sum / self.lat_n * 1000)
                    if self.lat_n else 0.0
                ),
            }


class InstrumentedBuffer(StageBuffer[T]):
    def __init__(
            self,
            inner: StageBuffer[T],
            metrics: FrameBufferMetrics,
            timestamp: bool = True
    ) -> None:
        self.inner = inner
        self.metrics = metrics
        self.timestamp = timestamp

    def __len__(self) -> int:
        return len(self.inner)

    def write(self, item: T) -> None:
        self.inner.write(
            (item, time.perf_counter())
            if self.timestamp else item
        )
        self.metrics.on_write(1)

    def read(self) -> T | None:
        item = self.inner.read()

        if item is None:
            return None

        self.metrics.on_read(1)

        if self.timestamp:
            value, ts = item
            self.metrics.on_latency(time.perf_counter() - ts)

            return value

        return item


class LosslessFIFO(StageBuffer[T]):
    def __init__(self) -> None:
        self.queue = asyncio.Queue()

    def __len__(self) -> int:
        return self.queue.qsize()

    def write(self, item: T) -> None:
        self.queue.put_nowait(item)

    async def read(self) -> T:
        return await self.queue.get()

    async def read_batch(self, batch_size: int, timeout: float | None = None) -> list[T]:
        batch = [await self.read()]

        if batch_size == 1:
            return batch

        t0 = asyncio.get_event_loop().time()

        while len(batch) < batch_size:
            if timeout is not None:
                dt = timeout - (asyncio.get_event_loop().time() - t0)
                if dt <= 0:
                    break
            else:
                dt = None

            try:
                item = await asyncio.wait_for(self.read(), timeout=dt)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch


class BoundedFIFO(StageBuffer[T]):
    def __init__(self, capacity: int) -> None:
        self.buf = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def __len__(self) -> int:
        with self.lock:
            return len(self.buf)

    def write(self, item: T) -> None:
        with self.lock:
            if len(self.buf) < self.buf.maxlen:
                self.buf.append(item)

    def write_batch(self, batch: Iterable[T]) -> None:
        with self.lock:
            for item in batch:
                self.buf.append(item)

    def read(self) -> T | None:
        with self.lock:
            return (
                self.buf.popleft()
                if self.buf else None
            )
