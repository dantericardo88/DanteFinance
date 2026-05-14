"""Internal event bus — asyncio queues for intra-process, Redis Streams for persistence."""
from __future__ import annotations
import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Awaitable
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

Handler = Callable[[Any], Awaitable[None]]


class EventBus:
    """
    Dual-mode event bus:
    - asyncio queues for low-latency intra-process fan-out
    - Redis Streams for cross-process durability (persist=True)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    async def publish(
        self,
        event_type: str,
        payload: Any,
        persist: bool = False,
    ) -> None:
        handlers = self._subscribers.get(event_type, [])
        tasks = [asyncio.create_task(h(payload)) for h in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if persist and self._redis:
            try:
                await self._redis.xadd(
                    f"sentinel:{event_type}",
                    {
                        "payload": json.dumps(payload, default=str),
                        "ts": datetime.utcnow().isoformat(),
                    },
                    maxlen=10_000,
                    approximate=True,
                )
            except Exception as exc:
                logger.warning("Redis stream publish failed: %s", exc)

    async def consume_stream(
        self,
        event_type: str,
        handler: Handler,
        group: str = "sentinel",
        consumer: str = "worker-1",
        block_ms: int = 5000,
    ) -> None:
        """Consume from Redis Stream for cross-process durable delivery."""
        if not self._redis:
            raise RuntimeError("EventBus not connected — call await bus.connect() first")
        stream = f"sentinel:{event_type}"
        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
        except aioredis.ResponseError:
            pass  # group already exists

        while True:
            try:
                messages = await self._redis.xreadgroup(
                    group, consumer, {stream: ">"}, count=10, block=block_ms
                )
                for _, entries in messages or []:
                    for msg_id, fields in entries:
                        try:
                            payload = json.loads(fields["payload"])
                            await handler(payload)
                            await self._redis.xack(stream, group, msg_id)
                        except Exception as exc:
                            logger.error("Stream handler error msg=%s: %s", msg_id, exc)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Stream consume error: %s", exc)
                await asyncio.sleep(1)


_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        from sentinel.core.config import get_settings
        _bus = EventBus(redis_url=get_settings().redis_url)
    return _bus
