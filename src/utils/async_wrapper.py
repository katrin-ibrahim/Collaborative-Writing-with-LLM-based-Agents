import asyncio
import time

import logging
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Union

logger = logging.getLogger(__name__)


# --- Protocols / types -------------------------------------------------------


class SyncTool(Protocol):
    def __call__(self, *args, **kwargs) -> Any: ...


class AsyncTool(Protocol):
    async def __call__(self, *args, **kwargs) -> Any: ...


ToolCallable = Union[SyncTool, AsyncTool]


class HasInvoke(Protocol):
    # e.g., LangChain-style: tool.invoke(payload)
    def invoke(self, *args, **kwargs) -> Any: ...


class HasAInvoke(Protocol):
    # some toolkits expose async invoke
    async def ainvoke(self, *args, **kwargs) -> Any: ...


# --- Retry helper ------------------------------------------------------------


def _exponential_backoff_delays(
    attempts: int,
    base: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.2,
) -> Iterable[float]:
    # yields: base * 2^(n-1) with jitter, capped at max_delay
    for i in range(attempts):
        delay = min(base * (2**i), max_delay)
        if jitter:
            delay = delay * (
                1 - jitter + 2 * jitter * (time.time() % 1.0)
            )  # cheap pseudo-jitter
        yield delay


# --- Async wrapper -----------------------------------------------------------


class AsyncWrapper:
    """
    A small async helper for calling:
      - sync functions (run in thread)
      - async functions
      - tool.invoke(...) or tool.ainvoke(...)
    Features:
      - concurrency limit (Semaphore)
      - retries with backoff
      - per-call timeout
    """

    def __init__(
        self,
        max_concurrency: int = 4,
        default_timeout_s: Optional[float] = 60.0,
        default_retries: int = 2,
    ) -> None:
        self.sem = asyncio.Semaphore(max_concurrency)
        self.default_timeout_s = default_timeout_s
        self.default_retries = default_retries

    async def call_api_async(
        self,
        fn: Callable[..., Any],
        *args,
        timeout_s: Optional[float] = None,
        retries: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Generic async caller for **sync** LLM/tool clients (runs in thread).
        Example: await aw.call_api_async(self.api_client.call_api, prompt)
        """
        timeout_s = self.default_timeout_s if timeout_s is None else timeout_s
        retries = self.default_retries if retries is None else retries

        async with self.sem:
            attempt = 0
            for delay in [0.0, *list(_exponential_backoff_delays(retries))]:
                if delay:
                    await asyncio.sleep(delay)
                attempt += 1
                try:
                    coro = asyncio.to_thread(fn, *args, **kwargs)
                    if timeout_s:
                        return await asyncio.wait_for(coro, timeout=timeout_s)
                    return await coro
                except Exception as e:
                    logger.warning(f"[call_api_async] attempt {attempt} failed: {e}")
                    last_exc = e
            raise last_exc  # type: ignore[name-defined]

    async def invoke_tool_async(
        self,
        tool: Any,
        payload: Dict[str, Any],
        timeout_s: Optional[float] = None,
        retries: Optional[int] = None,
        method: Optional[str] = None,
    ) -> Any:
        """
        Call ANY tool that looks like:
          - tool.invoke(payload)    (sync)
          - tool.ainvoke(payload)   (async)
          - tool(payload)           (sync callable)
          - await tool(payload)     (async callable)

        Args:
          tool: the tool object or callable
          payload: dict payload to pass
          timeout_s: per-call timeout (default from ctor if None)
          retries: number of retries (default from ctor if None)
          method: force one of {"invoke","ainvoke","__call__"} if you need
        """
        timeout_s = self.default_timeout_s if timeout_s is None else timeout_s
        retries = self.default_retries if retries is None else retries

        # Resolve how to call
        async def _do_call():
            # Forced method
            if method == "ainvoke" and hasattr(tool, "ainvoke"):
                return await tool.ainvoke(payload)
            if method == "invoke" and hasattr(tool, "invoke"):
                return await asyncio.to_thread(tool.invoke, payload)
            if method == "__call__":
                if asyncio.iscoroutinefunction(tool):
                    return await tool(payload)
                return await asyncio.to_thread(tool, payload)

            # Auto-detect best method
            if hasattr(tool, "ainvoke"):
                return await tool.ainvoke(payload)
            if hasattr(tool, "invoke"):
                return await asyncio.to_thread(tool.invoke, payload)
            if asyncio.iscoroutinefunction(tool):
                return await tool(payload)  # type: ignore[misc]
            # Fallback: sync callable
            return await asyncio.to_thread(tool, payload)

        async with self.sem:
            attempt = 0
            for delay in [0.0, *list(_exponential_backoff_delays(retries))]:
                if delay:
                    await asyncio.sleep(delay)
                attempt += 1
                try:
                    if timeout_s:
                        return await asyncio.wait_for(_do_call(), timeout=timeout_s)
                    return await _do_call()
                except Exception as e:
                    logger.warning(f"[invoke_tool_async] attempt {attempt} failed: {e}")
                    last_exc = e
            raise last_exc  # type: ignore[name-defined]
