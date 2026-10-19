"""On-disk cache for raw data pulls, plus polite-fetch helpers.

Everything we pull from a network source lands in ``data/raw/`` before it is
parsed, so re-runs and backtests never re-hit an API. The cache is deliberately
dumb: one file per request, named by the caller, no expiry. Pass
``refresh=True`` (``--refresh`` on the CLI) to re-fetch and overwrite.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

#: Where raw pulls are cached, relative to the working directory.
DEFAULT_CACHE_DIR = Path("data/raw")

#: Sent on NHL API requests so the operator can see who is calling.
USER_AGENT = "massey-ratings/0.1 (personal project; python-httpx)"

T = TypeVar("T")

logger = logging.getLogger(__name__)


class Cache:
    """A read-through cache of raw payloads on local disk.

    Args:
        root: Directory to store cached files under.
        refresh: When ``True``, ignore any cached copy and re-fetch, then
            overwrite the cache.
    """

    def __init__(self, root: Path | str = DEFAULT_CACHE_DIR, *, refresh: bool = False) -> None:
        self.root = Path(root)
        self.refresh = refresh

    def path(self, *parts: str) -> Path:
        """Return the full path of a cache entry without touching the disk."""
        return self.root.joinpath(*parts)

    def has(self, *parts: str) -> bool:
        """Return whether a cache entry exists (ignoring ``refresh``)."""
        return self.path(*parts).is_file()

    def json(self, parts: tuple[str, ...], fetch: Callable[[], Any]) -> Any:
        """Return cached JSON, calling ``fetch`` and caching the result on a miss.

        Args:
            parts: Path components of the cache entry, e.g. ``("nhl", "x.json")``.
            fetch: Zero-argument callable returning JSON-serialisable data. Only
                called on a cache miss (or when ``refresh`` is set).

        Returns:
            The cached or freshly fetched payload.
        """
        target = self.path(*parts)
        if target.is_file() and not self.refresh:
            logger.debug("cache hit: %s", target)
            return json.loads(target.read_text())

        logger.debug("cache miss: %s", target)
        payload = fetch()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
        return payload

    def parquet(self, parts: tuple[str, ...], fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        """Return a cached DataFrame, calling ``fetch`` and caching on a miss.

        Args:
            parts: Path components of the cache entry.
            fetch: Zero-argument callable returning the DataFrame to cache.

        Returns:
            The cached or freshly fetched DataFrame.
        """
        target = self.path(*parts)
        if target.is_file() and not self.refresh:
            logger.debug("cache hit: %s", target)
            return pd.read_parquet(target)

        logger.debug("cache miss: %s", target)
        frame = fetch()
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        return frame


def with_retry(
    fetch: Callable[[], T],
    *,
    attempts: int = 4,
    backoff: float = 1.0,
    description: str = "request",
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call ``fetch``, retrying transient failures with exponential backoff.

    Waits ``backoff``, ``2 * backoff``, ``4 * backoff``, ... between attempts.

    Args:
        fetch: Zero-argument callable to invoke.
        attempts: Maximum number of attempts, including the first.
        backoff: Seconds to wait before the first retry.
        description: Used in log messages and the final error message.
        retry_on: Exception types treated as transient.

    Returns:
        Whatever ``fetch`` returns.

    Raises:
        Exception: Re-raises the last failure once attempts are exhausted. Any
            exception not listed in ``retry_on`` propagates immediately.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fetch()
        except retry_on as error:
            last_error = error
            if attempt == attempts:
                break
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                description,
                attempt,
                attempts,
                error,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"{description} failed after {attempts} attempts") from last_error
