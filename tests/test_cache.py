"""Tests for the disk cache and the polite-fetch retry helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from massey.data.cache import Cache, with_retry


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def test_json_fetches_once_then_serves_from_disk(tmp_path: Path):
    cache = Cache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return {"games": [1, 2, 3]}

    first = cache.json(("nhl", "x.json"), fetch)
    second = cache.json(("nhl", "x.json"), fetch)

    assert first == second == {"games": [1, 2, 3]}
    assert len(calls) == 1, "second call should have been served from disk"
    assert (tmp_path / "nhl" / "x.json").is_file()


def test_refresh_bypasses_the_cache(tmp_path: Path):
    calls = []

    def fetch():
        calls.append(1)
        return {"n": len(calls)}

    assert Cache(tmp_path).json(("a.json",), fetch) == {"n": 1}
    assert Cache(tmp_path, refresh=True).json(("a.json",), fetch) == {"n": 2}
    # The refreshed value was written back.
    assert Cache(tmp_path).json(("a.json",), fetch) == {"n": 2}
    assert len(calls) == 2


def test_parquet_round_trips_a_frame(tmp_path: Path):
    cache = Cache(tmp_path)
    frame = pd.DataFrame({"team": ["AAA", "BBB"], "score": [24, 17]})
    calls = []

    def fetch():
        calls.append(1)
        return frame

    cache.parquet(("nfl", "s.parquet"), fetch)
    loaded = cache.parquet(("nfl", "s.parquet"), fetch)

    pd.testing.assert_frame_equal(loaded, frame)
    assert len(calls) == 1


def test_has_reports_presence_without_fetching(tmp_path: Path):
    cache = Cache(tmp_path)
    assert not cache.has("nhl", "x.json")
    cache.json(("nhl", "x.json"), lambda: {"ok": True})
    assert cache.has("nhl", "x.json")


def test_has_ignores_refresh(tmp_path: Path):
    # `has` answers "is it on disk", which `load_raw_games` relies on to decide
    # whether it needs a live client. `refresh` must not change that answer.
    Cache(tmp_path).json(("x.json",), lambda: {})
    assert Cache(tmp_path, refresh=True).has("x.json")


def test_path_does_not_create_anything(tmp_path: Path):
    cache = Cache(tmp_path)
    path = cache.path("nhl", "nope.json")
    assert path == tmp_path / "nhl" / "nope.json"
    assert not path.exists()


def test_nested_directories_are_created(tmp_path: Path):
    cache = Cache(tmp_path / "deep" / "deeper")
    cache.json(("a", "b", "c.json"), lambda: {"ok": True})
    assert (tmp_path / "deep" / "deeper" / "a" / "b" / "c.json").is_file()


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #

def test_with_retry_returns_the_first_success():
    assert with_retry(lambda: "ok", backoff=0) == "ok"


def test_with_retry_recovers_from_a_transient_failure(monkeypatch):
    monkeypatch.setattr("massey.data.cache.time.sleep", lambda _: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("reset")
        return "ok"

    assert with_retry(flaky, backoff=0) == "ok"
    assert len(attempts) == 3


def test_with_retry_gives_up_and_chains_the_cause(monkeypatch):
    monkeypatch.setattr("massey.data.cache.time.sleep", lambda _: None)

    def always_fails():
        raise ConnectionError("still down")

    with pytest.raises(RuntimeError, match="failed after 4 attempts") as caught:
        with_retry(always_fails, backoff=0, description="nhl fetch")

    assert isinstance(caught.value.__cause__, ConnectionError)


def test_with_retry_backs_off_exponentially(monkeypatch):
    waits = []
    monkeypatch.setattr("massey.data.cache.time.sleep", waits.append)

    with pytest.raises(RuntimeError):
        with_retry(lambda: 1 / 0, attempts=4, backoff=2.0)

    assert waits == [2.0, 4.0, 8.0]


def test_with_retry_does_not_swallow_unexpected_errors():
    # Only the listed exception types are treated as transient.
    with pytest.raises(KeyError):
        with_retry(
            lambda: (_ for _ in ()).throw(KeyError("bug")),
            retry_on=(ConnectionError,),
            backoff=0,
        )


def test_with_retry_rejects_zero_attempts():
    with pytest.raises(ValueError, match="at least 1"):
        with_retry(lambda: "ok", attempts=0)
