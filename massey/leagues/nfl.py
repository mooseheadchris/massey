"""NFL data loading and margin logic.

Results come from ``nflreadpy.load_schedules()``, which returns a polars
DataFrame of one row per scheduled game; we convert it to pandas and cache the
season's frame to ``data/raw/nfl/`` so re-runs and backtests stay offline.

Column names below were read off a live 2025/2026 pull rather than assumed:

* ``game_id``, ``season``, ``game_type`` (``REG``/``WC``/``DIV``/``CON``/``SB``)
* ``week``, ``gameday`` (``"YYYY-MM-DD"`` strings)
* ``home_team``/``away_team``, ``home_score``/``away_score``
* ``location`` -- ``"Home"`` or ``"Neutral"``
* ``result`` -- home score minus away score (verified identical to computing it
  by hand on every completed 2025 and 2026 game)
* ``overtime`` -- 1 when the game went to overtime
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from massey.data.cache import DEFAULT_CACHE_DIR, Cache, with_retry
from massey.leagues.base import GameRecord, filter_by_cutoff

logger = logging.getLogger(__name__)

name = "nfl"
margin_units = "points"
default_skip_weeks: int | None = 3
default_skip_games: int | None = None

#: Starting logistic scale for turning a predicted point margin into a home
#: win probability, used only until the backtest has enough out-of-sample
#: predictions to fit its own scale.
default_logistic_scale = 0.13

#: Knobs for the NFL margin definition. Edit these rather than passing
#: arguments around.
NFL_CONFIG: dict[str, Any] = {
    # Cap on the absolute point margin, to test diminishing returns on
    # blowouts. ``None`` means no cap -- use the real point differential.
    "MAX_MARGIN": None,
}

#: ``game_type`` values that make up the regular season.
REGULAR_SEASON_TYPES = ("REG",)

#: ``game_type`` values for the postseason, in order.
PLAYOFF_TYPES = ("WC", "DIV", "CON", "SB")

#: ``location`` value marking a game played at neither team's home venue.
NEUTRAL_LOCATION = "Neutral"


def current_season() -> int:
    """Return the NFL season that is current today.

    A season is named for the calendar year it starts in, so the 2025 season
    runs from September 2025 into February 2026.

    Returns:
        The current season year.
    """
    try:
        import nflreadpy

        return int(nflreadpy.get_current_season())
    except Exception:  # pragma: no cover - only on an nflreadpy change
        logger.debug("nflreadpy.get_current_season() unavailable; using date math")
        today = date.today()
        return today.year if today.month >= 3 else today.year - 1


def load_schedule_frame(
    season: int | str,
    *,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load one season's raw schedule as a pandas DataFrame, via the cache.

    Args:
        season: Season year, e.g. ``2026``. A string is accepted, since that is
            what the command line supplies.
        cache_dir: Root of the raw-data cache. Defaults to ``data/raw``.
        refresh: Re-fetch even if a cached copy exists.

    Returns:
        The season's schedule, one row per scheduled game, including games that
        have not been played yet.

    Raises:
        ValueError: If ``season`` is not a year.
    """
    # nflreadpy filters its season column, which is an integer, so a string
    # year reaches polars as a type mismatch rather than a lookup. Coerce here
    # so every caller -- the CLI included -- can pass either.
    try:
        year = int(season)
    except (TypeError, ValueError):
        raise ValueError(f"cannot read {season!r} as an NFL season year") from None

    cache = Cache(cache_dir or DEFAULT_CACHE_DIR, refresh=refresh)

    def fetch() -> pd.DataFrame:
        import nflreadpy

        logger.info("fetching NFL schedules for %s", year)
        return with_retry(
            lambda: nflreadpy.load_schedules(seasons=[year]).to_pandas(),
            description=f"nflreadpy.load_schedules({year})",
        )

    return cache.parquet(("nfl", f"schedules_{year}.parquet"), fetch)


def apply_margin_cap(margin: float, max_margin: float | None) -> float:
    """Clamp a margin symmetrically to ``+/- max_margin``.

    Args:
        margin: The raw home-relative margin.
        max_margin: The cap, or ``None`` for no cap.

    Returns:
        The capped margin.
    """
    if max_margin is None:
        return margin
    if max_margin <= 0:
        raise ValueError(f"max_margin must be positive, got {max_margin}")
    return max(-max_margin, min(max_margin, margin))


def load_games(
    season: int | str,
    *,
    as_of: date | None = None,
    week: int | None = None,
    include_playoffs: bool = False,
    max_margin: float | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> list[GameRecord]:
    """Load completed NFL games for a season as rating inputs.

    Args:
        season: Season year, e.g. ``2026``. Accepts a string year too.
        as_of: Keep only games played strictly before this date.
        week: Keep only games from weeks strictly before this one, i.e. the
            ratings a forecaster would carry *into* that week.
        include_playoffs: Include postseason games as well as the regular season.
        max_margin: Cap on the absolute point margin. ``None`` (the default)
            falls back to ``NFL_CONFIG["MAX_MARGIN"]``.
        cache_dir: Root of the raw-data cache. Defaults to ``data/raw``.
        refresh: Re-fetch even if a cached copy exists.

    Returns:
        Completed games, oldest first.
    """
    frame = load_schedule_frame(season, cache_dir=cache_dir, refresh=refresh)

    wanted_types = list(REGULAR_SEASON_TYPES)
    if include_playoffs:
        wanted_types += list(PLAYOFF_TYPES)

    played = frame[
        frame["game_type"].isin(wanted_types)
        & frame["home_score"].notna()
        & frame["away_score"].notna()
    ].copy()

    cap = NFL_CONFIG["MAX_MARGIN"] if max_margin is None else max_margin
    played["parsed_date"] = pd.to_datetime(played["gameday"]).dt.date

    records = [
        GameRecord(
            game_id=str(row.game_id),
            date=row.parsed_date,
            home=str(row.home_team),
            away=str(row.away_team),
            margin=apply_margin_cap(float(row.home_score) - float(row.away_score), cap),
            neutral=str(row.location) == NEUTRAL_LOCATION,
            home_score=int(row.home_score),
            away_score=int(row.away_score),
            week=int(row.week),
            outcome="OT" if _is_truthy(row.overtime) else "REG",
        )
        for row in played.itertuples(index=False)
    ]

    records.sort(key=lambda record: (record.date, record.game_id))
    return filter_by_cutoff(records, as_of=as_of, week=week)


def _is_truthy(value: Any) -> bool:
    """Interpret nflreadpy's 0/1/NaN flag columns as a bool."""
    return bool(value) and not pd.isna(value)
