"""League adapters and the registry the CLI and backtest look them up in."""

from __future__ import annotations

from typing import Any

from massey.leagues import nfl, nhl
from massey.leagues.base import GameRecord, LeagueAdapter, filter_by_cutoff

#: League key -> adapter module. Add a league by writing a module that matches
#: :class:`~massey.leagues.base.LeagueAdapter` and registering it here.
LEAGUES: dict[str, Any] = {
    nfl.name: nfl,
    nhl.name: nhl,
}


def get_league(key: str) -> Any:
    """Look up a league adapter by key.

    Args:
        key: League key, e.g. ``"nfl"``. Case-insensitive.

    Returns:
        The adapter module.

    Raises:
        KeyError: If no league is registered under that key.
    """
    try:
        return LEAGUES[key.lower()]
    except KeyError:
        raise KeyError(f"unknown league {key!r}; expected one of {sorted(LEAGUES)}") from None


__all__ = ["LEAGUES", "GameRecord", "LeagueAdapter", "filter_by_cutoff", "get_league", "nfl", "nhl"]
