"""Tests for the NFL loader and margin handling.

These run against a seeded cache (see ``tests/conftest.py``) using the real
column names ``nflreadpy`` returns, so no network access is needed.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from massey.leagues import nfl


# --------------------------------------------------------------------------- #
# Margin cap
# --------------------------------------------------------------------------- #

def test_no_cap_by_default():
    assert nfl.NFL_CONFIG["MAX_MARGIN"] is None
    assert nfl.apply_margin_cap(42.0, None) == 42.0


@pytest.mark.parametrize(
    ("margin", "cap", "expected"),
    [
        (37.0, 20.0, 20.0),
        (-37.0, 20.0, -20.0),  # symmetric
        (7.0, 20.0, 7.0),  # under the cap, untouched
        (20.0, 20.0, 20.0),  # exactly at the cap
        (0.0, 20.0, 0.0),
    ],
)
def test_apply_margin_cap(margin, cap, expected):
    assert nfl.apply_margin_cap(margin, cap) == expected


def test_apply_margin_cap_rejects_a_nonsense_cap():
    with pytest.raises(ValueError, match="must be positive"):
        nfl.apply_margin_cap(10.0, 0.0)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def test_load_games_keeps_only_completed_regular_season_games(nfl_cache: Path):
    games = nfl.load_games(2025, cache_dir=nfl_cache)

    # Five played regular-season games; the unplayed week 5 game and the
    # wild-card game are both excluded.
    assert [game.game_id for game in games] == [
        "2025_01_AAA_BBB",
        "2025_01_CCC_DDD",
        "2025_02_BBB_AAA",
        "2025_03_AAA_CCC",
        "2025_04_DDD_BBB",
    ]


def test_margin_is_the_point_differential(nfl_cache: Path):
    games = {game.game_id: game for game in nfl.load_games(2025, cache_dir=nfl_cache)}

    assert games["2025_01_AAA_BBB"].margin == 7.0  # BBB 24, AAA 17, at home
    assert games["2025_01_CCC_DDD"].margin == -10.0  # DDD 10, CCC 20, at home
    assert games["2025_04_DDD_BBB"].margin == 37.0  # uncapped by default


def test_neutral_site_is_flagged(nfl_cache: Path):
    games = {game.game_id: game for game in nfl.load_games(2025, cache_dir=nfl_cache)}
    assert games["2025_04_DDD_BBB"].neutral is True
    assert games["2025_01_AAA_BBB"].neutral is False


def test_overtime_is_recorded(nfl_cache: Path):
    games = {game.game_id: game for game in nfl.load_games(2025, cache_dir=nfl_cache)}
    assert games["2025_02_BBB_AAA"].outcome == "OT"
    assert games["2025_01_AAA_BBB"].outcome == "REG"


def test_a_tie_has_no_winner(nfl_cache: Path):
    games = {game.game_id: game for game in nfl.load_games(2025, cache_dir=nfl_cache)}
    tie = games["2025_03_AAA_CCC"]
    assert tie.margin == 0.0
    assert tie.home_won is None


def test_scores_and_weeks_are_kept(nfl_cache: Path):
    games = {game.game_id: game for game in nfl.load_games(2025, cache_dir=nfl_cache)}
    game = games["2025_01_AAA_BBB"]
    assert (game.home_score, game.away_score) == (24, 17)
    assert game.week == 1
    assert game.date == date(2025, 9, 7)


def test_max_margin_caps_blowouts(nfl_cache: Path):
    games = {
        game.game_id: game
        for game in nfl.load_games(2025, cache_dir=nfl_cache, max_margin=20)
    }
    assert games["2025_04_DDD_BBB"].margin == 20.0
    # Games under the cap are untouched.
    assert games["2025_01_AAA_BBB"].margin == 7.0


def test_max_margin_falls_back_to_the_config(nfl_cache: Path, monkeypatch):
    monkeypatch.setitem(nfl.NFL_CONFIG, "MAX_MARGIN", 14)
    games = {game.game_id: game for game in nfl.load_games(2025, cache_dir=nfl_cache)}
    assert games["2025_04_DDD_BBB"].margin == 14.0


def test_include_playoffs_adds_the_postseason(nfl_cache: Path):
    games = nfl.load_games(2025, cache_dir=nfl_cache, include_playoffs=True)
    assert "2025_19_AAA_BBB" in {game.game_id for game in games}


def test_week_cutoff_is_exclusive(nfl_cache: Path):
    games = nfl.load_games(2025, cache_dir=nfl_cache, week=3)

    # "--week 3" means the ratings going *into* week 3, so weeks 1 and 2 only.
    assert {game.week for game in games} == {1, 2}


def test_as_of_cutoff_is_exclusive(nfl_cache: Path):
    games = nfl.load_games(2025, cache_dir=nfl_cache, as_of=date(2025, 9, 14))

    assert [game.game_id for game in games] == ["2025_01_AAA_BBB", "2025_01_CCC_DDD"]


def test_games_are_chronological(nfl_cache: Path):
    games = nfl.load_games(2025, cache_dir=nfl_cache)
    assert [game.date for game in games] == sorted(game.date for game in games)


def test_games_feed_the_solver(nfl_cache: Path):
    from massey.models import solve_massey

    games = nfl.load_games(2025, cache_dir=nfl_cache)
    result = solve_massey([game.to_game() for game in games])

    assert set(result.ratings) == {"AAA", "BBB", "CCC", "DDD"}
    assert sum(result.ratings.values()) == pytest.approx(0.0, abs=1e-9)
    # BBB won every game it played, including a 37-point blowout.
    assert max(result.ratings, key=lambda team: result.ratings[team]) == "BBB"


def test_season_reaches_nflreadpy_as_an_integer(tmp_path: Path, monkeypatch):
    # The command line hands the season over as a string. nflreadpy filters an
    # integer season column, so a string year reaches polars as a type
    # mismatch ("'is_in' cannot check for Int32 values in List(String) data")
    # rather than as a lookup that returns nothing.
    import sys
    import types

    captured = {}

    fake = types.ModuleType("nflreadpy")

    def load_schedules(seasons):
        captured["seasons"] = seasons
        raise RuntimeError("stop before the download")

    fake.load_schedules = load_schedules
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    monkeypatch.setattr("massey.data.cache.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError):
        nfl.load_schedule_frame("2025", cache_dir=tmp_path)

    assert captured["seasons"] == [2025]
    assert isinstance(captured["seasons"][0], int)


def test_a_season_that_is_not_a_year_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="NFL season year"):
        nfl.load_schedule_frame("not-a-year", cache_dir=tmp_path)


def test_string_and_int_seasons_share_one_cache_entry(nfl_cache: Path):
    by_string = nfl.load_games("2025", cache_dir=nfl_cache)
    by_int = nfl.load_games(2025, cache_dir=nfl_cache)
    assert [game.game_id for game in by_string] == [game.game_id for game in by_int]


def test_loader_does_not_touch_the_network_when_cached(nfl_cache: Path, monkeypatch):
    # Guards the cache contract: a cached season must not import or call
    # nflreadpy at all.
    def explode(*args, **kwargs):
        raise AssertionError("the loader hit the network despite a warm cache")

    monkeypatch.setattr(nfl, "with_retry", explode)
    assert nfl.load_games(2025, cache_dir=nfl_cache)
