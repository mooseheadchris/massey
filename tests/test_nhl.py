"""Tests for the NHL margin mapping and loader.

The loader tests run against hand-authored fixtures seeded into the cache (see
``tests/fixtures/README.md``), so they never touch the network.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from massey.leagues import nhl


# --------------------------------------------------------------------------- #
# Margin mapping -- the heart of the NHL model
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("home_score", "away_score", "outcome", "expected"),
    [
        # Home wins: the winner's points, positive.
        (5, 2, "REG", 3),
        (4, 3, "OT", 2),
        (3, 2, "SO", 1),
        # Home loses: the same magnitudes, negative.
        (2, 5, "REG", -3),
        (3, 4, "OT", -2),
        (2, 3, "SO", -1),
    ],
)
def test_margin_from_outcome(home_score, away_score, outcome, expected):
    assert nhl.margin_from_outcome(home_score, away_score, outcome) == expected


def test_margin_ignores_goal_differential():
    # A 6-1 romp and a 2-1 regulation win are the same result in the standings,
    # and so must be the same rating margin.
    assert nhl.margin_from_outcome(6, 1, "REG") == nhl.margin_from_outcome(2, 1, "REG")


def test_margin_is_antisymmetric():
    for outcome in ("REG", "OT", "SO"):
        home_win = nhl.margin_from_outcome(4, 1, outcome)
        away_win = nhl.margin_from_outcome(1, 4, outcome)
        assert home_win == -away_win


def test_margin_magnitudes_are_ordered():
    # A regulation win must be worth strictly more than an overtime win, which
    # must be worth strictly more than a shootout win.
    regulation = nhl.margin_from_outcome(3, 1, "REG")
    overtime = nhl.margin_from_outcome(3, 2, "OT")
    shootout = nhl.margin_from_outcome(3, 2, "SO")
    assert regulation > overtime > shootout > 0


def test_margins_come_from_the_config_dict(monkeypatch):
    # The point values are a single config dict so they can be tuned without
    # touching the loader.
    monkeypatch.setitem(nhl.NHL_POINTS, "REG", 10)
    assert nhl.margin_from_outcome(5, 2, "REG") == 10


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("REG", "REG"),
        ("reg", "REG"),
        ("  OT  ", "OT"),
        ("Overtime", "OT"),
        ("SHOOTOUT", "SO"),
        ("so", "SO"),
    ],
)
def test_normalise_outcome_accepts_known_spellings(raw, expected):
    assert nhl.normalise_outcome(raw) == expected


def test_normalise_outcome_rejects_the_unknown():
    with pytest.raises(ValueError, match="unrecognised NHL outcome"):
        nhl.normalise_outcome("PENALTY_SHOTS")


def test_normalise_outcome_rejects_none():
    with pytest.raises(ValueError, match="no lastPeriodType"):
        nhl.normalise_outcome(None)


def test_level_score_raises():
    # No NHL game ends tied, so a level score means bad input, not a tie.
    with pytest.raises(ValueError, match="ended level"):
        nhl.margin_from_outcome(2, 2, "REG")


# --------------------------------------------------------------------------- #
# Season identifiers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("given", "expected"),
    [(2026, "20262027"), ("2026", "20262027"), ("20262027", "20262027"), (2025, "20252026")],
)
def test_season_id(given, expected):
    assert nhl.season_id(given) == expected


@pytest.mark.parametrize("bad", ["20262028", "26", "", "abcd", "2026-27"])
def test_season_id_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        nhl.season_id(bad)


def test_current_season_rolls_over_in_july():
    # Documents the convention rather than today's answer: a season is named
    # for the year it starts in.
    assert nhl.season_id(nhl.current_season()).startswith(str(nhl.current_season()))


# --------------------------------------------------------------------------- #
# Loading, against seeded fixtures
# --------------------------------------------------------------------------- #

def test_load_games_from_cache(nhl_cache: Path):
    games = nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])

    # Five completed regular-season games. The unplayed game, the in-progress
    # game and the playoff game are all excluded, and the two games that appear
    # in both clubs' schedules are counted once.
    assert len(games) == 5
    assert [game.game_id for game in games] == [
        "2025020001",
        "2025020002",
        "2025020003",
        "2025020007",
        "2025020005",
    ]


def test_load_games_maps_outcomes_to_margins(nhl_cache: Path):
    games = {
        game.game_id: game
        for game in nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    }

    # TOR 5-2 MTL in regulation, at home.
    assert games["2025020001"].margin == 3
    assert games["2025020001"].outcome == "REG"
    assert games["2025020001"].home == "TOR"

    # MTL lost at home in overtime.
    assert games["2025020002"].margin == -2
    assert games["2025020002"].outcome == "OT"
    assert games["2025020002"].home == "MTL"

    # TOR lost at home in a shootout.
    assert games["2025020003"].margin == -1
    assert games["2025020003"].outcome == "SO"


def test_load_games_flags_neutral_sites(nhl_cache: Path):
    games = {
        game.game_id: game
        for game in nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    }
    assert games["2025020005"].neutral is True
    assert games["2025020001"].neutral is False


def test_load_games_keeps_raw_scores_for_reporting(nhl_cache: Path):
    games = {
        game.game_id: game
        for game in nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    }
    blowout = games["2025020005"]
    assert (blowout.home_score, blowout.away_score) == (6, 1)
    # The margin is still the outcome-points value, not the goal differential.
    assert blowout.margin == 3


def test_load_games_is_chronological(nhl_cache: Path):
    games = nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    assert [game.date for game in games] == sorted(game.date for game in games)


def test_include_playoffs_adds_postseason_games(nhl_cache: Path):
    regular = nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    with_playoffs = nhl.load_games(
        2025, cache_dir=nhl_cache, teams=["TOR", "MTL"], include_playoffs=True
    )

    added = {game.game_id for game in with_playoffs} - {game.game_id for game in regular}
    assert added == {"2025030001"}
    playoff = next(game for game in with_playoffs if game.game_id == "2025030001")
    assert playoff.margin == 2  # home overtime win


def test_as_of_cutoff_is_exclusive(nhl_cache: Path):
    games = nhl.load_games(
        2025, cache_dir=nhl_cache, teams=["TOR", "MTL"], as_of=date(2025, 10, 14)
    )
    # The 14 October game is excluded: "as of" means the ratings you carry
    # *into* that date.
    assert [game.game_id for game in games] == ["2025020001", "2025020002"]


def test_week_argument_is_ignored_for_the_nhl(nhl_cache: Path):
    unfiltered = nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    with_week = nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"], week=3)
    assert [game.game_id for game in with_week] == [game.game_id for game in unfiltered]


def test_home_won_is_derived_from_the_scores(nhl_cache: Path):
    games = {
        game.game_id: game
        for game in nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    }
    assert games["2025020001"].home_won is True
    assert games["2025020003"].home_won is False


def test_games_feed_the_solver(nhl_cache: Path):
    from massey.models import solve_massey

    games = nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    result = solve_massey([game.to_game() for game in games])

    assert set(result.ratings) == {"TOR", "MTL", "BOS"}
    assert sum(result.ratings.values()) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Defensive field reads
# --------------------------------------------------------------------------- #

def test_renamed_score_field_is_tolerated():
    # "goals" is a registered alternate spelling for the score.
    game = {
        "id": 1,
        "gameType": 2,
        "gameDate": "2025-10-08",
        "gameState": "OFF",
        "homeTeam": {"abbrev": "TOR", "goals": 4},
        "awayTeam": {"abbrev": "MTL", "goals": 1},
        "gameOutcome": {"lastPeriodType": "REG"},
    }
    record = nhl._to_record(game, {2})
    assert record is not None
    assert record.margin == 3


def test_unknown_field_name_fails_loudly():
    # A renamed field must raise an error naming the keys that were present,
    # never silently produce a wrong rating.
    game = {
        "id": 1,
        "gameType": 2,
        "gameDate": "2025-10-08",
        "gameState": "OFF",
        "homeTeam": {"abbrev": "TOR", "score": 4},
        "awayTeam": {"abbrev": "MTL", "score": 1},
        "howIsItOver": {"period": "REG"},
    }
    with pytest.raises(KeyError, match="could not find 'outcome'"):
        nhl._to_record(game, {2})


def test_timestamp_only_date_is_parsed():
    game = {
        "id": 1,
        "gameType": 2,
        "startTimeUTC": "2025-10-08T23:00:00Z",
        "gameState": "OFF",
        "homeTeam": {"abbrev": "TOR", "score": 4},
        "awayTeam": {"abbrev": "MTL", "score": 1},
        "gameOutcome": {"lastPeriodType": "REG"},
    }
    record = nhl._to_record(game, {2})
    assert record is not None
    assert record.date == date(2025, 10, 8)


def test_missing_game_state_falls_back_to_score_presence():
    game = {
        "id": 1,
        "gameType": 2,
        "gameDate": "2025-10-08",
        "homeTeam": {"abbrev": "TOR", "score": 4},
        "awayTeam": {"abbrev": "MTL", "score": 1},
        "gameOutcome": {"lastPeriodType": "REG"},
    }
    record = nhl._to_record(game, {2})
    assert record is not None and record.margin == 3


def test_team_discovery_falls_back_when_too_few_teams(nhl_cache: Path):
    # The fixture only names three teams, well short of a real league, so
    # discovery must fall back to the built-in list rather than rate a
    # three-team NHL.
    teams = nhl.discover_team_abbrevs(2025, cache_dir=nhl_cache)
    assert teams == list(nhl.NHL_TEAM_ABBREVS)


def test_a_settled_failure_is_not_retried(tmp_path: Path, monkeypatch):
    # A bad team code gets a 404, which no amount of retrying will fix. It
    # should fail on the first attempt rather than after four waits.
    from nhlpy.http_client import ResourceNotFoundException

    monkeypatch.setattr("massey.leagues.nhl.time.sleep", lambda _: None)
    monkeypatch.setattr("massey.data.cache.time.sleep", lambda _: None)
    attempts = []

    class FailingApi:
        def team_season_schedule(self, team_abbr, season):
            attempts.append(team_abbr)
            raise ResourceNotFoundException("no such team")

    with pytest.raises(ResourceNotFoundException):
        nhl.load_club_schedule(
            "ZZZ", 2025, cache_dir=tmp_path, schedule_api=FailingApi(), delay=0
        )

    assert len(attempts) == 1


def test_a_transient_failure_is_retried(tmp_path: Path, monkeypatch):
    from nhlpy.http_client import ServerErrorException

    monkeypatch.setattr("massey.leagues.nhl.time.sleep", lambda _: None)
    monkeypatch.setattr("massey.data.cache.time.sleep", lambda _: None)
    attempts = []

    class FlakyApi:
        def team_season_schedule(self, team_abbr, season):
            attempts.append(team_abbr)
            if len(attempts) < 3:
                raise ServerErrorException("upstream hiccup", status_code=503)
            return {"games": []}

    payload = nhl.load_club_schedule(
        "TOR", 2025, cache_dir=tmp_path, schedule_api=FlakyApi(), delay=0
    )

    assert payload == {"games": []}
    assert len(attempts) == 3


def test_inspect_season_reports_resolved_fields(nhl_cache: Path):
    report = nhl.inspect_season(2025, cache_dir=nhl_cache)

    assert report["season"] == "20252026"
    assert report["n_games"] == 7
    assert report["game_states"] == {"FINAL": 1, "FUT": 1, "LIVE": 1, "OFF": 4}
    assert report["sample_resolved_fields"]["home_abbrev"] == "TOR"
    assert report["sample_resolved_fields"]["outcome"] == "REG"
