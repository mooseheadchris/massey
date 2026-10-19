"""Shared pytest fixtures."""

from __future__ import annotations

import itertools
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

#: A miniature NFL season using the real column names and values returned by
#: ``nflreadpy.load_schedules()``, verified against a live 2025/2026 pull.
#: Four teams, one unplayed game, one tie, one neutral-site blowout and one
#: playoff game.
NFL_SCHEDULE_ROWS = [
    # game_id, game_type, week, gameday, away_team, away_score, home_team,
    # home_score, location, overtime
    ("2025_01_AAA_BBB", "REG", 1, "2025-09-07", "AAA", 17.0, "BBB", 24.0, "Home", 0.0),
    ("2025_01_CCC_DDD", "REG", 1, "2025-09-07", "CCC", 20.0, "DDD", 10.0, "Home", 0.0),
    ("2025_02_BBB_AAA", "REG", 2, "2025-09-14", "BBB", 27.0, "AAA", 30.0, "Home", 1.0),
    ("2025_03_AAA_CCC", "REG", 3, "2025-09-21", "AAA", 13.0, "CCC", 13.0, "Home", 1.0),
    ("2025_04_DDD_BBB", "REG", 4, "2025-09-28", "DDD", 3.0, "BBB", 40.0, "Neutral", 0.0),
    ("2025_05_AAA_DDD", "REG", 5, "2025-10-05", "AAA", None, "DDD", None, "Home", None),
    ("2025_19_AAA_BBB", "WC", 19, "2026-01-11", "AAA", 14.0, "BBB", 21.0, "Home", 0.0),
]


@pytest.fixture
def nhl_cache(tmp_path: Path) -> Path:
    """A raw-data cache pre-seeded with the NHL schedule fixtures.

    The loader is read-through, so seeding the cache lets the NHL tests run the
    real code path end to end with no network access.

    Returns:
        The cache root to pass as ``cache_dir``.
    """
    target = tmp_path / "raw" / "nhl"
    target.mkdir(parents=True)
    for fixture in FIXTURE_DIR.glob("nhl_club_schedule_*.json"):
        shutil.copy(fixture, target / fixture.name.removeprefix("nhl_"))
    return tmp_path / "raw"


@pytest.fixture
def nhl_full_cache(tmp_path: Path) -> Path:
    """A raw-data cache holding a synthetic but complete 32-team NHL season.

    The three-team fixture files are enough to test the margin mapping, but not
    the parts that need a whole league: team discovery, deduplication across 32
    club schedules, and the CLI end to end. This builds that, with a hidden
    team strength the ratings should recover and roughly one upset in seven so
    the signal is not perfectly separable.

    Returns:
        The cache root to pass as ``cache_dir``.
    """
    from massey.leagues.nhl import NHL_TEAM_ABBREVS

    teams = list(NHL_TEAM_ABBREVS)
    strength = {team: 15.5 - index for index, team in enumerate(teams)}
    outcome_cycle = ["REG", "REG", "REG", "OT", "SO"]

    games: list[dict] = []
    for index, (home, away) in enumerate(itertools.permutations(teams, 2)):
        # A home edge on top of the strength gap, upset every seventh game.
        # The edge has to exceed the 1-point spacing between strengths to flip
        # any outcome, otherwise the fixture contains no home advantage at all.
        favoured_home = (strength[home] - strength[away] + 2.5) > 0
        home_wins = favoured_home != (index % 7 == 0)
        outcome = outcome_cycle[index % len(outcome_cycle)]
        winner, loser = (4, 1) if outcome == "REG" else (3, 2)

        games.append(
            {
                "id": 2025020001 + index,
                "season": 20252026,
                "gameType": 2,
                # Eight games a day, as the real schedule roughly runs.
                "gameDate": (date(2025, 10, 8) + timedelta(days=index // 8)).isoformat(),
                "gameState": "OFF",
                "neutralSite": False,
                "homeTeam": {"abbrev": home, "score": winner if home_wins else loser},
                "awayTeam": {"abbrev": away, "score": loser if home_wins else winner},
                "gameOutcome": {"lastPeriodType": outcome},
            }
        )

    target = tmp_path / "raw" / "nhl"
    target.mkdir(parents=True)
    for team in teams:
        involved = [
            game
            for game in games
            if team in (game["homeTeam"]["abbrev"], game["awayTeam"]["abbrev"])
        ]
        path = target / f"club_schedule_{team}_20252026.json"
        path.write_text(json.dumps({"currentSeason": 20252026, "games": involved}))

    return tmp_path / "raw"


@pytest.fixture
def nfl_cache(tmp_path: Path) -> Path:
    """A raw-data cache pre-seeded with a miniature 2025 NFL schedule.

    Lets the NFL tests exercise the real loader with no call to nflreadpy.

    Returns:
        The cache root to pass as ``cache_dir``.
    """
    frame = pd.DataFrame(
        NFL_SCHEDULE_ROWS,
        columns=[
            "game_id",
            "game_type",
            "week",
            "gameday",
            "away_team",
            "away_score",
            "home_team",
            "home_score",
            "location",
            "overtime",
        ],
    )
    target = tmp_path / "raw" / "nfl"
    target.mkdir(parents=True)
    frame.to_parquet(target / "schedules_2025.parquet", index=False)
    return tmp_path / "raw"
