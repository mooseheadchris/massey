"""End-to-end tests over a complete synthetic NHL season.

These use the ``nhl_full_cache`` fixture (32 teams, a full home-and-away
round-robin) to cover what the small fixtures cannot: team discovery,
deduplication across 32 club schedules, and both CLI commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from massey.cli import main
from massey.leagues import nhl
from massey.models import solve_massey

#: 32 teams playing each other home and away.
EXPECTED_GAMES = 32 * 31


def test_team_discovery_finds_the_whole_league(nhl_full_cache: Path):
    teams = nhl.discover_team_abbrevs(2025, cache_dir=nhl_full_cache)
    assert len(teams) == 32
    assert teams == sorted(nhl.NHL_TEAM_ABBREVS)


def test_games_are_deduplicated_across_club_schedules(nhl_full_cache: Path):
    # Each game appears in two clubs' schedules, so a naive concatenation would
    # double-count every one of them.
    games = nhl.load_games(2025, cache_dir=nhl_full_cache)
    assert len(games) == EXPECTED_GAMES
    assert len({game.game_id for game in games}) == EXPECTED_GAMES


def test_margins_only_ever_take_the_configured_values(nhl_full_cache: Path):
    games = nhl.load_games(2025, cache_dir=nhl_full_cache)
    assert {game.margin for game in games} == {-3, -2, -1, 1, 2, 3}


def test_every_team_plays_the_same_number_of_games(nhl_full_cache: Path):
    games = nhl.load_games(2025, cache_dir=nhl_full_cache)
    result = solve_massey([game.to_game() for game in games])
    assert set(result.games_played.values()) == {62}


def test_ratings_recover_the_hidden_strength_order(nhl_full_cache: Path):
    games = nhl.load_games(2025, cache_dir=nhl_full_cache)
    result = solve_massey([game.to_game() for game in games])

    # The fixture's strength runs down the team list in order, so the fitted
    # ranking should correlate almost perfectly with it.
    ranked = sorted(result.ratings, key=lambda team: -result.ratings[team])
    true_order = list(nhl.NHL_TEAM_ABBREVS)
    rank_errors = [abs(ranked.index(team) - true_order.index(team)) for team in true_order]

    assert max(rank_errors) <= 3
    assert sum(result.ratings.values()) == pytest.approx(0.0, abs=1e-9)


def test_hfa_is_positive_and_modest(nhl_full_cache: Path):
    games = nhl.load_games(2025, cache_dir=nhl_full_cache)
    result = solve_massey([game.to_game() for game in games])

    # The fixture gives the home side a one-point edge on a +/-3 scale.
    assert 0.0 < result.hfa < 1.5


def test_cli_rate_runs_end_to_end(nhl_full_cache: Path, tmp_path: Path, capsys):
    output_dir = tmp_path / "output"
    code = main(
        [
            "rate",
            "--league", "nhl",
            "--season", "2025",
            "--cache-dir", str(nhl_full_cache),
            "--output-dir", str(output_dir),
        ]
    )
    assert code == 0

    printed = capsys.readouterr().out
    assert "NHL 2025 Massey ratings" in printed
    assert "outcome points" in printed
    assert f"games fitted       : {EXPECTED_GAMES}" in printed

    import pandas as pd

    frame = pd.read_csv(output_dir / "nhl_2025_ratings.csv")
    assert list(frame.columns) == ["rank", "team", "rating", "games_played"]
    assert len(frame) == 32

    meta = json.loads((output_dir / "nhl_2025_ratings_meta.json").read_text())
    assert meta["margin_units"] == "outcome points"
    assert meta["games_fitted"] == EXPECTED_GAMES


def test_cli_backtest_beats_the_home_baseline(nhl_full_cache: Path, tmp_path: Path, capsys):
    output_dir = tmp_path / "output"
    code = main(
        [
            "backtest",
            "--league", "nhl",
            "--season", "2025",
            "--cache-dir", str(nhl_full_cache),
            "--output-dir", str(output_dir),
        ]
    )
    assert code == 0
    assert "walk-forward backtest" in capsys.readouterr().out

    summary = json.loads((output_dir / "nhl_2025_backtest_summary.json").read_text())

    # The league's own default warm-up is 100 games.
    assert summary["n_warmup_skipped"] == 100
    assert summary["n_evaluated"] == EXPECTED_GAMES - 100
    assert summary["n_ties_excluded"] == 0  # no NHL game ends tied

    # The fixture upsets one game in seven, so 6/7 ~= 0.857 is the ceiling; the
    # remaining gap is the near-even matchups the ratings cannot separate. The
    # point is that the model lands close to that ceiling and far clear of the
    # home baseline, which this fixture holds near a coin flip.
    assert summary["accuracy"] > 0.75
    assert summary["accuracy"] > summary["home_baseline_accuracy"] + 0.2
    assert summary["log_loss"] < summary["baseline_log_loss"]


def test_backtest_predictions_csv_has_one_row_per_evaluated_game(
    nhl_full_cache: Path, tmp_path: Path
):
    import pandas as pd

    output_dir = tmp_path / "output"
    main(
        [
            "backtest",
            "--league", "nhl",
            "--season", "2025",
            "--cache-dir", str(nhl_full_cache),
            "--output-dir", str(output_dir),
        ]
    )
    frame = pd.read_csv(output_dir / "nhl_2025_backtest_games.csv")

    assert len(frame) == EXPECTED_GAMES - 100
    assert {"predicted_margin", "actual_margin", "home_win_prob", "correct"} <= set(frame.columns)
    # Probabilities stay strictly inside (0, 1) so log loss is always finite.
    assert frame["home_win_prob"].between(0.0, 1.0, inclusive="neither").all()
