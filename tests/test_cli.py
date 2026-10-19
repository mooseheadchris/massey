"""Tests for the command-line interface.

Driven through ``main()`` with a seeded cache, so these cover argument parsing,
output files and formatting without any network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from massey.cli import build_parser, format_ratings_table, main, ratings_table
from massey.leagues import nfl
from massey.models import solve_massey


@pytest.fixture
def run_cli(nfl_cache: Path, tmp_path: Path):
    """Return a callable that runs the CLI against the seeded NFL cache."""
    output_dir = tmp_path / "output"

    def run(*args: str) -> int:
        return main([*args, "--cache-dir", str(nfl_cache), "--output-dir", str(output_dir)])

    run.output_dir = output_dir  # type: ignore[attr-defined]
    return run


# --------------------------------------------------------------------------- #
# rate
# --------------------------------------------------------------------------- #

def test_rate_prints_and_saves(run_cli, capsys):
    assert run_cli("rate", "--league", "nfl", "--season", "2025") == 0

    printed = capsys.readouterr().out
    assert "NFL 2025 Massey ratings" in printed
    assert "home-field advantage" in printed
    assert "ridge penalty" in printed

    csv_path = run_cli.output_dir / "nfl_2025_ratings.csv"
    meta_path = run_cli.output_dir / "nfl_2025_ratings_meta.json"
    assert csv_path.is_file() and meta_path.is_file()

    meta = json.loads(meta_path.read_text())
    assert meta["league"] == "nfl"
    assert meta["ridge"] == 1.0
    assert meta["games_fitted"] == 5
    assert "home_field_advantage" in meta


def test_rate_csv_has_the_requested_columns(run_cli):
    import pandas as pd

    run_cli("rate", "--league", "nfl", "--season", "2025")
    frame = pd.read_csv(run_cli.output_dir / "nfl_2025_ratings.csv")

    assert list(frame.columns) == ["rank", "team", "rating", "games_played"]
    assert frame["rank"].tolist() == [1, 2, 3, 4]
    assert frame["rating"].is_monotonic_decreasing


def test_rate_no_save_writes_nothing(run_cli, capsys):
    assert run_cli("rate", "--league", "nfl", "--season", "2025", "--no-save") == 0
    assert "Massey ratings" in capsys.readouterr().out
    assert not run_cli.output_dir.exists()


def test_rate_honours_the_ridge_flag(run_cli, capsys):
    run_cli("rate", "--league", "nfl", "--season", "2025", "--ridge", "25", "--no-save")
    assert "ridge penalty       : 25" in capsys.readouterr().out


def test_rate_honours_the_margin_cap(run_cli, capsys):
    run_cli(
        "rate", "--league", "nfl", "--season", "2025", "--max-margin", "10", "--no-save"
    )
    assert "margin cap          : 10.0" in capsys.readouterr().out


def test_rate_honours_the_week_cutoff(run_cli, capsys):
    run_cli("rate", "--league", "nfl", "--season", "2025", "--week", "3", "--no-save")
    printed = capsys.readouterr().out
    assert "weeks before 3" in printed
    assert "games fitted       : 3" in printed


def test_rate_honours_the_as_of_cutoff(run_cli, capsys):
    run_cli(
        "rate", "--league", "nfl", "--season", "2025", "--as-of", "2025-09-14", "--no-save"
    )
    printed = capsys.readouterr().out
    assert "games before 2025-09-14" in printed
    assert "games fitted       : 2" in printed


def test_rate_reports_when_a_season_has_no_games(run_cli, capsys):
    code = run_cli(
        "rate", "--league", "nfl", "--season", "2025", "--as-of", "2000-01-01", "--no-save"
    )
    assert code == 1
    assert "no completed NFL games" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #

def test_backtest_prints_and_saves(run_cli, capsys):
    code = run_cli(
        "backtest", "--league", "nfl", "--season", "2025", "--skip-weeks", "1"
    )
    assert code == 0

    printed = capsys.readouterr().out
    assert "walk-forward backtest" in printed
    assert "home-team baseline" in printed
    assert "edge over baseline" in printed

    games_path = run_cli.output_dir / "nfl_2025_backtest_games.csv"
    summary_path = run_cli.output_dir / "nfl_2025_backtest_summary.json"
    assert games_path.is_file() and summary_path.is_file()

    summary = json.loads(summary_path.read_text())
    assert summary["league"] == "nfl"
    assert summary["refit"] == "per-day"
    assert summary["n_games"] == 5


def test_backtest_uses_the_league_default_warmup(run_cli, capsys):
    # The NFL adapter asks for three warm-up weeks, which in this tiny
    # five-game season leaves only the week 4 game.
    run_cli("backtest", "--league", "nfl", "--season", "2025", "--no-save")
    printed = capsys.readouterr().out
    assert "skipped (warm-up)   : 4" in printed
    assert "evaluated           : 1" in printed


def test_backtest_refit_flag_is_recorded(run_cli, capsys):
    run_cli(
        "backtest",
        "--league", "nfl",
        "--season", "2025",
        "--skip-weeks", "1",
        "--refit", "per-game",
        "--no-save",
    )
    assert "per-game" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #

def test_league_is_required():
    with pytest.raises(SystemExit):
        main(["rate", "--season", "2025"])


def test_unknown_league_is_rejected():
    with pytest.raises(SystemExit):
        main(["rate", "--league", "mlb"])


def test_bad_date_is_rejected():
    with pytest.raises(SystemExit):
        main(["rate", "--league", "nfl", "--as-of", "last tuesday"])


def test_bad_refit_cadence_is_rejected():
    with pytest.raises(SystemExit):
        main(["backtest", "--league", "nfl", "--refit", "hourly"])


def test_season_defaults_to_the_current_one():
    parser = build_parser()
    args = parser.parse_args(["rate", "--league", "nfl"])
    assert args.season is None  # resolved later, from the league adapter


def test_inspect_is_unsupported_for_the_nfl(capsys, tmp_path):
    # nflreadpy hands back a typed table, so there are no field names to go
    # hunting for. Saying so beats a traceback.
    code = main(
        [
            "inspect",
            "--league", "nfl",
            "--season", "2025",
            "--cache-dir", str(tmp_path / "empty"),
        ]
    )
    assert code == 1
    assert "no inspect support" in capsys.readouterr().err


def test_a_failing_fetch_is_reported_without_a_traceback(capsys, tmp_path, monkeypatch):
    # Cold cache and an unreachable API: the user should get one line, not a
    # stack trace.
    monkeypatch.setattr("massey.data.cache.time.sleep", lambda _: None)
    monkeypatch.setattr("massey.leagues.nhl.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "massey.leagues.nhl.build_schedule_api",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("api-web.nhle.com unreachable")),
    )

    code = main(
        [
            "rate",
            "--league", "nhl",
            "--season", "2025",
            "--cache-dir", str(tmp_path / "empty"),
            "--no-save",
        ]
    )
    assert code == 1
    printed = capsys.readouterr()
    assert "error:" in printed.err
    assert "Traceback" not in printed.err


def test_nhl_inspect_renders_json(nhl_cache: Path, capsys):
    code = main(["inspect", "--league", "nhl", "--season", "2025", "--cache-dir", str(nhl_cache)])
    assert code == 0

    report = json.loads(capsys.readouterr().out)
    assert report["season"] == "20252026"
    assert report["sample_resolved_fields"]["home_abbrev"] == "TOR"


# --------------------------------------------------------------------------- #
# Table helpers
# --------------------------------------------------------------------------- #

def test_ratings_table_is_ranked(nfl_cache: Path):
    games = nfl.load_games(2025, cache_dir=nfl_cache)
    table = ratings_table(solve_massey([game.to_game() for game in games]))

    assert table["rank"].tolist() == [1, 2, 3, 4]
    assert table["rating"].is_monotonic_decreasing
    assert table.loc[0, "team"] == "BBB"


def test_format_ratings_table_includes_every_team(nfl_cache: Path):
    games = nfl.load_games(2025, cache_dir=nfl_cache)
    text = format_ratings_table(
        ratings_table(solve_massey([game.to_game() for game in games]))
    )

    for team in ("AAA", "BBB", "CCC", "DDD"):
        assert team in text
    assert "relative to an average team" in text
