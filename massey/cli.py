"""Command-line interface.

Three commands::

    python -m massey.cli rate     --league nfl --season 2026
    python -m massey.cli backtest --league nhl --season 2025
    python -m massey.cli inspect  --league nhl --season 2025

``rate`` prints a ranked table and writes it to ``output/``. ``backtest`` walks
forward through a season and reports accuracy against the home-team baseline.
``inspect`` dumps the raw field names a league's source returned, which is how
you confirm the NHL payload shape after an API change.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from massey.backtest import (
    REFIT_CADENCES,
    BacktestConfig,
    BacktestSummary,
    GamePrediction,
    run_backtest,
)
from massey.leagues import LEAGUES, get_league
from massey.leagues.base import GameRecord
from massey.models import DEFAULT_RIDGE, MasseyResult, solve_massey

#: Where CSV and JSON outputs are written.
DEFAULT_OUTPUT_DIR = Path("output")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all commands."""
    parser = argparse.ArgumentParser(
        prog="python -m massey.cli",
        description="Massey-style power ratings for the NFL and NHL.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log data fetching to stderr"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rate = subparsers.add_parser("rate", help="fit and print ratings for a season")
    _add_common_arguments(rate)
    rate.add_argument(
        "--ridge",
        type=float,
        default=DEFAULT_RIDGE,
        help=f"ridge penalty on team ratings (default: {DEFAULT_RIDGE})",
    )
    rate.add_argument(
        "--max-margin",
        type=float,
        default=None,
        help="cap the absolute margin at this value (NFL only; default: no cap)",
    )
    rate.add_argument(
        "--no-save", action="store_true", help="print only, do not write to output/"
    )

    backtest = subparsers.add_parser(
        "backtest", help="walk-forward evaluation against the home-team baseline"
    )
    _add_common_arguments(backtest)
    backtest.add_argument(
        "--ridge",
        type=float,
        default=DEFAULT_RIDGE,
        help=f"ridge penalty on team ratings (default: {DEFAULT_RIDGE})",
    )
    backtest.add_argument(
        "--max-margin",
        type=float,
        default=None,
        help="cap the absolute margin at this value (NFL only; default: no cap)",
    )
    backtest.add_argument(
        "--skip-weeks",
        type=int,
        default=None,
        help="use games in these opening weeks for training only (default: the league's own)",
    )
    backtest.add_argument(
        "--skip-games",
        type=int,
        default=None,
        help="use this many opening games for training only (default: the league's own)",
    )
    backtest.add_argument(
        "--refit",
        choices=REFIT_CADENCES,
        default="per-day",
        help="how often to refit ratings (default: per-day)",
    )
    backtest.add_argument(
        "--no-save", action="store_true", help="print only, do not write to output/"
    )

    inspect = subparsers.add_parser(
        "inspect", help="dump the raw field names the league's data source returned"
    )
    _add_common_arguments(inspect, cutoffs=False)

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser, *, cutoffs: bool = True) -> None:
    """Add the arguments every command shares."""
    parser.add_argument(
        "--league",
        required=True,
        choices=sorted(LEAGUES),
        help="which league to rate",
    )
    parser.add_argument(
        "--season",
        default=None,
        help=(
            "season, named for the year it starts in (e.g. 2026 means the NFL "
            "2026 season or the NHL 2026-27 season; an NHL id like 20262027 "
            "also works). Defaults to the current season."
        ),
    )
    parser.add_argument(
        "--include-playoffs",
        action="store_true",
        help="include postseason games (default: regular season only)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="root of the raw-data cache (default: data/raw)",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="re-fetch raw data, ignoring the cache"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"where to write CSV output (default: {DEFAULT_OUTPUT_DIR})",
    )
    if cutoffs:
        parser.add_argument(
            "--as-of",
            type=_parse_date_argument,
            default=None,
            metavar="YYYY-MM-DD",
            help="use only games played strictly before this date",
        )
        parser.add_argument(
            "--week",
            type=int,
            default=None,
            help="use only games before this week, i.e. ratings going into it (NFL only)",
        )


def _parse_date_argument(text: str) -> date:
    """Parse a ``YYYY-MM-DD`` command-line date."""
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    league = get_league(args.league)
    season = args.season if args.season is not None else league.current_season()

    try:
        if args.command == "rate":
            return _command_rate(args, league, season)
        if args.command == "backtest":
            return _command_backtest(args, league, season)
        if args.command == "inspect":
            return _command_inspect(args, league, season)
    except Exception as error:  # noqa: BLE001 - a CLI should not show a traceback
        print(f"error: {error}", file=sys.stderr)
        return 1

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _command_rate(args: argparse.Namespace, league: Any, season: Any) -> int:
    """Fit ratings for a season, print them, and optionally save them."""
    games = _load_games(args, league, season, as_of=args.as_of, week=args.week)
    if not games:
        print(f"no completed {league.name.upper()} games found for {season}")
        return 1

    result = solve_massey([game.to_game() for game in games], ridge=args.ridge)
    table = ratings_table(result)

    print(_rate_header(league, season, games, result, args))
    print()
    print(format_ratings_table(table, units=league.margin_units))

    if not args.no_save:
        paths = _save_ratings(table, result, league, season, args)
        print()
        for path in paths:
            print(f"wrote {path}")
    return 0


def _command_backtest(args: argparse.Namespace, league: Any, season: Any) -> int:
    """Walk forward through a season and report prediction metrics."""
    games = _load_games(args, league, season, as_of=args.as_of, week=args.week)
    if not games:
        print(f"no completed {league.name.upper()} games found for {season}")
        return 1

    config = BacktestConfig(
        ridge=args.ridge,
        skip_weeks=args.skip_weeks if args.skip_weeks is not None else league.default_skip_weeks,
        skip_games=args.skip_games if args.skip_games is not None else league.default_skip_games,
        refit=args.refit,
        logistic_scale=league.default_logistic_scale,
    )
    summary, predictions = run_backtest(
        games, league=league.name, season=str(season), config=config
    )

    print(format_backtest_summary(summary, units=league.margin_units))

    if not args.no_save:
        paths = _save_backtest(summary, predictions, league, season, args)
        print()
        for path in paths:
            print(f"wrote {path}")
    return 0


def _command_inspect(args: argparse.Namespace, league: Any, season: Any) -> int:
    """Dump the raw field names a league's data source returned."""
    if not hasattr(league, "inspect_season"):
        print(f"{league.name} has no inspect support", file=sys.stderr)
        return 1

    report = league.inspect_season(
        season, cache_dir=args.cache_dir, refresh=args.refresh
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load_games(
    args: argparse.Namespace,
    league: Any,
    season: Any,
    *,
    as_of: date | None,
    week: int | None,
) -> list[GameRecord]:
    """Load a season's games, passing only the options the league accepts."""
    options: dict[str, Any] = {
        "as_of": as_of,
        "week": week,
        "include_playoffs": args.include_playoffs,
        "cache_dir": args.cache_dir,
        "refresh": args.refresh,
    }
    # --max-margin is an NFL-side knob; the NHL margin is already a small
    # integer scale and has nothing to cap.
    if getattr(args, "max_margin", None) is not None and league.name == "nfl":
        options["max_margin"] = args.max_margin

    return league.load_games(season, **options)


# --------------------------------------------------------------------------- #
# Tables and formatting
# --------------------------------------------------------------------------- #

def ratings_table(result: MasseyResult) -> pd.DataFrame:
    """Build the ranked ratings table.

    Args:
        result: A fitted :class:`~massey.models.MasseyResult`.

    Returns:
        A DataFrame of ``rank``, ``team``, ``rating`` and ``games_played``,
        strongest team first.
    """
    frame = pd.DataFrame(
        {
            "team": list(result.ratings),
            "rating": list(result.ratings.values()),
            "games_played": [result.games_played[team] for team in result.ratings],
        }
    )
    frame = frame.sort_values("rating", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def format_ratings_table(table: pd.DataFrame, *, units: str = "points") -> str:
    """Render the ratings table as fixed-width console text."""
    lines = [f"{'rank':>4}  {'team':<6} {'rating':>9}  {'gp':>3}"]
    lines.append("-" * len(lines[0]))
    for row in table.itertuples(index=False):
        lines.append(f"{row.rank:>4}  {row.team:<6} {row.rating:>+9.3f}  {row.games_played:>3}")
    lines.append("-" * len(lines[0]))
    lines.append(f"ratings are in {units}, relative to an average team")
    return "\n".join(lines)


def _rate_header(
    league: Any,
    season: Any,
    games: Sequence[GameRecord],
    result: MasseyResult,
    args: argparse.Namespace,
) -> str:
    """Build the metadata block printed above the ratings table."""
    neutral = sum(1 for game in games if game.neutral)
    cutoff = []
    if args.as_of is not None:
        cutoff.append(f"games before {args.as_of.isoformat()}")
    if args.week is not None:
        cutoff.append(f"weeks before {args.week}")

    lines = [
        f"{league.name.upper()} {season} Massey ratings",
        f"  games fitted       : {result.n_games} "
        f"({neutral} at neutral sites, {games[0].date} to {games[-1].date})",
        f"  home-field advantage: {result.hfa:+.3f} {league.margin_units} (fitted)",
        f"  ridge penalty       : {result.ridge:g}",
        f"  in-sample margin MAE: {result.residual_mae:.3f} {league.margin_units}",
    ]
    if cutoff:
        lines.append(f"  cutoff              : {', and '.join(cutoff)}")
    if league.name == "nfl":
        from massey.leagues.nfl import NFL_CONFIG

        cap = args.max_margin if args.max_margin is not None else NFL_CONFIG["MAX_MARGIN"]
        lines.append(f"  margin cap          : {cap if cap is not None else 'none'}")
    return "\n".join(lines)


def format_backtest_summary(summary: BacktestSummary, *, units: str = "points") -> str:
    """Render a backtest summary as console text."""
    lines = [
        f"{summary.league.upper()} {summary.season} walk-forward backtest",
        f"  ridge / refit       : {summary.ridge:g} / {summary.refit}",
        f"  games in season     : {summary.n_games}",
        f"  evaluated           : {summary.n_evaluated}",
        f"  skipped (warm-up)   : {summary.n_warmup_skipped}",
        f"  skipped (unrated)   : {summary.n_unrated_skipped}",
        f"  ties excluded       : {summary.n_ties_excluded}",
        f"  mean fitted HFA     : {_fmt(summary.mean_hfa)} {units}",
        "",
        "  winner prediction",
        f"    model accuracy    : {_fmt_pct(summary.accuracy)}",
        f"    home-team baseline: {_fmt_pct(summary.home_baseline_accuracy)}",
        f"    edge over baseline: {_fmt_edge(summary.accuracy, summary.home_baseline_accuracy)}",
        "",
        f"  margin ({units})",
        f"    MAE               : {_fmt(summary.mae)}",
        f"    RMSE              : {_fmt(summary.rmse)}",
        "",
        "  probability calibration",
        f"    log loss          : {_fmt(summary.log_loss)}",
        f"    baseline log loss : {_fmt(summary.baseline_log_loss)} (constant home-win rate)",
        f"    Brier score       : {_fmt(summary.brier)}",
    ]
    return "\n".join(lines)


def _fmt(value: float | None, places: int = 3) -> str:
    """Format an optional float."""
    return "n/a" if value is None else f"{value:.{places}f}"


def _fmt_pct(value: float | None) -> str:
    """Format an optional proportion as a percentage."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_edge(model: float | None, baseline: float | None) -> str:
    """Format the model's percentage-point edge over the baseline."""
    if model is None or baseline is None:
        return "n/a"
    return f"{(model - baseline) * 100:+.1f} pts"


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def _save_ratings(
    table: pd.DataFrame,
    result: MasseyResult,
    league: Any,
    season: Any,
    args: argparse.Namespace,
) -> list[Path]:
    """Write the ratings CSV and a metadata JSON file."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{league.name}_{season}_ratings"

    csv_path = output_dir / f"{stem}.csv"
    table.to_csv(csv_path, index=False)

    meta_path = output_dir / f"{stem}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "league": league.name,
                "season": str(season),
                "margin_units": league.margin_units,
                "home_field_advantage": result.hfa,
                "ridge": result.ridge,
                "games_fitted": result.n_games,
                "in_sample_margin_mae": result.residual_mae,
                "include_playoffs": args.include_playoffs,
                "as_of": args.as_of.isoformat() if args.as_of else None,
                "week": args.week,
                "generated_at": date.today().isoformat(),
            },
            indent=2,
        )
    )
    return [csv_path, meta_path]


def _save_backtest(
    summary: BacktestSummary,
    predictions: Sequence[GamePrediction],
    league: Any,
    season: Any,
    args: argparse.Namespace,
) -> list[Path]:
    """Write per-game predictions and the summary."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{league.name}_{season}_backtest"

    games_path = output_dir / f"{stem}_games.csv"
    pd.DataFrame([asdict(row) for row in predictions]).to_csv(games_path, index=False)

    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    return [games_path, summary_path]


if __name__ == "__main__":
    raise SystemExit(main())
