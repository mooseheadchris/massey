"""Builds the standalone HTML pages for viewing ratings in a browser.

The pages are self-contained: the season's data is embedded as JSON, there are
no external scripts or fonts, and nothing is fetched at runtime. That is
deliberate -- a page opened by double-clicking has a ``null`` origin, so any
cross-origin request it tried to make would be refused, and the rating model
itself is Python. Everything the page needs is computed here and baked in.

Several seasons can be embedded in one page; the viewer switches between them
from a dropdown. A page can also read another season's JSON from disk through a
file picker, which needs no server either.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from massey.backtest import BacktestConfig, GamePrediction, run_backtest
from massey.leagues.base import GameRecord
from massey.models import DEFAULT_RIDGE, solve_massey

#: The page template, alongside this module.
TEMPLATE_PATH = Path(__file__).parent / "templates" / "league_page.html"

#: Replaced with the embedded JSON payload when the page is rendered.
PAYLOAD_PLACEHOLDER = "__MASSEY_PAYLOAD__"


def build_season_payload(
    league: Any,
    season: Any,
    games: Sequence[GameRecord],
    *,
    ridge: float = DEFAULT_RIDGE,
    backtest_config: BacktestConfig | None = None,
    prior: Mapping[str, float] | None = None,
    prior_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble everything the page shows for one season.

    Args:
        league: The league adapter module.
        season: Season as given on the command line.
        games: The season's completed games.
        ridge: Ridge penalty for the full-season fit.
        backtest_config: When given, a walk-forward backtest is run and its
            metrics and per-game predictions are included.
        prior: Ratings to shrink toward instead of zero.
        prior_meta: What that prior was built from, for the page to display.

    Returns:
        A JSON-serialisable payload.

    Raises:
        ValueError: If ``games`` is empty.
    """
    if not games:
        raise ValueError(f"no completed {league.name.upper()} games for {season}")

    ordered = sorted(games, key=lambda record: (record.date, record.game_id))
    result = solve_massey(
        [record.to_game() for record in ordered], ridge=ridge, prior=prior
    )

    summary = None
    predictions: list[GamePrediction] = []
    if backtest_config is not None:
        summary, predictions = run_backtest(
            ordered, league=league.name, season=str(season), config=backtest_config
        )

    by_game = {row.game_id: row for row in predictions}
    records = team_records(ordered, league.name)

    scale = (
        summary.final_logistic_scale
        if summary is not None
        else league.default_logistic_scale
    )

    return {
        "league": league.name,
        "league_label": league.name.upper(),
        "season": str(season),
        "margin_units": league.margin_units,
        "meta": {
            "games_fitted": result.n_games,
            "neutral_games": sum(1 for record in ordered if record.neutral),
            "first_date": ordered[0].date.isoformat(),
            "last_date": ordered[-1].date.isoformat(),
            "hfa": result.hfa,
            "ridge": result.ridge,
            "in_sample_mae": result.residual_mae,
            "logistic_scale": scale,
            "prior_season": (prior_meta or {}).get("season"),
            "prior_retention": (prior_meta or {}).get("retention"),
            "prior_teams": result.prior_teams,
            "generated_at": date.today().isoformat(),
        },
        "teams": [
            {
                "rank": rank,
                "team": team,
                "rating": result.ratings[team],
                "games_played": result.games_played[team],
                **records[team],
            }
            for rank, team in enumerate(
                sorted(result.ratings, key=lambda name: -result.ratings[name]), start=1
            )
        ],
        "games": [
            {
                "game_id": record.game_id,
                "date": record.date.isoformat(),
                "week": record.week,
                "home": record.home,
                "away": record.away,
                "home_score": record.home_score,
                "away_score": record.away_score,
                "margin": record.margin,
                "neutral": record.neutral,
                "outcome": record.outcome,
                "home_won": record.home_won,
                "predicted_margin": _rounded(by_game.get(record.game_id), "predicted_margin"),
                "home_win_prob": _rounded(by_game.get(record.game_id), "home_win_prob", 4),
                "correct": by_game[record.game_id].correct
                if record.game_id in by_game
                else None,
            }
            for record in ordered
        ],
        "backtest": asdict(summary) if summary is not None else None,
    }


def team_records(games: Sequence[GameRecord], league_name: str) -> dict[str, dict[str, Any]]:
    """Tally each team's win-loss record in that league's own terms.

    The NFL gets ties and points for and against. The NHL gets overtime losses
    broken out and its real standings points (2 for a win, 1 for an overtime or
    shootout loss), which is worth showing next to the rating: where the two
    disagree is where the model thinks the standings are flattering a team.

    Args:
        games: Completed games.
        league_name: League key, which decides the extra columns.

    Returns:
        Team identifier -> record fields.
    """
    records: dict[str, dict[str, Any]] = {}

    def blank() -> dict[str, Any]:
        base: dict[str, Any] = {"wins": 0, "losses": 0, "scored": 0, "allowed": 0}
        if league_name == "nhl":
            base["ot_losses"] = 0
        else:
            base["ties"] = 0
        return base

    for game in games:
        for team in (game.home, game.away):
            records.setdefault(team, blank())

        home = records[game.home]
        away = records[game.away]

        if game.home_score is not None and game.away_score is not None:
            home["scored"] += game.home_score
            home["allowed"] += game.away_score
            away["scored"] += game.away_score
            away["allowed"] += game.home_score

        if game.home_won is None:
            home["ties"] += 1
            away["ties"] += 1
            continue

        winner, loser = (home, away) if game.home_won else (away, home)
        winner["wins"] += 1
        # An NHL team that loses past regulation still banks a standings point,
        # so those losses are counted separately.
        if league_name == "nhl" and game.outcome in ("OT", "SO"):
            loser["ot_losses"] += 1
        else:
            loser["losses"] += 1

    if league_name == "nhl":
        for record in records.values():
            record["standings_points"] = 2 * record["wins"] + record["ot_losses"]

    return records


def render_page(payloads: Sequence[dict[str, Any]], destination: Path) -> Path:
    """Write a self-contained HTML page embedding one or more seasons.

    Args:
        payloads: Season payloads from :func:`build_season_payload`, in the
            order they should appear in the season dropdown.
        destination: Path of the HTML file to write.

    Returns:
        The path written.

    Raises:
        ValueError: If ``payloads`` is empty or mixes leagues.
    """
    if not payloads:
        raise ValueError("need at least one season to render a page")

    leagues = {payload["league"] for payload in payloads}
    if len(leagues) > 1:
        raise ValueError(f"cannot mix leagues on one page: {sorted(leagues)}")

    template = TEMPLATE_PATH.read_text()
    if PAYLOAD_PLACEHOLDER not in template:
        raise ValueError(f"{TEMPLATE_PATH} is missing the {PAYLOAD_PLACEHOLDER} placeholder")

    document = template.replace(
        PAYLOAD_PLACEHOLDER, _embeddable_json({"seasons": list(payloads)})
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)
    return destination


def _embeddable_json(data: Any) -> str:
    """Serialise data for embedding inside a ``<script>`` tag.

    Escapes the characters that could otherwise end the enclosing tag early or
    be read as a line terminator by the JSON parser.

    Args:
        data: Any JSON-serialisable value.

    Returns:
        The JSON text, safe to place inside a script element.
    """
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def _rounded(row: GamePrediction | None, field: str, places: int = 3) -> float | None:
    """Round a prediction field, or return ``None`` when the game was not scored."""
    if row is None:
        return None
    return round(getattr(row, field), places)
