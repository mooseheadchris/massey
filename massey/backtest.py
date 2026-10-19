"""Walk-forward backtesting of Massey ratings.

For each game we fit ratings on games played *strictly earlier*, predict the
margin and the winner, then compare against what happened. Nothing in a
prediction has seen its own game, so the accuracy numbers are honest
out-of-sample ones.

Two knobs matter for interpreting the output:

* **Warm-up.** Ratings are meaningless after two games, so the first few weeks
  (NFL) or games (NHL) of a season are used for training only and excluded from
  the metrics.
* **Refit cadence.** By default ratings are refit once per calendar date and
  used for every game that day. Games on the same day never see each other, so
  this leaks nothing; it is just much faster than refitting per game. Pass
  ``refit="per-game"`` for the strict version.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import groupby
from typing import Iterable, Sequence

import numpy as np

from massey.leagues.base import GameRecord
from massey.models import DEFAULT_RIDGE, Game, predict_margin, solve_massey

#: Refit cadences accepted by :func:`run_backtest`.
REFIT_CADENCES = ("per-day", "per-game")

#: Reported win probabilities are kept this far away from 0 and 1. A lopsided
#: predicted margin can otherwise underflow the logistic to exactly 0.0 or 1.0,
#: which would make the log loss infinite the first time such a pick was wrong.
PROBABILITY_EPSILON = 1e-6


@dataclass(frozen=True)
class BacktestConfig:
    """Settings for a walk-forward run.

    Attributes:
        ridge: Ridge penalty passed to the solver on every refit.
        skip_weeks: Treat games in weeks at or below this as training only.
            Ignored for leagues without week numbers.
        skip_games: Treat the first this-many games of the season as training
            only.
        refit: ``"per-day"`` or ``"per-game"``; see the module docstring.
        logistic_scale: Starting scale for converting a predicted margin into a
            home win probability, used until ``min_calibration_games``
            out-of-sample predictions have accumulated.
        min_calibration_games: How many out-of-sample predictions to collect
            before fitting the logistic scale from the data.
    """

    ridge: float = DEFAULT_RIDGE
    skip_weeks: int | None = None
    skip_games: int | None = None
    refit: str = "per-day"
    logistic_scale: float = 0.13
    min_calibration_games: int = 50

    def __post_init__(self) -> None:
        if self.refit not in REFIT_CADENCES:
            raise ValueError(f"refit must be one of {REFIT_CADENCES}, got {self.refit!r}")


@dataclass(frozen=True)
class GamePrediction:
    """One out-of-sample prediction and the result it is scored against.

    Attributes:
        game_id: League-native game identifier.
        date: When the game was played.
        week: Week number, where the league has them.
        home: Home team identifier.
        away: Away team identifier.
        neutral: Whether the game was at a neutral site.
        n_training_games: How many games the ratings were fitted on.
        predicted_margin: Predicted home-relative margin.
        actual_margin: Actual home-relative margin.
        home_win_prob: Calibrated probability that the home team won.
        predicted_home_win: Whether the model favoured the home team.
        home_won: Whether the home team won; ``None`` for a tie.
        correct: Whether the pick was right; ``None`` for a tie.
    """

    game_id: str
    date: date
    week: int | None
    home: str
    away: str
    neutral: bool
    n_training_games: int
    predicted_margin: float
    actual_margin: float
    home_win_prob: float
    predicted_home_win: bool
    home_won: bool | None
    correct: bool | None


@dataclass(frozen=True)
class BacktestSummary:
    """Aggregate metrics for a walk-forward run.

    ``None`` appears wherever there was nothing to compute a metric from.

    Attributes:
        league: League key.
        season: Season as given on the command line.
        ridge: Ridge penalty used.
        refit: Refit cadence used.
        n_games: Games available in the season.
        n_evaluated: Games actually predicted and scored.
        n_warmup_skipped: Games held back as warm-up training data.
        n_unrated_skipped: Games skipped because a team had no rating yet.
        n_ties_excluded: Tied games, excluded from winner and probability
            metrics but kept in the margin ones.
        accuracy: Share of winner picks that were right.
        home_baseline_accuracy: Share the "always pick the home team" baseline
            got right, over the same games.
        mae: Mean absolute error of the predicted margin.
        rmse: Root mean squared error of the predicted margin.
        log_loss: Mean negative log likelihood of the calibrated win
            probabilities.
        baseline_log_loss: Same, for a constant predictor set to the home-win
            rate over the evaluated games. A deliberately generous baseline --
            it knows the season's final home-win rate in advance.
        brier: Brier score of the calibrated win probabilities.
        mean_hfa: Mean fitted home-field advantage across refits.
        final_hfa: Home-field advantage from the last refit.
    """

    league: str
    season: str
    ridge: float
    refit: str
    n_games: int
    n_evaluated: int
    n_warmup_skipped: int
    n_unrated_skipped: int
    n_ties_excluded: int
    accuracy: float | None
    home_baseline_accuracy: float | None
    mae: float | None
    rmse: float | None
    log_loss: float | None
    baseline_log_loss: float | None
    brier: float | None
    mean_hfa: float | None
    final_hfa: float | None


def run_backtest(
    games: Sequence[GameRecord],
    *,
    league: str,
    season: str,
    config: BacktestConfig | None = None,
) -> tuple[BacktestSummary, list[GamePrediction]]:
    """Walk forward through a season, predicting each game from its past.

    Args:
        games: The season's completed games. Sorted internally, so the input
            order does not matter.
        league: League key, recorded on the summary.
        season: Season label, recorded on the summary.
        config: Run settings. Defaults to :class:`BacktestConfig`.

    Returns:
        A ``(summary, predictions)`` pair. ``predictions`` holds one row per
        evaluated game, in chronological order.
    """
    config = config or BacktestConfig()
    ordered = sorted(games, key=lambda record: (record.date, record.game_id))

    history: list[Game] = []
    predictions: list[GamePrediction] = []
    calibration: list[tuple[float, bool]] = []
    fitted_hfas: list[float] = []

    warmup_skipped = 0
    unrated_skipped = 0
    position = 0

    for batch in _batches(ordered, config.refit):
        result = solve_massey(history, ridge=config.ridge) if history else None
        if result is not None:
            fitted_hfas.append(result.hfa)
        scale = _logistic_scale(calibration, config)

        for record in batch:
            index = position
            position += 1

            if _is_warmup(record, index, config):
                warmup_skipped += 1
                continue
            if result is None:
                unrated_skipped += 1
                continue

            try:
                predicted = predict_margin(
                    result, record.home, record.away, neutral=record.neutral
                )
            except KeyError:
                # A team with no prior games has no rating, so there is nothing
                # to predict from. Counted, not silently dropped.
                unrated_skipped += 1
                continue

            probability = float(
                np.clip(_sigmoid(scale * predicted), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
            )
            home_won = record.home_won
            predictions.append(
                GamePrediction(
                    game_id=record.game_id,
                    date=record.date,
                    week=record.week,
                    home=record.home,
                    away=record.away,
                    neutral=record.neutral,
                    n_training_games=result.n_games,
                    predicted_margin=predicted,
                    actual_margin=record.margin,
                    home_win_prob=probability,
                    # A dead-even prediction is broken toward the home team,
                    # matching the naive baseline.
                    predicted_home_win=predicted >= 0,
                    home_won=home_won,
                    correct=None if home_won is None else (predicted >= 0) == home_won,
                )
            )
            if home_won is not None:
                calibration.append((predicted, home_won))

        history.extend(record.to_game() for record in batch)

    summary = summarise(
        predictions,
        league=league,
        season=season,
        config=config,
        n_games=len(ordered),
        n_warmup_skipped=warmup_skipped,
        n_unrated_skipped=unrated_skipped,
        fitted_hfas=fitted_hfas,
    )
    return summary, predictions


def summarise(
    predictions: Sequence[GamePrediction],
    *,
    league: str,
    season: str,
    config: BacktestConfig,
    n_games: int,
    n_warmup_skipped: int,
    n_unrated_skipped: int,
    fitted_hfas: Sequence[float],
) -> BacktestSummary:
    """Aggregate per-game predictions into a :class:`BacktestSummary`.

    Args:
        predictions: Scored predictions from :func:`run_backtest`.
        league: League key.
        season: Season label.
        config: The config the run used.
        n_games: Total games available in the season.
        n_warmup_skipped: Games held back as warm-up.
        n_unrated_skipped: Games skipped for want of a rating.
        fitted_hfas: Home-field advantage from each refit.

    Returns:
        The summary.
    """
    decided = [row for row in predictions if row.home_won is not None]

    accuracy = _mean([float(row.correct) for row in decided])
    baseline = _mean([float(row.home_won) for row in decided])

    errors = np.array([row.predicted_margin - row.actual_margin for row in predictions])
    mae = float(np.mean(np.abs(errors))) if errors.size else None
    rmse = float(np.sqrt(np.mean(errors**2))) if errors.size else None

    log_loss = brier = baseline_log_loss = None
    if decided:
        probabilities = np.clip(
            np.array([row.home_win_prob for row in decided]), 1e-12, 1 - 1e-12
        )
        outcomes = np.array([float(row.home_won) for row in decided])
        log_loss = float(
            -np.mean(outcomes * np.log(probabilities) + (1 - outcomes) * np.log(1 - probabilities))
        )
        brier = float(np.mean((probabilities - outcomes) ** 2))

        rate = float(np.clip(outcomes.mean(), 1e-12, 1 - 1e-12))
        baseline_log_loss = float(
            -(rate * np.log(rate) + (1 - rate) * np.log(1 - rate))
        )

    return BacktestSummary(
        league=league,
        season=season,
        ridge=config.ridge,
        refit=config.refit,
        n_games=n_games,
        n_evaluated=len(predictions),
        n_warmup_skipped=n_warmup_skipped,
        n_unrated_skipped=n_unrated_skipped,
        n_ties_excluded=len(predictions) - len(decided),
        accuracy=accuracy,
        home_baseline_accuracy=baseline,
        mae=mae,
        rmse=rmse,
        log_loss=log_loss,
        baseline_log_loss=baseline_log_loss,
        brier=brier,
        mean_hfa=_mean(list(fitted_hfas)),
        final_hfa=float(fitted_hfas[-1]) if fitted_hfas else None,
    )


def _batches(games: Sequence[GameRecord], refit: str) -> Iterable[list[GameRecord]]:
    """Group games into the units the ratings are refit between."""
    if refit == "per-game":
        return ([game] for game in games)
    return (list(group) for _, group in groupby(games, key=lambda record: record.date))


def _is_warmup(record: GameRecord, index: int, config: BacktestConfig) -> bool:
    """Return whether a game is warm-up (training only, never scored)."""
    if config.skip_weeks is not None and record.week is not None:
        if record.week <= config.skip_weeks:
            return True
    if config.skip_games is not None and index < config.skip_games:
        return True
    return False


def _logistic_scale(calibration: Sequence[tuple[float, bool]], config: BacktestConfig) -> float:
    """Fit the margin-to-probability scale on out-of-sample history.

    Minimises log loss over ``p = sigmoid(scale * predicted_margin)`` using the
    predictions already made this run, so the calibration is itself
    walk-forward. Falls back to ``config.logistic_scale`` until there is enough
    history to fit.

    Args:
        calibration: ``(predicted_margin, home_won)`` pairs from earlier games.
        config: The run config, for the fallback scale and the minimum count.

    Returns:
        The logistic scale to use for the next batch of predictions.
    """
    if len(calibration) < config.min_calibration_games:
        return config.logistic_scale

    from scipy.optimize import minimize_scalar

    margins = np.array([margin for margin, _ in calibration])
    outcomes = np.array([float(won) for _, won in calibration])

    def negative_log_likelihood(scale: float) -> float:
        probabilities = np.clip(_sigmoid(scale * margins), 1e-12, 1 - 1e-12)
        return float(
            -np.mean(
                outcomes * np.log(probabilities) + (1 - outcomes) * np.log(1 - probabilities)
            )
        )

    best = minimize_scalar(negative_log_likelihood, bounds=(1e-4, 5.0), method="bounded")
    return float(best.x) if best.success else config.logistic_scale


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(value, dtype=float)))


def _mean(values: list[float]) -> float | None:
    """Mean of ``values``, or ``None`` when empty."""
    return float(np.mean(values)) if values else None
