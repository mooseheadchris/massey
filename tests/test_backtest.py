"""Tests for the walk-forward backtest.

The headline test builds a synthetic league where one team is genuinely the
best and checks the backtest recovers that out of sample, beating the
home-team baseline. The rest pin down the no-leakage guarantees and the
bookkeeping.
"""

from __future__ import annotations

import itertools
from datetime import date, timedelta

import pytest

from massey.backtest import BacktestConfig, run_backtest
from massey.leagues.base import GameRecord

TEAM_STRENGTH = {"AAA": 12.0, "BBB": 4.0, "CCC": -4.0, "DDD": -12.0}
TRUE_HFA = 2.0
START = date(2025, 9, 7)


def build_season(
    *,
    rounds: int = 6,
    noise: float = 0.0,
    neutral_every: int | None = None,
) -> list[GameRecord]:
    """Build a synthetic season whose margins follow the Massey model.

    Args:
        rounds: How many times the full home-and-away round-robin repeats.
        noise: Amplitude of a deterministic, alternating wobble added to each
            margin, so the fit is not perfect.
        neutral_every: Make every nth game a neutral-site game.

    Returns:
        Games in chronological order, one calendar day apart.
    """
    games: list[GameRecord] = []
    pairings = list(itertools.permutations(TEAM_STRENGTH, 2))

    for round_number in range(rounds):
        for pair_number, (home, away) in enumerate(pairings):
            index = len(games)
            neutral = neutral_every is not None and index % neutral_every == 0
            wobble = noise * (1 if index % 2 == 0 else -1)
            margin = (
                TEAM_STRENGTH[home]
                - TEAM_STRENGTH[away]
                + (0.0 if neutral else TRUE_HFA)
                + wobble
            )
            # Keep the raw scores consistent with the margin so home_won agrees.
            home_score = 20 + round(margin)
            games.append(
                GameRecord(
                    game_id=f"g{index:03d}",
                    date=START + timedelta(days=index),
                    home=home,
                    away=away,
                    margin=margin,
                    neutral=neutral,
                    home_score=home_score,
                    away_score=20,
                    week=1 + index // len(pairings),
                    outcome="REG",
                )
            )
    return games


def run(games, **overrides):
    """Run a backtest with test-friendly defaults."""
    config = BacktestConfig(
        **{"skip_games": 12, "min_calibration_games": 10, **overrides}
    )
    return run_backtest(games, league="test", season="2025", config=config)


# --------------------------------------------------------------------------- #
# Headline behaviour
# --------------------------------------------------------------------------- #

def test_recovers_a_knowable_league_out_of_sample():
    summary, predictions = run(build_season(noise=3.0))

    assert summary.n_evaluated == len(predictions) > 0
    # With real signal in the data the model should be close to perfect, and
    # must at least beat "always pick the home team".
    assert summary.accuracy > 0.9
    assert summary.accuracy > summary.home_baseline_accuracy
    assert summary.mae < 5.0


def test_a_coin_flip_league_cannot_be_beaten():
    # Every team identical, so the only real signal is home advantage. The
    # model should match the home baseline, not beat it.
    flat = [
        GameRecord(
            game_id=game.game_id,
            date=game.date,
            home=game.home,
            away=game.away,
            margin=TRUE_HFA,
            neutral=False,
            home_score=23,
            away_score=20,
            week=game.week,
        )
        for game in build_season()
    ]
    summary, _ = run(flat)

    assert summary.accuracy == pytest.approx(1.0)
    assert summary.home_baseline_accuracy == pytest.approx(1.0)


def test_hfa_is_recovered_across_refits():
    summary, _ = run(build_season())
    assert summary.final_hfa == pytest.approx(TRUE_HFA, abs=0.5)


# --------------------------------------------------------------------------- #
# No leakage
# --------------------------------------------------------------------------- #

def test_training_set_grows_and_never_includes_the_predicted_game():
    _, predictions = run(build_season(), refit="per-game")

    counts = [row.n_training_games for row in predictions]
    assert counts == sorted(counts)
    # Per-game refit: the nth evaluated game trains on everything before it,
    # which is its own index in the season.
    assert counts[0] == 12  # the skipped warm-up games
    assert all(second - first == 1 for first, second in zip(counts, counts[1:]))


def test_per_day_refit_shares_one_fit_within_a_day():
    games = build_season()
    # Put the first three post-warm-up games on the same date.
    same_day = [
        GameRecord(**{**game.__dict__, "date": START + timedelta(days=100)})
        if 12 <= index < 15
        else game
        for index, game in enumerate(games)
    ]
    _, predictions = run(same_day, refit="per-day")

    on_that_day = [row for row in predictions if row.date == START + timedelta(days=100)]
    assert len(on_that_day) == 3
    # One fit for the day: all three trained on exactly the same history.
    assert len({row.n_training_games for row in on_that_day}) == 1


def test_per_game_and_per_day_agree_when_no_two_games_share_a_day():
    games = build_season()
    per_day, _ = run(games, refit="per-day")
    per_game, _ = run(games, refit="per-game")
    assert per_day.accuracy == per_game.accuracy
    assert per_day.mae == pytest.approx(per_game.mae)


# --------------------------------------------------------------------------- #
# Warm-up and skipping
# --------------------------------------------------------------------------- #

def test_skip_games_holds_back_the_opening_games():
    summary, predictions = run(build_season(), skip_games=20)

    assert summary.n_warmup_skipped == 20
    assert summary.n_evaluated == summary.n_games - 20
    assert predictions[0].game_id == "g020"


def test_skip_weeks_holds_back_the_opening_weeks():
    summary, predictions = run(build_season(), skip_games=None, skip_weeks=2)

    assert all(row.week > 2 for row in predictions)
    assert summary.n_warmup_skipped == 24  # two rounds of 12 games


def test_no_warmup_still_skips_the_very_first_game():
    # The first game of a season has no history to fit, so it cannot be
    # predicted. It is counted as unrated rather than quietly dropped.
    summary, _ = run(build_season(), skip_games=None, skip_weeks=None)
    assert summary.n_unrated_skipped >= 1
    assert summary.n_evaluated + summary.n_unrated_skipped == summary.n_games


def test_unrated_teams_are_counted_not_dropped():
    # A team that appears only in the final game has no rating when that game
    # is predicted.
    games = build_season(rounds=2)
    games.append(
        GameRecord(
            game_id="g999",
            date=START + timedelta(days=500),
            home="NEW",
            away="AAA",
            margin=1.0,
            neutral=False,
            home_score=21,
            away_score=20,
            week=99,
        )
    )
    summary, predictions = run(games, skip_games=6)

    assert "g999" not in {row.game_id for row in predictions}
    assert summary.n_unrated_skipped == 1


# --------------------------------------------------------------------------- #
# Metrics and bookkeeping
# --------------------------------------------------------------------------- #

def test_ties_are_excluded_from_winner_metrics_but_not_margin_ones():
    games = build_season()
    games[20] = GameRecord(
        game_id=games[20].game_id,
        date=games[20].date,
        home=games[20].home,
        away=games[20].away,
        margin=0.0,
        neutral=False,
        home_score=20,
        away_score=20,
        week=games[20].week,
    )
    summary, predictions = run(games)

    assert summary.n_ties_excluded == 1
    tie = next(row for row in predictions if row.game_id == games[20].game_id)
    assert tie.home_won is None and tie.correct is None
    # The tie is still in the margin error, so the counts differ by one.
    decided = [row for row in predictions if row.home_won is not None]
    assert len(decided) == summary.n_evaluated - 1


def test_probabilities_are_calibrated_better_than_a_constant():
    summary, _ = run(build_season(noise=6.0))
    assert summary.log_loss < summary.baseline_log_loss
    assert 0.0 < summary.brier < 0.25


def test_probabilities_track_the_predicted_margin():
    _, predictions = run(build_season(noise=3.0))

    for row in predictions:
        assert 0.0 < row.home_win_prob < 1.0
        assert (row.home_win_prob > 0.5) == (row.predicted_margin > 0)


def test_summary_counts_add_up():
    summary, predictions = run(build_season())
    assert (
        summary.n_evaluated + summary.n_warmup_skipped + summary.n_unrated_skipped
        == summary.n_games
    )
    assert summary.n_evaluated == len(predictions)


def test_summary_records_the_settings():
    summary, _ = run(build_season(), ridge=2.5, refit="per-game")
    assert summary.ridge == 2.5
    assert summary.refit == "per-game"
    assert summary.league == "test"
    assert summary.season == "2025"


def test_input_order_does_not_matter():
    games = build_season()
    forward, _ = run(games)
    backward, _ = run(list(reversed(games)))
    assert forward == backward


def test_neutral_games_are_predicted_without_hfa():
    _, predictions = run(build_season(neutral_every=7, noise=1.0))

    neutral = [row for row in predictions if row.neutral]
    assert neutral, "expected some neutral-site games"
    # Margins were generated with no HFA at neutral sites, so the errors there
    # should not be systematically biased toward the home team.
    bias = sum(row.predicted_margin - row.actual_margin for row in neutral) / len(neutral)
    assert abs(bias) < 1.0


def test_empty_season_summarises_without_crashing():
    summary, predictions = run_backtest([], league="test", season="2025")
    assert predictions == []
    assert summary.n_evaluated == 0
    assert summary.accuracy is None
    assert summary.mae is None
    assert summary.log_loss is None


def test_rejects_an_unknown_refit_cadence():
    with pytest.raises(ValueError, match="refit must be one of"):
        BacktestConfig(refit="hourly")
