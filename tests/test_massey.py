"""Tests for the league-agnostic Massey solver."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from massey.models import Game, MasseyResult, predict_margin, solve_massey

# A tiny synthetic league with known ratings (they sum to zero) and a known
# home-field advantage. Every pair meets twice, once at each venue, which makes
# the system well-connected enough that an unpenalised fit must recover the
# true values exactly.
TRUE_RATINGS = {"Aces": 10.0, "Bears": 3.0, "Cubs": -4.0, "Ducks": -9.0}
TRUE_HFA = 2.5


def synthetic_games(
    ratings: dict[str, float] = TRUE_RATINGS,
    hfa: float = TRUE_HFA,
    *,
    neutral: bool = False,
) -> list[Game]:
    """Build a double round-robin whose margins follow the model exactly."""
    games = []
    for home, away in itertools.permutations(ratings, 2):
        margin = ratings[home] - ratings[away] + (0.0 if neutral else hfa)
        games.append(Game(home=home, away=away, margin=margin, neutral=neutral))
    return games


def test_recovers_known_ratings_and_hfa():
    result = solve_massey(synthetic_games(), ridge=0.0)

    for team, expected in TRUE_RATINGS.items():
        assert result.ratings[team] == pytest.approx(expected, abs=1e-9)
    assert result.hfa == pytest.approx(TRUE_HFA, abs=1e-9)
    assert result.residual_mae == pytest.approx(0.0, abs=1e-9)


def test_ratings_sum_to_zero():
    # True ratings offset by a constant: the fit must shift them back so they
    # sum to zero, preserving only the differences.
    shifted = {team: rating + 100.0 for team, rating in TRUE_RATINGS.items()}
    result = solve_massey(synthetic_games(shifted), ridge=0.0)

    assert sum(result.ratings.values()) == pytest.approx(0.0, abs=1e-9)
    for team, expected in TRUE_RATINGS.items():
        assert result.ratings[team] == pytest.approx(expected, abs=1e-9)


def test_sum_to_zero_holds_with_ridge():
    result = solve_massey(synthetic_games(), ridge=5.0)
    assert sum(result.ratings.values()) == pytest.approx(0.0, abs=1e-9)


def test_neutral_games_carry_no_hfa():
    # Margins generated with no home-field effect at all. If the solver applied
    # an HFA term to neutral games it would fit a non-zero value here.
    result = solve_massey(synthetic_games(neutral=True), ridge=0.0)

    assert result.hfa == pytest.approx(0.0, abs=1e-9)
    for team, expected in TRUE_RATINGS.items():
        assert result.ratings[team] == pytest.approx(expected, abs=1e-9)


def test_mixed_neutral_and_home_games_still_recover_hfa():
    games = []
    for home, away in itertools.permutations(TRUE_RATINGS, 2):
        edge = TRUE_RATINGS[home] - TRUE_RATINGS[away]
        games.append(Game(home, away, edge + TRUE_HFA, neutral=False))
        games.append(Game(home, away, edge, neutral=True))

    result = solve_massey(games, ridge=0.0)

    assert result.hfa == pytest.approx(TRUE_HFA, abs=1e-9)
    for team, expected in TRUE_RATINGS.items():
        assert result.ratings[team] == pytest.approx(expected, abs=1e-9)


def test_all_neutral_schedule_reports_zero_hfa():
    games = [Game("A", "B", 4.0, neutral=True), Game("B", "A", -4.0, neutral=True)]
    result = solve_massey(games)
    assert result.hfa == 0.0


def test_estimate_hfa_false_drops_the_term():
    result = solve_massey(synthetic_games(), estimate_hfa=False, ridge=0.0)
    assert result.hfa == 0.0
    # Ratings still recover, because the HFA is a constant on every game and so
    # cancels out of the (sum-to-zero) differences.
    for team, expected in TRUE_RATINGS.items():
        assert result.ratings[team] == pytest.approx(expected, abs=1e-9)


def test_ridge_shrinks_ratings_toward_zero():
    unpenalised = solve_massey(synthetic_games(), ridge=0.0)
    penalised = solve_massey(synthetic_games(), ridge=50.0)

    for team in TRUE_RATINGS:
        assert abs(penalised.ratings[team]) < abs(unpenalised.ratings[team])
    # Ordering must survive the shrinkage.
    assert _ranking(penalised) == _ranking(unpenalised)


def test_single_game_is_absorbed_by_hfa():
    # One game cannot separate team strength from home advantage. Because the
    # ridge penalises ratings but not HFA, the cheapest explanation is "the
    # home team won by the league-average home margin" -- so HFA takes all of
    # it and both ratings stay at zero. This is intended: early in a season the
    # model should not read a single blowout as a huge rating gap.
    result = solve_massey([Game("A", "B", 10.0)], ridge=1.0)

    assert result.hfa == pytest.approx(10.0)
    assert result.ratings == pytest.approx({"A": 0.0, "B": 0.0})


def test_ridge_keeps_a_sparse_league_finite_and_ordered():
    # Home-and-away between two teams, both won by A. HFA alone cannot explain
    # that, so the ratings have to separate the teams.
    result = solve_massey([Game("A", "B", 10.0), Game("B", "A", -10.0)], ridge=1.0)

    assert np.isfinite(list(result.ratings.values())).all()
    assert sum(result.ratings.values()) == pytest.approx(0.0, abs=1e-9)
    assert result.hfa == pytest.approx(0.0)
    assert result.ratings["A"] > 0 > result.ratings["B"]
    # Shrunk toward zero from the unpenalised answer of +/-5.
    assert result.ratings["A"] < 5.0


def test_disconnected_league_is_handled_without_ridge():
    # Two pairs that never meet: unpenalised, this system has no unique answer.
    # The solver must still return finite numbers instead of raising.
    games = [
        Game("A", "B", 6.0),
        Game("B", "A", -6.0),
        Game("C", "D", 6.0),
        Game("D", "C", -6.0),
    ]
    result = solve_massey(games, ridge=0.0)
    assert np.isfinite(list(result.ratings.values())).all()


def test_games_played_counts_both_sides():
    result = solve_massey(synthetic_games())
    # Double round-robin of four teams: six games per team.
    assert result.games_played == {team: 6 for team in TRUE_RATINGS}
    assert result.n_games == 12


def test_metadata_is_reported():
    result = solve_massey(synthetic_games(), ridge=2.0)
    assert result.ridge == 2.0
    assert result.teams == sorted(TRUE_RATINGS)


def test_empty_games_raises():
    with pytest.raises(ValueError, match="no games"):
        solve_massey([])


def test_negative_ridge_raises():
    with pytest.raises(ValueError, match="non-negative"):
        solve_massey(synthetic_games(), ridge=-1.0)


def test_self_game_raises():
    with pytest.raises(ValueError, match="both home and away"):
        solve_massey([Game("A", "A", 0.0)])


def test_predict_margin_uses_hfa_only_at_home():
    result = solve_massey(synthetic_games(), ridge=0.0)

    at_home = predict_margin(result, "Aces", "Ducks")
    at_neutral = predict_margin(result, "Aces", "Ducks", neutral=True)

    assert at_home == pytest.approx(TRUE_RATINGS["Aces"] - TRUE_RATINGS["Ducks"] + TRUE_HFA)
    assert at_neutral == pytest.approx(TRUE_RATINGS["Aces"] - TRUE_RATINGS["Ducks"])
    assert at_home - at_neutral == pytest.approx(TRUE_HFA)


def test_predict_margin_is_antisymmetric_on_neutral_ground():
    result = solve_massey(synthetic_games(), ridge=0.0)
    forward = predict_margin(result, "Bears", "Cubs", neutral=True)
    reverse = predict_margin(result, "Cubs", "Bears", neutral=True)
    assert forward == pytest.approx(-reverse)


def test_predict_margin_unknown_team_raises():
    result = solve_massey(synthetic_games())
    with pytest.raises(KeyError, match="Sharks"):
        predict_margin(result, "Sharks", "Aces")


def _ranking(result: MasseyResult) -> list[str]:
    return sorted(result.ratings, key=lambda team: -result.ratings[team])
