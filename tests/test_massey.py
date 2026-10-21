"""Tests for the league-agnostic Massey solver."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from massey.models import (
    Game,
    MasseyResult,
    carry_over_prior,
    predict_margin,
    solve_massey,
)

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


# --------------------------------------------------------------------------- #
# Shrinking toward a prior instead of toward zero
# --------------------------------------------------------------------------- #

def test_prior_does_nothing_without_a_ridge():
    # The prior rides on the ridge penalty, so switching the penalty off
    # switches the prior off with it.
    absurd = {team: 500.0 for team in TRUE_RATINGS}
    with_prior = solve_massey(synthetic_games(), ridge=0.0, prior=absurd)
    without = solve_massey(synthetic_games(), ridge=0.0)

    assert with_prior.ratings == pytest.approx(without.ratings)


def test_prior_pulls_ratings_toward_itself():
    games = synthetic_games()
    plain = solve_massey(games, ridge=20.0)
    # A prior that disagrees with the data: reverse the true order.
    reversed_prior = {team: -rating for team, rating in TRUE_RATINGS.items()}
    pulled = solve_massey(games, ridge=20.0, prior=reversed_prior)

    for team in TRUE_RATINGS:
        assert pulled.ratings[team] < plain.ratings[team] if TRUE_RATINGS[team] > 0 \
            else pulled.ratings[team] > plain.ratings[team]


def test_a_heavy_ridge_lands_close_to_the_prior():
    # With little weight on the data, the fit should sit near the prior. This
    # is the behaviour that makes early-season ratings sane.
    prior = {"Aces": 6.0, "Bears": 2.0, "Cubs": -2.0, "Ducks": -6.0}
    result = solve_massey([Game("Aces", "Ducks", 1.0)], ridge=500.0, prior=prior)

    for team in ("Aces", "Ducks"):
        assert result.ratings[team] == pytest.approx(prior[team], abs=0.3)


def test_evidence_outweighs_the_prior_as_games_accumulate():
    # One game barely moves off the prior; a full double round-robin should
    # leave it behind.
    prior = {team: -rating for team, rating in TRUE_RATINGS.items()}
    sparse = solve_massey([Game("Aces", "Ducks", 20.0)], ridge=5.0, prior=prior)
    dense = solve_massey(synthetic_games(), ridge=5.0, prior=prior)

    # The prior says Aces is the worst team; plenty of evidence says otherwise.
    assert sparse.ratings["Aces"] < 0
    assert dense.ratings["Aces"] > 0


def test_prior_leaves_hfa_free():
    games = synthetic_games()
    prior = {team: -rating for team, rating in TRUE_RATINGS.items()}

    without = solve_massey(games, ridge=10.0)
    with_prior = solve_massey(games, ridge=10.0, prior=prior)

    # HFA carries no ridge penalty, so the prior cannot reach it.
    assert with_prior.hfa == pytest.approx(without.hfa)


def test_ratings_still_sum_to_zero_under_a_lopsided_prior():
    # A prior that does not itself sum to zero must not break the constraint.
    result = solve_massey(synthetic_games(), ridge=10.0, prior={"Aces": 100.0})
    assert sum(result.ratings.values()) == pytest.approx(0.0, abs=1e-9)


def test_a_team_missing_from_the_prior_is_treated_as_average():
    # An expansion team has no history, and "average" is the right assumption.
    prior = {"Aces": 9.0}
    result = solve_massey(synthetic_games(), ridge=25.0, prior=prior)

    assert result.ratings["Aces"] > TRUE_RATINGS["Aces"] * 0.2
    assert result.prior_teams == 1


def test_a_prior_team_that_never_played_is_ignored():
    # A relocated or folded club in last season's ratings must not create a
    # column, or ask the API for a schedule that does not exist.
    prior = {**TRUE_RATINGS, "Gone": 50.0}
    result = solve_massey(synthetic_games(), ridge=1.0, prior=prior)

    assert "Gone" not in result.ratings
    assert result.prior_teams == len(TRUE_RATINGS)


def test_empty_prior_is_the_same_as_none():
    games = synthetic_games()
    assert solve_massey(games, prior={}).ratings == pytest.approx(
        solve_massey(games).ratings
    )
    assert solve_massey(games, prior={}).prior_teams == 0


def test_non_finite_prior_is_rejected():
    # A NaN survives multiplication by a zero penalty and would spread through
    # the entire solve, so it has to be caught at the door.
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="not finite"):
            solve_massey(synthetic_games(), prior={"Aces": bad})


def test_carry_over_prior_regresses_toward_the_mean():
    fitted = solve_massey(synthetic_games(), ridge=0.0)
    prior = carry_over_prior(fitted, 0.3)

    for team, rating in fitted.ratings.items():
        assert prior[team] == pytest.approx(0.3 * rating)
    # Scaling preserves sum-to-zero, so the prior agrees with the constraint.
    assert sum(prior.values()) == pytest.approx(0.0, abs=1e-9)


def test_carry_over_prior_at_zero_retention_forgets_last_season():
    fitted = solve_massey(synthetic_games(), ridge=0.0)
    prior = carry_over_prior(fitted, 0.0)

    assert all(value == 0.0 for value in prior.values())
    # Which makes it equivalent to an ordinary fit.
    assert solve_massey(synthetic_games(), prior=prior).ratings == pytest.approx(
        solve_massey(synthetic_games()).ratings
    )


def test_carry_over_prior_at_full_retention_keeps_last_season():
    fitted = solve_massey(synthetic_games(), ridge=0.0)
    assert carry_over_prior(fitted, 1.0) == pytest.approx(fitted.ratings)


@pytest.mark.parametrize("retention", [-0.1, 1.5])
def test_carry_over_prior_rejects_a_nonsense_retention(retention):
    fitted = solve_massey(synthetic_games())
    with pytest.raises(ValueError, match="between 0 and 1"):
        carry_over_prior(fitted, retention)


def _ranking(result: MasseyResult) -> list[str]:
    return sorted(result.ratings, key=lambda team: -result.ratings[team])
