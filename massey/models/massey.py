"""League-agnostic Massey least-squares rating solver.

Each game contributes one equation::

    rating_home - rating_away + hfa = margin

for games at the home team's venue, and::

    rating_home - rating_away = margin

for neutral-site games (no home-field term). The system is solved as a ridge
regression with the ratings constrained to sum to zero. The home-field
advantage (HFA) is a single free parameter estimated from the data and is
deliberately *not* penalised by the ridge term -- shrinking HFA toward zero
would bias the one quantity we are trying to measure.

This module knows nothing about football or hockey. Both leagues convert their
own results into :class:`Game` rows and share the solver, so a new league (or a
new margin definition) only needs a converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

#: Default ridge penalty applied to the team-rating block. Modest on purpose:
#: enough to keep early-season ratings from exploding when teams have played
#: only a game or two, small enough to barely move a full-season fit.
DEFAULT_RIDGE: float = 1.0


@dataclass(frozen=True)
class Game:
    """One completed game, reduced to what the solver needs.

    Attributes:
        home: Identifier for the home (or first-listed) team.
        away: Identifier for the away (or second-listed) team.
        margin: Home-relative margin. Positive means the home team did better.
            The units are the league's choice -- points for the NFL,
            outcome-based standings-style points for the NHL.
        neutral: ``True`` for games at a neutral site, which get no HFA term.
    """

    home: str
    away: str
    margin: float
    neutral: bool = False


@dataclass(frozen=True)
class MasseyResult:
    """Output of a single Massey fit.

    Attributes:
        ratings: Team identifier -> rating. Ratings sum to zero, so a rating is
            read as "points better than an average team".
        hfa: Fitted home-field advantage, in the same units as the margins.
            ``0.0`` when there was nothing to estimate it from (every game
            neutral) or when HFA estimation was switched off.
        games_played: Team identifier -> number of games in the fit.
        ridge: The ridge penalty that was used.
        n_games: Number of games in the fit.
        residual_mae: In-sample mean absolute error of the fitted margins. A
            sanity check on the fit, not an out-of-sample metric -- use
            :mod:`massey.backtest` for that.
        prior_teams: How many teams the ridge was pulled toward a prior rating
            for, rather than toward zero. ``0`` for an ordinary fit.
    """

    ratings: dict[str, float]
    hfa: float
    games_played: dict[str, int]
    ridge: float
    n_games: int
    residual_mae: float
    prior_teams: int = 0

    @property
    def teams(self) -> list[str]:
        """Team identifiers in the fit, alphabetically."""
        return sorted(self.ratings)


def solve_massey(
    games: Iterable[Game],
    *,
    ridge: float = DEFAULT_RIDGE,
    estimate_hfa: bool = True,
    prior: Mapping[str, float] | None = None,
) -> MasseyResult:
    """Fit Massey ratings to a collection of games.

    Args:
        games: Completed games. Order does not matter.
        ridge: Ridge penalty on the team-rating block. ``0.0`` gives an
            unpenalised least-squares fit, which recovers exact ratings on a
            well-connected schedule but can be ill-posed on a sparse one.
        estimate_hfa: When ``False``, no home-field term is fitted and
            ``hfa`` is reported as ``0.0``.
        prior: Ratings to shrink toward instead of zero, usually last season's
            regressed toward the mean -- see :func:`carry_over_prior`. A team
            missing from the prior is shrunk toward zero, which for an
            expansion team is exactly right. A team in the prior that played no
            games is ignored. Has no effect when ``ridge`` is ``0``, since the
            penalty it rides on is then switched off.

    Returns:
        The fitted :class:`MasseyResult`.

    Raises:
        ValueError: If ``games`` is empty, ``ridge`` is negative, a game lists
            the same team as both home and away, or the prior holds a value
            that is not finite.
    """
    if ridge < 0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")

    rows: Sequence[Game] = list(games)
    if not rows:
        raise ValueError("cannot fit ratings with no games")

    for game in rows:
        if game.home == game.away:
            raise ValueError(f"game lists {game.home!r} as both home and away")

    teams = sorted({team for game in rows for team in (game.home, game.away)})
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    # Only fit an HFA term if at least one game actually had a home venue;
    # otherwise its column would be all zeros and the system singular.
    has_hfa = estimate_hfa and any(not game.neutral for game in rows)
    n_params = n_teams + (1 if has_hfa else 0)
    hfa_col = n_teams  # only meaningful when has_hfa

    design = np.zeros((len(rows), n_params))
    margins = np.zeros(len(rows))
    for i, game in enumerate(rows):
        design[i, index[game.home]] = 1.0
        design[i, index[game.away]] = -1.0
        if has_hfa and not game.neutral:
            design[i, hfa_col] = 1.0
        margins[i] = game.margin

    prior_vector, prior_teams = _prior_vector(prior, index, n_params)
    params = _solve_constrained_ridge(
        design, margins, n_teams=n_teams, ridge=ridge, prior=prior_vector
    )

    fitted = design @ params
    residual_mae = float(np.mean(np.abs(margins - fitted)))

    games_played: dict[str, int] = {team: 0 for team in teams}
    for game in rows:
        games_played[game.home] += 1
        games_played[game.away] += 1

    return MasseyResult(
        ratings={team: float(params[index[team]]) for team in teams},
        hfa=float(params[hfa_col]) if has_hfa else 0.0,
        games_played=games_played,
        ridge=float(ridge),
        n_games=len(rows),
        residual_mae=residual_mae,
        prior_teams=prior_teams,
    )


def _prior_vector(
    prior: Mapping[str, float] | None,
    index: Mapping[str, int],
    n_params: int,
) -> tuple[np.ndarray | None, int]:
    """Lay a team-keyed prior out in parameter order.

    The caller supplies a mapping because it cannot know the solver's column
    order, which comes from sorting the teams found in the games.

    Args:
        prior: Team -> prior rating, or ``None``.
        index: Team -> column, from the fit being set up.
        n_params: Width of the parameter vector, including any HFA column.

    Returns:
        ``(vector, teams_matched)``. The vector is ``None`` when there is no
        prior to apply. The home-field column is left at zero, which is what
        keeps HFA freely estimated: its ridge penalty is zero, so whatever sits
        in that slot is multiplied away.

    Raises:
        ValueError: If a prior rating is not finite. A NaN would survive the
            multiply by a zero penalty and spread through the whole solve.
    """
    if not prior:
        return None, 0

    vector = np.zeros(n_params)
    matched = 0
    for team, rating in prior.items():
        column = index.get(team)
        if column is None:
            continue  # in the prior but played no games: a folded or moved club
        value = float(rating)
        if not np.isfinite(value):
            raise ValueError(f"prior rating for {team!r} is not finite: {rating!r}")
        vector[column] = value
        matched += 1

    return (vector, matched) if matched else (None, 0)


def carry_over_prior(
    previous: MasseyResult,
    retention: float,
) -> dict[str, float]:
    """Regress a finished season's ratings toward the mean for use as a prior.

    Teams change over an offseason, so last season's ratings are evidence about
    this one rather than a starting point to be taken at face value.
    ``retention`` says how much of the gap from average survives: ``0.0``
    discards last season entirely (identical to no prior at all), ``1.0``
    assumes nothing changed. Somewhere around a third is the usual
    neighbourhood for the NFL; tune it against the backtest rather than taking
    that on faith.

    Because the ratings sum to zero, scaling them keeps that true, so the prior
    never fights the solver's sum-to-zero constraint.

    Args:
        previous: A fit of the preceding season.
        retention: Fraction of each rating to keep, from 0 to 1.

    Returns:
        Team -> prior rating, ready to pass to :func:`solve_massey`.

    Raises:
        ValueError: If ``retention`` is outside ``[0, 1]``.
    """
    if not 0.0 <= retention <= 1.0:
        raise ValueError(f"retention must be between 0 and 1, got {retention}")

    return {team: retention * rating for team, rating in previous.ratings.items()}


def _solve_constrained_ridge(
    design: np.ndarray,
    margins: np.ndarray,
    *,
    n_teams: int,
    ridge: float,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Minimise ``||Xb - y||^2 + ridge * ||b_teams - p||^2`` with the team block
    constrained to sum to zero.

    The sum-to-zero constraint is enforced exactly with a Lagrange multiplier
    rather than approximated with a heavily weighted extra row, which keeps the
    ratings interpretable (they really do sum to zero) and the system
    well-conditioned. Writing ``A = X'X + ridge*D`` and ``c`` for the indicator
    of the team columns, we solve the KKT system::

        [ A   c ] [ b ]   [ X'y ]
        [ c'  0 ] [ u ] = [  0  ]

    Args:
        design: ``(n_games, n_params)`` design matrix.
        margins: ``(n_games,)`` target margins.
        n_teams: Number of leading columns that are team ratings. Any further
            column (the HFA term) is left unpenalised and unconstrained.
        ridge: Ridge penalty for the team block.
        prior: Values to shrink toward instead of zero, in parameter order.
            Shrinking toward ``p`` rather than ``0`` leaves the left-hand side
            untouched and only adds ``ridge * p`` to the target, since
            minimising ``||Xb - y||^2 + L||b - p||^2`` gives
            ``(X'X + L)b = X'y + Lp``.

    Returns:
        The ``(n_params,)`` parameter vector.
    """
    n_params = design.shape[1]

    penalty = np.zeros(n_params)
    penalty[:n_teams] = ridge

    normal = design.T @ design + np.diag(penalty)
    target = design.T @ margins
    if prior is not None:
        # Elementwise, so the HFA column contributes nothing: its penalty is
        # zero, which multiplies away whatever the prior holds there.
        target = target + penalty * prior

    constraint = np.zeros(n_params)
    constraint[:n_teams] = 1.0

    kkt = np.zeros((n_params + 1, n_params + 1))
    kkt[:n_params, :n_params] = normal
    kkt[:n_params, n_params] = constraint
    kkt[n_params, :n_params] = constraint
    rhs = np.concatenate([target, [0.0]])

    try:
        solution = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        # Reachable with ridge=0 on a schedule too sparse to pin down every
        # rating (a disconnected league, or one with no cycles). The minimum
        # norm least-squares solution is the sensible answer there.
        solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)

    return solution[:n_params]


def predict_margin(
    result: MasseyResult,
    home: str,
    away: str,
    *,
    neutral: bool = False,
) -> float:
    """Predict the home-relative margin of a matchup from fitted ratings.

    Args:
        result: A fitted :class:`MasseyResult`.
        home: Home (or first-listed) team identifier.
        away: Away (or second-listed) team identifier.
        neutral: When ``True``, the HFA term is omitted.

    Returns:
        The predicted home-relative margin.

    Raises:
        KeyError: If either team is absent from the fit, which means it has no
            games yet and therefore no rating.
    """
    for team in (home, away):
        if team not in result.ratings:
            raise KeyError(f"{team!r} has no rating in this fit")

    edge = result.ratings[home] - result.ratings[away]
    return edge if neutral else edge + result.hfa
