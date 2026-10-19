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
from typing import Iterable, Sequence

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
    """

    ratings: dict[str, float]
    hfa: float
    games_played: dict[str, int]
    ridge: float
    n_games: int
    residual_mae: float

    @property
    def teams(self) -> list[str]:
        """Team identifiers in the fit, alphabetically."""
        return sorted(self.ratings)


def solve_massey(
    games: Iterable[Game],
    *,
    ridge: float = DEFAULT_RIDGE,
    estimate_hfa: bool = True,
) -> MasseyResult:
    """Fit Massey ratings to a collection of games.

    Args:
        games: Completed games. Order does not matter.
        ridge: Ridge penalty on the team-rating block. ``0.0`` gives an
            unpenalised least-squares fit, which recovers exact ratings on a
            well-connected schedule but can be ill-posed on a sparse one.
        estimate_hfa: When ``False``, no home-field term is fitted and
            ``hfa`` is reported as ``0.0``.

    Returns:
        The fitted :class:`MasseyResult`.

    Raises:
        ValueError: If ``games`` is empty, ``ridge`` is negative, or a game
            lists the same team as both home and away.
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

    params = _solve_constrained_ridge(design, margins, n_teams=n_teams, ridge=ridge)

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
    )


def _solve_constrained_ridge(
    design: np.ndarray,
    margins: np.ndarray,
    *,
    n_teams: int,
    ridge: float,
) -> np.ndarray:
    """Minimise ``||Xb - y||^2 + ridge * ||b_teams||^2`` with the team block
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

    Returns:
        The ``(n_params,)`` parameter vector.
    """
    n_params = design.shape[1]

    penalty = np.zeros(n_params)
    penalty[:n_teams] = ridge

    normal = design.T @ design + np.diag(penalty)
    target = design.T @ margins

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
