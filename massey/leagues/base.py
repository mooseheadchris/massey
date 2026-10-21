"""Shared types for league adapters.

A league adapter's only job is to turn a season's results into a list of
:class:`GameRecord` rows. Everything downstream -- the solver, the backtest,
the CLI -- works off those rows and never imports a league module directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

from massey.models import Game


@dataclass(frozen=True)
class GameRecord:
    """A completed game with everything the pipeline needs.

    This is :class:`~massey.models.Game` plus the bookkeeping the backtest and
    the reports need: when it happened, what the raw result was, and how it
    ended.

    Attributes:
        game_id: League-native game identifier.
        date: Calendar date the game was played.
        home: Home (or first-listed) team identifier.
        away: Away (or second-listed) team identifier.
        margin: Home-relative margin in the league's rating units. Positive
            means the home team did better.
        neutral: ``True`` for neutral-site games, which get no HFA term.
        home_score: Raw home score, kept for reporting.
        away_score: Raw away score, kept for reporting.
        week: Week number where the league has them (NFL), else ``None``.
        outcome: How the game ended -- ``"REG"``, ``"OT"`` or ``"SO"`` for the
            NHL; ``"REG"`` or ``"OT"`` for the NFL.
    """

    game_id: str
    date: date
    home: str
    away: str
    margin: float
    neutral: bool
    home_score: int | None = None
    away_score: int | None = None
    week: int | None = None
    outcome: str | None = None

    def to_game(self) -> Game:
        """Reduce to the solver's input row."""
        return Game(home=self.home, away=self.away, margin=self.margin, neutral=self.neutral)

    @property
    def home_won(self) -> bool | None:
        """``True``/``False`` for a home win/loss, ``None`` for a tie.

        Decided on the raw scores when they are available and on the margin
        otherwise. Only the NFL can return ``None`` -- NHL games are always
        settled.
        """
        if self.home_score is not None and self.away_score is not None:
            if self.home_score == self.away_score:
                return None
            return self.home_score > self.away_score
        if self.margin == 0:
            return None
        return self.margin > 0


@runtime_checkable
class LeagueAdapter(Protocol):
    """What the CLI and backtest require of a league module."""

    #: Short lowercase league key, e.g. ``"nfl"``.
    name: str
    #: Units the margin is expressed in, for report headers.
    margin_units: str
    #: Backtest warm-up in weeks, or ``None`` when the league has no weeks.
    default_skip_weeks: int | None
    #: Backtest warm-up in games, or ``None``.
    default_skip_games: int | None
    #: Starting logistic scale for win probabilities, used until the backtest
    #: has enough out-of-sample predictions to fit its own.
    default_logistic_scale: float
    #: Ridge penalty used when the command line does not override it.
    default_ridge: float
    #: Fraction of the previous season's ratings to shrink toward by default.
    #: ``0.0`` switches the carry-over prior off.
    default_prior_retention: float

    def current_season(self) -> int:
        """Return the season that is current today, in CLI form."""

    def previous_season(self, season: int | str) -> int:
        """Return the season before ``season``, for a carry-over prior."""

    def load_games(
        self,
        season: int,
        *,
        as_of: date | None = None,
        week: int | None = None,
        include_playoffs: bool = False,
        cache_dir: Path | str | None = None,
        refresh: bool = False,
    ) -> list[GameRecord]:
        """Load completed games for a season, oldest first."""


def filter_by_cutoff(
    games: list[GameRecord],
    *,
    as_of: date | None = None,
    week: int | None = None,
) -> list[GameRecord]:
    """Apply the "as of" cutoffs shared by every league.

    Both cutoffs are exclusive, which gives you the ratings a forecaster would
    have had *going into* that date or week rather than ones that peek at the
    results they are about to predict.

    Args:
        games: Games to filter.
        as_of: Keep only games played strictly before this date.
        week: Keep only games from weeks strictly before this week. Ignored for
            games with no week number.

    Returns:
        The surviving games, in the order they came in.
    """
    kept = games
    if as_of is not None:
        kept = [game for game in kept if game.date < as_of]
    if week is not None:
        kept = [game for game in kept if game.week is None or game.week < week]
    return kept
