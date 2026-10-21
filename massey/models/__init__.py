"""Rating models. Currently just the Massey least-squares solver."""

from massey.models.massey import (
    DEFAULT_RIDGE,
    Game,
    MasseyResult,
    carry_over_prior,
    predict_margin,
    solve_massey,
)

__all__ = [
    "DEFAULT_RIDGE",
    "Game",
    "MasseyResult",
    "carry_over_prior",
    "predict_margin",
    "solve_massey",
]
