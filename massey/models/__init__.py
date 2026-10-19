"""Rating models. Currently just the Massey least-squares solver."""

from massey.models.massey import (
    DEFAULT_RIDGE,
    Game,
    MasseyResult,
    predict_margin,
    solve_massey,
)

__all__ = [
    "DEFAULT_RIDGE",
    "Game",
    "MasseyResult",
    "predict_margin",
    "solve_massey",
]
