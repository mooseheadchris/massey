"""Tests for the HTML page builder and the export command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from massey.backtest import BacktestConfig
from massey.cli import main
from massey.leagues import nfl, nhl
from massey.web import (
    PAYLOAD_PLACEHOLDER,
    TEMPLATE_PATH,
    build_season_payload,
    render_page,
    team_records,
)


def nfl_games(nfl_cache: Path):
    return nfl.load_games(2025, cache_dir=nfl_cache)


# --------------------------------------------------------------------------- #
# Team records
# --------------------------------------------------------------------------- #

def test_nfl_records_count_wins_losses_and_ties(nfl_cache: Path):
    records = team_records(nfl_games(nfl_cache), "nfl")

    # BBB won at home and won the 40-3 neutral-site blowout, and lost the
    # overtime game away at AAA.
    assert records["BBB"]["wins"] == 2
    assert records["BBB"]["losses"] == 1
    assert records["BBB"]["ties"] == 0
    # AAA tied once.
    assert records["AAA"]["ties"] == 1
    assert "ot_losses" not in records["AAA"]


def test_nfl_records_tally_points_for_and_against(nfl_cache: Path):
    records = team_records(nfl_games(nfl_cache), "nfl")
    # BBB: 24 + 27(as away) + 40 scored; 17 + 30 + 3 allowed.
    assert records["BBB"]["scored"] == 24 + 27 + 40
    assert records["BBB"]["allowed"] == 17 + 30 + 3


def test_nhl_records_break_out_overtime_losses(nhl_cache: Path):
    games = nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    records = team_records(games, "nhl")

    # MTL lost at home in overtime, which is an OT loss rather than a loss.
    assert records["MTL"]["ot_losses"] == 1
    # TOR lost at home in a shootout -- also an overtime loss.
    assert records["TOR"]["ot_losses"] == 1
    assert "ties" not in records["TOR"]


def test_nhl_standings_points_follow_the_real_system(nhl_cache: Path):
    games = nhl.load_games(2025, cache_dir=nhl_cache, teams=["TOR", "MTL"])
    records = team_records(games, "nhl")

    for record in records.values():
        # Two for a win, one for a loss past regulation -- the NHL's own rule,
        # which is deliberately not the symmetric scale the ratings use.
        assert record["standings_points"] == 2 * record["wins"] + record["ot_losses"]


def test_records_cover_every_team_that_played(nfl_cache: Path):
    records = team_records(nfl_games(nfl_cache), "nfl")
    assert set(records) == {"AAA", "BBB", "CCC", "DDD"}


# --------------------------------------------------------------------------- #
# Season payload
# --------------------------------------------------------------------------- #

def test_payload_has_the_pieces_the_page_needs(nfl_cache: Path):
    payload = build_season_payload(nfl, 2025, nfl_games(nfl_cache))

    assert payload["league"] == "nfl"
    assert payload["season"] == "2025"
    assert payload["margin_units"] == "points"
    assert {"games_fitted", "hfa", "ridge", "logistic_scale"} <= set(payload["meta"])
    assert len(payload["teams"]) == 4
    assert len(payload["games"]) == 5


def test_payload_teams_are_ranked_best_first(nfl_cache: Path):
    payload = build_season_payload(nfl, 2025, nfl_games(nfl_cache))

    ratings = [team["rating"] for team in payload["teams"]]
    assert ratings == sorted(ratings, reverse=True)
    assert [team["rank"] for team in payload["teams"]] == [1, 2, 3, 4]
    assert payload["teams"][0]["team"] == "BBB"


def test_payload_is_json_serialisable_without_nan(nfl_cache: Path):
    payload = build_season_payload(nfl, 2025, nfl_games(nfl_cache))
    # allow_nan=False is what the embedder uses; NaN would produce invalid JSON.
    json.dumps(payload, allow_nan=False)


def test_payload_without_a_backtest_has_no_predictions(nfl_cache: Path):
    payload = build_season_payload(nfl, 2025, nfl_games(nfl_cache))

    assert payload["backtest"] is None
    assert all(game["predicted_margin"] is None for game in payload["games"])
    # Falls back to the league's starting scale so the predictor still works.
    assert payload["meta"]["logistic_scale"] == nfl.default_logistic_scale


def test_payload_with_a_backtest_joins_predictions_onto_games(nfl_cache: Path):
    payload = build_season_payload(
        nfl, 2025, nfl_games(nfl_cache), backtest_config=BacktestConfig(skip_weeks=1)
    )

    assert payload["backtest"]["n_evaluated"] > 0
    scored = [game for game in payload["games"] if game["predicted_margin"] is not None]
    assert len(scored) == payload["backtest"]["n_evaluated"]
    for game in scored:
        assert 0.0 < game["home_win_prob"] < 1.0


def test_warmup_games_carry_no_prediction(nfl_cache: Path):
    payload = build_season_payload(
        nfl, 2025, nfl_games(nfl_cache), backtest_config=BacktestConfig(skip_weeks=3)
    )

    early = [game for game in payload["games"] if game["week"] <= 3]
    assert early and all(game["predicted_margin"] is None for game in early)
    assert all(game["correct"] is None for game in early)


def test_payload_uses_the_backtest_logistic_scale(nfl_cache: Path):
    payload = build_season_payload(
        nfl, 2025, nfl_games(nfl_cache), backtest_config=BacktestConfig(skip_weeks=1)
    )
    assert payload["meta"]["logistic_scale"] == payload["backtest"]["final_logistic_scale"]


def test_payload_rejects_an_empty_season():
    with pytest.raises(ValueError, match="no completed"):
        build_season_payload(nfl, 2025, [])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def test_render_writes_a_self_contained_page(nfl_cache: Path, tmp_path: Path):
    payload = build_season_payload(nfl, 2025, nfl_games(nfl_cache))
    page = render_page([payload], tmp_path / "nfl.html")
    html = page.read_text()

    assert PAYLOAD_PLACEHOLDER not in html
    assert html.startswith("<!doctype html>")
    assert "BBB" in html
    # Nothing is fetched at runtime: no external scripts, styles or fonts.
    assert "src=\"http" not in html
    assert "href=\"http" not in html
    assert "fonts.googleapis" not in html


def test_rendered_page_embeds_every_season(nfl_cache: Path, tmp_path: Path):
    games = nfl_games(nfl_cache)
    payloads = [
        build_season_payload(nfl, 2025, games),
        {**build_season_payload(nfl, 2025, games), "season": "2024"},
    ]
    page = render_page(payloads, tmp_path / "nfl.html")
    embedded = _embedded_payload(page)

    assert [season["season"] for season in embedded["seasons"]] == ["2025", "2024"]


def test_embedded_json_cannot_close_the_script_tag(tmp_path: Path):
    # A team code will never contain markup, but the embedder must not depend
    # on that: a stray "</script>" in the data would end the tag early.
    payload = {
        "league": "nfl",
        "league_label": "NFL",
        "season": "2025",
        "margin_units": "points",
        "meta": {},
        "teams": [{"team": "</script><script>boom()</script>"}],
        "games": [],
        "backtest": None,
    }
    page = render_page([payload], tmp_path / "nfl.html")
    html = page.read_text()

    assert "<script>boom()" not in html
    assert "\\u003c/script\\u003e" in html
    # It survives the round trip intact.
    assert _embedded_payload(page)["seasons"][0]["teams"][0]["team"] == (
        "</script><script>boom()</script>"
    )


def test_render_rejects_no_seasons(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one season"):
        render_page([], tmp_path / "nfl.html")


def test_render_rejects_mixed_leagues(nfl_cache: Path, tmp_path: Path):
    payload = build_season_payload(nfl, 2025, nfl_games(nfl_cache))
    with pytest.raises(ValueError, match="cannot mix leagues"):
        render_page([payload, {**payload, "league": "nhl"}], tmp_path / "x.html")


def test_template_carries_the_placeholder():
    assert PAYLOAD_PLACEHOLDER in TEMPLATE_PATH.read_text()


# --------------------------------------------------------------------------- #
# The export command
# --------------------------------------------------------------------------- #

def test_export_writes_the_page_and_a_season_file(nfl_cache: Path, tmp_path: Path, capsys):
    output_dir = tmp_path / "output"
    code = main(
        [
            "export",
            "--league", "nfl",
            "--season", "2025",
            "--cache-dir", str(nfl_cache),
            "--output-dir", str(output_dir),
        ]
    )
    assert code == 0

    page = output_dir / "nfl.html"
    season_file = output_dir / "nfl_2025_page.json"
    assert page.is_file() and season_file.is_file()
    assert "NFL 2025" in capsys.readouterr().out

    # The season file is exactly what the page's file picker expects.
    loaded = json.loads(season_file.read_text())
    assert loaded["league"] == "nfl" and loaded["teams"]


def test_export_embeds_several_seasons_for_the_dropdown(
    nfl_cache: Path, tmp_path: Path, monkeypatch
):
    real = nfl.load_games
    # Pretend 2024 exists too, so the page gets a two-entry dropdown.
    monkeypatch.setattr(
        nfl, "load_games", lambda season, **kw: real(2025, **{**kw, "cache_dir": nfl_cache})
    )

    output_dir = tmp_path / "output"
    code = main(
        [
            "export",
            "--league", "nfl",
            "--season", "2025", "2024",
            "--cache-dir", str(nfl_cache),
            "--output-dir", str(output_dir),
        ]
    )
    assert code == 0
    assert (output_dir / "nfl_2024_page.json").is_file()

    embedded = _embedded_payload(output_dir / "nfl.html")
    # Newest first, so the page opens on the most recent season.
    assert [season["season"] for season in embedded["seasons"]] == ["2025", "2024"]


def test_export_skips_a_season_with_no_games(nfl_cache: Path, tmp_path: Path, capsys):
    real = nfl.load_games
    output_dir = tmp_path / "output"

    def only_2025(season, **kw):
        return real(2025, **kw) if str(season) == "2025" else []

    import massey.leagues.nfl as nfl_module

    original = nfl_module.load_games
    nfl_module.load_games = only_2025
    try:
        code = main(
            [
                "export",
                "--league", "nfl",
                "--season", "2025", "1999",
                "--cache-dir", str(nfl_cache),
                "--output-dir", str(output_dir),
            ]
        )
    finally:
        nfl_module.load_games = original

    assert code == 0
    assert "no completed NFL games found for 1999, skipping" in capsys.readouterr().err
    embedded = _embedded_payload(output_dir / "nfl.html")
    assert [season["season"] for season in embedded["seasons"]] == ["2025"]


def test_export_no_backtest_skips_predictions(nfl_cache: Path, tmp_path: Path):
    output_dir = tmp_path / "output"
    main(
        [
            "export",
            "--league", "nfl",
            "--season", "2025",
            "--cache-dir", str(nfl_cache),
            "--output-dir", str(output_dir),
            "--no-backtest",
        ]
    )
    season = _embedded_payload(output_dir / "nfl.html")["seasons"][0]

    assert season["backtest"] is None
    assert all(game["predicted_margin"] is None for game in season["games"])


def test_export_reports_when_there_is_nothing_to_write(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setattr(nfl, "load_games", lambda *a, **k: [])
    code = main(
        [
            "export",
            "--league", "nfl",
            "--season", "2025",
            "--cache-dir", str(tmp_path / "empty"),
            "--output-dir", str(tmp_path / "output"),
        ]
    )
    assert code == 1
    assert "nothing to export" in capsys.readouterr().out


def test_exported_nhl_page_uses_the_leagues_own_columns(nhl_full_cache: Path, tmp_path: Path):
    output_dir = tmp_path / "output"
    code = main(
        [
            "export",
            "--league", "nhl",
            "--season", "2025",
            "--cache-dir", str(nhl_full_cache),
            "--output-dir", str(output_dir),
        ]
    )
    assert code == 0

    season = _embedded_payload(output_dir / "nhl.html")["seasons"][0]
    assert season["margin_units"] == "outcome points"
    team = season["teams"][0]
    assert "ot_losses" in team and "standings_points" in team
    assert "ties" not in team
    # NHL games have no week number, so the page omits that column.
    assert all(game["week"] is None for game in season["games"])


def _embedded_payload(page: Path) -> dict:
    """Pull the JSON payload back out of a rendered page."""
    html = page.read_text()
    start = html.index('<script type="application/json" id="payload">') + len(
        '<script type="application/json" id="payload">'
    )
    end = html.index("</script>", start)
    return json.loads(html[start:end])
