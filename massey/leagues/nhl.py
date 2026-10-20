"""NHL data loading and outcome-based margin logic.

Results come from the ``nhl-api-py`` package (``nhlpy``), which wraps the free,
undocumented ``api-web.nhle.com`` endpoints. We use one endpoint only --
``club-schedule-season/{team}/{season}``, via
``NHLClient().schedule.team_season_schedule()`` -- and cache each team's
response to ``data/raw/nhl/`` so a season is pulled once.

Margins here are **not** goal differentials. A 6-1 romp and a 2-1 win are the
same result in the standings, so the rating target is an outcome-based points
differential from :data:`NHL_POINTS`.

Field names and defensive reads
-------------------------------
The payload shape below is what ``club-schedule-season`` documents and returns:
``id``, ``gameType`` (2 = regular season, 3 = playoffs), ``gameDate``,
``gameState``, ``neutralSite``, ``homeTeam``/``awayTeam`` objects carrying
``abbrev`` and ``score``, and ``gameOutcome.lastPeriodType`` holding ``"REG"``,
``"OT"`` or ``"SO"``.

Because this is an undocumented API that has renamed fields before, every read
goes through :func:`_resolve`, which tries a short list of known spellings and
raises an error naming the keys it actually found rather than silently
producing a wrong rating. Run ``python -m massey.cli inspect --league nhl`` to
dump a real payload and confirm the resolution for yourself.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

from massey.data.cache import DEFAULT_CACHE_DIR, USER_AGENT, Cache, with_retry
from massey.leagues.base import GameRecord, filter_by_cutoff

logger = logging.getLogger(__name__)

name = "nhl"
margin_units = "outcome points"
default_skip_weeks: int | None = None
default_skip_games: int | None = 100

#: Starting logistic scale for turning a predicted outcome-point margin into a
#: home win probability, used only until the backtest has fitted its own.
default_logistic_scale = 0.35

#: The rating margin for a win by each outcome type, from the winner's
#: perspective; the loser gets the negative. Deliberately symmetric, which is
#: *not* the NHL's own standings system (that awards 2 for any win and a
#: consolation 1 for an overtime or shootout loss). A symmetric scale keeps the
#: least-squares system centred and makes a regulation win count for three
#: times a shootout win. Tweak these to taste.
NHL_POINTS: dict[str, int] = {
    "REG": 3,  # regulation win / loss
    "OT": 2,  # overtime win / loss
    "SO": 1,  # shootout win / loss
}

#: Spellings of ``lastPeriodType`` mapped onto the keys of :data:`NHL_POINTS`.
OUTCOME_ALIASES: dict[str, str] = {
    "REG": "REG",
    "REGULATION": "REG",
    "OT": "OT",
    "OVERTIME": "OT",
    "SO": "SO",
    "SHOOTOUT": "SO",
}

#: ``gameType`` for preseason games. A club's season schedule includes these,
#: and they are never rated: they are exhibitions, and the opponent is not
#: always an NHL club (the league has played European teams in preseason).
PRESEASON_GAME_TYPE = 1

#: ``gameType`` for regular-season games.
REGULAR_SEASON_GAME_TYPE = 2

#: ``gameType`` for playoff games.
PLAYOFF_GAME_TYPE = 3

#: ``gameState`` values that mean the game is over and the score is final.
#: A completed season returns a mix of these -- ``"OFF"`` for games the league
#: has finalised and ``"FINAL"`` for ones it has not yet archived.
FINAL_GAME_STATES = frozenset({"OFF", "FINAL", "OVER"})

#: Team whose schedule is fetched first to discover the league's teams. In the
#: current NHL format every team plays every other team, so one club's season
#: schedule names all of them -- which keeps this correct through relocations
#: and expansion without a hardcoded list. Toronto has existed continuously
#: since 1917, so it is a safe seed.
SEED_TEAM = "TOR"

#: Fallback team list, used only if seed discovery comes back implausibly
#: short. Current as of the 2025-26 season.
NHL_TEAM_ABBREVS: tuple[str, ...] = (
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
)

#: Known key paths for each field we need, tried in order.
FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "game_id": ("id", "gameId"),
    "game_type": ("gameType", "gameTypeId"),
    "game_date": ("gameDate", "gameDateEst", "startTimeUTC"),
    "game_state": ("gameState", "gameStateId"),
    "home_abbrev": ("homeTeam.abbrev", "homeTeam.triCode", "homeTeam.abbreviation"),
    "away_abbrev": ("awayTeam.abbrev", "awayTeam.triCode", "awayTeam.abbreviation"),
    "home_score": ("homeTeam.score", "homeTeam.goals"),
    "away_score": ("awayTeam.score", "awayTeam.goals"),
    "outcome": ("gameOutcome.lastPeriodType", "periodDescriptor.periodType"),
    "neutral": ("neutralSite", "neutralSiteGame"),
}

_MISSING = object()


# --------------------------------------------------------------------------- #
# Seasons
# --------------------------------------------------------------------------- #

def current_season() -> int:
    """Return the NHL season that is current today, in CLI form.

    A season is named for the calendar year it starts in, matching the NFL, so
    2026 means the 2026-27 season. The NHL year rolls over in July.

    Returns:
        The starting year of the current season.
    """
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_id(season: int | str) -> str:
    """Convert a CLI season into the API's ``YYYYYYYY`` season identifier.

    Args:
        season: Either the starting year (``2026``) or a full season identifier
            (``"20262027"``), which is passed through unchanged.

    Returns:
        The eight-digit season identifier, e.g. ``"20262027"``.

    Raises:
        ValueError: If ``season`` is neither a four- nor eight-digit number.
    """
    text = str(season).strip()
    if len(text) == 8 and text.isdigit():
        first, second = int(text[:4]), int(text[4:])
        if second != first + 1:
            raise ValueError(f"{season!r} is not a consecutive-year season id")
        return text
    if len(text) == 4 and text.isdigit():
        start = int(text)
        return f"{start}{start + 1}"
    raise ValueError(f"cannot read {season!r} as an NHL season")


# --------------------------------------------------------------------------- #
# Margin logic
# --------------------------------------------------------------------------- #

def normalise_outcome(last_period_type: Any) -> str:
    """Map an API ``lastPeriodType`` onto a :data:`NHL_POINTS` key.

    Args:
        last_period_type: The raw value, e.g. ``"REG"`` or ``"shootout"``.

    Returns:
        One of ``"REG"``, ``"OT"`` or ``"SO"``.

    Raises:
        ValueError: If the value is not a recognised outcome type.
    """
    if last_period_type is None:
        raise ValueError("game has no lastPeriodType; it may not be finished")

    key = str(last_period_type).strip().upper()
    if key not in OUTCOME_ALIASES:
        raise ValueError(
            f"unrecognised NHL outcome type {last_period_type!r}; "
            f"expected one of {sorted(OUTCOME_ALIASES)}"
        )
    return OUTCOME_ALIASES[key]


def margin_from_outcome(
    home_score: int,
    away_score: int,
    last_period_type: Any,
) -> int:
    """Return the home-relative outcome-points margin for one game.

    The winner's share comes from :data:`NHL_POINTS` and the loser takes the
    negative, so a home regulation win is ``+3`` and a home shootout loss is
    ``-1``. Goal differential is deliberately ignored.

    Args:
        home_score: Final home goals.
        away_score: Final away goals.
        last_period_type: How the game ended -- ``"REG"``, ``"OT"`` or ``"SO"``.

    Returns:
        The signed margin, positive when the home team won.

    Raises:
        ValueError: If the outcome type is unrecognised, or the scores are
            level (no NHL game ends tied).
    """
    points = NHL_POINTS[normalise_outcome(last_period_type)]

    if home_score > away_score:
        return points
    if away_score > home_score:
        return -points
    raise ValueError(
        f"NHL game ended level at {home_score}-{away_score}; "
        "every game is settled in regulation, overtime or a shootout"
    )


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #

def build_schedule_api(*, timeout: int = 20, user_agent: str = USER_AGENT) -> Any:
    """Build the ``nhlpy`` schedule API with a descriptive User-Agent.

    ``nhlpy`` builds its own ``httpx`` client per request and exposes no hook
    for request headers, so to identify ourselves politely we subclass its
    ``HttpClient`` to add the header and inject that into the schedule API --
    which the library already accepts as a constructor argument.

    Args:
        timeout: Per-request timeout in seconds.
        user_agent: Value for the ``User-Agent`` header.

    Returns:
        An ``nhlpy.api.schedule.Schedule`` instance.
    """
    import httpx
    from nhlpy.api.schedule import Schedule
    from nhlpy.config import ClientConfig
    from nhlpy.http_client import HttpClient

    class _UserAgentHttpClient(HttpClient):
        """``nhlpy`` HTTP client that sends a ``User-Agent`` header."""

        def get(self, endpoint: Any, resource: str, query_params: dict | None = None) -> Any:
            with httpx.Client(
                headers={"User-Agent": user_agent},
                verify=self._config.ssl_verify,
                timeout=self._config.timeout,
                follow_redirects=self._config.follow_redirects,
            ) as client:
                response = client.get(url=f"{endpoint.value}{resource}", params=query_params)
            self._handle_response(response, resource)
            return response

    return Schedule(http_client=_UserAgentHttpClient(ClientConfig(timeout=timeout)))


def load_club_schedule(
    team_abbrev: str,
    season: int | str,
    *,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    delay: float = 1.0,
    schedule_api: Any | None = None,
) -> dict:
    """Load one club's season schedule, via the cache.

    Args:
        team_abbrev: Three-letter team code, e.g. ``"TOR"``.
        season: Season year or identifier; see :func:`season_id`.
        cache_dir: Root of the raw-data cache. Defaults to ``data/raw``.
        refresh: Re-fetch even if a cached copy exists.
        delay: Seconds to wait before a live request, to be polite to the API.
            Not applied on a cache hit.
        schedule_api: Pre-built schedule API, so a batch of teams can share one.

    Returns:
        The raw ``club-schedule-season`` payload.
    """
    sid = season_id(season)
    cache = Cache(cache_dir or DEFAULT_CACHE_DIR, refresh=refresh)
    parts = ("nhl", f"club_schedule_{team_abbrev}_{sid}.json")

    def fetch() -> dict:
        api = schedule_api if schedule_api is not None else build_schedule_api()
        if delay > 0:
            time.sleep(delay)
        logger.info("fetching NHL schedule for %s %s", team_abbrev, sid)
        return with_retry(
            lambda: api.team_season_schedule(team_abbr=team_abbrev, season=sid),
            description=f"club-schedule-season/{team_abbrev}/{sid}",
            retry_on=transient_errors(),
        )

    return cache.json(parts, fetch)


def transient_errors() -> tuple[type[BaseException], ...]:
    """Return the exception types worth retrying an NHL request on.

    Rate limits, server errors and transport failures are worth another go
    after a wait. A 400, 401, 403 or 404 is a settled answer -- a bad team
    code, or a host we are not allowed to reach -- so retrying it only turns a
    fast failure into a slow one.

    Returns:
        Exception types to treat as transient.
    """
    import httpx
    from nhlpy.http_client import RateLimitExceededException, ServerErrorException

    return (httpx.TransportError, RateLimitExceededException, ServerErrorException)


def discover_team_abbrevs(
    season: int | str,
    *,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    delay: float = 1.0,
    schedule_api: Any | None = None,
) -> list[str]:
    """Return the season's team codes, discovered from the seed team's schedule.

    Falls back to :data:`NHL_TEAM_ABBREVS` if discovery turns up implausibly
    few teams.

    Args:
        season: Season year or identifier.
        cache_dir: Root of the raw-data cache.
        refresh: Re-fetch even if a cached copy exists.
        delay: Seconds to wait before a live request.
        schedule_api: Pre-built schedule API to reuse.

    Returns:
        Sorted team codes.
    """
    payload = load_club_schedule(
        SEED_TEAM,
        season,
        cache_dir=cache_dir,
        refresh=refresh,
        delay=delay,
        schedule_api=schedule_api,
    )

    found: set[str] = set()
    for game in payload.get("games", []):
        # Regular-season games only. A preseason opponent is not necessarily an
        # NHL club, and inventing a team code here would send us asking the API
        # for a club schedule that does not exist.
        if _game_type(game) != REGULAR_SEASON_GAME_TYPE:
            continue
        for field in ("home_abbrev", "away_abbrev"):
            value = _dig(game, FIELD_PATHS[field])
            if value is not _MISSING:
                found.add(str(value))

    if len(found) < 20:
        logger.warning(
            "team discovery from %s found only %d teams; falling back to the built-in list",
            SEED_TEAM,
            len(found),
        )
        return list(NHL_TEAM_ABBREVS)

    return sorted(found)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_raw_games(
    season: int | str,
    *,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    delay: float = 1.0,
    teams: Sequence[str] | None = None,
) -> list[dict]:
    """Fetch every team's schedule for a season and return deduplicated games.

    Each game appears in both clubs' schedules, so results are deduplicated on
    the game id.

    Args:
        season: Season year or identifier.
        cache_dir: Root of the raw-data cache.
        refresh: Re-fetch even if a cached copy exists.
        delay: Seconds between live requests.
        teams: Explicit team codes to fetch. Defaults to discovery.

    Returns:
        Raw game dicts, one per game, in no particular order.
    """
    cache = Cache(cache_dir or DEFAULT_CACHE_DIR, refresh=refresh)
    sid = season_id(season)

    # Only build a live client if something is actually missing from the cache,
    # so a fully cached season needs no network access at all.
    probe = teams or (SEED_TEAM,)
    needs_network = refresh or any(
        not cache.has("nhl", f"club_schedule_{team}_{sid}.json") for team in probe
    )
    schedule_api = build_schedule_api() if needs_network else None

    if teams is None:
        teams = discover_team_abbrevs(
            season,
            cache_dir=cache_dir,
            refresh=refresh,
            delay=delay,
            schedule_api=schedule_api,
        )

    by_id: dict[str, dict] = {}
    for team in teams:
        if schedule_api is None and not cache.has("nhl", f"club_schedule_{team}_{sid}.json"):
            schedule_api = build_schedule_api()
        payload = load_club_schedule(
            team,
            season,
            cache_dir=cache_dir,
            refresh=refresh,
            delay=delay,
            schedule_api=schedule_api,
        )
        for game in payload.get("games", []):
            game_id = _dig(game, FIELD_PATHS["game_id"])
            if game_id is not _MISSING:
                by_id[str(game_id)] = game

    return list(by_id.values())


def load_games(
    season: int | str,
    *,
    as_of: date | None = None,
    week: int | None = None,
    include_playoffs: bool = False,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    delay: float = 1.0,
    teams: Sequence[str] | None = None,
) -> list[GameRecord]:
    """Load completed NHL games for a season as rating inputs.

    Args:
        season: Season year (``2026``) or identifier (``"20262027"``).
        as_of: Keep only games played strictly before this date.
        week: Unused -- the NHL has no week numbers. Accepted so the CLI can
            treat both leagues alike.
        include_playoffs: Include playoff games as well as the regular season.
        cache_dir: Root of the raw-data cache. Defaults to ``data/raw``.
        refresh: Re-fetch even if a cached copy exists.
        delay: Seconds between live requests.
        teams: Explicit team codes to fetch. Defaults to discovery.

    Returns:
        Completed games, oldest first.
    """
    if week is not None:
        logger.warning("--week does not apply to the NHL; ignoring it")

    wanted_types = {REGULAR_SEASON_GAME_TYPE}
    if include_playoffs:
        wanted_types.add(PLAYOFF_GAME_TYPE)

    records = [
        record
        for game in load_raw_games(
            season, cache_dir=cache_dir, refresh=refresh, delay=delay, teams=teams
        )
        if (record := _to_record(game, wanted_types)) is not None
    ]

    records.sort(key=lambda record: (record.date, record.game_id))
    return filter_by_cutoff(records, as_of=as_of)


def _to_record(game: dict, wanted_types: Iterable[int]) -> GameRecord | None:
    """Convert one raw game dict into a :class:`GameRecord`.

    Args:
        game: A raw game dict from ``club-schedule-season``.
        wanted_types: ``gameType`` values to keep.

    Returns:
        The record, or ``None`` if the game is the wrong type or not finished.
    """
    if _game_type(game) not in set(wanted_types):
        return None

    if not _has_score(game):
        return None  # not played yet

    # Prefer the explicit state flag; fall back to "it has scores" if the API
    # ever drops the field.
    state = _dig(game, FIELD_PATHS["game_state"])
    if state is not _MISSING and str(state).upper() not in FINAL_GAME_STATES:
        return None

    home_score = int(_resolve(game, "home_score"))
    away_score = int(_resolve(game, "away_score"))
    outcome = normalise_outcome(_resolve(game, "outcome"))
    neutral = _dig(game, FIELD_PATHS["neutral"])

    return GameRecord(
        game_id=str(_resolve(game, "game_id")),
        date=_parse_date(_resolve(game, "game_date")),
        home=str(_resolve(game, "home_abbrev")),
        away=str(_resolve(game, "away_abbrev")),
        margin=margin_from_outcome(home_score, away_score, outcome),
        neutral=bool(neutral) if neutral is not _MISSING else False,
        home_score=home_score,
        away_score=away_score,
        week=None,
        outcome=outcome,
    )


def inspect_season(
    season: int | str,
    *,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    delay: float = 1.0,
) -> dict[str, Any]:
    """Summarise a raw NHL payload so field names can be eyeballed.

    Fetches only the seed team's schedule, then reports the keys the API
    actually returned alongside the values this module resolved from them.
    Backs the ``inspect`` CLI command.

    Args:
        season: Season year or identifier.
        cache_dir: Root of the raw-data cache.
        refresh: Re-fetch even if a cached copy exists.
        delay: Seconds to wait before a live request.

    Returns:
        A dictionary of diagnostics: top-level payload keys, per-game keys,
        counts by game state and type, and the resolved fields of one
        completed game.
    """
    payload = load_club_schedule(
        SEED_TEAM, season, cache_dir=cache_dir, refresh=refresh, delay=delay
    )
    games = payload.get("games", [])

    report: dict[str, Any] = {
        "season": season_id(season),
        "seed_team": SEED_TEAM,
        "payload_keys": sorted(payload),
        "n_games": len(games),
        "game_keys": sorted(games[0]) if games else [],
        "game_states": _counts(_dig(game, FIELD_PATHS["game_state"]) for game in games),
        "game_types": _counts(_dig(game, FIELD_PATHS["game_type"]) for game in games),
        "outcome_types": _counts(_dig(game, FIELD_PATHS["outcome"]) for game in games),
    }

    # Show a regular-season game if there is one, since that is what actually
    # gets rated; a club's schedule opens with preseason exhibitions.
    played = [game for game in games if _has_score(game)]
    sample = next(
        (game for game in played if _game_type(game) == REGULAR_SEASON_GAME_TYPE),
        played[0] if played else None,
    )
    if sample is not None:
        report["sample_resolved_fields"] = {
            field: _render(_dig(sample, paths)) for field, paths in FIELD_PATHS.items()
        }
        report["sample_raw_game"] = sample

    return report


# --------------------------------------------------------------------------- #
# Defensive field access
# --------------------------------------------------------------------------- #

def _dig(payload: dict, paths: tuple[str, ...]) -> Any:
    """Return the first value found at any of ``paths``, else :data:`_MISSING`.

    Args:
        payload: The dict to read.
        paths: Dotted key paths to try in order, e.g. ``"homeTeam.score"``.

    Returns:
        The value at the first path that resolves, or :data:`_MISSING`.
    """
    for path in paths:
        current: Any = payload
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                current = _MISSING
                break
            current = current[key]
        if current is not _MISSING:
            return current
    return _MISSING


def _has_score(game: dict) -> bool:
    """Return whether both teams' scores are present, i.e. the game was played."""
    return all(
        _dig(game, FIELD_PATHS[field]) not in (_MISSING, None)
        for field in ("home_score", "away_score")
    )


def _game_type(game: dict) -> int | None:
    """Return a game's ``gameType`` as an int, or ``None`` if unreadable.

    The API sends it as a number, but it is the field every filter hinges on,
    so an unexpected value returns ``None`` (excluding the game) rather than
    raising.

    Args:
        game: A raw game dict.

    Returns:
        The game type, or ``None``.
    """
    value = _dig(game, FIELD_PATHS["game_type"])
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("game %s has an unreadable gameType %r", game.get("id"), value)
        return None


def _resolve(game: dict, field: str) -> Any:
    """Read a required field, failing loudly if the API has renamed it.

    Args:
        game: A raw game dict.
        field: A key of :data:`FIELD_PATHS`.

    Returns:
        The field's value.

    Raises:
        KeyError: If none of the known paths resolve, with the keys that were
            actually present so the fix is obvious.
    """
    value = _dig(game, FIELD_PATHS[field])
    if value is _MISSING:
        raise KeyError(
            f"could not find {field!r} in the NHL payload: tried "
            f"{list(FIELD_PATHS[field])}, game has keys {sorted(game)}. "
            "The API shape may have changed -- run "
            "`python -m massey.cli inspect --league nhl` and update FIELD_PATHS."
        )
    return value


def _parse_date(value: Any) -> date:
    """Parse an API date, accepting a bare date or a full UTC timestamp."""
    return date.fromisoformat(str(value)[:10])


def _counts(values: Iterable[Any]) -> dict[str, int]:
    """Count occurrences of each value, rendered as strings."""
    counts: dict[str, int] = {}
    for value in values:
        key = _render(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _render(value: Any) -> str:
    """Render a value (including the missing sentinel) for diagnostics."""
    return "<missing>" if value is _MISSING else str(value)
