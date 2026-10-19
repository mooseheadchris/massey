# Test fixtures

## `nhl_club_schedule_*.json`

These are **hand-authored**, not captured from the live API. The session that
wrote them had no network route to `api-web.nhle.com`, so they were built to
match the documented `club-schedule-season/{team}/{season}` response shape
rather than recorded from a real response.

They exercise the NHL loader end to end offline: every outcome type
(regulation / overtime / shootout), both home and away wins, an unplayed game,
a playoff game, a neutral-site game, and a game whose score fields are
present but whose state is not final.

If the NHL API changes shape, replace these with a real capture:

```bash
python -m massey.cli inspect --league nhl --season 2025 > /tmp/nhl.json
cp data/raw/nhl/club_schedule_TOR_20252026.json \
   tests/fixtures/nhl_club_schedule_TOR_20252026.json
```

then update `FIELD_PATHS` in `massey/leagues/nhl.py` if the key names moved,
and trim the fixture down to the games the tests need.
