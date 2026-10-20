# Massey power ratings — NFL and NHL

Least-squares team ratings for the NFL and NHL, with a fitted home-field
advantage, a ridge penalty for early-season stability, and a walk-forward
backtest that scores the ratings against the "always pick the home team"
baseline.

The solver is league-agnostic: both leagues reduce their results to
`(home, away, margin, neutral)` rows and share the same code. Adding Elo or an
efficiency model later means writing a new rating module against that same
interface, not touching this one.

## Install

```bash
pip install -r requirements.txt
```

Python 3.11+. Verified on 3.13 with numpy 2.5, pandas 3.0, nflreadpy 0.1.5 and
nhl-api-py 3.3.0.

## Quick start

```bash
# Ratings for the current season, printed and written to output/
python -m massey.cli rate --league nfl
python -m massey.cli rate --league nhl

# A specific season
python -m massey.cli rate --league nfl --season 2025
python -m massey.cli rate --league nhl --season 2025        # the 2025-26 season

# Ratings as they stood going into week 7, or going into a date
python -m massey.cli rate --league nfl --season 2026 --week 7
python -m massey.cli rate --league nhl --season 2025 --as-of 2026-01-15

# Walk-forward backtest
python -m massey.cli backtest --league nfl --season 2025
python -m massey.cli backtest --league nhl --season 2025

# Build a browser page (see "Viewing in a browser")
python -m massey.cli export --league nfl --season 2026 2025 2024
python -m massey.cli export --league nhl --season 2025

# Dump the raw field names the NHL API returned (see "NHL payload shape")
python -m massey.cli inspect --league nhl --season 2025
```

Add `-v` to watch the data fetching, and `--refresh` to re-pull data instead of
reading the cache.

Sample output:

```
NFL 2025 Massey ratings
  games fitted       : 272 (7 at neutral sites, 2025-09-04 to 2026-01-04)
  home-field advantage: +2.176 points (fitted)
  ridge penalty       : 1
  in-sample margin MAE: 9.053 points
  margin cap          : none

rank  team      rating   gp
---------------------------
   1  SEA      +10.942   17
   2  LA        +9.001   17
...
```

## The model

Each game is one equation:

```
rating_home - rating_away + hfa = margin        (at a home venue)
rating_home - rating_away       = margin        (at a neutral site)
```

Stacked over every game, that is an over-determined linear system solved by
least squares. Three details matter:

- **One home-field parameter**, estimated from the data rather than assumed.
  Neutral-site games contribute no HFA term, so they help pin down team
  strength without polluting the home-field estimate.
- **A ridge penalty on the team ratings** (`--ridge`, default `1.0`). Early in a
  season a team with two games has almost no information behind it; the penalty
  pulls such ratings toward average instead of letting one blowout dominate.
  HFA is deliberately *not* penalised — shrinking it toward zero would bias the
  one quantity we are trying to measure.
- **Ratings sum to zero**, enforced exactly with a Lagrange multiplier rather
  than approximated by a weighted extra row. A rating therefore reads directly
  as "margin better than an average team".

One consequence worth knowing: because HFA is unpenalised, with very little
data it absorbs the average home margin before the ratings move at all. Fit a
single game and you get `hfa = margin`, both ratings zero. That is the intended
behaviour — a lone result should not read as a huge rating gap.

## Margins

**NFL** — actual point differential, home minus away. `NFL_CONFIG["MAX_MARGIN"]`
in `massey/leagues/nfl.py` caps the absolute margin to test diminishing returns
on blowouts; the default is `None`, meaning no cap. `--max-margin 28` overrides
it per run.

**NHL** — *not* goal differential. A 6-1 romp and a 2-1 win are the same result
in the standings, so the target is an outcome-based points differential from
`NHL_POINTS` in `massey/leagues/nhl.py`:

| How it ended | Winner | Loser |
| ------------ | -----: | ----: |
| Regulation   |   `+3` |  `-3` |
| Overtime     |   `+2` |  `-2` |
| Shootout     |   `+1` |  `-1` |

Note this is a symmetric scale and **not** the NHL's own standings system,
which awards 2 for any win and a consolation 1 for an overtime or shootout
loss. A symmetric scale keeps the least-squares system centred and makes a
regulation win worth three times a shootout win. Edit the dict to taste; the
loader and the solver pick the change up with no other edits.

## Season and cutoff conventions

- A season is **named for the year it starts in**, for both leagues.
  `--season 2026` means the NFL 2026 season (Sept 2026 – Feb 2027) or the NHL
  2026-27 season. The NHL's own `20262027` identifier is also accepted.
- `--as-of DATE` and `--week N` are both **exclusive**. `--week 7` gives the
  ratings a forecaster would have carried *into* week 7, built from weeks 1-6.
  This is what you want for prediction; it is not "through week 7".
- `--week` applies to the NFL only; the NHL has no week numbers and warns if
  you pass it.
- Regular season only by default. `--include-playoffs` adds the postseason.
  For the NHL the points scale is a regular-season idea, so playoff overtime
  games are approximated at `±2`.

## Outputs

`rate` writes to `output/`:

- `{league}_{season}_ratings.csv` — `rank, team, rating, games_played`
- `{league}_{season}_ratings_meta.json` — the fitted HFA, the ridge setting, the
  game count, the cutoffs used

`backtest` writes:

- `{league}_{season}_backtest_games.csv` — one row per evaluated game, with the
  predicted and actual margin, the win probability, and the training-set size
- `{league}_{season}_backtest_summary.json` — the metrics below

`export` writes:

- `{league}.html` — the self-contained browser page
- `{league}_{season}_page.json` — one per season, for the page's file picker

`--no-save` prints without writing. Raw data pulls are cached under `data/raw/`;
both directories are gitignored.

## Backtest

For each game, ratings are fitted on games played strictly earlier, then used
to predict that game. Nothing in a prediction has seen its own result.

Reported per league:

- **Winner accuracy**, against the **home-team baseline** over the same games,
  and the edge between them
- **MAE and RMSE** of the predicted margin
- **Log loss and Brier score** of a calibrated win probability, against a
  constant-rate baseline

Two knobs shape the numbers:

- **Warm-up.** `--skip-weeks` (NFL, default 3) and `--skip-games` (NHL, default
  100) hold back the opening stretch of the season as training data only.
  Ratings there are too unstable to score fairly.
- **Refit cadence.** `--refit per-day` (the default) refits once per calendar
  date and predicts every game that day from it. Same-day games never see each
  other, so this leaks nothing and is much faster. `--refit per-game` does the
  strict version; on a schedule where no two games share a date the two give
  identical results.

Win probabilities come from `p = sigmoid(k · predicted_margin)`, with `k`
refitted on the predictions already made this run — so the calibration is
itself walk-forward rather than fitted on the whole season. Until 50
out-of-sample predictions have accumulated, each league's
`default_logistic_scale` is used.

The log-loss baseline is a constant predictor set to the home-win rate over the
evaluated games. That is deliberately generous: it knows the season's final
home-win rate in advance, which the model does not.

**NHL caveat.** Predicting the outcome-points margin is a proxy. Wins and
losses are the thing to judge, so read accuracy and log loss first; the margin
MAE is only a sanity check on a `±3` scale.

### Measured results

NFL, default settings (ridge 1.0, 3 warm-up weeks, 224 games scored):

| Season | Accuracy | Home baseline |   Edge | Margin MAE | Mean HFA |
| -----: | -------: | ------------: | -----: | ---------: | -------: |
|   2023 |    59.8% |         57.6% | +2.2pt |      11.00 |     1.52 |
|   2024 |    68.8% |         54.0% | +14.7  |      10.68 |     1.20 |
|   2025 |    59.6% |         52.9% | +6.7   |      10.65 |     2.96 |

Beating the home-team baseline every season, by a margin that varies a lot
year to year. A ~10.7-point margin MAE is about what a pure
points-differential model gets; closing that gap is what an efficiency model
(EPA) would be for.

No NHL figures are listed here yet: the field names are confirmed against a
live payload, but no full season has been pulled and backtested.

## Viewing in a browser

```bash
python -m massey.cli export --league nfl --season 2026 2025 2024
```

Writes `output/nfl.html` (and `output/nhl.html` for the NHL), a single
self-contained file. Double-click it — there is no server to run and nothing is
fetched when it opens. The page has:

- the fitted HFA, ridge, game count and backtest accuracy as tiles
- a **matchup predictor**: pick two teams, tick neutral site, get the predicted
  margin and win probability
- a diverging bar chart of the ratings, with hover detail
- a sortable table using each league's own columns — W-L-T and points for and
  against for the NFL; W-L-OTL, standings points and goals for the NHL
- a filterable list of every game with the actual result beside the
  walk-forward prediction, marked hit or miss

Pass several seasons and the page gets a dropdown to switch between them
instantly. Each season is also written as `output/<league>_<season>_page.json`,
and the page's **Load season file…** button reads one of those from disk, so a
season generated later can be viewed without rebuilding the page.

Two deliberate constraints are worth knowing, because they explain why the page
works the way it does. A page opened by double-clicking has a `null` origin, so
any request it made to the NHL API or to GitHub would be refused by CORS — and
the rating model is Python, which a browser cannot run. So the page never
fetches anything: every number in it is computed by `export` and baked in. The
file-picker button sidesteps this because `FileReader` reads local files
without a network request of any kind.

Size: a season is roughly 40KB of JSON for the NFL and 400KB for the NHL, so
embedding many NHL seasons in one page adds up.

A note on the colours, since they were chosen rather than picked: ratings use a
blue-to-red diverging scale around zero, which is the right encoding for a
value whose sign is the point. Hits and misses in the results list are marked
by a glyph and a word, with red used only for a miss — the obvious green/red
pairing measures a colour-difference of 2.5 under deuteranopia, meaning
red-green colourblind readers could not tell the two apart.

## NHL payload shape

`api-web.nhle.com` is free but undocumented. The field names below have been
confirmed against a live `club-schedule-season` response for the 2025-26
season; the loader resolves all ten of the fields it needs from their primary
paths, with no fallbacks required.

If the API shape ever moves, this prints the payload's top-level keys, the
per-game keys, the counts of each `gameState`, `gameType` and `lastPeriodType`,
and the fields the loader resolved from one completed regular-season game:

```bash
python -m massey.cli inspect --league nhl --season 2025
```

The fields used are `id`, `gameType`, `gameDate`, `gameState`, `neutralSite`,
`homeTeam.abbrev`, `homeTeam.score`, `awayTeam.abbrev`, `awayTeam.score` and
`gameOutcome.lastPeriodType`.

Every read goes through `_resolve()` in `massey/leagues/nhl.py`, which tries a
short list of known spellings and raises an error naming the keys it actually
found. A renamed field fails loudly rather than producing a quietly wrong
rating. If something has moved, add the new path to `FIELD_PATHS`.

Two things the live payload confirmed, both of which the loader handles:

- **`gameType` 1 is preseason**, and a club's season schedule includes those
  games. They are never rated. They are also excluded from team discovery,
  because a preseason opponent is not always an NHL club — the league has
  played European teams in preseason, and taking a code like `EBB` for a real
  team would send the loader after a club schedule that does not exist.
- **A finished game's `gameState` is either `OFF` or `FINAL`**, depending on
  whether the league has archived it. Both count as complete.

Politeness, since the API is free and undocumented: one request per team with a
one-second gap (`delay`), a descriptive `User-Agent`, a 20-second timeout, and
retry with exponential backoff. Everything is cached, so a season costs ~32
requests once. `nhlpy` builds its own HTTP client per request with no header
hook, so `build_schedule_api()` subclasses its `HttpClient` to attach the
`User-Agent` and injects that — which the library already supports as a
constructor argument.

Team discovery: the league's teams are read off one club's schedule, since in
the current format every team plays every other team. That stays correct
through relocation and expansion without a hardcoded list, and falls back to
`NHL_TEAM_ABBREVS` if it turns up implausibly few teams.

## Layout

```
massey/
  models/massey.py     league-agnostic solver; knows nothing about either sport
  data/cache.py        read-through disk cache + retry/backoff helper
  leagues/base.py      GameRecord, the adapter protocol, shared cutoff logic
  leagues/nfl.py       nflreadpy loading, point margins, the margin cap
  leagues/nhl.py       api-web loading, outcome-points margins, defensive reads
  backtest.py          walk-forward evaluation and metrics
  cli.py               argument parsing, console tables, CSV/JSON output
  web.py               builds the page payload and renders the HTML
  templates/           the page itself: one file, no external dependencies
tests/                 182 tests; see below
data/raw/              cached raw pulls (gitignored)
output/                generated CSV and JSON (gitignored)
```

## Tests

```bash
pytest -q
```

182 tests, no network access required — the league tests seed the cache and run
the real loaders against it.

- `test_massey.py` — the solver against a synthetic four-team league with known
  ratings and a known HFA: exact recovery with no ridge, sum-to-zero, neutral
  games carrying no HFA, ridge shrinkage, singular-system handling
- `test_nhl.py` — the regulation/overtime/shootout margin mapping in both
  directions, outcome-string normalisation, and the loader against fixtures
- `test_nhl_season.py` — a full synthetic 32-team season end to end: team
  discovery, deduplication across club schedules, both CLI commands
- `test_nfl.py` — completed-game filtering, point margins, the cap, neutral
  sites, ties, and the week/date cutoffs
- `test_backtest.py` — no-leakage guarantees (training sets grow and never
  include the game being predicted), warm-up accounting, tie handling,
  calibration
- `test_cache.py` — fetch-once caching, `--refresh`, exponential backoff
- `test_cli.py` — every command end to end, output files, error paths
- `test_web.py` — the page payload, each league's own record columns, and
  that the embedded JSON cannot break out of its script tag

## Known limitations

- **NFL neutral-site labelling** is taken from nflreadpy's `location` column as
  given. In some seasons it marks games that look like ordinary home games;
  that is the upstream data's call, and the loader does not second-guess it.
- **Team abbreviations are not normalised across seasons.** Relocations (OAK to
  LV, SD to LAC) mean a team can appear under two codes in different seasons.
  Only one season is rated at a time, so this does not affect results, but do
  not join two seasons' CSVs on `team` without mapping first.
- **Ratings are a single-season fit** with no carry-over prior from last
  season, so week-1 ratings are genuinely uninformative rather than
  slightly-informative.

## Extending it

The solver's input is a list of `Game(home, away, margin, neutral)` and its
output is a `MasseyResult`. Anything that produces ratings from games can sit
beside it:

- **Elo** — a new module in `models/`, driven from the same `GameRecord` list
  the backtest already walks in chronological order.
- **EPA (NFL) / xG (NHL)** — a new `margin` definition in the league adapter.
  Swap point differential for an efficiency-based margin and the solver needs
  no changes at all; `GameRecord` already carries the raw scores alongside the
  rating margin.
- **Ensembles** — `run_backtest` takes games and returns per-game predictions,
  so two models can be scored on the same walk-forward split and blended.
