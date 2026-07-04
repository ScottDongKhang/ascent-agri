# Depth-first: decision-grade output for the Robusta Coffee Monitor

**Date:** 2026-07-04
**Status:** Draft — awaiting Scott's review
**Goal:** Raise the product from a well-built *nowcast* to a *decision tool* a
green-coffee buyer, roaster, or co-op would pay for — a forward look, a scored
track record, and an adoptable weekly artifact — without touching the strategy
or weakening any honesty guarantee.

Deliberately out of scope (each gets its own spec later): multi-province /
multi-country coverage, the outreach sprint, the robusta contract backfill
(Scott-only task), and any payment mechanism.

---

## Component 1 — The 14-day forecast layer

The one feature that changes the product's class. Today every number on the
site describes the past. This adds: *"Next 14 days: 38 mm expected vs 92 mm
seasonal norm during fruit fill — stage-weighted stress projected to rise from
low to elevated."* No free source publishes this for the Vietnamese robusta
belt.

**Data.** Open-Meteo's forecast API (same provider as the existing history
cache; free, keyless): daily `precipitation_sum`, `temperature_2m_max/min`,
`forecast_days=16`, at the existing Buon Ma Thuot coordinates (12.68 N,
108.04 E). Fetched by the daily workflow alongside the existing fetches;
cached to `data/processed/forecast_vn.csv`. `build_site.py` reads the cache
and never fetches (existing pattern).

**Computation** (new module `ascentagri/agronomy/forecast.py`):

- **Seasonal norm:** for the 14 calendar days ahead, mean total rainfall over
  the same calendar window in all prior years of the existing historical
  cache (2018→, ≥3 years required; error out below that rather than fake a
  norm). Same-source norm (Open-Meteo history) so forecast and norm are
  comparable.
- **Forward anomaly:** expected 14-day total vs norm, expressed both in mm
  and as a z-score using the historical distribution of same-window totals.
- **Projected stress:** apply the existing phenology weights
  (`stage_for()`, drought/wetness weights from `agronomy/phenology.py`) to
  the forward anomaly → projected stress index + band via `stress_label()`.
  If the 14-day window spans a stage boundary, weight by days in each stage.

**The forecast snapshot ledger** — the piece that makes Component 2 possible.
Every weekday the issued forecast is appended to
`data/ledger/weather_forecasts.jsonl` (append-only, committed, same
discipline as `forecasts.jsonl`): `{date_issued, window_start, window_end,
expected_mm, norm_mm, anom_z, projected_stress, projected_band, schema}`.
Entries are never edited.

**Surfaces:**

- New site section "The next two weeks" (chart: daily forecast bars vs norm
  band, plus the projected-stress sentence).
- `api/latest.json` gains a `forecast` block (add fields; keep
  `schema_version: 1` — additive change, documented in the `#data` section).
- One sentence appended to `daily_brief()` and `daily_brief_vi()` (English
  sentence built in code, Vietnamese via the existing `*_VI` string-table
  pattern).
- RSS carries the extended brief automatically.

**Honesty rules:** the section names the source ("Open-Meteo forecast model")
and carries its issue date; until Component 2 has scored ≥5 closed windows,
the section says in-line that forecast skill is not yet verified and links to
the verification panel where the score will appear.

**Fail-safe:** a failed forecast fetch must not block publishing. The daily
workflow treats the forecast fetch as optional: on failure, the site builds
without the forecast section (or with the last cache if <3 days old, marked
with its as-of date). The existing all-or-nothing fail-safe continues to
govern the core fetches.

---

## Component 2 — Forecast verification: the scored track record

The ledger is the moat — calls time-stamped before outcomes. Extend it from
"positions" to "forecasts," using the standard verification framing: **skill
vs climatology**.

**Scoring** (`ascentagri/agronomy/forecast.py`, CLI
`python -m ascentagri.agronomy.forecast score`): for every snapshot in
`weather_forecasts.jsonl` whose 14-day window has fully closed, compare
`expected_mm` against realized rainfall from the historical cache:

- MAE and bias of the forecast (mm).
- MAE of the climatology baseline (`norm_mm` as the prediction).
- **Skill score = 1 − MAE_forecast / MAE_climatology** (positive = beats
  climatology).
- Band hit rate: did the projected stress band match the band computed from
  realized rainfall?

Reproducible from committed files alone — the score reads only
`weather_forecasts.jsonl` and the committed historical cache, mirroring
`ledger.score_ledger()`.

**Surfaces:** the existing ledger section becomes "The track record" with two
panels: the current position ledger (unchanged, `LedgerScore.summary_line()`)
and forecast verification. Until ≥5 closed windows exist, the verification
panel shows "first verification lands ~{date}" (computable: first snapshot
date + 14 days + weekly cadence). Whatever the numbers are — including a
negative skill score — they are published. If the forecast can't beat
climatology, the site says so; that honesty is the brand.

**Cadence:** scoring runs at site-build time from the committed snapshot
ledger and the weather cache — the same pattern as `score_ledger()` — so the
panel is always as fresh as the page and needs no cross-workflow handoff.

---

## Component 3 — The weekly one-pager and change alerts

The artifact that gets adopted, plus the machine surface that makes "they
depend on it" literally true (a poller on a schedule).

**One-pager** (`site/build_onepager.py`, invoked from every daily site
build — it's cheap): a single-page, print-styled HTML brief —
regime + posture, 1w/1m price change, current stress + the 14-day outlook,
farm-gate VND/kg line, ledger summary line, attribution + CC BY 4.0 footer,
every panel with its as-of date. Output `site/_build/brief/latest.html` on every build, plus a dated archive
copy (`brief/YYYY-MM-DD.html`) written on Fridays only. Print CSS makes Cmd+P produce
a clean PDF; no headless-browser dependency in CI. The main page links it
("This week on one page — print or forward it").

**Change alerts:**

- `api/changes.json` — the most recent state *changes* (not daily state):
  a list of `{date, type, from, to}` where type ∈ `regime_change`,
  `stress_band_change`, `projected_stress_band_change`. Derived
  deterministically by diffing consecutive ledger/history entries at build
  time — no new state file to corrupt.
- `alerts.xml` — a second RSS feed that gets an item **only** when something
  changes. The existing daily `feed.xml` is untouched. The site's data
  section documents both: "poll `changes.json`, or subscribe to `alerts.xml`
  (works with any RSS-to-email bridge)."

No email infrastructure is built. The one-pager + alerts feed are the
adoption hooks the (separate, later) outreach spec will point at.

---

## Error handling

- Forecast fetch failure → publish without the forecast section (or ≤3-day-old
  cache, labeled); never block the site.
- Norm requires ≥3 years of history at the location → hard error at compute
  time (config mistake, not a runtime condition).
- Verification with zero closed windows → "not yet scoreable" output, never
  a fabricated zero.
- `changes.json` on first build (no prior state) → empty list, valid JSON.

## Testing (matches existing discipline: synthetic fixtures, no network)

- `tests/agronomy/test_forecast.py`: norm computation across leap/short
  windows, z-score math, stage-boundary weighting, snapshot append + no-edit
  invariant, verification math (MAE, skill score, band hit rate) on synthetic
  forecasts with known answers, zero-closed-window behavior.
- `tests/site/`: `changes.json` diffing (no change, single change, first
  build), one-pager renders with every as-of date present, forecast section
  omitted cleanly when cache is missing.
- Existing 120+ tests keep passing; the API additions are additive.

## Order of work

1. Forecast fetch + cache + norm/anomaly/projected-stress math + snapshot
   ledger (Component 1 core).
2. Site section, API block, brief sentences EN/VI, workflow wiring.
3. Verification scoring + track-record panel (Component 2).
4. One-pager + `changes.json` + `alerts.xml` (Component 3).
5. README + `docs/ROADMAP-to-Nov-1.md` updated to reflect the new surfaces.

Each step lands with its tests; the site must build green after every step.

## What this never does

- No strategy or backtest tuning; the negative WFE stays published as-is.
- No forecast claim without an issue date, a source label, and a path to
  public verification.
- No adoption/usage claims anywhere — that bar is defined in
  `docs/OUTREACH.md` and is out of scope here.
