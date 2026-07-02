# Contributing to ascent-agri

This is an open agricultural market-intelligence project. Contributions that
extend the science or the coverage are welcome; contributions that make the
backtest look better are not (see "Ground rules").

## Get running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                     # full suite, no network needed for most tests
python demo.py                # end-to-end backtest with printed report
python site/build_site.py     # build the monitor locally (fetches caches once)
```

## Good first contributions

- **Add a growing region.** `ascentagri/macro_fetch.py` has a generic
  `fetch_weather_at(lat, lon, ...)` — Indonesia (Lampung, robusta), Colombia
  (Huila, arabica), or Uganda are natural next stations. Wire a new cache +
  a feature group and the regime engine picks it up.
- **Add a crop.** The pipeline is single-series by design: cocoa (another
  weather-driven soft with a famous 2024 squeeze) or sugar would exercise it
  with almost no code change beyond a fetcher.
- **Production-weighted weather composites.** Each region is currently one
  grid cell; a multi-station composite weighted by production share is a
  straightforward, high-value refinement (and is named in the paper).
- **Severity-weighted event definitions** for the weather study — the
  pre-registered refinement motivated by the 2021 frost finding.
- Anything labeled in the
  [issue tracker](https://github.com/ScottDongKhang/ascent-agri/issues).

## Ground rules (the integrity constraints)

1. **Causality is non-negotiable.** Every feature and anomaly uses trailing
   data only. Tests assert this — keep them passing.
2. **No post-hoc event redefinition.** Study parameters are fixed a priori;
   refinements are added as new, clearly labeled analyses, never by editing
   old ones after seeing outcomes.
3. **The ledger is append-only.** Entries are never edited or deleted.
4. **No data under misleading names.** Provenance is labeled everywhere
   (the arabica stand-in is loudly not-robusta); keep it that way.
5. **Compliant data only.** No scraping. Keyless public APIs (Open-Meteo,
   FRED) or personal-use downloads that stay out of git (`data/raw/` is
   ignored for a reason).
6. **Honest reporting.** Negative and null results ship. The project's
   credibility is that it publishes its own unflattering numbers.

## How the daily loop works

- `.github/workflows/update-site.yml` (weekdays 21:20 UTC): fetch → append
  the day's ledger entry (committed to main) → rebuild the site → publish.
  A failed fetch publishes nothing — the site goes stale, never wrong.
- `.github/workflows/weekly-research.yml` (Mondays): re-runs the weather
  study and walk-forward on the newest data and commits refreshed artifacts.

Tests live in `tests/`, mirroring the package layout. New code needs tests;
synthetic fixtures, no network in unit tests.
