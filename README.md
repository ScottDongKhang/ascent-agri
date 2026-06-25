# ascent-agri

A rigorous, backtested case study applying **regime detection + multi-agent debate**
to **ICE Robusta Coffee futures** (Vietnamese robusta benchmark). Research / backtesting
only — **not** live trading.

This repository is a standalone Cornell CALS application artifact. It does not import
from or depend on any other project.

## What this session built

The **data foundation**: a clean, roll-adjusted *continuous* front-month price series
for ICE Robusta Coffee, spliced from individual expired contracts using **proportional
(ratio) back-adjustment**. Everything downstream (regime model, debate layer) consumes
this series, so it has to be honest first.

## Why roll adjustment matters

Robusta contracts expire every 2 months (delivery months Jan/Mar/May/Jul/Sep/Nov —
6 per year). No single contract trades for more than ~2 years. To get long history you
splice many contracts together. Naively concatenating them injects a fake price jump at
every roll, caused by the shape of the futures curve (contango/backwardation), not by
the market. Left uncorrected, a regime model hallucinates a "regime change" every
~2 months. We remove these jumps with ratio adjustment, which preserves percentage
returns across each splice exactly — and returns are what the regime model consumes.

## Data acquisition — two compliant paths into `data/raw/`

Both paths produce the same thing: per-contract OHLCV CSVs named `RM<code><yy>.csv` in
`data/raw/`. Nothing downstream changes.

**Path 1 — Databento API (recommended).** Compliant programmatic access (an API, not
scraping). New accounts get **$125 in promo credits**; daily OHLCV for the whole chain
costs cents. Reaches further back than Barchart's free tier. Individual contracts, our
exact granularity.

```bash
export DATABENTO_API_KEY=db-xxxxxxxx
python -m ascentagri.databento_fetch --list   # coverage + cost + real symbols (free metadata)
python -m ascentagri.databento_fetch          # writes RM*.csv into data/raw/
```

**Path 2 — manual Barchart download.** We do **not** scrape Barchart (ToS prohibits
automated extraction; `robots.txt` disallows the `/proxies/` CSV endpoint). The compliant
route is to **hand-download** each contract's CSV from the web UI and drop it in
`data/raw/`. Note the **free tier only serves ~2 years prior to today**, so this path
yields ~2 years of history, not more. See **`data/raw/README.md`** for steps + naming.

Either way, then run `python -m ascentagri.build_series`.

**Dev stand-in (while robusta data is being gathered).** A free, daily, continuous
*robusta* series does not exist in any compliant programmatic form (Yahoo has no
downloadable robusta; Stooq is JS-gated; Barchart robusta is paywalled past ~1/day).
To unblock downstream work, `ascentagri/vendor_fetch.py` pulls **`KC=F` (ICE Arabica
coffee)** from Yahoo as a clearly-labeled development stand-in — same tidy columns, so
the real robusta series swaps in later with no code change:

```bash
python -m ascentagri.vendor_fetch   # -> data/processed/coffee_KCF_yahoo.csv (+ PROVENANCE.md)
```
This is **Arabica, not the robusta deliverable** — for building/tuning the pipeline only.

## Layout

```
ascentagri/
  contracts.py        parse contract codes (e.g. RMK24), First Notice Day + roll date, chain gaps
  loader.py           read per-contract CSVs from data/raw/ into tidy OHLCV frames
  roll.py             proportional (ratio) back-adjustment; returns ALL intermediates
  validate.py         returns-around-each-roll diagnostics, offset sensitivity, plots
  databento_fetch.py  optional: pull individual contracts from the Databento API into data/raw/
  build_series.py     real-data runner (load -> roll -> reports -> CSV/plot)
  worked_example.py   synthetic proof of the pipeline (no data needed)
data/raw/        <- you drop manually-downloaded Barchart CSVs here (git-ignored)
data/interim/    naive spliced (un-adjusted) series
data/processed/  final ratio-adjusted continuous series
notebooks/       01_build_continuous_series.ipynb  (load -> roll -> validate)
tests/           pytest suite + synthetic fixtures with a known curve shape
docs/            design spec
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                  # runs the synthetic worked example + property tests
python -m ascentagri.worked_example        # prints the worked example + offset sensitivity
```

## Method summary

- **Roll trigger:** roll `roll_offset_bd` business days before **First Notice Day** (FND).
  FND = 4th business day before the 1st business day of the delivery month (ICE rule).
  Default `roll_offset_bd = 5`. (Calendar-based, deterministic, reproducible.)
- **Ratio window:** average the close over the **4 trading days immediately before the
  roll date** on both the expiring and next contract; `ratio = avg_next / avg_exp`.
- **Back-adjustment:** the most recent contract is left unadjusted; every earlier segment
  is multiplied by the cumulative product of ratios at/after its rolls. Percentage
  returns are preserved across every splice.

Single source of truth for the method and decisions:
`docs/superpowers/specs/2026-06-24-ascent-agri-roll-adjustment-design.md`.
