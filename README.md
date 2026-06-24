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

## Data acquisition (READ THIS — it is manual on purpose)

We do **not** scrape Barchart. Their Terms of Use prohibit automated extraction, and
`robots.txt` disallows the `/proxies/` path their CSV endpoint uses. The compliant path
is to **manually download** each contract's CSV from the Barchart web UI (personal /
academic use) and drop the files into `data/raw/`.

See **`data/raw/README.md`** for the exact download steps and file-naming convention.

## Layout

```
ascentagri/
  contracts.py   parse contract codes (e.g. RMK24), compute First Notice Day + roll date
  loader.py      read Barchart per-contract CSVs from data/raw/ into tidy OHLCV frames
  roll.py        proportional (ratio) back-adjustment; returns ALL intermediates
  validate.py    returns-around-each-roll diagnostics, offset sensitivity, plots
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
