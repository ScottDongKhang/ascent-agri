# ascent-agri

A rigorous, backtested case study applying **regime detection + systematic
position sizing** to **ICE Robusta Coffee futures** (the Vietnamese robusta
benchmark), with **BRL/USD** and **Vietnam Central Highlands weather** as
regime inputs. Research / backtesting only — **not** live trading, and
**long-only by design** (the score-to-position mapping never shorts coffee).

This repository is a standalone Cornell CALS application artifact. It does not
import from or depend on any other project.

## The public monitor (live site)

**https://scottdongkhang.github.io/ascent-agri/** — the *Robusta Coffee
Monitor*: a daily, model-driven read on coffee markets and Vietnamese growing
conditions. Current regime posture in plain English, price with regime
shading, Central Highlands rainfall anomalies, and the BRL/USD driver.

- Rebuilt every weekday by `.github/workflows/update-site.yml` (fetch → build
  → publish). **Fail-safe:** if any data fetch fails, nothing publishes and
  the previous page stays live — panels carry their own as-of dates, so the
  site can go stale but never wrong.
- Generator: `python site/build_site.py` (reads the caches, never fetches).
- Analytics: the page carries a [GoatCounter](https://www.goatcounter.com)
  snippet pointed at `ascent-agri.goatcounter.com`. Claim that code (free,
  ~2 minutes: sign up → site code `ascent-agri`) to see visitor counts;
  until then the snippet is a harmless no-op.
- Feedback from users arrives as GitHub Issues (footer link on the page);
  daily briefs are also available as an RSS feed (`feed.xml`).
- Forward look: a 14-day rainfall outlook for the robusta belt (Open-Meteo
  forecast × the crop phenology calendar). Every issued forecast is committed
  to an append-only snapshot ledger (`data/ledger/weather_forecasts.jsonl`)
  before the outcome is known and scored against realized rainfall once the
  window closes — skill vs climatology, published whatever it says
  (`python -m ascentagri.agronomy.forecast score` reproduces it).
- Machine surfaces for adopters: `api/latest.json` (now with the `forecast`
  block), `api/changes.json` + `alerts.xml` (fire only on regime/stress-band
  changes — poll on your schedule), and `brief/latest.html` (the week on one
  printable page, archived by date on Fridays).

## Research

**[Do growing-region weather anomalies predict coffee futures returns?](docs/research/weather-and-coffee-returns.md)**
([PDF](docs/research/weather-and-coffee-returns.pdf)) — an event study with
a-priori definitions, Monte-Carlo permutation inference, and a built-in
placebo structure (Vietnam weather × robusta, Brazil weather × arabica, and
the cross-pairs as placebos). Headlines: Brazilian dry events precede
arabica rallies with the right sign at every horizon (+5.1pp/5d, n=4,
uncorrected p=0.04 — does not survive multiple-comparisons correction, and
says so); the placebo is null; the primary Vietnam→robusta test is
data-limited until the contract backfill lands; and the July 2021 frost
(+33.8%/5d) was absorbed by the event-cooldown rule — a documented lesson in
event-definition risk. Reproduce with
`python -m ascentagri.research.weather_study`.

Project critical path to the Nov 1 application: `docs/ROADMAP-to-Nov-1.md`.

## Daily operations (the Ascent pattern)

The repository runs itself on the parent platform's daily-agent discipline:

| Cadence | Workflow | What it does |
|---|---|---|
| Weekdays 21:20 UTC | `update-site.yml` | fetch → **append the day's ledger entry** (committed to main) → rebuild → publish |
| Mondays 22:00 UTC | `weekly-research.yml` | re-run the weather study + walk-forward on the newest data, commit refreshed artifacts |

**The ledger** (`data/ledger/forecasts.jsonl`) is the project's public,
append-only track record: every weekday the model's regime call and target
exposure are written down *before* the outcome is known, and scored later
using only prices recorded in the ledger itself (1-day execution delay).
Entries are never edited — `python -m ascentagri.ledger score` reproduces
the track record from the committed file alone. The live site renders it in
"The ledger — the model in public."

Want to contribute (new growing regions, new crops, weather composites)?
See `CONTRIBUTING.md`.

## Run the demo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                # 120 tests: data layer + full pipeline

python demo.py                           # full walk-forward on the stand-in series
python demo.py --series robusta          # the real robusta series (thin, see below)

jupyter nbconvert --to notebook --execute --inplace demo.ipynb   # headless
# …or just open demo.ipynb and Run All (it ships with executed outputs)
```

`demo.py` runs the whole pipeline from the command line — data → regime engine
→ alpha stack → long-only positions → walk-forward out-of-sample backtest —
and prints CAGR, Sharpe, max drawdown, and walk-forward efficiency vs
buy-and-hold, writing artifacts to `outputs/`. `demo.ipynb` walks the same
pipeline with charts: the regime overlay on price, the BRL/USD and weather
features, sleeve scores and exposure, equity vs buy-and-hold, and the
walk-forward report. First run fetches BRL/USD + weather into
`data/processed/` (network needed once; cached after).

## Data sources — and their honest limitations

| Input | Source | Notes |
|---|---|---|
| Robusta continuous series | hand-built from per-contract CSVs via ratio roll-adjustment | **currently 1 contract (RMU26, ~17 months)** — see below |
| Coffee dev stand-in | Yahoo `KC=F` (ICE **Arabica**) via yfinance | 2018–2026; labeled substitute, NOT the robusta deliverable |
| BRL/USD | FRED `DEXBZUS`, **fallback Yahoo `BRL=X`** | FRED was unreachable from the build network, so the cache was built from the Yahoo fallback (same USD/BRL orientation); the fetcher tries FRED first every time |
| Growing-region weather | Open-Meteo Historical Weather API | daily rain/temp at Buon Ma Thuot, Dak Lak (12.68 N, 108.04 E) — free, keyless |

**The robusta data bottleneck.** Only 1 of ~13 needed contracts has been
downloaded, so the real series spans ~17 months and supports only ~2
walk-forward folds. The Databento fetcher (`python -m ascentagri.databento_fetch`)
can backfill the full chain for cents once a `DATABENTO_API_KEY` exists (none
was available at build time), or contracts can be hand-downloaded from Barchart
(`data/raw/README.md`). **Every result on the robusta series is therefore
provisional**; the main demonstration runs on the arabica stand-in, loudly
labeled. Nothing downstream changes when real contracts arrive — rebuild with
`python -m ascentagri.build_series` and rerun.

**Execution convention.** The robusta continuous series is close-only, so
rebalance trades execute at the t+1 close (signal at t). The stand-in has real
opens, which the engine uses (signal at t close, execution at t+1 open).

## What the demo actually shows (read this before quoting numbers)

On the stand-in (26 folds, 2020–2026 OOS, net of 10 bps costs, 1-day execution
delay): **CAGR +0.9%, Sharpe 0.18, max drawdown −9.2%, vs buy-and-hold CAGR
+14.2% — and walk-forward efficiency is negative.** Coffee 2024–25 was a
violent, volatility-fueled bull market; the HMM associates high volatility with
its worst state and the risk overlays cut exposure exactly when the market ran,
so the long-only defensive system under-participates. That is reported plainly
because the deliverable here is the *evaluation machinery that makes lying to
yourself hard* — fold-local regime refits, causal features, purge gaps,
execution delay, transaction costs, and visible failed folds — not a trading
result. (The parent project's own honest post-audit number was a modest Sharpe
~0.41; modest-and-true beats impressive-and-leaky.)

## Layout

```
ascentagri/
  contracts.py / loader.py / roll.py /      data foundation: parse contract codes,
    validate.py / build_series.py             load raw CSVs, ratio roll-adjustment,
    databento_fetch.py / vendor_fetch.py      diagnostics, fetchers
  macro_fetch.py      BRL/USD (FRED→Yahoo fallback) + Open-Meteo weather caches
  agronomy/           phenology calendar + stage-weighted stress, farm-gate
                        economics, forecast.py (14-day outlook, snapshot
                        ledger, verification vs climatology)
  config.py           one dataclass of coffee-tuned knobs (AgriConfig)
  types.py            AgentOutput — standardized strategy-run record
  regime/             features (price + BRL + weather anomalies), HMM/Markov model,
                        asymmetric-hysteresis decision layer, crisis override,
                        posture, structural breaks, engine
  alpha/              trend + meanrev sleeves (rolling TS z-scores — the
                        cross-sectional z-score zeroes out at N=1), vol_sizing
                        (vol target + 200d MA), Bayesian sleeve meta_learner,
                        stack (long-only combiner)
  backtest/           engine (1-day delay, costs, cash-aware drift), costs
  research/           walk-forward splits, CPCV, evaluation metrics + WFE,
                        walk_forward_runner (single-series fold loop)
demo.py               runnable end-to-end script
demo.ipynb            the same pipeline with charts (ships executed)
data/{raw,interim,processed}/   raw contracts (git-ignored) → caches
outputs/{wf_results,backtest}/  generated reports (git-ignored)
tests/                120 tests mirroring the packages; synthetic fixtures, no network
docs/                 design specs
```

## Method summary

- **Roll adjustment** (data foundation): roll 5 business days before First
  Notice Day; ratio back-adjustment preserves percentage returns across every
  splice. Spec: `docs/superpowers/specs/2026-06-24-ascent-agri-roll-adjustment-design.md`.
- **Regime**: Gaussian HMM (K=3 default, walk-forward K selection available) on
  trailing price/FX/weather features; entropy filter → asymmetric hysteresis
  (downgrades at 0.40, upgrades at 0.70, 5-day dwell) → rule-based crisis
  override (5-day return < −10% AND 21d vol > 45%).
- **Alpha**: trend (momentum composite incl. 11-1 skip-month, MACD, SMA cross)
  75% + short-term mean-reversion 25%, each normalized by a rolling 252-day
  time-series z-score; regime tilts the mix per date.
- **Sizing**: long-only `clip(score, 0, 1.5)/1.5` × regime risk multiplier ×
  20%-vol-target overlay × 200d-MA filter (×0.70 below).
- **Evaluation**: rolling walk-forward (train 504d / test 63d / purge 5d),
  regime refit on each training slice only, backtest with 1-day execution delay
  and 10 bps costs, WFE = mean(OOS Sharpe / IS Sharpe) over folds with IS
  Sharpe ≥ 0.1. Failed folds are logged with fold id, stage, and exception —
  never silently zeroed.

## Provenance of the code

Ported from the Ascent Capital equity platform by copying/rewriting into this
namespace (no imports, no shared code). Near-verbatim: regime model/decision/
posture/breaks, splits/CPCV/evaluation, backtest engine/costs, meta-learner.
Rewritten for a single instrument: regime features (universe breadth → BRL +
weather), alpha normalization (cross-sectional → rolling time-series z-score),
walk-forward runner (universe/sector machinery → single-series fold loop).
Two latent source bugs surfaced and fixed during the port: the Markov backend
returned train-window probabilities regardless of prediction input, and the
backtest drift renormalization dropped the cash bucket at fractional exposure.
