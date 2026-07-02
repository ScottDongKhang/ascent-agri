"""Do growing-region weather anomalies predict coffee futures returns?

A hypothesis-driven event study with a built-in control structure:

    signal                          target            expectation
    ─────────────────────────────   ───────────────   ─────────────────────────
    Vietnam (Buon Ma Thuot) dry     robusta           positive fwd returns
    Brazil (Sul de Minas) dry       arabica (KC=F)    positive fwd returns
    Brazil cold snap (tmin < 6°C)   arabica (KC=F)    positive fwd returns
    Vietnam dry                     arabica           weak/none (cross-placebo)
    Brazil dry                      robusta           weak/none (cross-placebo)

Design choices, all fixed a priori (no tuning on outcomes):
  * Anomalies are CAUSAL: each 30-day rainfall window is z-scored against the
    location's own trailing 365 days (same helper the regime engine uses).
  * A "dry event" is the first day the anomaly closes below −1.25σ after
    ≥30 days above it (cooldown prevents double-counting one drought).
  * A "cold event" is the first day 2m tmin < 6.0°C (ground-frost risk proxy —
    ERA5-scale 2m temperatures over this grid cell never reach 0°C) after
    ≥30 days without one.
  * Outcome: forward 5/21/63 trading-day cumulative returns.
  * Inference: the mean forward return over events is compared with the
    distribution of means over 2,000 same-size random draws of trading days
    (a Monte-Carlo permutation test); we report the two-sided p-value.
  * Lead-lag: daily Spearman correlation between the anomaly and forward
    returns, with a moving-block bootstrap CI (block = 21 days) to respect
    serial dependence.

Event counts are small (single-digit for some pairs) and are reported next to
every estimate. Run:  python -m ascentagri.research.weather_study
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..regime.features import _trailing_anomaly_z

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "research"

# ── a-priori parameters (fixed before looking at outcomes) ─────────────────
DRY_THRESHOLD = -1.25          # 30d rainfall anomaly z-score
COLD_TMIN_C = 6.0              # 2m tmin proxy for ground-frost risk
EVENT_COOLDOWN_DAYS = 30       # calendar days between events
HORIZONS = (5, 21, 63)         # forward trading days
N_PERMUTATIONS = 2000
BLOCK_BOOTSTRAP_N = 1000
BLOCK_SIZE = 21
SEED = 20260702


@dataclass
class EventStudyResult:
    signal: str
    target: str
    n_events: int
    event_dates: List[str]
    horizons: Dict[str, Dict]          # {"5d": {mean_fwd, baseline_mean, p_value}}


@dataclass
class LeadLagResult:
    signal: str
    target: str
    horizons: Dict[str, Dict]          # {"21d": {spearman, ci_lo, ci_hi, n}}


@dataclass
class StudyReport:
    params: Dict
    event_studies: List[EventStudyResult] = field(default_factory=list)
    lead_lags: List[LeadLagResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "params": self.params,
            "event_studies": [asdict(e) for e in self.event_studies],
            "lead_lags": [asdict(l) for l in self.lead_lags],
        }, indent=2)


# ── building blocks ─────────────────────────────────────────────────────────

def rain_anomaly(weather: pd.DataFrame, window: int = 30,
                 baseline: int = 365) -> pd.Series:
    """Causal 30d rainfall anomaly z-score (vs the location's trailing year)."""
    return _trailing_anomaly_z(weather["rain_mm"].astype(float),
                               window=window, baseline=baseline)


def detect_threshold_events(series: pd.Series, threshold: float,
                            below: bool = True,
                            cooldown_days: int = EVENT_COOLDOWN_DAYS) -> pd.DatetimeIndex:
    """First crossing events: the day `series` closes beyond `threshold`
    after at least `cooldown_days` without doing so."""
    s = series.dropna()
    hit = s < threshold if below else s > threshold
    events = []
    last_event: Optional[pd.Timestamp] = None
    prev_hit = False
    for date, h in hit.items():
        if h and not prev_hit:
            if last_event is None or (date - last_event).days >= cooldown_days:
                events.append(date)
                last_event = date
        prev_hit = bool(h)
    return pd.DatetimeIndex(events)


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Cumulative return over the NEXT `horizon` trading days (t+1..t+h)."""
    close = close.sort_index()
    return close.shift(-horizon) / close - 1


def _align_events_to_trading(events: pd.DatetimeIndex,
                             trading_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map each (calendar-day) event to the first trading day at or after it."""
    aligned = []
    for e in events:
        pos = trading_index.searchsorted(e)
        if pos < len(trading_index):
            aligned.append(trading_index[pos])
    return pd.DatetimeIndex(sorted(set(aligned)))


def event_study(close: pd.Series, events: pd.DatetimeIndex,
                horizons=HORIZONS, n_perm: int = N_PERMUTATIONS,
                seed: int = SEED) -> Dict[str, Dict]:
    """Mean forward return after events vs a permutation baseline.

    p_value is two-sided: the fraction of random same-size date draws whose
    mean forward return is at least as extreme (in absolute distance from the
    baseline mean) as the observed one.
    """
    rng = np.random.default_rng(seed)
    close = close.sort_index()
    out: Dict[str, Dict] = {}
    for h in horizons:
        fwd = forward_returns(close, h).dropna()
        ev = _align_events_to_trading(events, fwd.index)
        ev_fwd = fwd.reindex(ev).dropna()
        n = len(ev_fwd)
        if n == 0:
            out[f"{h}d"] = {"n": 0}
            continue
        obs_mean = float(ev_fwd.mean())
        pool = fwd.values
        base_mean = float(fwd.mean())
        perm_means = np.array([
            pool[rng.integers(0, len(pool), size=n)].mean()
            for _ in range(n_perm)
        ])
        p = float(np.mean(np.abs(perm_means - base_mean)
                          >= abs(obs_mean - base_mean)))
        out[f"{h}d"] = {
            "n": n,
            "mean_fwd_return": round(obs_mean, 5),
            "baseline_mean": round(base_mean, 5),
            "excess": round(obs_mean - base_mean, 5),
            "p_value": round(p, 4),
            "hit_rate_positive": round(float((ev_fwd > 0).mean()), 3),
        }
    return out


def lead_lag(anomaly: pd.Series, close: pd.Series, horizons=HORIZONS,
             n_boot: int = BLOCK_BOOTSTRAP_N, block: int = BLOCK_SIZE,
             seed: int = SEED) -> Dict[str, Dict]:
    """Spearman correlation of the daily anomaly with forward returns,
    moving-block-bootstrap 95% CI. Dry (negative anomaly) preceding price
    rises shows up as a NEGATIVE correlation."""
    rng = np.random.default_rng(seed)
    close = close.sort_index()
    anomaly_t = anomaly.reindex(close.index, method="ffill")
    out: Dict[str, Dict] = {}
    for h in horizons:
        fwd = forward_returns(close, h)
        df = pd.DataFrame({"a": anomaly_t, "f": fwd}).dropna()
        n = len(df)
        if n < 100:
            out[f"{h}d"] = {"n": n}
            continue
        rho = float(df["a"].rank().corr(df["f"].rank()))
        # moving-block bootstrap
        n_blocks = int(np.ceil(n / block))
        stats = []
        for _ in range(n_boot):
            starts = rng.integers(0, n - block, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            sample = df.iloc[idx]
            stats.append(float(sample["a"].rank().corr(sample["f"].rank())))
        lo, hi = np.percentile(stats, [2.5, 97.5])
        out[f"{h}d"] = {
            "n": n,
            "spearman": round(rho, 4),
            "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4),
        }
    return out


# ── the study itself ─────────────────────────────────────────────────────────

def run_study(arabica_close: pd.Series,
              robusta_close: Optional[pd.Series],
              weather_vn: pd.DataFrame,
              weather_br: pd.DataFrame,
              verbose: bool = True) -> StudyReport:
    report = StudyReport(params={
        "dry_threshold_z": DRY_THRESHOLD,
        "cold_tmin_c": COLD_TMIN_C,
        "cooldown_days": EVENT_COOLDOWN_DAYS,
        "horizons_days": list(HORIZONS),
        "n_permutations": N_PERMUTATIONS,
        "block_bootstrap": {"n": BLOCK_BOOTSTRAP_N, "block": BLOCK_SIZE},
        "seed": SEED,
        "arabica_span": f"{arabica_close.index[0].date()}→{arabica_close.index[-1].date()}",
        "robusta_span": (f"{robusta_close.index[0].date()}→{robusta_close.index[-1].date()}"
                         if robusta_close is not None else "unavailable"),
    })

    anom_vn = rain_anomaly(weather_vn)
    anom_br = rain_anomaly(weather_br)
    dry_vn = detect_threshold_events(anom_vn, DRY_THRESHOLD, below=True)
    dry_br = detect_threshold_events(anom_br, DRY_THRESHOLD, below=True)
    cold_br = detect_threshold_events(weather_br["tmin_c"].astype(float),
                                      COLD_TMIN_C, below=True)

    targets = {"arabica": arabica_close}
    if robusta_close is not None and len(robusta_close) > 120:
        targets["robusta"] = robusta_close

    signal_sets = {
        "vn_dry": dry_vn,
        "br_dry": dry_br,
        "br_cold": cold_br,
    }
    anomalies = {"vn_rain_anom": anom_vn, "br_rain_anom": anom_br}

    for sig_name, events in signal_sets.items():
        for tgt_name, close in targets.items():
            in_span = events[(events >= close.index[0]) & (events <= close.index[-1])]
            res = EventStudyResult(
                signal=sig_name, target=tgt_name,
                n_events=len(in_span),
                event_dates=[str(d.date()) for d in in_span],
                horizons=event_study(close, in_span),
            )
            report.event_studies.append(res)
            if verbose:
                h21 = res.horizons.get("21d", {})
                print(f"[study] {sig_name:>8} → {tgt_name:<8} "
                      f"events={res.n_events:>2}  21d excess="
                      f"{h21.get('excess', float('nan')):+.4f}  "
                      f"p={h21.get('p_value', float('nan'))}")

    for a_name, anom in anomalies.items():
        for tgt_name, close in targets.items():
            res = LeadLagResult(
                signal=a_name, target=tgt_name,
                horizons=lead_lag(anom, close),
            )
            report.lead_lags.append(res)
            if verbose:
                h21 = res.horizons.get("21d", {})
                print(f"[study] {a_name:>13} ~ {tgt_name:<8} 21d Spearman="
                      f"{h21.get('spearman', float('nan')):+.3f} "
                      f"CI [{h21.get('ci_lo', float('nan')):+.3f}, "
                      f"{h21.get('ci_hi', float('nan')):+.3f}]")

    return report


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    from ..macro_fetch import load_weather, load_weather_brazil

    arabica = (pd.read_csv(ROOT / "data/processed/coffee_KCF_yahoo.csv",
                           parse_dates=["date"])
               .set_index("date").sort_index()["close"])
    robusta_csv = ROOT / "data/processed/robusta_continuous.csv"
    robusta = None
    if robusta_csv.exists():
        robusta = (pd.read_csv(robusta_csv, parse_dates=["date"])
                   .set_index("date").sort_index()["close"])

    report = run_study(arabica, robusta, load_weather(), load_weather_brazil())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "weather_study.json"
    out_path.write_text(report.to_json())
    print(f"[study] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
