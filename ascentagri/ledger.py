"""The daily ledger — the model's public, unedited track record.

Mirrors the parent platform's daily-agent discipline: every weekday the
pipeline records what the model believes BEFORE the outcome is known
(regime call, target exposure, the plain-English brief), appends it to
data/ledger/forecasts.jsonl, and the publish workflow commits that file to
main. Scoring then marks every past entry against realized prices. Entries
are append-only and idempotent per date; nothing is ever edited or deleted —
if the model is wrong, the ledger says so forever.

Scoring convention (conservative, matches the backtest engine's 1-day
execution delay): the exposure recorded on signal date t is assumed filled
at the NEXT close, so it earns the close-to-close return of the day after
that. Formally, using the ledger's own recorded closes,
    strategy_return(t_{i+2}) = exposure(t_i) applied over t_{i+1} → t_{i+2}.
With daily entries this is exposure(d-2) × r(d).

Usage:
    python -m ascentagri.ledger append    # compute today's entry, append
    python -m ascentagri.ledger score     # print the scored track record
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "ledger" / "forecasts.jsonl"
SCHEMA_VERSION = 1


# ── entries ─────────────────────────────────────────────────────────────────

def read_ledger(path: Path = LEDGER_PATH) -> List[Dict]:
    """All entries, oldest first. Malformed lines are skipped loudly."""
    if not path.exists():
        return []
    entries = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("ledger: skipping malformed line %d", i + 1)
    entries.sort(key=lambda e: e["date"])
    return entries


def append_entry(entry: Dict, path: Path = LEDGER_PATH) -> bool:
    """Append one entry. Returns False (no write) if the date already has
    one — the ledger never rewrites history."""
    existing = {e["date"] for e in read_ledger(path)}
    if entry["date"] in existing:
        log.info("ledger: entry for %s already exists — skipping", entry["date"])
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return True


def build_todays_entry() -> Dict:
    """Compute the model's current view from the caches (no fetching here —
    the workflow fetches first, same fail-safe as the site builder)."""
    from .alpha.stack import build_positions
    from .config import get_config
    from .macro_fetch import load_brlusd, load_weather
    from .regime.engine import RegimeEngine

    price_csv = ROOT / "data" / "processed" / "coffee_KCF_yahoo.csv"
    close = (pd.read_csv(price_csv, parse_dates=["date"])
             .set_index("date").sort_index()["close"])
    brl = load_brlusd()
    weather = load_weather()

    engine = RegimeEngine()
    engine.fit(close, brl_usd=brl, weather=weather,
               run_model_selection=False, k_override=3, hmm_restarts=5)
    signals = engine.get_signal_series()
    positions, _ = build_positions(close, regime_signal_df=signals,
                                   config=get_config())
    last = signals.iloc[-1]
    return {
        "schema": SCHEMA_VERSION,
        "date": str(close.index[-1].date()),      # signal date = last close
        "close": round(float(close.iloc[-1]), 2),
        "label": str(last["label"]),
        "risk_multiplier": round(float(last["risk_multiplier"]), 4),
        "exposure": round(float(positions.iloc[-1]), 4),
        "series": "KC=F (arabica benchmark stand-in)",
    }


# ── scoring ─────────────────────────────────────────────────────────────────

@dataclass
class LedgerScore:
    n_entries: int
    n_scored_days: int
    start: Optional[str]
    end: Optional[str]
    strategy_return: float          # cumulative, net of nothing (no cost model)
    bh_return: float                # buy-and-hold same window
    strategy_daily: pd.Series
    bh_daily: pd.Series
    mean_exposure: float
    label_counts: Dict[str, int]

    def summary_line(self) -> str:
        if self.n_scored_days == 0:
            return (f"{self.n_entries} entries since {self.start} — too young "
                    f"to score (needs 3+ consecutive entries).")
        return (f"{self.n_entries} entries · {self.n_scored_days} scored days "
                f"({self.start} → {self.end}) · strategy {self.strategy_return:+.2%} "
                f"vs buy-and-hold {self.bh_return:+.2%} · mean exposure "
                f"{self.mean_exposure:.2f}")


def score_ledger(entries: Optional[List[Dict]] = None) -> LedgerScore:
    """Mark all past entries against the closes recorded in later entries.

    Uses ONLY prices recorded inside the ledger itself, so the score is
    reproducible from the committed file alone. 1-day execution delay:
    exposure(t_i) earns the return t_{i+1} → t_{i+2}.
    """
    entries = read_ledger() if entries is None else sorted(
        entries, key=lambda e: e["date"])
    n = len(entries)
    if n < 3:
        return LedgerScore(
            n_entries=n, n_scored_days=0,
            start=entries[0]["date"] if entries else None,
            end=entries[-1]["date"] if entries else None,
            strategy_return=0.0, bh_return=0.0,
            strategy_daily=pd.Series(dtype=float),
            bh_daily=pd.Series(dtype=float),
            mean_exposure=float(np.mean([e["exposure"] for e in entries])) if entries else 0.0,
            label_counts=_label_counts(entries),
        )

    dates = [pd.Timestamp(e["date"]) for e in entries]
    closes = np.array([float(e["close"]) for e in entries])
    expo = np.array([float(e["exposure"]) for e in entries])

    strat, bh, idx = [], [], []
    for i in range(2, n):
        r = closes[i] / closes[i - 1] - 1.0
        strat.append(expo[i - 2] * r)
        bh.append(r)
        idx.append(dates[i])

    strat_s = pd.Series(strat, index=idx)
    bh_s = pd.Series(bh, index=idx)
    return LedgerScore(
        n_entries=n,
        n_scored_days=len(strat_s),
        start=entries[0]["date"], end=entries[-1]["date"],
        strategy_return=float((1 + strat_s).prod() - 1),
        bh_return=float((1 + bh_s).prod() - 1),
        strategy_daily=strat_s, bh_daily=bh_s,
        mean_exposure=float(expo.mean()),
        label_counts=_label_counts(entries),
    )


def _label_counts(entries: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in entries:
        out[e.get("label", "?")] = out.get(e.get("label", "?"), 0) + 1
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    import sys
    logging.basicConfig(level=logging.ERROR)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "append"
    if cmd == "append":
        entry = build_todays_entry()
        wrote = append_entry(entry)
        print(f"[ledger] {'appended' if wrote else 'already present'}: "
              f"{entry['date']} label={entry['label']} "
              f"exposure={entry['exposure']:.2f} close={entry['close']}")
        return 0
    if cmd == "score":
        score = score_ledger()
        print("[ledger]", score.summary_line())
        print("[ledger] regime calls:", score.label_counts)
        return 0
    print(f"unknown command '{cmd}' (use: append | score)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
