"""ascent-agri demo — full backtest pipeline, end to end, from the command line.

Runs: data load → regime engine → alpha stack → long-only positions →
walk-forward out-of-sample evaluation → summary stats (CAGR, Sharpe,
drawdown, walk-forward efficiency) vs buy-and-hold.

Usage:
    python demo.py                     # arabica KC=F dev stand-in (2018–2026, full WF)
    python demo.py --series robusta    # the real robusta series (358 days, thin WF)
    python demo.py --fast              # smaller windows / fewer HMM restarts

Series choice, honestly stated:
  * `standin` (default): ICE Arabica KC=F from Yahoo — NOT the robusta
    deliverable, but the only coffee series long enough (2018→2026) for a
    meaningful walk-forward. Established in this repo as the labeled dev
    stand-in while robusta contracts are collected.
  * `robusta`: the real roll-adjusted robusta series — currently built from
    a single contract (RMU26, ~17 months), so the walk-forward has only ~2
    folds. This is the data bottleneck, not a code limitation.

External regime inputs (BRL/USD + Central Highlands weather) are fetched
once into data/processed/ (see ascentagri/macro_fetch.py) and cached.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

from ascentagri.config import get_config
from ascentagri.macro_fetch import ensure_caches, load_brlusd, load_weather
from ascentagri.regime.posture import compute_posture_from_regime
from ascentagri.research.walk_forward_runner import print_wf_report, run_walk_forward
from ascentagri.types import AgentOutput

STANDIN_CSV = ROOT / "data" / "processed" / "coffee_KCF_yahoo.csv"
ROBUSTA_CSV = ROOT / "data" / "processed" / "robusta_continuous.csv"
OUT_WF = ROOT / "outputs" / "wf_results"
OUT_BT = ROOT / "outputs" / "backtest"


def load_series(name: str) -> "tuple[pd.Series, pd.Series | None, str]":
    """Return (close, open_or_None, provenance_label) for a series name."""
    if name == "standin":
        if not STANDIN_CSV.exists():
            sys.exit(f"Missing {STANDIN_CSV} — run: python -m ascentagri.vendor_fetch")
        df = pd.read_csv(STANDIN_CSV, parse_dates=["date"]).set_index("date").sort_index()
        label = ("ICE Arabica KC=F via Yahoo — DEV STAND-IN, NOT the robusta "
                 "deliverable (used for its 2018-2026 history)")
        return df["close"], df.get("open"), label
    if name == "robusta":
        if not ROBUSTA_CSV.exists():
            sys.exit(f"Missing {ROBUSTA_CSV} — run: python -m ascentagri.build_series")
        df = pd.read_csv(ROBUSTA_CSV, parse_dates=["date"]).set_index("date").sort_index()
        label = ("roll-adjusted continuous ICE Robusta — REAL deliverable, but "
                 "currently a single contract (~17 months): walk-forward is thin")
        return df["close"], None, label
    sys.exit(f"Unknown series '{name}' (use: standin | robusta)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="standin", choices=["standin", "robusta"])
    ap.add_argument("--fast", action="store_true",
                    help="smaller windows / fewer HMM restarts (quicker, noisier)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-fold lines")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")

    print("=" * 55)
    print("  ASCENT-AGRI DEMO — robusta coffee regime backtest")
    print("=" * 55)

    # ── data ────────────────────────────────────────────────────────────
    close, open_, label = load_series(args.series)
    print(f"\n[data] price series : {args.series} ({len(close)} rows, "
          f"{close.index[0].date()} → {close.index[-1].date()})")
    print(f"[data] provenance   : {label}")

    try:
        ensure_caches()
        brl = load_brlusd()
        weather = load_weather()
        print(f"[data] BRL/USD      : {len(brl)} rows "
              f"({brl.index[0].date()} → {brl.index[-1].date()})")
        print(f"[data] weather      : {len(weather)} rows, Buon Ma Thuot "
              f"(Central Highlands robusta belt)")
    except Exception as exc:
        print(f"[data] WARNING: external inputs unavailable ({exc}) — "
              f"regime features fall back to price-only")
        brl, weather = None, None

    # ── config ──────────────────────────────────────────────────────────
    if args.series == "robusta":
        # short series: shrink windows so at least a couple of folds exist
        cfg = get_config(wf_train_days=252, wf_test_days=63, wf_step_days=63,
                         wf_min_train_days=200)
        print("\n[config] short-series mode: train=252d test=63d "
              "(the 358-row series allows ~2 folds — disclosed limitation)")
    elif args.fast:
        cfg = get_config(wf_train_days=378, wf_test_days=63, wf_step_days=126,
                         wf_hmm_restarts=3)
        print("\n[config] fast mode: train=378d test=63d step=126d restarts=3")
    else:
        cfg = get_config()
        print(f"\n[config] train={cfg.wf_train_days}d test={cfg.wf_test_days}d "
              f"step={cfg.wf_step_days}d purge={cfg.wf_purge_days}d "
              f"K={cfg.wf_regime_k} rebalance every {cfg.rebalance_freq_days}d")

    # ── walk-forward ────────────────────────────────────────────────────
    print()
    result = run_walk_forward(
        close=close,
        open_prices=open_,
        brl_usd=brl,
        weather=weather,
        config=cfg,
        verbose=not args.quiet,
    )

    print()
    print_wf_report(result)

    # ── latest posture (plain-English summary of the last OOS regime) ───
    last_fold = next((s for s in reversed(result.fold_summaries)
                      if s.get("status") == "ok"), None)
    if last_fold:
        posture = compute_posture_from_regime(
            asof=str(result.oos_returns.index[-1].date()),
            regime_label=str(last_fold["dominant_regime"]),
            probs={},  # per-state probs not carried in the fold summary
            days_in_regime=0,
            min_confidence=0.0,
        )
        print(f"\n[posture] last OOS fold regime: {posture.regime_label} "
              f"→ {posture.posture} (risk multiplier {posture.risk_multiplier})")
        print(f"[posture] {posture.notes}")

    # ── standardized strategy output (AgentOutput contract) ─────────────
    agent_out = AgentOutput(
        agent_id=f"coffee_{args.series}",
        as_of_date=date.today(),
        target_weights={cfg.series_name: round(float(result.positions.iloc[-1]), 4)},
        regime_signal=str(last_fold["dominant_regime"]) if last_fold else None,
        skill_score=round(float(result.report["arithmetic_sharpe"]), 4),
        metadata={"wfe": result.report["wfe"], "n_folds": result.report["n_folds"]},
    )
    print(f"\n{agent_out.summary()}")

    # ── artifacts ────────────────────────────────────────────────────────
    OUT_WF.mkdir(parents=True, exist_ok=True)
    OUT_BT.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    report_path = OUT_WF / f"wf_report_{args.series}_{stamp}.json"
    payload = dict(result.report)
    payload["series"] = args.series
    payload["provenance"] = label
    payload["fold_summaries"] = result.fold_summaries
    report_path.write_text(json.dumps(payload, indent=2, default=str))

    equity = result.equity_curve.rename("equity")
    bh = (1 + result.benchmark_returns).cumprod().rename("buy_and_hold")
    curve_path = OUT_BT / f"wf_equity_{args.series}_{stamp}.csv"
    pd.concat([equity, bh, result.positions.rename("exposure")], axis=1) \
        .to_csv(curve_path, index_label="date")

    print(f"\n[artifacts] {report_path.relative_to(ROOT)}")
    print(f"[artifacts] {curve_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
