"""The weekly one-pager: the whole monitor on a single printable page.

Self-contained HTML (no images, print-first CSS) so it can be forwarded,
attached, or Cmd+P'd into a clean PDF. Rebuilt on every daily run;
a dated archive copy is kept on Fridays.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

SITE_URL = "https://scottdongkhang.github.io/ascent-agri/"


def archive_name(now: "Optional[_dt.datetime]" = None) -> "Optional[str]":
    """Dated archive filename on Fridays (UTC), else None."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return f"{now:%Y-%m-%d}.html" if now.weekday() == 4 else None


def render_onepager(s, brief: str, ledger_line: str,
                    verification_line: str, updated: str) -> str:
    pct = lambda x: f"{x:+.1%}" if x is not None else "n/a"
    o = getattr(s, "outlook", None)
    outlook_row = ""
    if o is not None:
        outlook_row = f"""
  <tr><th>Next 14 days</th><td>{o.expected_mm:.0f} mm expected vs
    {o.norm_mm:.0f} mm norm ({o.anom_z:+.1f}σ) — projected stress:
    <strong>{o.projected_band}</strong>
    <span class="dim">(Open-Meteo forecast, issued {o.issued})</span></td></tr>"""
    farm_row = (f"""
  <tr><th>Farm gate</th><td>{s.farm_gate_line}
    <span class="dim">({s.farm_gate_asof})</span></td></tr>"""
                if s.farm_gate_line else "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robusta Coffee Monitor — weekly one-pager</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font: 13px/1.5 -apple-system, "Segoe UI", Helvetica, Arial,
         sans-serif; color: #1a1a19; background: #fff; padding: 32px; }}
  .sheet {{ max-width: 720px; margin: 0 auto; }}
  h1 {{ font: 500 22px/1.2 Georgia, serif; }}
  .kicker {{ font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
            color: #666; margin-bottom: 6px; }}
  .dim {{ color: #666; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
  th, td {{ text-align: left; padding: 7px 10px 7px 0; vertical-align: top;
           border-bottom: 1px solid #e2e0d8; }}
  th {{ width: 130px; font-weight: 600; font-size: 12px; color: #444; }}
  .brief {{ margin-top: 14px; padding: 12px 14px; background: #f6f5f0;
           border-left: 3px solid #999; }}
  footer {{ margin-top: 18px; font-size: 11px; color: #666; }}
  footer p {{ margin-top: 4px; }}
  @media print {{ body {{ padding: 0; }} .sheet {{ max-width: none; }} }}
</style>
</head>
<body>
<div class="sheet">
  <div class="kicker">ascent-agri · robusta coffee monitor · weekly brief</div>
  <h1>This week in one page</h1>
  <p class="dim">generated {updated} · live version, data files and methods:
    <a href="{SITE_URL}">{SITE_URL.replace("https://", "")}</a></p>
  <table>
  <tr><th>Regime</th><td><strong>{s.label}</strong> ({s.posture.posture}
    posture, ×{s.posture.risk_multiplier:.2f} exposure guide) —
    {max(s.dwell, 1)} sessions
    <span class="dim">(price through {s.price_asof})</span></td></tr>
  <tr><th>Price</th><td>{s.price:,.0f}¢/lb · {pct(s.chg_1w)} on the week ·
    {pct(s.chg_1m)} on the month</td></tr>
  <tr><th>Growing belt</th><td>30-day rainfall anomaly
    {f"{s.rain_z:+.1f}σ" if s.rain_z is not None else "n/a"} at Buon Ma Thuot ·
    crop stage: {s.crop_stage} · stress: {s.crop_stress_band or "n/a"}
    <span class="dim">(weather through {s.weather_asof})</span></td></tr>{outlook_row}{farm_row}
  <tr><th>Track record</th><td>{ledger_line}<br>
    <span class="dim">Forecast verification: {verification_line}</span></td></tr>
  </table>
  <div class="brief">{brief}</div>
  <footer>
    <p>Data: Yahoo Finance · Open-Meteo. Derived series (regimes, anomalies,
    briefs) CC BY 4.0 — free to reuse with attribution to ascent-agri.</p>
    <p>This is not investment advice. An open-source agricultural
    market-intelligence project — every model call is committed to a public,
    append-only ledger before the outcome is known.</p>
  </footer>
</div>
</body>
</html>
"""
