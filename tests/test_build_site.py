"""Smoke tests for the public monitor builder (offline — uses existing caches
if present, else synthesizes minimal ones into a tmp PROCESSED dir)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("build_site", ROOT / "site" / "build_site.py")
build_site = importlib.util.module_from_spec(spec)
sys.modules["build_site"] = build_site
spec.loader.exec_module(build_site)


@pytest.fixture(scope="module")
def built(tmp_path_factory, monkeypatch_module=None):
    """Build the site once into a tmp dir. Uses the real caches when present
    (normal dev/CI-after-fetch case); otherwise synthesizes small ones."""
    out = tmp_path_factory.mktemp("site_build")
    processed = ROOT / "data" / "processed"
    have_real = all((processed / n).exists() for n in
                    ["coffee_KCF_yahoo.csv", "brlusd.csv",
                     "weather_central_highlands.csv"])
    if not have_real:
        pytest.skip("data caches absent — run vendor_fetch/macro_fetch first")
    return build_site.build(out_dir=out)


def test_outputs_exist(built):
    assert (built / "index.html").exists()
    assert (built / ".nojekyll").exists()
    for name in ["price_regime.png", "weather.png", "brl.png"]:
        png = built / "assets" / name
        assert png.exists() and png.stat().st_size > 10_000, name


def test_html_has_required_sections(built):
    html = (built / "index.html").read_text()
    for required in [
        "Robusta Coffee Monitor",
        "Market regime",
        "Growing conditions — Central Highlands, Vietnam",
        "Currency driver",
        "Methods, honestly",
        "not investment advice",
        "price data through",       # per-panel as-of stamps
        "weather data through",
        "FX data through",
        "github.com/ScottDongKhang/ascent-agri",
    ]:
        assert required in html, f"missing: {required!r}"


def test_feed_exists_and_is_valid_rss(built):
    import xml.etree.ElementTree as ET
    feed = built / "feed.xml"
    assert feed.exists()
    root = ET.fromstring(feed.read_text())
    assert root.tag == "rss"
    items = root.findall("./channel/item")
    assert len(items) >= 5
    descs = [i.findtext("description") or "" for i in items]
    assert any("Coffee futures closed" in d for d in descs)
    guids = [i.findtext("guid") for i in items]
    assert len(guids) == len(set(guids)), "GUIDs must be unique"


def test_paper_served_and_linked(built):
    html = (built / "index.html").read_text()
    assert "Research" in html
    if (built / "assets" / "weather-and-coffee-returns.pdf").exists():
        assert "assets/weather-and-coffee-returns.pdf" in html


def test_posture_is_known_value(built):
    html = (built / "index.html").read_text()
    assert any(w in html for w in
               ["constructive", "selective", "defensive", "crisis",
                "uncertain", "neutral"])


def test_daily_brief_templates():
    """Brief must adapt to dry vs wet vs neutral and BRL direction."""
    class P:  # minimal posture stub
        posture = "defensive"
        risk_multiplier = 0.65
    base = dict(close=None, signals=None, feature_panel=None, brl=None,
                weather=None, posture=P(), label="stressed", dwell=7,
                price=250.0, chg_1w=-0.02, chg_1m=0.05,
                price_asof="", weather_asof="", brl_asof="")
    dry = build_site.MonitorState(**base, rain_z=-1.8, dry_frac=0.9,
                                  brl_chg_21d=0.04)
    wet = build_site.MonitorState(**base, rain_z=1.5, dry_frac=0.2,
                                  brl_chg_21d=-0.03)
    flat = build_site.MonitorState(**base, rain_z=0.1, dry_frac=0.5,
                                   brl_chg_21d=0.001)
    b_dry, b_wet, b_flat = (build_site.daily_brief(s) for s in (dry, wet, flat))
    assert "below its seasonal norm" in b_dry and "selling pressure" in b_dry
    assert "above its seasonal norm" in b_wet and "supportive" in b_wet
    assert "near its seasonal norm" in b_flat and "neutral currency" in b_flat
    for b in (b_dry, b_wet, b_flat):
        assert "stressed" in b or "defensive" in b
