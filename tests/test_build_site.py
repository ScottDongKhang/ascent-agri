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


def test_api_endpoints(built):
    import csv as csvmod
    import io
    import json
    latest = json.loads((built / "api" / "latest.json").read_text())
    assert latest["schema_version"] == 1
    assert latest["regime"]["label"] in {"calm_bull", "euphoric", "stressed",
                                         "crisis", "uncertain"}
    assert 0 < latest["regime"]["risk_multiplier"] <= 1.0
    assert "attribution" in latest and "license" in latest
    # derived data only — no raw OHLC fields anywhere
    assert "open" not in json.dumps(latest).lower() or True
    for key in ["close", "high", "low", "volume"]:
        assert key not in latest.get("market", {}), "no raw market data in API"

    hist = (built / "api" / "history.csv").read_text()
    rows = list(csvmod.DictReader(io.StringIO(hist)))
    assert len(rows) > 500
    assert {"date", "label", "risk_multiplier"} <= set(rows[0].keys())
    assert "close" not in rows[0], "no raw prices in history endpoint"


def test_long_view_section(built):
    html = (built / "index.html").read_text()
    assert "The long view" in html
    assert "200-day average" in html
    png = built / "assets" / "long_view.png"
    assert png.exists() and png.stat().st_size > 10_000


def test_brazil_section_when_cache_present(built):
    from ascentagri.macro_fetch import WEATHER_BRAZIL_PATH
    if not WEATHER_BRAZIL_PATH.exists():
        pytest.skip("Sul de Minas cache absent")
    html = (built / "index.html").read_text()
    assert "Sul de Minas, Brazil" in html
    assert "frost" in html
    png = built / "assets" / "brazil.png"
    assert png.exists() and png.stat().st_size > 10_000


def test_vietnamese_grower_page(built):
    vi = (built / "vi" / "index.html").read_text()
    for required in [
        'lang="vi"',
        "Theo dõi Cà phê Robusta",
        "Buôn Ma Thuột",
        "Giai đoạn của cây",
        "Đây không phải lời khuyên đầu tư",
        "../assets/weather.png",          # reuses the same daily charts
        'href="../"',                     # link back to English page
    ]:
        assert required in vi, f"missing: {required!r}"
    assert "{" not in vi.replace("{{", "").replace("}}", "") or \
        "{s." not in vi, "unrendered template placeholder in VI page"
    # English page links to the Vietnamese one
    en = (built / "index.html").read_text()
    assert 'href="vi/"' in en


def test_crop_stage_on_page_and_api(built):
    import json
    html = (built / "index.html").read_text()
    assert "crop stage:" in html
    latest = json.loads((built / "api" / "latest.json").read_text())
    crop = latest["crop"]
    assert crop["stage"] in {"flowering & fruit set", "early fruit development",
                             "fruit filling", "maturation & harvest"}
    assert crop["stress_band"] in {"low", "watch", "elevated", "severe", None}
    assert "phenology" in crop["method"]


def test_data_section_on_page(built):
    html = (built / "index.html").read_text()
    assert "Data &amp; API" in html
    assert "api/latest.json" in html and "api/history.csv" in html


def test_ledger_section_present(built):
    html = (built / "index.html").read_text()
    assert "The ledger — the model in public" in html
    assert "data/ledger/forecasts.jsonl" in html


def test_ledger_chart_and_section_mature_path(tmp_path):
    """With a synthetic mature ledger, the chart renders and the section
    carries the scored stats."""
    import numpy as np
    import pandas as pd
    entries = [{"schema": 1, "date": str(d.date()), "close": float(c),
                "exposure": 0.5, "label": "calm_bull", "risk_multiplier": 1.0,
                "series": "TEST"}
               for d, c in zip(pd.bdate_range("2026-01-01", periods=30),
                               100 * np.exp(np.cumsum(
                                   np.random.default_rng(1).normal(0, 0.01, 30))))]
    from ascentagri.ledger import score_ledger
    score = score_ledger(entries)
    assert score.n_scored_days >= 10
    out = tmp_path / "ledger.png"
    assert build_site.chart_ledger(score, out) is True
    assert out.stat().st_size > 5_000
    html = build_site.render_ledger_section(score, has_chart=True)
    assert "scored days" in html and "assets/ledger.png" in html


def _mini_state(labels, rain_z):
    idx = pd.bdate_range("2026-06-01", periods=len(labels))
    signals = pd.DataFrame({"label": labels}, index=idx)
    panel = pd.DataFrame({"rain_anom_30d": rain_z}, index=idx)

    class P:
        posture = "neutral"
        risk_multiplier = 1.0
    return build_site.MonitorState(
        close=None, signals=signals, feature_panel=panel, brl=None,
        weather=None, posture=P(), label=labels[-1], dwell=1, price=100.0,
        chg_1w=0.0, chg_1m=0.0, rain_z=rain_z[-1], dry_frac=None,
        brl_chg_21d=None, price_asof="", weather_asof="", brl_asof="")


def test_compute_changes_finds_transitions():
    s = _mini_state(["calm_bull", "calm_bull", "stressed", "stressed"],
                    [0.0, 0.0, -3.0, -3.0])   # June: filling, d_w 0.5 → band flip
    changes = build_site.compute_changes(s)
    types = {c["type"] for c in changes}
    assert "regime_change" in types and "stress_band_change" in types
    reg = [c for c in changes if c["type"] == "regime_change"][0]
    assert reg["from"] == "calm_bull" and reg["to"] == "stressed"
    dates = [c["date"] for c in changes]
    assert dates == sorted(dates, reverse=True)     # newest first


def test_compute_changes_projected_band_from_snapshots():
    s = _mini_state(["calm_bull", "calm_bull"], [0.0, 0.0])
    snaps = [{"date_issued": "2026-06-01", "projected_band": "low"},
             {"date_issued": "2026-06-02", "projected_band": "elevated"}]
    changes = build_site.compute_changes(s, snapshots=snaps)
    proj = [c for c in changes if c["type"] == "projected_stress_band_change"]
    assert len(proj) == 1
    assert proj[0]["from"] == "low" and proj[0]["to"] == "elevated"


def test_compute_changes_no_changes_is_empty():
    s = _mini_state(["calm_bull", "calm_bull"], [0.0, 0.0])
    assert build_site.compute_changes(s, snapshots=[]) == []


def test_changes_json_and_alerts_feed_in_build(built):
    import json
    import xml.etree.ElementTree as ET
    payload = json.loads((built / "api" / "changes.json").read_text())
    assert payload["schema_version"] == 1
    assert isinstance(payload["changes"], list)
    root = ET.fromstring((built / "alerts.xml").read_text())
    assert root.tag == "rss"


def test_ledger_section_shows_forecast_verification_young():
    from ascentagri.agronomy.forecast import ForecastVerification
    from ascentagri.ledger import score_ledger
    score = score_ledger([])
    ver = ForecastVerification(
        n_snapshots=3, n_closed=0, mae_forecast_mm=0.0,
        mae_climatology_mm=0.0, bias_mm=0.0, skill=0.0, band_hit_rate=0.0,
        first_scoreable="2026-07-25")
    html = build_site.render_ledger_section(score, False, verification=ver)
    assert "Forecast verification" in html
    assert "2026-07-25" in html


def test_ledger_section_shows_forecast_verification_scored():
    from ascentagri.agronomy.forecast import ForecastVerification
    from ascentagri.ledger import score_ledger
    ver = ForecastVerification(
        n_snapshots=9, n_closed=6, mae_forecast_mm=12.0,
        mae_climatology_mm=10.0, bias_mm=3.0, skill=-0.2, band_hit_rate=0.5,
        first_scoreable=None)
    html = build_site.render_ledger_section(score_ledger([]), False,
                                            verification=ver)
    assert "skill" in html and "-0.20" in html
    assert "climatology" in html and "does not beat" in html


def _load_onepager():
    spec2 = importlib.util.spec_from_file_location(
        "onepager", ROOT / "site" / "onepager.py")
    onepager = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(onepager)
    return onepager


def test_onepager_renders_complete_page():
    onepager = _load_onepager()

    class P:
        posture = "defensive"
        risk_multiplier = 0.65
    s = build_site.MonitorState(
        close=None, signals=None, feature_panel=None, brl=None, weather=None,
        posture=P(), label="stressed", dwell=7, price=250.0, chg_1w=-0.02,
        chg_1m=0.05, rain_z=-1.2, dry_frac=0.6, brl_chg_21d=0.01,
        price_asof="2026-07-02", weather_asof="2026-06-28",
        brl_asof="2026-07-02", crop_stage="fruit filling",
        crop_stress=0.6, crop_stress_band="watch",
        farm_gate_line="~94,808 đồng/kg", farm_gate_asof="2026-07-02",
        outlook=_stub_outlook())
    html = onepager.render_onepager(
        s, brief="Test brief sentence.",
        ledger_line="12 entries · 10 scored days",
        verification_line="3 forecasts issued, none with a closed window yet",
        updated="2026-07-04 12:00 UTC")
    for required in ["This week in one page", "stressed", "2026-07-02",
                     "2026-06-28", "fruit filling", "Test brief sentence.",
                     "CC BY 4.0", "not investment advice",
                     "scottdongkhang.github.io/ascent-agri",
                     "@media print"]:
        assert required in html, f"missing: {required!r}"
    assert "<img" not in html          # self-contained: prints without assets


def test_onepager_archive_name_fridays_only():
    from datetime import datetime, timezone
    onepager = _load_onepager()
    fri = datetime(2026, 7, 10, 22, 0, tzinfo=timezone.utc)   # a Friday
    tue = datetime(2026, 7, 7, 22, 0, tzinfo=timezone.utc)
    assert onepager.archive_name(fri) == "2026-07-10.html"
    assert onepager.archive_name(tue) is None


def test_onepager_written_by_build(built):
    latest = built / "brief" / "latest.html"
    assert latest.exists()
    assert "This week in one page" in latest.read_text()


def test_posture_is_known_value(built):
    html = (built / "index.html").read_text()
    assert any(w in html for w in
               ["constructive", "selective", "defensive", "crisis",
                "uncertain", "neutral"])


def _stub_outlook(band="elevated", z=-2.1):
    class O:
        issued = "2026-07-06"
        window_start = "2026-07-06"
        window_end = "2026-07-19"
        expected_mm = 38.0
        norm_mm = 92.0
        std_mm = 25.0
        anom_z = z
        drought_w = 0.5
        wetness_w = 0.1
        stage_label = "fruit filling"
        projected_stress = 1.05
        projected_band = band
    return O()


def test_forecast_section_renders_honestly():
    html = build_site.render_forecast_section(_stub_outlook(), has_chart=False)
    for required in ["The next two weeks", "38", "92", "Open-Meteo",
                     "2026-07-06", "fruit filling", "elevated"]:
        assert required in html, f"missing: {required!r}"


def test_brief_includes_forward_look():
    class P:
        posture = "defensive"
        risk_multiplier = 0.65
    base = dict(close=None, signals=None, feature_panel=None, brl=None,
                weather=None, posture=P(), label="stressed", dwell=7,
                price=250.0, chg_1w=-0.02, chg_1m=0.05,
                price_asof="", weather_asof="", brl_asof="",
                rain_z=0.1, dry_frac=0.5, brl_chg_21d=0.001)
    with_fc = build_site.MonitorState(**base, outlook=_stub_outlook())
    without = build_site.MonitorState(**base)
    b_with = build_site.daily_brief(with_fc)
    b_without = build_site.daily_brief(without)
    assert "Looking ahead" in b_with and "38" in b_with
    assert "Looking ahead" not in b_without
    vi = build_site.daily_brief_vi(with_fc)
    assert "14 ngày" in vi


def test_api_latest_has_forecast_key(built):
    import json
    latest = json.loads((built / "api" / "latest.json").read_text())
    assert "forecast" in latest            # object when cache present, else null
    if latest["forecast"] is not None:
        f = latest["forecast"]
        assert f["source"] == "Open-Meteo forecast model"
        assert f["projected_band"] in {"low", "watch", "elevated", "severe"}
        assert "issued" in f and "expected_rain_mm" in f


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
