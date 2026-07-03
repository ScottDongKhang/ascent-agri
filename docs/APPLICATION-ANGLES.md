# Application angles — one project, two honest framings

The same artifact supports two different applications because the work is
genuinely interdisciplinary. The rule: **lead with the frame the school
cares about; never rename the work.** Everything below is checkable by an
admissions reader who clicks.

## Frame 1 — Agricultural science (Cornell CALS, and top schools as ag/enviro)

**Thesis sentence:** *I built an open agricultural intelligence system for
Vietnamese robusta coffee — crop phenology, growing-region climate stress,
and the market signals that carry them to farmers — and I run it in public,
every day.*

What maps to what CALS actually values:

| CALS value | Your receipt |
|---|---|
| Science applied to a real crop system | Phenology-weighted stress index: the same rainfall deficit is scored by the developmental stage it hits (flowering ≫ filling; harvest inverts to wetness risk). `ascentagri/agronomy/phenology.py`, live on the site daily. |
| Climate resilience | Causal, seasonal-aware weather anomalies at Buon Ma Thuot; the research paper's drought/frost event studies; the Coffee Watch groundwater context in your outreach. |
| Land-grant / extension mission | The monitor IS extension: free, daily, plain-English, open data (CC BY 4.0), farmers'-margin framing (đồng/kg), and a contribution path for new regions. |
| Smallholder economics / food systems | Farm-gate price transmission: futures USD/tonne → VND/kg with honest "before local basis" labeling. |
| Scientific integrity | Published negative backtest result; pre-registered event definitions; the 2021 frost that broke the method, reported instead of patched. |

**The genuine-passion test** (what makes it read as real, not resume-built):
it runs every day whether or not anyone is watching; it publishes results
that make it look worse; its hardest problem (robusta data) is being solved
by hand, one contract per day; and the personal *why* — that part only you
can write. If there's a family or heritage thread to Vietnamese coffee,
that's the essay's spine and no committee can mistake it for manufactured
spike. Don't overclaim it; one true paragraph beats a theme.

## Frame 2 — Economics (UCs and similar)

**Thesis sentence:** *I study how information moves into agricultural
commodity prices — weather shocks, currency pass-through, and market
regimes — with an event-study methodology I built and run in public.*

| Econ concept | Your receipt |
|---|---|
| Price discovery & market efficiency | Lead-lag nulls in the paper ("markets price gradual weather continuously") vs the event-level Brazil-dry effect — a clean efficient-markets discussion with your own data. |
| Identification & causal inference | A-priori thresholds, permutation tests, placebo pairs (Vietnam weather × arabica must be null, and is). |
| Exchange-rate pass-through | BRL/USD producer-revenue channel in the regime model; USD/VND farm-gate transmission. |
| Institutions & market structure | Futures-vs-farm-gate basis, roll-adjustment of contract chains, tariff episode (Royal Coffee) in your outreach research. |
| Public goods | Free data API used by real organizations (measured). |

## The interdisciplinary bridge (works in both essays)

One sentence that carries the whole project: **"A rainfall z-score is
statistics; knowing that the same deficit ruins February flowering but
speeds November drying is agronomy; measuring whether the market already
knows is economics."** That's your project in one line — and each clause is
implemented, tested, and public.

## Claims inventory — what you may say, with receipts

- "Runs autonomously every weekday since July 2026" → Actions history + ledger commits
- "Public, append-only forecast track record" → `data/ledger/forecasts.jsonl`
- "Original event study with pre-registered design" → paper §3, seed + params in code
- "Published our own negative result" → README + site, WFE section
- "Free data product under CC BY 4.0" → `api/` endpoints + LICENSE
- "X users" → ONLY per the evidence checklist in `docs/OUTREACH.md`
- Never claim: trading profitability, robusta-series completeness (until the
  contracts land), or peer review (until JEI accepts).

## What still needs a human (you), in priority order

1. The personal-why paragraph (heritage/curiosity origin — yours alone).
2. Robusta contracts → makes the primary hypothesis testable → paper v2 is
   the strongest possible September update for both frames.
3. Named users + outreach log (both frames cite them).
4. Optional: JEI submission (either frame: "under review at a peer-reviewed
   journal for pre-college research").
