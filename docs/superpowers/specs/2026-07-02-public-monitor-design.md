# Robusta Coffee Monitor — public site design

**Date:** 2026-07-02
**Goal:** give ascent-agri real (unpaid) users as a living public artifact for a
Cornell CALS application: a zero-friction URL that anyone interested in coffee
or Vietnamese agriculture can visit daily.

> Process note: brainstormed and implemented in one autonomous session at
> Scott's instruction ("think of a way… and implement it"); approaches
> considered and the rationale are recorded here in lieu of an interactive
> approval gate.

## Approaches considered

| Option | Users | Ongoing effort | Reliability | CALS fit |
|---|---|---|---|---|
| **A. Auto-updating GitHub Pages monitor (chosen)** | anyone with the URL; measurable | none after setup | static site — nothing to keep alive | leads with agronomy (weather-driven supply risk) |
| B. Streamlit app | needs the app awake; free tier sleeps | some | flaky | tool-ish, trading-flavored |
| C. Weekly Substack brief | slow accumulation | weekly writing forever | n/a | good voice, bad automation |
| D. PyPI package | developers | some | fine | wrong audience |

A wins on every axis that matters for the application: verifiable real users
(analytics + GitHub stars/issues), zero ongoing labor, and the same
infrastructure pattern already proven by Ascent Capital's public dashboard.

## What the site is

**"Robusta Coffee Monitor"** — a single dark-editorial page, updated daily:

1. **Hero / posture** — the regime engine's current read of the coffee market
   in plain English (constructive / defensive / crisis / uncertain), with the
   latest price and 1-week / 1-month changes, and an auto-written 2–3 sentence
   daily brief (deterministic template — no LLM).
2. **Price + regime chart** — last ~2 years of coffee futures with regime
   shading (the HMM + hysteresis layer from the pipeline).
3. **Growing-conditions panel** — 30-day rainfall anomaly and dry-spell
   fraction at Buon Ma Thuot (Central Highlands robusta belt) — the
   agricultural heart of the page.
4. **Currency driver** — BRL/USD (producer-currency pressure).
5. **Methods & honesty** — short: what the model is, what data feeds it, a
   link to the repo, and the disclosure that the backtest's own walk-forward
   verdict is modest. Honesty is the brand.
6. **Footer** — data attribution (Yahoo, Open-Meteo), "not investment advice,"
   feedback via GitHub Issues, GoatCounter analytics snippet.

**Honest labeling:** the price series is ICE **arabica** KC=F (the compliant
daily-fetchable benchmark) until enough robusta contracts are collected; the
page says so in the chart caption. The hand-built robusta series stays off the
public site for now (derived from personal-use Barchart downloads — not
redistributable; and it's absent in CI anyway).

## Architecture

```
site/build_site.py       reads data/processed caches (NEVER fetches) →
                         runs RegimeEngine (K=3, no model selection) →
                         renders 3 dark PNG charts + one self-contained
                         index.html into site/_build/   (git-ignored)
.github/workflows/
  update-site.yml        daily cron (21:20 UTC, after US close) + manual:
                         pip install → vendor_fetch (KC=F) + macro_fetch
                         --force → build_site.py → publish site/_build to
                         the gh-pages branch (force_orphan)
GitHub Pages             serves the gh-pages branch
```

**Fail-safe rule:** the builder never fetches, and the workflow publishes only
after a fully successful fetch+build. Any failure (Yahoo blocking the runner is
the known risk) leaves the previous deployment untouched — the site can go
stale but never wrong. Every panel carries its own "as of" date so staleness
is visible, and the page shows a "last updated" stamp.

First deploy is pushed from the local machine (where Yahoo works), so the site
is live immediately regardless of runner luck.

## Design language

Dark editorial, quiet professional (Scott's established dashboard language):
near-black surface, warm off-white ink, hairline rules, generous whitespace,
no emoji, no gradients, no AI-slop phrasing. Charts use the validated dark-mode
palette steps (blue #3987e5, aqua #199e70, yellow #c98500, orange #d95926) on
the dark surface; regime bands use the repo's posture status colors with a
text legend (never color alone). System font stacks only — no webfont
dependencies.

## Users & measurement

- **Audience:** coffee enthusiasts and home-roasters who track green prices,
  ag/climate-curious readers, classmates/teachers — anyone Scott shares the
  URL with. No accounts, no signup, nothing to install.
- **Measurement:** GoatCounter (free, no-cookie) pageview snippet is wired to
  `https://ascent-agri.goatcounter.com/count`; Scott claims that code with a
  2-minute signup (README explains). Until claimed, the snippet 404s
  harmlessly. GitHub stars/watches and Issues-based feedback are the second
  signal.
- **Feedback loop:** footer links "suggest a feature / report something wrong"
  → GitHub Issues.

## Testing

- Builder smoke tests (offline, tmp dirs): builds from existing caches; HTML
  contains hero, all three panels, as-of stamps, disclaimer; PNGs exist and
  are nonempty; posture word is one of the six known values.
- Local run on real caches + visual inspection of rendered charts before the
  first publish.
- Post-deploy: fetch the live URL and confirm 200 + expected content.

## Out of scope (YAGNI)

RSS/email subscriptions, interactive JS charts, historical archive pages,
robusta series on the public page (until contracts arrive), any LLM-written
copy, Reddit anything.
