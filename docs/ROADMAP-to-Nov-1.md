# Roadmap → Cornell CALS application, November 1, 2026

What exists today (all live, all verified):

- **Live public tool** — https://scottdongkhang.github.io/ascent-agri/ —
  self-updating weekday monitor with RSS feed and analytics hook.
- **Original research** — *"Do growing-region weather anomalies predict
  coffee futures returns?"* (docs/research/, PDF served from the site):
  a-priori event study, permutation inference, placebo controls, and two
  honest headline findings (correctly-signed but underpowered Brazil-dry
  effect; the 2021 frost exposing event-definition risk).
- **Engineering depth** — 133 tests, causal features throughout, fail-safe
  publishing, honest negative-WFE backtest, full reproduction commands.

The gap between this and a finished application is **things only Scott can
do**. Dated critical path below. Automated pieces keep running on their own.

## July (data month)

- [ ] **Claim GoatCounter code `ascent-agri`** (goatcounter.com, free, ~2 min).
      Do this FIRST — every week unclaimed is a week of uncounted visitors.
- [ ] **Robusta contract grind — the single highest-value task.** Barchart
      free tier ≈ 1 CSV/day: RMU24 → RMU26 chain (~13 contracts) lands in
      `data/raw/` by ~mid-July. Then `python -m ascentagri.build_series`.
      (Alternative: Databento — new accounts get $125 promo credit; the whole
      chain costs cents. One evening instead of two weeks.)
- [ ] **Start sharing the URL** where coffee/ag people actually are:
      home-roasting forums (Home-Barista, CoffeeGeek), school econ/ag
      teachers and clubs, local roasters (walk in, show the weather panel),
      Discord coffee servers, family/community networks in Vietnam if
      applicable. Goal: first 50 real visitors + first GitHub issue from a
      stranger.
- [ ] **Begin corporate outreach** — the playbook, targets, and email
      template are in `docs/OUTREACH.md`. The data product they consume:
      `api/latest.json`, `api/history.csv`, the RSS feed (CC BY 4.0, free
      commercial use with attribution). Log every send/reply; the evidence
      checklist in the playbook defines when you may claim "X users depend
      on my data."

## August (science month)

- [ ] **Re-run the study with the extended robusta series**
      (`python -m ascentagri.research.weather_study`). A ~2023-2026 robusta
      span brings the 2024 Central Highlands drought — which coincided with
      robusta's record run — into sample, making the PRIMARY hypothesis
      testable for the first time.
- [ ] **Paper v2** with the robusta results (whatever they are — a null on
      the primary hypothesis with the positive control intact is still a
      publishable-shaped finding). Pre-registered refinement already named
      in v1: severity-weighted event definitions.
- [ ] **Robusta panel joins the live site** once the series is real
      (site copy already promises this).
- [ ] Optional but high-leverage: **submit the paper** to a peer-reviewed
      venue for pre-college research — Journal of Emerging Investigators
      (jei.org, free, rolling) or National High School Journal of Science.
      Even "under review at JEI" is a strong application line. Start early;
      JEI review takes months.

## September (narrative month)

- [ ] **Essay drafts.** The arc writes itself: family/heritage or curiosity
      hook → hand-built roll-adjusted data when free data didn't exist →
      a system that publishes its own unflattering backtest → a hypothesis
      test where the famous frost breaks your event definition → a live
      tool real people use. CALS themes it touches: food systems, climate
      exposure of smallholder agriculture, honest quantitative science.
- [ ] **Two or three named users.** A roaster, a teacher, a club — anyone
      who'll say "I check this" — beats a hundred anonymous pageviews.
- [ ] **Get the Vietnamese grower page reviewed by a native speaker** — the
      page is live at `/vi/` (draft translation, flagged as such in its
      footer). A family member reviewing it is both quality control and a
      true story for the essay. Fold their corrections into
      `site/build_site.py` (the `*_VI` string tables and `daily_brief_vi`).

## October (freeze month)

- [ ] **Oct 15: feature freeze.** Only data updates and bug fixes after.
- [ ] **Metrics snapshot for the application**: GoatCounter totals, RSS
      subscribers if visible, GitHub stars/issues, uptime streak of the
      daily workflow (Actions history is the receipt).
- [ ] Verify every URL that appears in application materials from a
      logged-out browser and a phone.
- [ ] **Nov 1: submit.**

## Standing (automated — just glance weekly)

- Daily site updates: check the Actions tab shows green ~weekdays. A red
  run is harmless (fail-safe keeps the last good page) unless it stays red
  for a week — then the Yahoo fetch may need attention.
- Keep `git pull` before working; the repo is the single source of truth.

## What NOT to do

- Don't tune the strategy to make the backtest look better — the negative
  WFE, honestly disclosed, is the credibility of the whole project.
- Don't redefine study events after seeing outcomes (the paper explains why).
- Don't add features in October.
