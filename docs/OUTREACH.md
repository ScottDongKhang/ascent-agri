# Outreach playbook — turning the monitor into "X users depend on my data"

The claim you want to make to a corporation (or an admissions reader) has
three parts, and each needs receipts:

1. **"Users"** — measured, not asserted: GoatCounter uniques (claim the code
   `ascent-agri` first!), `data/ledger/traffic.jsonl` (weekly snapshots of
   repo views/clones/stars, committed automatically), RSS subscribers you
   hear from, and named organizations below.
2. **"Depend"** — they consume something on a schedule: `api/latest.json`,
   `api/history.csv`, or the RSS feed. One named organization that polls the
   API weekly is worth more than a thousand anonymous pageviews.
3. **"My data"** — the derived series this project computes (regime labels,
   risk multipliers, weather anomalies, briefs), CC BY 4.0, so companies are
   actually allowed to use it.

## Who to contact (in order of realistic yes)

**Tier 1 — specialty coffee (most likely to care, fastest replies)**
- Green coffee importers that publish market commentary: their newsletters
  need exactly the weather-anomaly angle this monitor automates.
- Local/regional roasters that buy Vietnamese robusta or blend with it.
  Walk-in or short email; show the growing-conditions panel on your phone.
- Coffee newsletters and podcasts (market-roundup genre) — offer the daily
  brief as a free syndicated blurb with attribution.

**Tier 2 — agriculture & data**
- Ag-econ teachers/professors (yours first) — classroom use is a legitimate
  "users depend on it" story and often the easiest yes.
- Student investment clubs and ag clubs (FFA chapters, commodity-trading
  clubs) — a standing data source for their meetings.
- Open-data aggregators and awesome-lists (awesome-agriculture,
  awesome-quant) — a PR adding this repo is distribution that compounds.

**Tier 3 — ambitious (fine to try, don't stake the story on it)**
- Vietnamese coffee exporters/co-ops (the Buon Ma Thuot angle is genuinely
  relevant to them; bilingual follow-up helps here).
- Commodity research desks — realistically they have Bloomberg, but a
  student-built open alternative is a conversation starter, not a sale.

## The email (short version — send this, not an essay)

> Subject: Free daily Vietnam robusta weather/market feed — built by a student
>
> Hi {name} — I'm a high-school senior who builds agricultural market
> models. I run a free, open-source daily monitor for coffee: market regime
> (hidden-Markov model), rainfall anomalies at Buon Ma Thuot in Vietnam's
> robusta belt, and BRL/USD pressure — updated automatically every weekday.
>
> Live: https://scottdongkhang.github.io/ascent-agri/
> Machine-readable: /api/latest.json and /api/history.csv (CC BY 4.0, free
> for commercial use with attribution).
>
> If {a daily weather-anomaly line / a weekly robusta blurb} would be useful
> for {your newsletter / your buying desk / your class}, it's yours — and I
> can add other growing regions (Indonesia, Colombia, Uganda) on request.
> Would you take a look and tell me what's missing?

Rules: one concrete offer, one question, no attachments, no jargon. Log
every send/reply in a spreadsheet — response rate is itself application
material ("contacted 40 organizations, 6 adopted the feed").

## What to ask adopters for

- Permission to name them ("used by …") on the site/README.
- Attribution when they quote the data (the license requires it anyway).
- One sentence of testimonial and one feature request. Feature requests from
  real organizations → GitHub issues → the community loop.

## Evidence checklist before you write "X users depend on my data" anywhere

- [ ] GoatCounter claimed and showing ≥4 weeks of uniques
- [ ] `data/ledger/traffic.jsonl` has ≥4 weekly snapshots
- [ ] ≥1 named organization consuming the feed/API on a schedule
- [ ] The number X is the *smallest defensible* one (weekly uniques, not
      lifetime pageviews) — under-claiming is credibility

## What NOT to do

- No scraping contact lists; write individually to people whose work you
  actually read.
- No Reddit campaigns.
- Don't promise trading performance — the data product is weather + regime
  *information*, and the project's own paper says the trading edge is
  unproven. That honesty is the pitch, not a weakness.
