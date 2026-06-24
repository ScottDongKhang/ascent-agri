# data/raw/ — manually-downloaded Barchart contract CSVs

**Do not commit the CSVs.** They are personal/academic-use-only under Barchart's Terms of
Use and are git-ignored. This folder is the single drop point for raw inputs; the code
never scrapes.

## Why manual

Barchart's Terms of Use prohibit automated extraction ("no data mining, robots, or
similar data gathering and extraction tools"), and `robots.txt` disallows the `/proxies/`
path their download endpoint uses. So we download by hand, as a human, through the
browser. The free tier limits downloads to ~5/day for a registered account — plan around
that (see history-depth note below).

## How to download one contract

1. Go to Barchart's Robusta Coffee futures page (root symbol **`RM`**, the 10-tonne
   contract): `https://www.barchart.com/futures/quotes/RM*0/all-futures` lists every
   listed/expired delivery month.
2. Click the specific contract (e.g. *Robusta Coffee May '24*, symbol `RMK24`).
3. Open its **Historical Data** / **Download** view, set the date range to the full life
   of the contract (Barchart free shows up to ~2 years of daily bars — enough, since no
   contract trades longer than that).
4. Download the **daily** CSV.
5. Save it here with the naming convention below.

## File-naming convention (REQUIRED)

`RM<MONTHCODE><YY>.csv`  — root + delivery-month code + 2-digit year.

Delivery-month codes (ICE robusta cycle, every 2 months):

| Month | Code |
|-------|------|
| January  | F |
| March    | H |
| May      | K |
| July     | N |
| September| U |
| November | X |

Examples: `RMK24.csv` (May 2024), `RMN24.csv` (Jul 2024), `RMU24.csv` (Sep 2024).

The loader parses delivery month + year directly from the filename, so the name must be
correct. If a downloaded file uses a different Barchart symbol spelling, rename it to this
convention.

## Which contracts to download (start small, expand later)

Per the agreed plan, start with **~3 years (~18 contracts)** to validate the pipeline,
then backfill older contracts into this same folder later — no code changes needed.

For a continuous front-month series you need an unbroken chain of consecutive delivery
months (…K24, N24, U24, X24, F25, H25, …). A missing contract in the middle breaks the
chain at that roll; the loader/validator will flag gaps.

## Expected CSV columns

Barchart daily historical CSVs are typically:

```
Time, Open, High, Low, Last, Change, Volume, Open Int
```

The loader maps `Time -> date` and `Last -> close` (and is tolerant of a `Close` column
and minor header variations). If your download differs, note it and the loader mapping can
be adjusted. Barchart often appends a non-data footer line ("Downloaded from Barchart.com
...") — the loader skips trailing non-parseable rows.
