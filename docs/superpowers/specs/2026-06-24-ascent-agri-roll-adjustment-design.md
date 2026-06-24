# ascent-agri — Roll-Adjusted Continuous Robusta Series (Design)

**Date:** 2026-06-24
**Status:** Approved — implementation in progress
**Scope of this session:** project skeleton + the data foundation (a clean, roll-adjusted
continuous front-month price series for ICE Robusta Coffee). Regime model and debate
layer are explicitly out of scope and will be ported in later sessions.

---

## 1. Purpose

Cornell CALS application artifact: a rigorous backtested case study applying
regime-detection + multi-agent debate to ICE Robusta Coffee futures. The regime model
consumes **returns** (5/21/63-day returns, realized vol, drawdown), so the price series it
runs on must be free of artificial roll jumps. This session builds that series and proves
it is clean, before any real data is downloaded.

Standalone project. No imports from, or references to, any other repository.

## 2. Data acquisition — compliance decision (settled)

**We do not scrape Barchart.**

- Barchart Terms of Use prohibit automated extraction ("no data mining, robots, or
  similar data gathering and extraction tools").
- `robots.txt` has `Disallow: /proxies/`, the path the CSV download endpoint uses.

**Compliant path:** the user manually downloads each contract's daily CSV from the
Barchart web UI (personal/academic use, ~5 downloads/day on the free tier) and drops files
into `data/raw/`. Acquisition (manual, human-driven) is decoupled from processing (code).
The same code path works unchanged if a paid API is added later. Download steps and the
file-naming convention live in `data/raw/README.md`.

**History depth:** start with ~3 years (~18 contracts) to validate the pipeline, expand by
backfilling into the same folder later.

## 3. Contract facts (verified against ICE spec)

- Root symbol: **`RM`** (the 10-tonne robusta contract; price quoted USD/tonne — continuous
  across the 2017 lot-size change).
- Delivery months: **Jan/Mar/May/Jul/Sep/Nov** → codes F/H/K/N/U/X → **6 contracts/year**.
- **First Notice Day (FND)** = 4th business day before the 1st business day of the delivery
  month.
- Last Trading Day = 4th business day before the last business day of the delivery month
  (later than FND; **not** our trigger).
- For a front-month series we roll **before FND** to avoid delivery risk.

## 4. Roll-trigger choice (settled)

**Calendar / FND-based**, deterministic and reproducible:
`roll_date = FND(expiring) − roll_offset_bd business days`, default `roll_offset_bd = 5`.

Rejected alternatives: volume/OI crossover (free historical volume on far months is
unreliable → non-reproducible) and fixed day-of-month heuristic (not tied to real
contract rules). Volume-based rolling is a documented possible future extension.

Business-day math uses `numpy.busday_offset` (Mon–Fri). Exchange holidays are **not**
modeled in v1 (documented limitation); the API accepts an optional holiday list so this
can be tightened later. Because the ratio window uses the *available* closes before the
roll date (not exact calendar slots), a holiday near the roll shifts the window by at most
a day and does not break the method.

## 5. Roll-adjustment method (proportional / ratio, back-adjusted)

For each consecutive pair (expiring contract *i* → next contract *i+1*):

1. `roll_date_i = FND(i) − roll_offset_bd` business days.
2. `avg_exp_i` = mean **Close** over the **4 most recent available trading days strictly
   before `roll_date_i`** on contract *i*. `avg_next_i` = same window on contract *i+1*.
3. `ratio_i = avg_next_i / avg_exp_i`.
4. **Naive front-month splice** (`raw_spliced`): contract *i* for dates `< roll_date_i`,
   contract *i+1* for dates `>= roll_date_i`. This series **contains** the fake jumps and
   is kept for inspection/validation.
5. **Back-adjustment** (`adjusted`): the most recent contract is left unadjusted
   (factor 1). The segment sourced from contract *i* is multiplied by the **cumulative
   product of ratios at/after its roll**:
   `cumulative_factor_i = Π_{j >= i} ratio_j`.
   The oldest contract therefore carries the product of all ratios.

### Why ratio (not additive)

Multiplying a price segment by a constant leaves its percentage returns unchanged, so
within-contract returns are preserved exactly and the boundary return becomes the genuine
roll-adjusted move (the curve spread is divided out). Additive (Panama) adjustment shifts
levels and distorts returns at low price levels — wrong for a returns-driven regime model.

### Boundary return check (the heart of validation)

At a roll boundary, naive return = `P_next(first)/P_exp(last) − 1` (carries the curve
jump). Adjusted return = `P_next(first) / (P_exp(last) · ratio_i) − 1`, which is ≈ 0 when
the boundary closes sit near their 4-day averages — i.e. the artificial jump is removed.

## 6. Modules and interfaces

All units are small, single-purpose, independently testable.

- **`contracts.py`**
  - `parse_contract(symbol_or_filename) -> Contract(root, month_code, year, delivery_date)`
  - `first_notice_day(year, month, holidays=None) -> date`
  - `roll_date(contract, roll_offset_bd=5, holidays=None) -> date`
  - `MONTH_CODE_TO_MONTH` / `MONTH_TO_CODE` maps.
- **`loader.py`**
  - `load_contract_csv(path) -> pd.DataFrame` (index=date, cols incl. `close`); maps
    Barchart `Time/Last`, tolerates `Close`, skips footer rows.
  - `load_raw_dir(dir) -> dict[Contract, DataFrame]` sorted by delivery date.
- **`roll.py`**
  - `@dataclass RollResult`: `raw_spliced`, `roll_table`, `adjusted`, `contract_map`.
  - `build_continuous(contracts: dict, roll_offset_bd=5, window=4, holidays=None) ->
    RollResult`. Pure function over in-memory frames — no I/O. **No black box:** every
    intermediate (spliced, per-roll ratios, cumulative factors, source map) is a field.
- **`validate.py`**
  - `roll_return_report(result) -> pd.DataFrame`: per roll, naive vs adjusted boundary
    return, and the abs improvement.
  - `offset_sensitivity(contracts, offsets=(3,5,10), **kw) -> pd.DataFrame`: how ratios,
    cumulative factors, and report numbers move across offsets.
  - `plot_rolls(result, out_path)`: naive vs adjusted series + marked roll dates.
- **`worked_example.py`** (`python -m ascentagri.worked_example`): builds the synthetic
  fixture, prints the worked example (March→May roll with a known curve shape and the
  backward propagation), and prints the offset-sensitivity table. No real data required.

## 7. Synthetic fixture (proves the pipeline before real data)

`tests/fixtures/` holds a short chain of synthetic contracts (e.g. K24→N24→U24→X24) with a
**known, constant contango** (each next contract a fixed % above the expiring one), built
so ratios land on clean, checkable numbers (e.g. ~1.03) and the backward propagation is
hand-verifiable. The fixture exercises overlapping contract lifetimes so the 4-day window
exists on both sides of every roll.

## 8. Validation deliverables this session

1. Worked example printed: the ratio at one roll, plus the cumulative-factor chain back to
   the oldest contract.
2. Returns-around-each-roll table: naive (fake jump present) vs adjusted (≈0) + plot.
3. **Offset sensitivity**: the same fixture run at `roll_offset_bd ∈ {3, 5, 10}`, reporting
   how much cumulative factors and validation numbers change — to judge robustness before
   committing a default for real data.
4. Honest data report: with the synthetic fixture the run is complete and gap-free; real
   downloaded data's gaps/quality can only be assessed once files are dropped in
   `data/raw/`, and the loader/validator will surface missing-contract chain breaks.

## 9. Testing strategy (TDD)

- `test_contracts.py`: FND and roll-date math against hand-computed dates; month-code
  parsing; weekend/business-day edges.
- `test_roll.py`: ratio equals known fixture value; **return-preservation property**
  (within-contract adjusted returns == raw returns to floating tolerance); cumulative
  factor == product of ratios; most-recent contract unadjusted; naive splice still shows
  the jump (guards against accidentally "fixing" the wrong series).

## 10. Out of scope (later sessions)

Regime model, walk-forward framework, multi-agent debate layer, any live/paper trading,
and any automated data download.

## 11. Non-negotiables

- No Barchart scraping; manual download only.
- Ratio (proportional) adjustment, back-adjusted so the most recent contract is unadjusted.
- 4-trading-day pre-roll averaging window for both legs of each ratio.
- All intermediates inspectable; no single black-box function.
- No dependency on any external repo.
