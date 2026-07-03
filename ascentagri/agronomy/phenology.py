"""Crop phenology for Dak Lak robusta — the agronomy layer.

A rainfall anomaly is not equally meaningful year-round: what matters to a
coffee tree is WHICH DEVELOPMENTAL STAGE the water lands on (or fails to).
This module encodes the widely documented Central Highlands robusta crop
calendar and converts the generic 30-day rainfall anomaly into a
stage-weighted crop stress index.

The Dak Lak robusta cycle (Coffea canephora, single main crop):

  Jan–Mar   FLOWERING & FRUIT SET — the dry season. Trees need a period of
            water deficit followed by rain (or farmer irrigation — typically
            several irrigation rounds) to trigger and set blossom. Water
            stress here is the classic yield killer: drought sensitivity is
            highest, because a failed flowering cannot be recovered later in
            the season.
  Apr–May   EARLY FRUIT DEVELOPMENT — monsoon onset. Rapid cell expansion;
            a late monsoon (continued dryness) directly limits bean size.
            Drought sensitivity high.
  Jun–Sep   FRUIT FILLING — peak monsoon. Moisture is normally ample, so a
            deficit is unusual but damaging; sensitivity moderate.
  Oct–Dec   MATURATION & HARVEST — monsoon retreat. Cherries are picked and
            sun-dried on tarps and drying yards; here the sign flips:
            EXCESS rain delays picking, causes cherry drop and mold in
            drying, and hurts quality. Wetness, not drought, is the risk.

The index is fully causal: it consumes the trailing rainfall anomaly
(z-scored against the location's own trailing year, computed elsewhere) and
the calendar month — nothing else.

Interpretation: 0 = no stage-relevant stress; positive values grow with
stress. Above ~1.0 means a ≥1σ anomaly is hitting the crop in a stage that
cares about it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Stage:
    name: str                 # machine name
    label: str                # human label for the site/API
    drought_weight: float     # how damaging a rainfall DEFICIT is (0..1)
    wetness_weight: float     # how damaging EXCESS rain is (0..1)
    note: str                 # one-line agronomy rationale


# month (1-12) → stage
_CALENDAR: Dict[int, Stage] = {}

_FLOWERING = Stage(
    "flowering", "flowering & fruit set",
    drought_weight=1.0, wetness_weight=0.0,
    note=("dry-season flowering: water deficit at blossom is the classic "
          "yield killer; farmers irrigate to trigger and hold fruit set"))
_EARLY_FRUIT = Stage(
    "early_fruit", "early fruit development",
    drought_weight=0.8, wetness_weight=0.0,
    note=("monsoon onset drives cell expansion; a late monsoon directly "
          "limits bean size"))
_FILLING = Stage(
    "fruit_filling", "fruit filling",
    drought_weight=0.5, wetness_weight=0.1,
    note=("peak monsoon: moisture normally ample, so a deficit is unusual "
          "but damaging"))
_HARVEST = Stage(
    "harvest", "maturation & harvest",
    drought_weight=0.1, wetness_weight=0.8,
    note=("harvest and sun-drying: excess rain delays picking and causes "
          "mold in drying yards — wetness is the risk, not drought"))

for _m in (1, 2, 3):
    _CALENDAR[_m] = _FLOWERING
for _m in (4, 5):
    _CALENDAR[_m] = _EARLY_FRUIT
for _m in (6, 7, 8, 9):
    _CALENDAR[_m] = _FILLING
for _m in (10, 11, 12):
    _CALENDAR[_m] = _HARVEST


def stage_for(date: pd.Timestamp) -> Stage:
    """The Dak Lak robusta crop stage in effect on a given date."""
    return _CALENDAR[pd.Timestamp(date).month]


def crop_stress_index(rain_anom_z: pd.Series) -> pd.Series:
    """Stage-weighted crop stress from the causal 30-day rainfall anomaly.

    stress(t) = drought_weight(month) × max(0, −z(t))
              + wetness_weight(month) × max(0, +z(t))

    A −2σ dry anomaly in February (flowering) scores 2.0; the same anomaly
    in August (filling) scores 1.0; in November (harvest) it scores 0.2 —
    while a +2σ wet anomaly in November scores 1.6. Same statistics,
    different biology.
    """
    z = rain_anom_z.astype(float)
    months = z.index.month
    d_w = np.array([_CALENDAR[m].drought_weight for m in months])
    w_w = np.array([_CALENDAR[m].wetness_weight for m in months])
    dry = np.clip(-z.values, 0, None)
    wet = np.clip(z.values, 0, None)
    return pd.Series(d_w * dry + w_w * wet, index=z.index, name="crop_stress")


def stress_label(value: float) -> str:
    """Plain-English band for the site and briefs."""
    if not np.isfinite(value):
        return "unknown"
    if value < 0.5:
        return "low"
    if value < 1.0:
        return "watch"
    if value < 2.0:
        return "elevated"
    return "severe"
