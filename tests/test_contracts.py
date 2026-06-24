"""Tests for contract-code parsing and ICE roll-date math.

FND rule: 4th business day before the 1st business day of the delivery month.
Roll date: roll_offset_bd business days before FND.
All hand-computed below (Mon-Fri business days, no exchange holidays in v1).
"""
import datetime as dt

import pytest

from ascentagri.contracts import (
    Contract,
    parse_contract,
    first_notice_day,
    roll_date,
    MONTH_CODE_TO_MONTH,
)


def test_parse_contract_from_symbol():
    c = parse_contract("RMK24")
    assert c.root == "RM"
    assert c.month_code == "K"
    assert c.month == 5
    assert c.year == 2024


def test_parse_contract_from_filename():
    c = parse_contract("RMN24.csv")
    assert c.root == "RM"
    assert c.month_code == "N"
    assert c.month == 7
    assert c.year == 2024


def test_parse_contract_rejects_bad_month_code():
    with pytest.raises(ValueError):
        parse_contract("RMZ24")  # Z is not a robusta delivery month


def test_month_code_map_is_the_robusta_cycle():
    assert MONTH_CODE_TO_MONTH == {
        "F": 1, "H": 3, "K": 5, "N": 7, "U": 9, "X": 11
    }


def test_fnd_may_2024():
    # May 1 2024 is a Wednesday (a business day). 4 business days before = Apr 25.
    assert first_notice_day(2024, 5) == dt.date(2024, 4, 25)


def test_fnd_july_2024():
    # Jul 1 2024 is Monday. 4 business days before = Jun 25.
    assert first_notice_day(2024, 7) == dt.date(2024, 6, 25)


def test_fnd_handles_weekend_first_of_month():
    # Sep 1 2024 is a Sunday -> first business day Sep 2 (Mon).
    # 4 business days before Sep 2 = Aug 27.
    assert first_notice_day(2024, 9) == dt.date(2024, 8, 27)


def test_roll_date_default_offset_5():
    # FND(May 2024) = Apr 25. 5 business days before = Apr 18.
    c = parse_contract("RMK24")
    assert roll_date(c, roll_offset_bd=5) == dt.date(2024, 4, 18)


def test_roll_date_offset_3_and_10():
    c = parse_contract("RMK24")  # FND = Apr 25 2024
    # 3 business days before Apr 25: Apr 24, 23, 22 -> Apr 22
    assert roll_date(c, roll_offset_bd=3) == dt.date(2024, 4, 22)
    # 10 business days before Apr 25: Apr 11
    assert roll_date(c, roll_offset_bd=10) == dt.date(2024, 4, 11)
