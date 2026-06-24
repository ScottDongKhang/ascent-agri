"""Tests for detecting gaps in the contract chain (data-quality / deliverable #5)."""
from ascentagri.contracts import parse_contract, next_in_cycle, chain_gaps


def test_next_in_cycle_within_year():
    assert next_in_cycle(parse_contract("RMH24")).symbol == "RMK24"  # Mar -> May


def test_next_in_cycle_wraps_year():
    assert next_in_cycle(parse_contract("RMX24")).symbol == "RMF25"  # Nov24 -> Jan25


def test_chain_gaps_none_when_consecutive():
    cs = [parse_contract(s) for s in ["RMH24", "RMK24", "RMN24"]]
    assert chain_gaps(cs) == []


def test_chain_gaps_flags_missing_middle_contract():
    cs = [parse_contract(s) for s in ["RMH24", "RMK24", "RMU24"]]  # missing N24 (Jul)
    assert chain_gaps(cs) == ["RMN24"]


def test_chain_gaps_flags_missing_across_year_boundary():
    cs = [parse_contract(s) for s in ["RMU24", "RMF25"]]  # missing X24 (Nov24)
    assert chain_gaps(cs) == ["RMX24"]
