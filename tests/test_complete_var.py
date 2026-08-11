"""Tests for Simulation.complete_var name completion, no fixture files needed."""

import pytest

from yaaps.simulation import Simulation


class FakeScrape:
    @staticmethod
    def debug_data_keys():
        return {
            ("hydro.aux.T", ("x1", "x2"), True): None,
            ("hydro.aux.T", ("x1", "x2"), False): None,
            ("hydro.prim.rho", ("x1", "x2"), False): None,
            ("foo.vel", ("x1", "x2"), False): None,
            ("bar.vel", ("x1", "x2"), False): None,
        }


class FakeSim:
    complete_var = Simulation.complete_var
    scrape = FakeScrape()


def test_same_var_with_and_without_ghosts_is_not_ambiguous():
    assert FakeSim().complete_var("T", ("x1v", "x2v")) == ("hydro.aux.T", False)


def test_unique_completion():
    assert FakeSim().complete_var("rho", ("x1v", "x2v")) == ("hydro.prim.rho", False)


def test_distinct_variables_still_raise():
    with pytest.raises(ValueError, match="More than one completion"):
        FakeSim().complete_var("vel", ("x1v", "x2v"))


def test_missing_sampling_raises():
    with pytest.raises(ValueError, match="Sampling"):
        FakeSim().complete_var("rho", ("x1v", "x3v"))
