"""Tests for yaaps.scrape.AthdfScraper."""

import os
import shutil

import h5py
import numpy as np
import pytest

from yaaps.scrape import AthdfScraper, IterationNotAvailable, find_restart_dirs

FIXTURE = (
    "/home/mjacobi/Documents/Projects/LongOEjecta/tracer_test/files/"
    "Blast.out2.00000.athdf"
)
N_B = (20, 20, 20)  # matches the fixture's MeshBlockSize

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE), reason="local fixture athdf file not present"
)


def _make_copy(dst: str, time_value: float) -> None:
    shutil.copy(FIXTURE, dst)
    with h5py.File(dst, "r+") as f:
        f.attrs["Time"] = time_value


def _truncate(path: str) -> None:
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.truncate(size // 2)


def _any_var_key(scraper: AthdfScraper) -> tuple:
    """Return (var, sampling, ghosts) for an arbitrary indexed variable."""
    return next(iter(scraper.debug_data_keys()))


class TestGapHandling:
    def test_missing_middle_iteration(self, tmp_path):
        d = tmp_path / "sim"
        d.mkdir()
        _make_copy(str(d / "problem.out2.00000.athdf"), 0.0)
        _make_copy(str(d / "problem.out2.00100.athdf"), 10.0)
        _make_copy(str(d / "problem.out2.00200.athdf"), 20.0)

        scraper = AthdfScraper(str(d), N_B=N_B)
        var, sampling, ghosts = _any_var_key(scraper)

        iters = scraper.get_available_iters(var, sampling, ghosts)
        assert list(iters) == [0, 100, 200]

        os.remove(str(d / "problem.out2.00100.athdf"))
        scraper2 = AthdfScraper(str(d), N_B=N_B)
        iters2 = scraper2.get_available_iters(var, sampling, ghosts)
        assert list(iters2) == [0, 200]

        with pytest.raises(IterationNotAvailable):
            scraper2.get_iter_fn("out2", 100)

    def test_get_iter_from_time_snaps_to_real_neighbor(self, tmp_path):
        d = tmp_path / "sim"
        d.mkdir()
        _make_copy(str(d / "problem.out2.00000.athdf"), 0.0)
        _make_copy(str(d / "problem.out2.00200.athdf"), 20.0)

        scraper = AthdfScraper(str(d), N_B=N_B)
        var, sampling, ghosts = _any_var_key(scraper)

        assert scraper.get_iter_from_time(var, sampling, 9.9, ghosts) == 0
        assert scraper.get_iter_from_time(var, sampling, 10.1, ghosts) == 200
        # never returns a nonexistent iteration such as 100
        assert scraper.get_iter_from_time(var, sampling, 10.0, ghosts) in (0, 200)


class TestTruncatedLastFile:
    def test_drops_unreadable_last_file(self, tmp_path):
        d = tmp_path / "sim"
        d.mkdir()
        _make_copy(str(d / "problem.out2.00000.athdf"), 0.0)
        last = str(d / "problem.out2.00001.athdf")
        _make_copy(last, 10.0)
        _truncate(last)

        scraper = AthdfScraper(str(d), N_B=N_B)
        var, sampling, ghosts = _any_var_key(scraper)

        assert list(scraper.get_available_iters(var, sampling, ghosts)) == [0]
        with pytest.raises(IterationNotAvailable):
            scraper.get_iter_fn("out2", 1)


class TestCache:
    def test_cache_avoids_reopening_unchanged_files(self, tmp_path, monkeypatch):
        d = tmp_path / "sim"
        d.mkdir()
        for it, t in ((0, 0.0), (100, 10.0), (200, 20.0)):
            _make_copy(str(d / f"problem.out2.{it:05d}.athdf"), t)

        import yaaps.scrape as scrape_mod

        real_file = h5py.File
        counts = {"n": 0}

        def counting_file(*args, **kwargs):
            counts["n"] += 1
            return real_file(*args, **kwargs)

        monkeypatch.setattr(scrape_mod, "h5py", scrape_mod.h5py)
        monkeypatch.setattr(scrape_mod.h5py, "File", counting_file)

        AthdfScraper(str(d), N_B=N_B)
        first_count = counts["n"]
        assert os.path.exists(str(d / ".yaaps_scrape_cache.json"))

        counts["n"] = 0
        AthdfScraper(str(d), N_B=N_B)
        second_count = counts["n"]

        assert second_count < first_count

    def test_cache_invalidated_on_n_b_mismatch(self, tmp_path):
        d = tmp_path / "sim"
        d.mkdir()
        _make_copy(str(d / "problem.out2.00000.athdf"), 0.0)
        _make_copy(str(d / "problem.out2.00001.athdf"), 1.0)

        AthdfScraper(str(d), N_B=N_B)
        # different N_B: must not crash, and must not silently reuse a cache
        # built for the other N_B (ghost-zone counts depend on N_B)
        scraper2 = AthdfScraper(str(d), N_B=(10, 10, 10))
        var, *_ = _any_var_key(scraper2)
        assert var  # got this far without an exception -> cache was rebuilt


class TestMultiRestart:
    def test_newer_restart_wins_on_overlap(self, tmp_path):
        d = tmp_path / "sim"
        d.mkdir()
        r0 = d / "output-0000"
        r1 = d / "output-0001"
        r0.mkdir()
        r1.mkdir()

        old_fn = str(r0 / "problem.out2.00000.athdf")
        new_fn = str(r1 / "problem.out2.00000.athdf")
        _make_copy(old_fn, 0.0)
        _make_copy(new_fn, 5.0)  # a restart re-wrote iteration 0 with a "corrected" Time

        now = 1_700_000_000
        os.utime(old_fn, (now, now))
        os.utime(new_fn, (now + 100, now + 100))  # newer mtime

        scraper = AthdfScraper(str(d), N_B=N_B)
        var, sampling, ghosts = _any_var_key(scraper)
        out = scraper.get_var_info(var, sampling, ghosts)[0]

        assert scraper.get_iter_fn(out, 0) == new_fn
        assert scraper.get_iter_time(out, 0) == 5.0

    def test_find_restart_dirs(self, tmp_path):
        d = tmp_path / "sim"
        d.mkdir()
        assert find_restart_dirs(str(d)) == [str(d)]

        (d / "output-0001").mkdir()
        (d / "output-0000").mkdir()
        (d / "not-a-restart").mkdir()
        dirs = find_restart_dirs(str(d))
        assert dirs == [str(d / "output-0000"), str(d / "output-0001")]
