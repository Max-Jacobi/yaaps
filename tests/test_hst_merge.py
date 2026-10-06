import numpy as np

from yaaps.athena_read import hst
from yaaps.simulation import _load_ascii_multi


def _write(d, version, names, rows):
    d.mkdir()
    head = " ".join(f"[{i + 1}]={n}" for i, n in enumerate(names))
    body = "".join(" ".join(map(str, r)) + "\n" for r in rows)
    (d / "x.hst").write_text(f"# GR-Athena++ ({version}) history data\n# {head}\n{body}")


def test_columns_matched_by_name_across_restarts(tmp_path):
    _write(tmp_path / "output-0000", "aaa", ["time", "mass"], [[0, 1.0], [1, 1.1], [2, 1.2]])
    # restart from t=1.5 with a new column inserted before mass
    _write(tmp_path / "output-0001", "bbb", ["time", "qdot", "mass"], [[1.5, 7.0, 2.0], [2.5, 8.0, 2.1]])
    dirs = [str(tmp_path / "output-0000"), str(tmp_path / "output-0001")]
    d = _load_ascii_multi(str(tmp_path), "x.hst", 2, hst, dirs)
    assert np.array_equal(d["time"], [0, 1, 1.5, 2.5])
    assert np.array_equal(d["mass"], [1.0, 1.1, 2.0, 2.1])
    assert np.isnan(d["qdot"][:2]).all() and np.array_equal(d["qdot"][2:], [7.0, 8.0])
