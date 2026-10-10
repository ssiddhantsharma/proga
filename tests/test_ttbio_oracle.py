"""Card-free tests for the tt-bio oracle: registry + the confidence reader against a fixture
results.json (tt-bio's stable list-form schema). No GPU, no card, no tt-bio install."""
import json

import pytest

from proga.config import DesignConfig
from proga.oracle_base import _ORACLE_BACKENDS, make_oracle


def _oracle(tmp_path):
    return make_oracle("ttbio", DesignConfig(), work_dir=str(tmp_path))


def _write(tmp_path, rec):
    (tmp_path / "results.json").write_text(json.dumps([rec]))


def test_ttbio_registered():
    assert _ORACLE_BACKENDS.get("ttbio") == "proga.ttbio_oracle:TTBioOracle"


def test_confidence_reads_stable_keys(tmp_path):
    o = _oracle(tmp_path)
    _write(tmp_path, {
        "id": "val", "status": "ok", "iptm": 0.493,
        "pair_chains_iptm": {"0": {"0": 0.906, "1": 0.493}, "1": {"0": 0.362, "1": 0.700}},
        "chains_ptm": {"0": 0.906, "1": 0.700},
    })
    iptm_global, cpi = o._parse_confidence_ttbio(tmp_path, tmp_path, ["A", "B"], "val")
    assert iptm_global == pytest.approx(0.493)
    assert cpi.shape == (2, 2)
    assert cpi[0, 1].item() == pytest.approx(0.493)
    assert cpi[1, 0].item() == pytest.approx(0.362)   # directional
    assert cpi[0, 0].item() == pytest.approx(0.906)   # per-chain pTM on the diagonal


def test_confidence_falls_back_to_confidence_score(tmp_path):
    o = _oracle(tmp_path)
    _write(tmp_path, {"id": "v", "status": "ok", "confidence_score": 0.7,
                      "pair_chains_iptm": {}, "chains_ptm": {}})
    iptm_global, _ = o._parse_confidence_ttbio(tmp_path, tmp_path, ["A", "B"], "v")
    assert iptm_global == pytest.approx(0.7)


def test_failed_status_raises(tmp_path):
    o = _oracle(tmp_path)
    _write(tmp_path, {"id": "v", "status": "failed"})
    with pytest.raises(RuntimeError):
        o._parse_confidence_ttbio(tmp_path, tmp_path, ["A", "B"], "v")
