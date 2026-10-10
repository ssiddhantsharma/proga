"""Card-free tests for the Nesso-1 affinity oracle: registry, ligand-required guard, affinity
parser, and the Scorer affinity-only path on a fold-free PredictionOutputs. No card/GPU/tt-bio."""
import json

import pytest
import torch

from proga.config import DesignConfig, ScoreWeights
from proga.oracle_base import _ORACLE_BACKENDS, ChainIndex, PredictionOutputs, make_oracle
from proga.scoring import Scorer


def _nesso(tmp_path, key="affinity_probability_binary"):
    return make_oracle(
        "nesso1",
        DesignConfig(nesso1_ligand_smiles="c1ccccc1", nesso1_affinity_key=key),
        work_dir=str(tmp_path),
    )


def test_nesso1_registered():
    assert _ORACLE_BACKENDS.get("nesso1") == "proga.nesso1_oracle:NessoAffinityOracle"


def test_requires_ligand(tmp_path):
    with pytest.raises(ValueError):
        make_oracle("nesso1", DesignConfig(), work_dir=str(tmp_path))


def test_parse_affinity_key(tmp_path):
    o = _nesso(tmp_path)
    (tmp_path / "d_affinity.json").write_text(
        json.dumps({"id": "d", "affinity_pred_value": 2.5, "affinity_probability_binary": 0.42})
    )
    assert o._parse_affinity(tmp_path, "d") == pytest.approx(0.42)


def test_parse_affinity_missing_key(tmp_path):
    o = _nesso(tmp_path, key="nope")
    (tmp_path / "d_affinity.json").write_text(json.dumps({"id": "d", "affinity_pred_value": 2.5}))
    with pytest.raises(KeyError):
        o._parse_affinity(tmp_path, "d")


def _foldfree_outputs(affinity):
    ci = ChainIndex(
        chain_ids=["A"],
        chain_tokens={"A": torch.zeros(0, dtype=torch.long)},
        chain_roles={"A": "binder"},
        chain_asym={"A": 0},
        binder_id="A",
        chain_names=None,
        design_tokens=torch.zeros(0, dtype=torch.long),
        residue_index=torch.zeros(0, dtype=torch.long),
        asym_id=torch.zeros(0, dtype=torch.long),
    )
    return PredictionOutputs(
        per_token_plddt=torch.zeros(0),
        pae=torch.zeros(0, 0),
        distogram_logits=None,
        ca_coords=torch.zeros(0, 3),
        chain_pair_iptm=torch.zeros(0, 0),
        chain_index=ci,
        distogram_bin_params={},
        iptm_global=0.0,
        backbone_coords=None,
        sequence="AAAA",
        affinity=affinity,
    )


def test_scorer_affinity_only_path():
    scorer = Scorer(config=DesignConfig(weights=ScoreWeights.affinity_only()))
    terms = scorer.score_terms(_foldfree_outputs(0.42))
    assert terms["affinity"] == pytest.approx(0.42)
    assert terms["iptm"] == 0.0  # structure term zeroed on a fold-free oracle
    assert scorer.fitness(terms) == pytest.approx(0.42)
