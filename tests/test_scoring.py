"""Card-free, protenix-free unit tests for the core selection math: the ipSAE term, ipTM
readout, interface-PAE stats, and that every scored term has a matching weight (fitness)."""
import torch

from proga.config import DesignConfig, ScoreWeights
from proga.oracle_base import ChainIndex, PredictionOutputs
from proga.scoring import Scorer, ipsae_d0res_max

BINDER = torch.tensor([0, 1, 2])
TARGET = torch.tensor([3, 4, 5])
N = 6


def _pae(interface_value: float) -> torch.Tensor:
    """6x6 PAE with the binder<->target interface blocks set to one value, intra blocks high."""
    pae = torch.full((N, N), 30.0)
    pae[BINDER.unsqueeze(1), TARGET] = interface_value
    pae[TARGET.unsqueeze(1), BINDER] = interface_value
    return pae


def _outputs(interface_value: float) -> PredictionOutputs:
    ci = ChainIndex(
        residue_index=torch.tensor([0, 1, 2, 0, 1, 2]),
        asym_id=torch.tensor([0, 0, 0, 1, 1, 1]),
        design_tokens=BINDER,
        chain_ids=["A", "B"],
        chain_tokens={"A": BINDER, "B": TARGET},
        chain_roles={"A": "binder", "B": "target"},
        chain_asym={"A": 0, "B": 1},
        binder_id="A",
    )
    return PredictionOutputs(
        per_token_plddt=torch.full((N,), 90.0),
        pae=_pae(interface_value),
        distogram_logits=None,
        ca_coords=torch.zeros(N, 3),
        chain_pair_iptm=torch.tensor([[0.90, 0.60], [0.50, 0.80]]),
        chain_index=ci,
        distogram_bin_params={},
        iptm_global=0.72,
    )


def test_ipsae_helper_bounds():
    # All interface PAE = 0 -> ptm = 1 -> ipSAE = 1. PAE >= cutoff -> no partners -> 0.
    assert ipsae_d0res_max(_pae(0.0), BINDER, TARGET, 10.0).item() == 1.0
    assert ipsae_d0res_max(_pae(20.0), BINDER, TARGET, 10.0).item() == 0.0


def test_ipsae_helper_exact_half():
    # PAE = 1 for every interface pair: n0 = 3 -> L floored at 26 -> d0 clamps to 1.0 ->
    # ptm = 1/(1+(1/1)^2) = 0.5, averaged and maxed stays 0.5.
    assert abs(ipsae_d0res_max(_pae(1.0), BINDER, TARGET, 10.0).item() - 0.5) < 1e-6


def test_term_ipsae_and_iptm():
    s = Scorer(config=DesignConfig())
    assert abs(s.term_iptm(_outputs(1.0)) - 0.60) < 1e-6  # chain_pair_iptm[binder=0, target=1]
    assert abs(s.term_ipsae(_outputs(1.0)) - 0.5) < 1e-6
    assert s.term_ipsae(_outputs(20.0)) == 0.0


def test_ipsae_cutoff_is_configurable():
    cfg = DesignConfig()
    cfg.ipsae_pae_cutoff = 0.5  # now PAE=1 pairs are no longer "valid"
    assert Scorer(config=cfg).term_ipsae(_outputs(1.0)) == 0.0


def test_interface_pae_stats():
    stats = Scorer(config=DesignConfig()).interface_pae_stats(_outputs(3.0))
    assert abs(stats["mean_ipae"] - 3.0) < 1e-6
    assert abs(stats["min_ipae"] - 3.0) < 1e-6


def test_iptm_ipsae_preset_and_fitness():
    w = ScoreWeights.iptm_ipsae()
    assert w.ipsae == 1.0 and w.iptm == 0.0 and w.mean_ipae == -1.0 / 31.0
    cfg = DesignConfig()
    cfg.weights = w
    terms = Scorer(config=cfg).score_terms(_outputs(1.0))
    assert "ipsae" in terms and abs(terms["ipsae"] - 0.5) < 1e-6
    # fitness must resolve a weight for every term (ipsae included) without raising.
    assert isinstance(Scorer(config=cfg).fitness(terms), float)
