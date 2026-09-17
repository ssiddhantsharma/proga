# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Protenix selectivity-aware genetic pipeline.

These tests do NOT load the Protenix model; scoring is exercised on synthetic
:class:`PredictionOutputs`. A single GPU smoke test is skipped when CUDA / the
model checkpoint are unavailable.
"""

import os
import random
from dataclasses import fields

import gemmi
import pytest
import torch
import torch.nn.functional as F

from proga.complex import (
    apply_sequence,
    build_sample_dict,
    discriminating_indices,
    extract_cdr_residues,
    parse_design_mask,
)
from proga.config import (
    DESIGN_ALPHABET,
    NANOBODY_SCAFFOLD,
    DesignConfig,
    ScoreWeights,
)
from proga.genetic import (
    GeneticOptimizer,
    crossover,
    crowding_distance,
    initialize_population,
    mutate,
    non_dominated_fronts,
    pareto_select,
    top_k,
    Individual,
    _dominates,
)
from proga import rmsd
from proga.oracle import ChainIndex, PredictionOutputs
from proga.scoring import Scorer, _TabsRef, build_tabs_ref, max_k, min_k
from proga.spec import (
    BinderSpec,
    FixedChain,
    OffTarget,
    ReferenceSpec,
    TargetSpec,
    pmhc_spec_from_cifs,
)


# --------------------------------------------------------------------------- #
# Aggregators                                                                  #
# --------------------------------------------------------------------------- #
def test_min_k_max_k():
    v = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0])
    assert min_k(v, 2).item() == pytest.approx(1.5)  # mean(1, 2)
    assert max_k(v, 2).item() == pytest.approx(4.5)  # mean(5, 4)
    # k larger than size clamps to all.
    assert min_k(v, 99).item() == pytest.approx(3.0)
    # empty -> 0
    assert min_k(torch.empty(0), 2).item() == 0.0


# --------------------------------------------------------------------------- #
# Design mask & sequence splicing                                              #
# --------------------------------------------------------------------------- #
def test_parse_design_mask():
    mask = parse_design_mask(NANOBODY_SCAFFOLD)
    assert len(mask.cdr_spans) == 3
    lengths = [end - start for start, end in mask.cdr_spans]
    assert lengths == [9, 10, 33]
    assert len(mask.positions) == 9 + 10 + 33
    # span covers from first CDR1 X to last CDR3 X.
    assert mask.span == (mask.positions[0], mask.positions[-1] + 1)
    # No 'X' remains outside the mask positions.
    assert all(NANOBODY_SCAFFOLD[p] == "X" for p in mask.positions)


def test_apply_and_extract_roundtrip():
    mask = parse_design_mask(NANOBODY_SCAFFOLD)
    genome = "".join(DESIGN_ALPHABET[i % len(DESIGN_ALPHABET)] for i in range(len(mask.positions)))
    seq = apply_sequence(mask, genome)
    assert "X" not in seq
    assert len(seq) == len(NANOBODY_SCAFFOLD)
    assert extract_cdr_residues(mask, seq) == genome
    # Framework positions are untouched.
    for fp in mask.framework_positions():
        assert seq[fp] == NANOBODY_SCAFFOLD[fp]


def test_apply_sequence_length_check():
    mask = parse_design_mask(NANOBODY_SCAFFOLD)
    with pytest.raises(ValueError):
        apply_sequence(mask, "AAA")


# --------------------------------------------------------------------------- #
# GA operators                                                                 #
# --------------------------------------------------------------------------- #
def test_initialize_population_excludes_cys():
    rng = random.Random(0)
    pop = initialize_population(20, 30, rng)
    assert len(pop) == 20
    for g in pop:
        assert len(g) == 30
        assert "C" not in g


def test_mutate_respects_rate_and_alphabet():
    rng = random.Random(1)
    g = "A" * 50
    assert mutate(g, 0.0, rng) == g  # no mutation
    fully = mutate(g, 1.0, rng)
    assert len(fully) == 50
    assert "C" not in fully


def test_crossover_breakpoint_within_span():
    mask = parse_design_mask(NANOBODY_SCAFFOLD)
    n = len(mask.positions)
    g1 = "A" * n
    g2 = "D" * n
    rng = random.Random(2)
    for _ in range(50):
        child = crossover(g1, g2, mask, rng)
        assert len(child) == n
        # Each child residue comes from g1 ('A') or g2 ('D'); prefix of A's then D's.
        assert set(child) <= {"A", "D"}
        a_count = child.count("A")
        assert child[:a_count] == "A" * a_count  # contiguous prefix from g1


def test_top_k_ordering():
    pop = [Individual("a", 1.0), Individual("b", 3.0), Individual("c", 2.0)]
    best = top_k(pop, 2)
    assert [ind.genome for ind in best] == ["b", "c"]


# --------------------------------------------------------------------------- #
# Multi-objective (Pareto / NSGA-II) selection                                #
# --------------------------------------------------------------------------- #
def test_dominates_maximisation():
    assert _dominates((1.0, 1.0), (0.0, 0.0))       # strictly better on both
    assert _dominates((1.0, 0.0), (0.0, 0.0))       # better on one, equal on other
    assert not _dominates((1.0, 0.0), (0.0, 1.0))   # trade-off: neither dominates
    assert not _dominates((0.0, 0.0), (0.0, 0.0))   # equal does not dominate


def test_non_dominated_fronts_partition():
    # Trade-off pair (mutually non-dominated) on F0; a point they both dominate on F1.
    #   idx0=(1,0) and idx1=(0,1): neither dominates the other  -> both F0
    #   idx2=(0.4,0.4): NOT dominated by either specialist (each worse on one axis)
    #                   -> also F0
    #   idx3=(0,0): dominated by all of the above               -> F1
    objs = [(1.0, 0.0), (0.0, 1.0), (0.4, 0.4), (0.0, 0.0)]
    fronts = non_dominated_fronts(objs)
    assert len(fronts) == 2
    assert sorted(fronts[0]) == [0, 1, 2]
    assert fronts[1] == [3]


def test_pareto_keeps_tradeoff_that_scalar_drops():
    """A design off the weighted-sum optimum but Pareto-optimal must survive."""
    # Axis values are already 'higher is better' (identity objective here).
    # Scalar fitness = a[0] + a[1] would rank 'balanced' top and could drop the
    # extreme specialists; Pareto keeps all three (they are mutually non-dominated).
    axes = {
        "spec_on": (1.0, 0.1),   # great on-target, weak off-target-clean
        "spec_off": (0.1, 1.0),  # weak on-target, great off-target-clean
        "balanced": (0.6, 0.6),
        "dominated": (0.05, 0.05),
    }
    pop = [Individual(g, a[0] + a[1]) for g, a in axes.items()]
    obj = lambda ind: axes[ind.genome]
    kept = {ind.genome for ind in pareto_select(pop, 3, obj)}
    # The dominated one is dropped; both specialists survive despite low scalar.
    assert "dominated" not in kept
    assert "spec_on" in kept and "spec_off" in kept


def test_pareto_select_drops_dominated_first():
    axes = {"a": (1.0, 1.0), "b": (0.9, 0.9), "c": (0.0, 0.0)}
    pop = [Individual(g, sum(a)) for g, a in axes.items()]
    obj = lambda ind: axes[ind.genome]
    # Only room for two: 'a' dominates all; 'c' is dominated by both → dropped.
    kept = [ind.genome for ind in pareto_select(pop, 2, obj)]
    assert kept[0] == "a"
    assert "c" not in kept


def test_crowding_distance_boundaries_infinite():
    objs = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    dist = crowding_distance(objs, [0, 1, 2])
    assert dist[0] == float("inf") and dist[2] == float("inf")  # extremes
    assert dist[1] < float("inf")                               # interior finite


def test_scalar_path_unchanged_by_default():
    """Default selection is 'scalar' → GA result identical to pre-Pareto behaviour."""
    cfg = DesignConfig(population=8, generations=6, top_n=5, mutation_rate=0.2, seed=3)
    assert cfg.selection == "scalar"
    mask = parse_design_mask(NANOBODY_SCAFFOLD)

    def evaluate(genome, use_ddg):
        return float(genome.count("A")), {"plddt": 0.0}

    result = GeneticOptimizer(cfg, mask).run(evaluate)
    bests = result.best_per_generation
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))  # elitism preserved
    assert len(result.top) == 5


def test_pareto_mixed_direction_axes():
    """Sign-folding: a maximized term (iptm) and a minimized penalty (mean_ipae)
    must both point 'higher-is-better' after signed_weight * term. This is the
    real selectivity case — a negative-weight axis was untested otherwise."""
    w = ScoreWeights(**{f.name: 0.0 for f in fields(ScoreWeights)})
    w.iptm = 1.0        # maximize
    w.mean_ipae = -1.0  # minimize (penalty)
    cfg = DesignConfig(
        selection="pareto", weights=w, pareto_objectives=["iptm", "mean_ipae"]
    )
    opt = GeneticOptimizer(cfg, parse_design_mask(NANOBODY_SCAFFOLD))
    axis = opt._pareto_objective

    # Lower mean_ipae must score higher on its (negative-weight) axis.
    low = Individual("low", 0.0, {"iptm": 0.0, "mean_ipae": 2.0})
    high = Individual("high", 0.0, {"iptm": 0.0, "mean_ipae": 8.0})
    assert axis(low) > axis(high)

    # A: strong iptm / weak ipae; B: weak iptm / strong ipae; C dominated on both.
    a = Individual("A", 0.0, {"iptm": 0.9, "mean_ipae": 8.0})
    b = Individual("B", 0.0, {"iptm": 0.5, "mean_ipae": 3.0})
    c = Individual("C", 0.0, {"iptm": 0.4, "mean_ipae": 9.0})
    kept = {ind.genome for ind in pareto_select([a, b, c], 2, axis)}
    assert kept == {"A", "B"}  # trade-off front kept, dominated C dropped


def test_ga_pareto_end_to_end_runs_and_is_deterministic():
    """Pareto GA runs on a fake two-objective oracle and is resume-safe/deterministic."""
    from proga.config import ScoreWeights

    # Two live axes: reward 'A' count (iptm proxy) and 'D' count (repulsion proxy).
    w = ScoreWeights(**{f.name: 0.0 for f in fields(ScoreWeights)})
    w.iptm = 1.0
    w.repulsion = 1.0  # positive here just to define a +direction for the proxy
    cfg = DesignConfig(
        population=8, generations=6, top_n=5, mutation_rate=0.2, seed=5,
        weights=w, selection="pareto", pareto_objectives=["iptm", "repulsion"],
    )
    mask = parse_design_mask(NANOBODY_SCAFFOLD)

    def evaluate_batch(genomes, use_ddg):
        out = []
        for g in genomes:
            terms = {"iptm": float(g.count("A")), "repulsion": float(g.count("D"))}
            fitness = terms["iptm"] + terms["repulsion"]
            out.append((g, fitness, terms))
        return out

    r1 = GeneticOptimizer(cfg, mask).run_resumable(evaluate_batch)
    r2 = GeneticOptimizer(cfg, mask).run_resumable(evaluate_batch)
    assert len(r1.top) == 5
    assert [i.genome for i in r1.top] == [i.genome for i in r2.top]  # deterministic


# --------------------------------------------------------------------------- #
# Discriminating residue detection                                            #
# --------------------------------------------------------------------------- #
def test_discriminating_indices():
    assert discriminating_indices("VVVGADGVGK", "VVVGAGGVGK") == [5]
    with pytest.raises(ValueError):
        discriminating_indices("AAA", "AAAA")


# --------------------------------------------------------------------------- #
# build_sample_dict                                                            #
# --------------------------------------------------------------------------- #
def test_build_sample_dict_chain_order():
    d = build_sample_dict("NANOBODY", "PEPTIDE", "MHCSEQ", name="x")
    seqs = d["sequences"]
    assert len(seqs) == 3
    assert seqs[0]["proteinChain"]["sequence"] == "NANOBODY"
    assert seqs[0]["proteinChain"]["id"] == ["A"]
    assert seqs[1]["proteinChain"]["id"] == ["B"]
    assert seqs[2]["proteinChain"]["id"] == ["C"]
    # No MSA fields unless provided.
    assert "pairedMsaPath" not in seqs[1]["proteinChain"]


def test_build_sample_dict_attaches_msa_by_seq():
    msa_by_seq = {
        "MHCSEQ": {"pairedMsaPath": "/x/pairing.a3m", "unpairedMsaPath": "/x/non_pairing.a3m"},
    }
    d = build_sample_dict("NANOBODY", "PEPTIDE", "MHCSEQ", msa_by_seq=msa_by_seq)
    nb, pep, mhc = (s["proteinChain"] for s in d["sequences"])
    # MHC gets the paths; nanobody and peptide stay MSA-free.
    assert mhc["unpairedMsaPath"] == "/x/non_pairing.a3m"
    assert mhc["pairedMsaPath"] == "/x/pairing.a3m"
    assert "unpairedMsaPath" not in nb and "pairedMsaPath" not in nb
    assert "unpairedMsaPath" not in pep


def test_msa_cache_hashing_and_reuse(tmp_path, monkeypatch):
    """get_or_compute searches only uncached seqs and reuses on the second call."""
    from proga import msa as msa_mod

    calls = {"n": 0}

    def fake_msa_search(seqs, msa_res_dir, mode="protenix"):
        calls["n"] += 1
        subdirs = []
        for i, _ in enumerate(seqs):
            sub = os.path.join(msa_res_dir, str(i))
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, "non_pairing.a3m"), "w") as f:
                f.write(">q\nAAAA\n")
            subdirs.append(sub)
        return subdirs

    # msa_search is imported inside MSACache._search_and_cache.
    import runner.msa_search as rms
    monkeypatch.setattr(rms, "msa_search", fake_msa_search)

    cache = msa_mod.MSACache(str(tmp_path), mode="protenix")
    out1 = cache.get_or_compute(["SEQONE", "SEQTWO"])
    assert calls["n"] == 1
    assert out1["SEQONE"]["unpairedMsaPath"].endswith("non_pairing.a3m")
    assert os.path.exists(out1["SEQTWO"]["unpairedMsaPath"])

    # Distinct sequences hash to distinct cache dirs.
    assert msa_mod.seq_hash("SEQONE") != msa_mod.seq_hash("SEQTWO")

    # Second call: everything cached -> no new search.
    out2 = cache.get_or_compute(["SEQONE", "SEQTWO"])
    assert calls["n"] == 1
    assert out2["SEQONE"] == out1["SEQONE"]

    # A new sequence triggers exactly one more search.
    cache.get_or_compute(["SEQONE", "SEQNEW"])
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# Synthetic PredictionOutputs for scoring tests                                #
# --------------------------------------------------------------------------- #
DISTO_PARAMS = {"min_bin": 2.3125, "max_bin": 21.6875, "no_bins": 64}


def _make_outputs(contact_pairs=()):
    """Build a tiny complex: 5 nanobody + 3 peptide + 4 mhc tokens.

    ``contact_pairs`` is a list of (i, j) token pairs forced into strong contact
    (high probability in the nearest distance bin).
    """
    n_token = 12
    n_bins = 64
    asym_id = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    residue_index = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3, 4])

    distogram_logits = torch.zeros(n_token, n_token, n_bins)
    # Default: mass in a far bin (no contact).
    distogram_logits[:, :, -1] = 5.0
    for i, j in contact_pairs:
        distogram_logits[i, j, :] = 0.0
        distogram_logits[j, i, :] = 0.0
        distogram_logits[i, j, 0] = 10.0  # near bin -> contact
        distogram_logits[j, i, 0] = 10.0

    chain_index = ChainIndex(
        nanobody_tokens=torch.tensor([0, 1, 2, 3, 4]),
        peptide_tokens=torch.tensor([5, 6, 7]),
        mhc_tokens=torch.tensor([8, 9, 10, 11]),
        design_tokens=torch.tensor([1, 2, 3]),
        residue_index=residue_index,
        asym_id=asym_id,
        nanobody_asym=0,
        peptide_asym=1,
        mhc_asym=2,
    )
    return PredictionOutputs(
        per_token_plddt=torch.full((n_token,), 0.9),
        pae=torch.full((n_token, n_token), 5.0),
        distogram_logits=distogram_logits,
        ca_coords=torch.randn(n_token, 3) * 5.0,
        chain_pair_iptm=torch.tensor(
            [[0.0, 0.8, 0.1], [0.8, 0.0, 0.2], [0.1, 0.2, 0.0]]
        ),
        iptm_global=0.75,
        chain_index=chain_index,
        distogram_bin_params=DISTO_PARAMS,
    )


def test_scoring_terms_ranges_and_iptm():
    scorer = Scorer(config=DesignConfig(), discriminating_positions=[1])
    out = _make_outputs()
    terms = scorer.score_terms(out)
    assert set(terms) == {
        "iptm", "iptm_global",
        "plddt", "i_plddt", "i_pae", "pae_intra", "con_intra",
        "rg", "paratope", "target_contact", "peptide_contact", "repulsion",
        "mpnn",
        "min_ipae", "min_2_ipae", "min_3_ipae", "min_4_ipae", "mean_ipae",
    }
    assert terms["iptm"] == pytest.approx(0.8)  # chain_pair_iptm[0, 1]
    assert terms["iptm_global"] == pytest.approx(0.75)  # summary global iptm
    assert terms["plddt"] == pytest.approx(0.1, abs=1e-6)  # 1 - 0.9
    assert terms["i_pae"] == pytest.approx(5.0 / 31.0, abs=1e-6)
    for v in terms.values():
        assert torch.isfinite(torch.tensor(v))


def test_interface_pae_family():
    """Binder<->target interface PAE (Angstrom): min/min-k/mean over both blocks."""
    scorer = Scorer(config=DesignConfig())
    out = _make_outputs()  # uniform pae = 5.0 everywhere
    stats = scorer.interface_pae_stats(out)
    # All interface entries are 5.0, so every statistic equals 5.0 (raw Angstrom).
    for key in ("min_ipae", "min_2_ipae", "min_3_ipae", "min_4_ipae", "mean_ipae"):
        assert stats[key] == pytest.approx(5.0)

    # Lowering a few binder<->target entries pulls min_ipae below the mean.
    out2 = _make_outputs()
    binder, peptide, mhc = (
        out2.chain_index.nanobody_tokens,
        out2.chain_index.peptide_tokens,
        out2.chain_index.mhc_tokens,
    )
    out2.pae[binder[0], peptide[0]] = 0.5  # one strong interface pair
    out2.pae[binder[1], mhc[0]] = 1.0
    s2 = scorer.interface_pae_stats(out2)
    assert s2["min_ipae"] == pytest.approx(0.5)
    assert s2["min_2_ipae"] == pytest.approx((0.5 + 1.0) / 2)
    assert s2["min_ipae"] <= s2["min_2_ipae"] <= s2["mean_ipae"]
    assert s2["mean_ipae"] < 5.0


def test_iptm_ipae_objective_preset():
    """Preset selects on global ipTM (+), interface mean ipAE (-), and contact (-)."""
    w = ScoreWeights.iptm_ipae()
    assert w.iptm_global == pytest.approx(1.0)
    assert w.mean_ipae == pytest.approx(-1.0 / 31.0)
    assert w.target_contact == pytest.approx(-0.5)  # keeps selectivity pressure
    # The binder<->target-pair ipTM is no longer weighted (global ipTM is used).
    zeroed = ["iptm", "plddt", "i_plddt", "i_pae", "pae_intra",
              "con_intra", "rg", "paratope", "peptide_contact", "repulsion",
              "ddg", "min_ipae", "min_2_ipae", "min_3_ipae", "min_4_ipae"]
    for name in zeroed:
        assert getattr(w, name) == 0.0
    # Opt out of discriminating-residue pressure for a pure interface objective.
    assert ScoreWeights.iptm_ipae(target_contact=0.0).target_contact == 0.0

    # Fitness rewards high global ipTM and penalises high mean interface PAE.
    cfg = DesignConfig(weights=w)
    scorer = Scorer(config=cfg)
    hi = scorer.fitness({"iptm_global": 0.9, "mean_ipae": 4.0})
    lo = scorer.fitness({"iptm_global": 0.9, "mean_ipae": 20.0})
    assert hi > lo
    # Report-only metric without a weight is ignored by fitness.
    assert scorer.fitness({"iptm_global": 0.5, "min_3_ipae": 99.0}) == pytest.approx(0.5)


def test_iptm_ipae_repulsion_objective_preset():
    """iptm_ipae plus a repulsion penalty; everything else matches iptm_ipae."""
    w = ScoreWeights.iptm_ipae_repulsion()
    base = ScoreWeights.iptm_ipae()
    assert w.iptm_global == pytest.approx(1.0)
    assert w.mean_ipae == pytest.approx(-1.0 / 31.0)
    assert w.target_contact == pytest.approx(-0.5)
    assert w.repulsion == pytest.approx(-0.5)  # the added selectivity pressure
    # Identical to iptm_ipae on every term other than repulsion.
    for f in fields(ScoreWeights):
        if f.name != "repulsion":
            assert getattr(w, f.name) == getattr(base, f.name)
    # repulsion=0.0 reduces exactly to iptm_ipae.
    assert ScoreWeights.iptm_ipae_repulsion(repulsion=0.0).repulsion == 0.0
    # Higher repulsion loss lowers fitness (negative weight).
    cfg = DesignConfig(weights=w)
    scorer = Scorer(config=cfg)
    assert (scorer.fitness({"iptm": 0.9, "repulsion": 0.0})
            > scorer.fitness({"iptm": 0.9, "repulsion": 2.0}))


def test_target_contact_rewards_discriminating_contact():
    """Lower target-contact loss when a CDR token contacts the discriminating residue."""
    scorer = Scorer(config=DesignConfig(), discriminating_positions=[1])
    # discriminating peptide local index 1 -> token 6.
    no_contact = scorer.term_target_contact(_make_outputs())
    with_contact = scorer.term_target_contact(_make_outputs(contact_pairs=[(2, 6)]))
    assert with_contact < no_contact


def test_repulsion_penalises_mhc_contact():
    cfg = DesignConfig()
    cfg.mhc_avoidance_residues = [1, 2, 3, 4]  # all mhc residues
    scorer = Scorer(config=cfg, discriminating_positions=[1])
    # design token 2 contacting mhc token 9 -> larger repulsion.
    base = scorer.term_repulsion(_make_outputs())
    contacting = scorer.term_repulsion(_make_outputs(contact_pairs=[(2, 9)]))
    assert contacting > base


def test_fitness_is_signed_weighted_sum():
    scorer = Scorer(config=DesignConfig())
    terms = {"iptm": 0.5, "plddt": 0.2, "peptide_contact": 1.0}
    w = scorer.config.weights
    expected = w.iptm * 0.5 + w.plddt * 0.2 + w.peptide_contact * 1.0
    assert scorer.fitness(terms) == pytest.approx(expected)


def test_chain_index_mhc_residue_lookup():
    out = _make_outputs()
    tokens = out.chain_index.mhc_tokens_for_residues([2, 3])
    assert tokens.tolist() == [9, 10]
    assert out.chain_index.mhc_tokens_for_residues([]).numel() == 0


# --------------------------------------------------------------------------- #
# GA end-to-end with a fake oracle                                             #
# --------------------------------------------------------------------------- #
def test_genetic_optimizer_improves_and_dedups():
    cfg = DesignConfig(population=8, generations=6, top_n=5, mutation_rate=0.2, seed=3)
    mask = parse_design_mask(NANOBODY_SCAFFOLD)
    opt = GeneticOptimizer(cfg, mask)

    # Fitness = number of 'A's in the genome (max = len(mask.positions)).
    def evaluate(genome, use_ddg):
        return float(genome.count("A")), {"plddt": 0.0}

    result = opt.run(evaluate)
    # Elitism (TopK(P ∪ O)) => best fitness is monotonic non-decreasing.
    bests = result.best_per_generation
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))
    assert len(result.top) == 5
    # Top sequences are unique.
    genomes = [ind.genome for ind in result.top]
    assert len(set(genomes)) == len(genomes)
    # Best-of-run equals the maximum recorded fitness.
    assert result.top[0].fitness == max(ind.fitness for ind in result.top)


def test_run_resumable_matches_run():
    """run_resumable (batched) reproduces run (per-genome) exactly."""
    cfg = DesignConfig(population=8, generations=6, top_n=5, mutation_rate=0.2, seed=7)
    mask = parse_design_mask(NANOBODY_SCAFFOLD)

    def evaluate(genome, use_ddg):
        return float(genome.count("A")), {"plddt": 0.0}

    def evaluate_batch(genomes, use_ddg):
        return [(g, float(g.count("A")), {"plddt": 0.0}) for g in genomes]

    serial = GeneticOptimizer(cfg, mask).run(evaluate)
    batched = GeneticOptimizer(cfg, mask).run_resumable(evaluate_batch)
    assert [i.genome for i in serial.top] == [i.genome for i in batched.top]
    assert serial.best_per_generation == batched.best_per_generation
    assert serial.num_evaluations == batched.num_evaluations


def test_run_resumable_checkpoint_resume():
    """Resuming from a mid-run checkpoint yields the same final result."""
    import json

    cfg = DesignConfig(population=8, generations=6, top_n=5, mutation_rate=0.2, seed=11)
    mask = parse_design_mask(NANOBODY_SCAFFOLD)

    def evaluate_batch(genomes, use_ddg):
        return [(g, float(g.count("A")), {"plddt": 0.0}) for g in genomes]

    # Uninterrupted reference run, capturing JSON-round-tripped checkpoints.
    checkpoints = []
    full = GeneticOptimizer(cfg, mask).run_resumable(
        evaluate_batch,
        checkpoint_cb=lambda s: checkpoints.append(json.loads(json.dumps(s))),
    )
    # Resume from the generation-3 checkpoint on a fresh optimizer.
    mid = next(c for c in checkpoints if c["generation"] == 3)
    resumed = GeneticOptimizer(cfg, mask).run_resumable(evaluate_batch, resume_state=mid)
    assert [i.genome for i in resumed.top] == [i.genome for i in full.top]
    assert resumed.best_per_generation == full.best_per_generation
    assert resumed.num_evaluations == full.num_evaluations


def test_contact_prob_monotonic_in_cutoff():
    from protenix.model.sample_confidence import compute_contact_prob

    logits = torch.randn(4, 4, 64)
    p8 = compute_contact_prob(logits, thres=8.0, **DISTO_PARAMS)
    p14 = compute_contact_prob(logits, thres=14.0, **DISTO_PARAMS)
    assert torch.all(p14 >= p8 - 1e-6)


# --------------------------------------------------------------------------- #
# Target-agnostic spec + generalized ChainIndex / scoring                      #
# --------------------------------------------------------------------------- #
def _make_general_outputs(contact_pairs=()):
    """Generic 2-chain complex: 5 binder (A) + 3 target (B) tokens.

    Built through the *general* ChainIndex constructor (no pMHC legacy fields).
    """
    n_token, n_bins = 8, 64
    asym_id = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    residue_index = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3])

    distogram_logits = torch.zeros(n_token, n_token, n_bins)
    distogram_logits[:, :, -1] = 5.0
    for i, j in contact_pairs:
        distogram_logits[i, j, :] = 0.0
        distogram_logits[j, i, :] = 0.0
        distogram_logits[i, j, 0] = 10.0
        distogram_logits[j, i, 0] = 10.0

    chain_index = ChainIndex(
        chain_ids=["A", "B"],
        chain_tokens={"A": torch.tensor([0, 1, 2, 3, 4]), "B": torch.tensor([5, 6, 7])},
        chain_roles={"A": "binder", "B": "target"},
        chain_asym={"A": 0, "B": 1},
        binder_id="A",
        design_tokens=torch.tensor([1, 2, 3]),
        residue_index=residue_index,
        asym_id=asym_id,
    )
    return PredictionOutputs(
        per_token_plddt=torch.full((n_token,), 0.9),
        pae=torch.full((n_token, n_token), 5.0),
        distogram_logits=distogram_logits,
        ca_coords=torch.randn(n_token, 3) * 5.0,
        chain_pair_iptm=torch.tensor([[0.0, 0.7], [0.7, 0.0]]),
        chain_index=chain_index,
        distogram_bin_params=DISTO_PARAMS,
    )


def test_chain_index_general_groups():
    ci = _make_general_outputs().chain_index
    assert ci.binder_tokens.tolist() == [0, 1, 2, 3, 4]
    assert ci.binder_asym == 0
    assert ci.target_tokens.tolist() == [5, 6, 7]
    assert ci.rest_of_complex_tokens.tolist() == [5, 6, 7]
    # res-id lookup into a named chain.
    assert ci.contact_tokens({"B": [2]}).tolist() == [6]
    assert ci.contact_tokens({"B": []}).numel() == 0


def test_general_path_matches_legacy_scoring():
    """The contact/avoidance res-id paths reproduce the legacy local-index paths."""
    cfg = DesignConfig()
    # target_contact: legacy discriminating_positions=[1] == general {"B":[2]}
    # (peptide local index 1 has residue_index 2 in _make_outputs).
    legacy = Scorer(config=cfg, discriminating_positions=[1])
    general = Scorer(config=cfg, contact_residues={"B": [2]})
    pairs = [(2, 6)]
    assert general.term_target_contact(_make_outputs(pairs)) == pytest.approx(
        legacy.term_target_contact(_make_outputs(pairs))
    )

    # repulsion: legacy config.mhc_avoidance_residues == general {"C": [...]}.
    cfg_legacy = DesignConfig()
    cfg_legacy.mhc_avoidance_residues = [1, 2, 3, 4]
    legacy_rep = Scorer(config=cfg_legacy)
    general_rep = Scorer(config=DesignConfig(), avoidance_residues={"C": [1, 2, 3, 4]})
    rp = [(2, 9)]
    assert general_rep.term_repulsion(_make_outputs(rp)) == pytest.approx(
        legacy_rep.term_repulsion(_make_outputs(rp))
    )


def test_generic_two_chain_scoring():
    cfg = DesignConfig()
    scorer = Scorer(config=cfg, contact_residues={"B": [2]}, iptm_target_id="B")
    terms = scorer.score_terms(_make_general_outputs())
    # Same full set of term keys as the pMHC case.
    assert set(terms) == {
        "iptm", "iptm_global",
        "plddt", "i_plddt", "i_pae", "pae_intra", "con_intra",
        "rg", "paratope", "target_contact", "peptide_contact", "repulsion",
        "mpnn",
        "min_ipae", "min_2_ipae", "min_3_ipae", "min_4_ipae", "mean_ipae",
    }
    assert terms["iptm"] == pytest.approx(0.7)  # chain_pair_iptm[binder=0, B=1]
    assert terms["repulsion"] == 0.0  # no avoidance set, no mhc shim
    for v in terms.values():
        assert torch.isfinite(torch.tensor(v))


# --------------------------------------------------------------- ProteinMPNN term

def _make_mpnn_outputs(seq="ACDEFGHK"):
    """``_make_general_outputs`` (8 tokens) + synthetic backbone + sequence."""
    out = _make_general_outputs()
    n = out.ca_coords.shape[0]
    torch.manual_seed(0)
    out.backbone_coords = torch.randn(n, 4, 3) * 5.0
    out.sequence = seq
    return out


def test_term_mpnn_inert_without_model():
    out = _make_mpnn_outputs()
    scorer = Scorer(config=DesignConfig())  # no mpnn_model attached
    assert scorer.term_mpnn(out) == 0.0
    assert scorer.score_terms(out)["mpnn"] == 0.0  # term present but inert


def test_fitness_inert_when_mpnn_weight_zero():
    scorer = Scorer(config=DesignConfig(weights=ScoreWeights(mpnn=0.0)))
    base = {"iptm": 0.7}
    assert scorer.fitness(base) == scorer.fitness({**base, "mpnn": -3.2})


def test_fitness_shifts_when_mpnn_weighted():
    scorer = Scorer(config=DesignConfig(weights=ScoreWeights(mpnn=0.5)))
    base = {"iptm": 0.7}
    assert scorer.fitness({**base, "mpnn": -3.2}) == pytest.approx(
        scorer.fitness(base) + 0.5 * -3.2
    )


@pytest.mark.parametrize("reduction", ["min", "mean"])
def test_term_mpnn_with_real_model(reduction):
    from proga.proteinmpnn import load_scoring_model, weights_path

    if not weights_path().exists():
        pytest.skip("ProteinMPNN weights not vendored")
    model = load_scoring_model("v_48_020", device="cpu")
    cfg = DesignConfig(mpnn_score_chain="target", mpnn_reduction=reduction)
    scorer = Scorer(config=cfg, mpnn_model=model)
    out = _make_mpnn_outputs()
    v1 = scorer.term_mpnn(out)
    v2 = scorer.term_mpnn(out)
    assert v1 == v2  # deterministic (augment_eps=0, fixed decode order)
    assert v1 < 0.0  # a log-probability
    assert torch.isfinite(torch.tensor(v1))


def test_targetspec_roundtrip(tmp_path):
    spec = TargetSpec(
        name="demo",
        binder=BinderSpec(id="A", denovo_length=20, name="minibinder"),
        chains=[
            FixedChain(id="B", sequence="MKTAYIAK", role="target", name="target",
                       contact_residues=[3, 4], avoidance_residues=[7]),
        ],
        off_targets=[
            OffTarget(name="off", chains=[FixedChain(id="B", sequence="MKTGYIAK", role="target")]),
        ],
        msa_chain_ids=["B"],
        iptm_target_id="B",
        reference=ReferenceSpec(path="ref.pdb", target_chains=["B"], binder_chains=["A"]),
    )
    again = TargetSpec.from_dict(spec.to_dict())
    assert again.to_dict() == spec.to_dict()
    assert again.reference == spec.reference
    assert again.roles() == {"A": "binder", "B": "target"}
    assert again.contact_residues_map() == {"B": [3, 4]}
    assert again.avoidance_residues_map() == {"B": [7]}
    assert again.target_chain_ids() == ["B"]
    # de-novo binder: whole chain designable.
    assert len(again.binder.mask().positions) == 20
    # JSON file round-trip.
    p = tmp_path / "spec.json"
    spec.to_json(str(p))
    assert TargetSpec.from_file(str(p)).to_dict() == spec.to_dict()


def test_pmhc_spec_from_cifs_builder(monkeypatch):
    """pmhc_spec_from_cifs assembles the legacy pMHC roles/sets (deps monkeypatched)."""
    from proga import spec as spec_mod
    from proga.complex import PMHCTarget

    def fake_load(path, name=None):
        if "off" in path:
            return PMHCTarget(name="off_target", peptide_seq="VVVGAGGVGK", mhc_seq="MHCSEQMHC")
        return PMHCTarget(name="target", peptide_seq="VVVGADGVGK", mhc_seq="MHCSEQMHC")

    monkeypatch.setattr(spec_mod, "load_pmhc_from_cif", fake_load)
    s = pmhc_spec_from_cifs("on.cif", "off.cif")
    assert s.binder.scaffold == NANOBODY_SCAFFOLD
    assert s.discriminating_positions == [5]  # G12D differs at local index 5
    pep, mhc = s.chains
    assert pep.id == "B" and pep.role == "target" and pep.name == "peptide"
    assert pep.contact_residues == [6]  # 1-based res_id of local index 5
    assert mhc.id == "C" and mhc.role == "context"
    assert mhc.avoidance_residues == list(range(182, 275))
    assert s.iptm_target_id == "B"
    assert s.off_targets[0].aligned_to == {"B": "B"}


# --------------------------------------------------------------------------- #
# Target-aligned binder RMSD (TABS)                                            #
# --------------------------------------------------------------------------- #
def _pos(x, y, z) -> gemmi.Position:
    return gemmi.Position(float(x), float(y), float(z))


def _rigid_transform() -> gemmi.Transform:
    mat = gemmi.Mat33([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90° about z
    return gemmi.Transform(mat, gemmi.Vec3(5.0, -3.0, 2.0))


def _apply(transform, positions) -> list:
    moved = []
    for p in positions:
        v = transform.apply(p)
        moved.append(gemmi.Position(v.x, v.y, v.z))
    return moved


def _write_ca_pdb(path, chains: dict) -> None:
    """chains: name -> list of (3-letter resname, (x, y, z))."""
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    for cname, residues in chains.items():
        chain = gemmi.Chain(cname)
        for i, (resname, xyz) in enumerate(residues, start=1):
            res = gemmi.Residue()
            res.name = resname
            res.seqid = gemmi.SeqId(i, " ")
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = _pos(*xyz)
            res.add_atom(atom)
            chain.add_residue(res)
        model.add_chain(chain)
    structure.add_model(model)
    structure.write_pdb(str(path))


def _reference_spec(path, target_seq="AAAA", binder_seq="GGG") -> TargetSpec:
    return TargetSpec(
        name="demo",
        binder=BinderSpec(id="A", scaffold=binder_seq, name="binder"),
        chains=[FixedChain(id="C", sequence=target_seq, role="target", name="target")],
        reference=ReferenceSpec(path=str(path), target_chains=["T"], binder_chains=["L"]),
    )


def _tabs_scorer_and_outputs(binder_shift, tol=2.5, weight=-0.1):
    """Scorer with a hand-built reference: identity target fit, binder shifted in z."""
    outputs = _make_general_outputs()  # binder tokens 0-4, target tokens 5-7
    coords = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        [0.0, 0.0, 9.0], [1.0, 0.0, 9.0], [0.0, 1.0, 9.0],
    ])
    outputs.ca_coords = coords
    tabs = _TabsRef(
        target_ca=[_pos(*coords[i].tolist()) for i in (5, 6, 7)],
        binder_ca=[_pos(coords[i][0], coords[i][1], coords[i][2] + binder_shift) for i in range(5)],
        target_pred_idx=[0, 1, 2],
        binder_pred_idx=[0, 1, 2, 3, 4],
    )
    weights = ScoreWeights()
    weights.target_aligned_binder_rmsd = weight
    return Scorer(config=DesignConfig(weights=weights, tabs_tol=tol), tabs=tabs), outputs


def test_matched_columns_handles_length_mismatch():
    idx_a, idx_b = rmsd.matched_columns("CDEFG", "AACDEFGHH")
    assert idx_a == [0, 1, 2, 3, 4]
    assert idx_b == [2, 3, 4, 5, 6]


def test_tabs_zero_for_rigid_body():
    ref_t = [_pos(0, 0, 0), _pos(1, 0, 0), _pos(0, 1, 0), _pos(0, 0, 1)]
    ref_b = [_pos(2, 2, 2), _pos(3, 2, 2), _pos(2, 3, 2)]
    T = _rigid_transform()
    val = rmsd.target_aligned_binder_rmsd(ref_t, _apply(T, ref_t), ref_b, _apply(T, ref_b))
    assert val == pytest.approx(0.0, abs=1e-6)


def test_tabs_equals_binder_shift():
    ref_t = [_pos(0, 0, 0), _pos(1, 0, 0), _pos(0, 1, 0), _pos(0, 0, 1)]
    ref_b = [_pos(2, 2, 2), _pos(3, 2, 2), _pos(2, 3, 2)]
    shifted = [_pos(p.x + 1.0, p.y + 2.0, p.z + 2.0) for p in ref_b]  # ||d|| = 3
    val = rmsd.target_aligned_binder_rmsd(ref_t, list(ref_t), ref_b, shifted)
    assert val == pytest.approx(3.0, abs=1e-6)


def test_term_tabs_hinge():
    scorer_hi, out_hi = _tabs_scorer_and_outputs(binder_shift=3.5, tol=2.5)
    expected_hi = float(F.elu(torch.tensor(3.5 - 2.5)))
    assert expected_hi == pytest.approx(1.0, abs=1e-5)  # linear above tolerance
    assert scorer_hi.term_target_aligned_binder_rmsd(out_hi) == pytest.approx(expected_hi, abs=1e-5)
    assert "target_aligned_binder_rmsd" in scorer_hi.score_terms(out_hi)

    scorer_lo, out_lo = _tabs_scorer_and_outputs(binder_shift=0.5, tol=2.5)
    val_lo = scorer_lo.term_target_aligned_binder_rmsd(out_lo)
    assert -1.0 < val_lo <= 0.0  # bounded below tolerance, never a large penalty
    assert val_lo < scorer_hi.term_target_aligned_binder_rmsd(out_hi)


def test_tabs_fitness_shifts_when_weighted():
    scorer = Scorer(config=DesignConfig(weights=ScoreWeights.iptm_ipae_tabs()))
    # Negative weight: a larger RMSD penalty lowers fitness.
    assert (scorer.fitness({"target_aligned_binder_rmsd": 0.0})
            > scorer.fitness({"target_aligned_binder_rmsd": 1.0}))


def test_tabs_inert_when_weight_zero():
    scorer = Scorer(config=DesignConfig())  # default weights -> tabs weight 0, tabs=None
    outputs = _make_general_outputs()
    assert scorer.tabs is None
    assert scorer.term_target_aligned_binder_rmsd(outputs) == 0.0
    assert scorer.score_terms(outputs)["target_aligned_binder_rmsd"] == 0.0
    assert scorer.fitness({"target_aligned_binder_rmsd": 5.0}) == pytest.approx(0.0)


def test_build_tabs_ref_aligns(tmp_path):
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, {
        "T": [("ALA", (i, 0, 0)) for i in range(4)],
        "L": [("GLY", (i, 5, 0)) for i in range(3)],
    })
    tabs = build_tabs_ref(_reference_spec(pdb), weight=-0.1)
    assert tabs is not None
    assert tabs.target_pred_idx == [0, 1, 2, 3]
    assert tabs.binder_pred_idx == [0, 1, 2]
    assert len(tabs.target_ca) == 4 and len(tabs.binder_ca) == 3


def test_build_tabs_ref_none_when_unweighted(tmp_path):
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, {"T": [("ALA", (0, 0, 0))], "L": [("GLY", (0, 5, 0))]})
    assert build_tabs_ref(_reference_spec(pdb, target_seq="A", binder_seq="G"), weight=0.0) is None


def test_tabs_requires_reference(tmp_path):
    spec = TargetSpec(
        name="d", binder=BinderSpec(id="A", scaffold="GGG"),
        chains=[FixedChain(id="C", sequence="AAAA", role="target")],
    )
    with pytest.raises(ValueError, match="reference"):
        build_tabs_ref(spec, weight=-0.1)  # no reference block

    spec.reference = ReferenceSpec(path="x.pdb", target_chains=[], binder_chains=[])
    with pytest.raises(ValueError):
        build_tabs_ref(spec, weight=-0.1)  # incomplete

    spec.reference = ReferenceSpec(
        path=str(tmp_path / "nope.pdb"), target_chains=["T"], binder_chains=["L"]
    )
    with pytest.raises(ValueError, match="not found"):
        build_tabs_ref(spec, weight=-0.1)  # file missing


def test_tabs_absent_chain_raises(tmp_path):
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, {"T": [("ALA", (0, 0, 0))], "L": [("GLY", (0, 5, 0))]})
    spec = _reference_spec(pdb, target_seq="A", binder_seq="G")
    spec.reference.binder_chains = ["Z"]  # not in the file
    with pytest.raises(ValueError, match="missing"):
        build_tabs_ref(spec, weight=-0.1)


def test_iptm_ipae_tabs_objective_preset():
    w = ScoreWeights.iptm_ipae_tabs()
    base = ScoreWeights.iptm_ipae()
    assert w.iptm_global == pytest.approx(1.0)
    assert w.mean_ipae == pytest.approx(-1.0 / 31.0)
    assert w.target_contact == pytest.approx(-0.5)
    assert w.target_aligned_binder_rmsd == pytest.approx(-0.1)  # the added restraint
    for f in fields(ScoreWeights):
        if f.name != "target_aligned_binder_rmsd":
            assert getattr(w, f.name) == getattr(base, f.name)
    assert ScoreWeights.iptm_ipae_tabs(
        target_aligned_binder_rmsd=0.0
    ).target_aligned_binder_rmsd == 0.0


# --------------------------------------------------------------------------- #
# Optional GPU smoke test                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA + model checkpoint")
def test_oracle_predict_smoke():
    from proga.oracle import ProtenixOracle

    cfg = DesignConfig(population=1, generations=0, n_cycle=2, n_step=20)
    mask = parse_design_mask(NANOBODY_SCAFFOLD)
    oracle = ProtenixOracle(cfg)
    oracle.design_positions = mask.positions
    genome = "A" * len(mask.positions)
    out = oracle.predict(apply_sequence(mask, genome), "VVVGADGVGK", "GSHSMRYFF")
    assert out.pae.ndim == 2
    assert out.distogram_logits.shape[-1] == DISTO_PARAMS["no_bins"]
    assert torch.isfinite(out.per_token_plddt).all()
