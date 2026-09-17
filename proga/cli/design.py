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
"""CLI entrypoint for Protenix selectivity-aware binder design.

Two input modes:

* pMHC CIFs (legacy / byte-identical to the original pipeline)::

    python runner/design.py \
        --target_cif targets/hla_a_03_01_truncated_kras_g12d_vvvgadgvgk.cif \
        --off_target_cif off_targets/hla_a_03_01_truncated_kras_vvvgaggvgk.cif \
        --output_dir ./design_out --population 8 --generations 5

* a target-agnostic spec (any chains / binder / off-targets)::

    python runner/design.py --spec my_target.json --output_dir ./design_out

Set --population 100 --generations 80 to reproduce the paper's scale.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from proga.complex import (
    PMHCTarget,
    apply_sequence,
    discriminating_indices,
    load_pmhc_from_cif,
    parse_design_mask,
)
from proga.config import NANOBODY_SCAFFOLD, DesignConfig, ScoreWeights
from proga.ddg import default_ddg_predictor
from proga.genetic import GeneticOptimizer
from proga.scoring import Scorer, build_tabs_ref
from proga.spec import TargetSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("protenix-evolve")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protenix selectivity-aware binder design")
    p.add_argument("--target_cif", help="on-target pMHC mmCIF (legacy pMHC mode)")
    p.add_argument("--off_target_cif", help="off-target pMHC mmCIF (legacy pMHC mode)")
    p.add_argument(
        "--spec",
        help="target-agnostic design spec (.json/.yaml); alternative to the CIF flags",
    )
    p.add_argument("--output_dir", default="./design_out")

    # GA hyperparameters.
    p.add_argument("--population", type=int, default=DesignConfig.population)
    p.add_argument("--generations", type=int, default=DesignConfig.generations)
    p.add_argument("--mutation_rate", type=float, default=DesignConfig.mutation_rate)
    p.add_argument("--crossover_rate", type=float, default=DesignConfig.crossover_rate)
    p.add_argument("--top_n", type=int, default=DesignConfig.top_n)
    p.add_argument("--t_ddg", type=int, default=DesignConfig.t_ddg)
    p.add_argument("--seed", type=int, default=DesignConfig.seed)
    p.add_argument(
        "--objective",
        choices=[
            "composite",
            "iptm_ipae",
            "mpnn",
            "iptm_ipae_mpnn",
            "iptm_ipae_repulsion",
            "iptm_ipae_paratope",
            "iptm_ipae_tabs",
            "iptm_ipae_target_peptide",
            "iptm_ipae_target_peptide_mpnn"
        ],
        default="composite",
        help="'composite' = full Table-1 fitness (paper); 'iptm_ipae' = select "
        "purely on high Protenix ipTM + low mean interface PAE; "
        "'iptm_ipae_repulsion' = iptm_ipae plus a repulsion penalty on the "
        "avoidance set M (selectivity pressure); "
        "'iptm_ipae_paratope' = iptm_ipae plus the paratope-formation term "
        "(CDR-vs-framework contact differential); "
        "'iptm_ipae_target_peptide_mpnn' = iptm_ipae_target_peptide plus a "
        "ProteinMPNN peptide-likelihood reward; "
        "'mpnn' = select purely on the ProteinMPNN peptide-likelihood term.",
    )

    p.add_argument(
        "--selection",
        choices=["scalar", "pareto"],
        default=DesignConfig.selection,
        help="'scalar' (default) = rank survivors by the weighted-sum fitness; "
        "'pareto' = NSGA-II non-dominated sorting + crowding over "
        "--pareto_objectives, preserving trade-offs (e.g. on-target ipTM vs "
        "off-target repulsion) instead of collapsing them to one weight ratio.",
    )
    p.add_argument(
        "--pareto_objectives",
        nargs="*",
        default=None,
        metavar="TERM",
        help="term names used as Pareto axes when --selection pareto (e.g. "
        "iptm_global mean_ipae repulsion). Omit to use every term with a "
        "non-zero weight; direction per axis follows the sign of that weight.",
    )

    # Oracle knobs.
    p.add_argument("--model_name", default=DesignConfig.model_name)
    p.add_argument("--n_cycle", type=int, default=DesignConfig.n_cycle)
    p.add_argument("--n_step", type=int, default=DesignConfig.n_step)
    p.add_argument("--n_sample", type=int, default=DesignConfig.n_sample)
    p.add_argument("--dtype", default=DesignConfig.dtype)
    p.add_argument(
        "--use_msa",
        action="store_true",
        help="compute MSAs for the target chains once and reuse them (cached by "
        "sequence hash); the nanobody binder stays MSA-free.",
    )
    p.add_argument("--msa_mode", default=DesignConfig.msa_mode,
                   choices=["protenix", "colabfold"], help="remote MMseqs2 mode")
    p.add_argument("--msa_cache_dir", default=DesignConfig.msa_cache_dir)
    p.add_argument("--msa_for_peptide", action="store_true",
                   help="also search an MSA for the (usually too short) peptide")
    p.add_argument(
        "--disable_mhc_avoidance",
        action="store_true",
        help="set the MHC avoidance set M to empty (disables the repulsion term)",
    )
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> DesignConfig:
    config = DesignConfig(
        population=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        top_n=args.top_n,
        t_ddg=args.t_ddg,
        seed=args.seed,
        model_name=args.model_name,
        n_cycle=args.n_cycle,
        n_step=args.n_step,
        n_sample=args.n_sample,
        dtype=args.dtype,
        use_msa=args.use_msa,
        msa_mode=args.msa_mode,
        msa_cache_dir=args.msa_cache_dir,
        msa_for_peptide=args.msa_for_peptide,
        selection=args.selection,
        pareto_objectives=list(args.pareto_objectives or []),
    )
    if args.objective == "iptm_ipae":
        config.weights = ScoreWeights.iptm_ipae()
    elif args.objective == "mpnn":
        config.weights = ScoreWeights.mpnn_only()
    elif args.objective == "iptm_ipae_mpnn":
        config.weights = ScoreWeights.iptm_ipae_mpnn()
    elif args.objective == "iptm_ipae_repulsion":
        config.weights = ScoreWeights.iptm_ipae_repulsion()
    elif args.objective == "iptm_ipae_paratope":
        config.weights = ScoreWeights.iptm_ipae_paratope()
    elif args.objective == "iptm_ipae_tabs":
        config.weights = ScoreWeights.iptm_ipae_tabs()
    elif args.objective == "iptm_ipae_target_peptide":
        config.weights = ScoreWeights.iptm_ipae_target_peptide()
    elif args.objective == "iptm_ipae_target_peptide_mpnn":
        config.weights = ScoreWeights.iptm_ipae_target_peptide_mpnn()
    return config


def main() -> None:
    args = parse_args()
    if not args.spec and not (args.target_cif and args.off_target_cif):
        raise SystemExit(
            "Provide either --spec or both --target_cif and --off_target_cif."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    config = _build_config(args)
    if args.spec:
        run_with_spec(args, config)
    else:
        run_with_cifs(args, config)


def run_with_cifs(args: argparse.Namespace, config: DesignConfig) -> None:
    """Legacy pMHC flow; byte-identical to the original pipeline."""
    if config.weights.target_aligned_binder_rmsd != 0:
        raise SystemExit(
            "The 'target_aligned_binder_rmsd' objective needs a reference structure; "
            "use --spec with a `reference` block instead of the legacy CIF mode."
        )
    if args.disable_mhc_avoidance:
        config.mhc_avoidance_residues = []

    # --- targets --------------------------------------------------------- #
    target: PMHCTarget = load_pmhc_from_cif(args.target_cif, name="target")
    off_target: PMHCTarget = load_pmhc_from_cif(args.off_target_cif, name="off_target")
    disc = discriminating_indices(target.peptide_seq, off_target.peptide_seq)
    logger.info("Target peptide:      %s", target.peptide_seq)
    logger.info("Off-target peptide:  %s", off_target.peptide_seq)
    logger.info("Discriminating positions (0-based): %s", disc)
    logger.info("MHC length: %d", len(target.mhc_seq))

    mask = parse_design_mask(NANOBODY_SCAFFOLD)
    logger.info(
        "Design mask: %d positions across %d CDRs %s",
        len(mask.positions), len(mask.cdr_spans), mask.cdr_spans,
    )
    logger.info("Objective: %s", args.objective)
    logger.info("Estimated Protenix forward passes: ~%d", config.num_oracle_calls())

    # --- oracle + scorer + ddG hook ------------------------------------- #
    from proga.oracle import ProtenixOracle  # heavy import (torch/model)

    oracle = ProtenixOracle(config, work_dir=os.path.join(args.output_dir, "_oracle"))
    oracle.design_positions = mask.positions
    if config.use_msa:
        target_seqs = [target.mhc_seq]
        if config.msa_for_peptide:
            target_seqs.append(target.peptide_seq)
        logger.info("Precomputing target MSA (%d chain(s), cached by seq hash)...",
                    len(target_seqs))
        oracle.precompute_target_msa(target_seqs)
    scorer = Scorer(config=config, discriminating_positions=disc)
    ddg_predictor = default_ddg_predictor(None)  # disabled by default
    logger.info("ddG predictor enabled: %s", ddg_predictor.enabled)

    def evaluate(genome: str, use_ddg: bool) -> tuple[float, dict[str, float]]:
        nanobody_seq = apply_sequence(mask, genome)
        outputs = oracle.predict(nanobody_seq, target.peptide_seq, target.mhc_seq)
        ddg_raw = None
        if use_ddg and ddg_predictor.enabled:
            ddg_raw = ddg_predictor.predict(
                nanobody_seq, target.peptide_seq, off_target.peptide_seq, target.mhc_seq
            )
        terms = scorer.score_terms(outputs, ddg_pred=ddg_raw)
        return scorer.fitness(terms), terms

    # --- run ------------------------------------------------------------- #
    optimizer = GeneticOptimizer(config, mask)
    result = optimizer.run(evaluate)

    logger.info("Evaluations: %d", result.num_evaluations)
    logger.info("Best fitness per generation: %s",
                [round(x, 4) for x in result.best_per_generation])

    # --- off-target scoring of the final top-N designs ------------------- #
    # The GA only folds the on-target complex (off-target folding is gated
    # behind the disabled ddG hook). To report selectivity, fold each reported
    # design once more against the off-target pMHC and score the same terms.
    # This is len(result.top) extra forward passes, not one per GA candidate.
    logger.info("Folding %d top designs against the off-target pMHC...",
                len(result.top))
    off_target_terms: list[dict[str, float]] = []
    for ind in result.top:
        nanobody_seq = apply_sequence(mask, ind.genome)
        off_outputs = oracle.predict(
            nanobody_seq, off_target.peptide_seq, off_target.mhc_seq
        )
        off_target_terms.append(scorer.score_terms(off_outputs))

    # --- write outputs --------------------------------------------------- #
    out = {
        "target": target.name,
        "off_target": off_target.name,
        "discriminating_positions": disc,
        "best_per_generation": result.best_per_generation,
        "num_evaluations": result.num_evaluations,
        "designs": [
            {
                "rank": i + 1,
                "fitness": ind.fitness,
                "cdr_residues": ind.genome,
                "nanobody_sequence": apply_sequence(mask, ind.genome),
                "first_generation": ind.generation,
                "terms": ind.terms,
                "off_target_terms": off_target_terms[i],
            }
            for i, ind in enumerate(result.top)
        ],
    }
    out_path = os.path.join(args.output_dir, "designs.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %d designs to %s", len(result.top), out_path)


def run_with_spec(args: argparse.Namespace, config: DesignConfig) -> None:
    """Target-agnostic flow driven by a :class:`TargetSpec` (.json/.yaml)."""
    spec = TargetSpec.from_file(args.spec)
    if args.disable_mhc_avoidance:
        for c in spec.chains:
            c.avoidance_residues = []

    mask = spec.binder.mask()
    logger.info("Target: %s | binder %r (%d design positions)",
                spec.name, spec.binder.id, len(mask.positions))
    logger.info("Fixed chains: %s",
                [(c.id, c.role, len(c.sequence)) for c in spec.chains])
    logger.info("Contact residues (T): %s", spec.contact_residues_map())
    logger.info("Avoidance residues (M): %s", spec.avoidance_residues_map())
    logger.info("Off-targets: %s", [o.name for o in spec.off_targets])
    logger.info("Objective: %s", args.objective)
    logger.info("Estimated Protenix forward passes: ~%d", config.num_oracle_calls())

    from proga.oracle import ProtenixOracle  # heavy import (torch/model)

    oracle = ProtenixOracle(config, work_dir=os.path.join(args.output_dir, "_oracle"))
    oracle.design_positions = mask.positions
    if config.use_msa:
        msa_seqs = [c.sequence for c in spec.chains if c.id in spec.msa_chain_ids]
        logger.info("Precomputing target MSA (%d chain(s))...", len(msa_seqs))
        oracle.precompute_target_msa(msa_seqs)

    scorer = Scorer(
        config=config,
        discriminating_positions=spec.discriminating_positions,
        contact_residues=spec.contact_residues_map(),
        avoidance_residues=spec.avoidance_residues_map(),
        iptm_target_id=spec.iptm_target_id,
        tabs=build_tabs_ref(spec, config.weights.target_aligned_binder_rmsd),
    )
    roles = spec.roles()
    chain_names = spec.chain_names() or None

    def evaluate(genome: str, use_ddg: bool) -> tuple[float, dict[str, float]]:
        binder_seq = apply_sequence(mask, genome)
        outputs = oracle.predict_complex(
            spec.assembly_chains(binder_seq), roles, mask.positions, chain_names=chain_names
        )
        terms = scorer.score_terms(outputs)
        return scorer.fitness(terms), terms

    optimizer = GeneticOptimizer(config, mask)
    result = optimizer.run(evaluate)
    logger.info("Evaluations: %d", result.num_evaluations)
    logger.info("Best fitness per generation: %s",
                [round(x, 4) for x in result.best_per_generation])

    # --- off-target scoring of the top-N designs against each off-target --- #
    off_terms: dict[str, list[dict[str, float]]] = {}
    for off in spec.off_targets:
        logger.info("Folding %d top designs against off-target %r...",
                    len(result.top), off.name)
        off_roles = spec.offtarget_roles(off)
        off_names = {c.name: c.id for c in off.chains if c.name} or None
        terms_list = []
        for ind in result.top:
            binder_seq = apply_sequence(mask, ind.genome)
            out = oracle.predict_complex(
                spec.offtarget_assembly_chains(off, binder_seq),
                off_roles,
                mask.positions,
                chain_names=off_names,
            )
            terms_list.append(scorer.score_terms(out))
        off_terms[off.name] = terms_list

    first_off = spec.off_targets[0].name if spec.off_targets else None
    designs = []
    for i, ind in enumerate(result.top):
        record = {
            "rank": i + 1,
            "fitness": ind.fitness,
            "cdr_residues": ind.genome,
            "binder_sequence": apply_sequence(mask, ind.genome),
            "first_generation": ind.generation,
            "terms": ind.terms,
            "off_targets": {name: lst[i] for name, lst in off_terms.items()},
        }
        if first_off is not None:
            # Legacy-compatible single off-target field.
            record["off_target_terms"] = off_terms[first_off][i]
            record_iptm = ind.terms.get("iptm")
            off_iptm = off_terms[first_off][i].get("iptm")
            if record_iptm is not None and off_iptm is not None:
                record["selectivity"] = record_iptm - off_iptm
        designs.append(record)

    out = {
        "target": spec.name,
        "off_target": first_off,
        "discriminating_positions": spec.discriminating_positions,
        "contact_residues": spec.contact_residues_map(),
        "best_per_generation": result.best_per_generation,
        "num_evaluations": result.num_evaluations,
        "designs": designs,
    }
    out_path = os.path.join(args.output_dir, "designs.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %d designs to %s", len(result.top), out_path)


if __name__ == "__main__":
    main()
