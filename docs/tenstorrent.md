# Running proga on Tenstorrent (tt-bio oracle)

proga's genetic-algorithm core is fold-engine agnostic: any structure predictor plugs in behind
the `Oracle` protocol. The `ttbio` backend runs each candidate's fold on a **Tenstorrent** card
through the [`tt-bio`](https://github.com/moritztng/tt-bio) `predict` CLI, so the whole
GA + NSGA-II loop evolves sequences against folds computed on Tenstorrent hardware.

This fills a gap tt-bio has no route to on its own: tt-bio ships gradient design (BindCraft 2) and
generative design (BoltzGen, RFd3), but no derivative-free, multi-objective optimiser over its
non-differentiable scorers (ipTM, interface PAE, affinity). A GA is the only way to optimise those,
and a population search is throughput-bound rather than latency-bound — which suits a card that is
cheap per second and slow per fold.

## Install

proga core (the oracle shells out to `tt-bio`, so proga's own env needs no GPU deps):

    pip install proga

tt-bio on the Tenstorrent host, with its launcher on `PATH` or at `$TTBIO`:

    pip install 'tt-bio[tenstorrent]'
    tt-bio weights --download boltz2      # or the model you select below

## Run

    proga --spec specs/example_ttbio_selectivity.yaml --output_dir ./out \
          --oracle-backend ttbio --ttbio-model boltz2 \
          --selection pareto --pareto_objectives iptm_global mean_ipae \
          --use_msa --ttbio-extra-args --msa_dir ~/.cache/ttbio_msa --msa_cache_only

`specs/example_kras_g12d.yaml` is a complete, runnable on/off (pMHC G12D-vs-WT) spec; run it with
the same flags.

### Flags the `ttbio` backend adds

| flag | meaning |
|---|---|
| `--oracle-backend ttbio` | select this backend |
| `--ttbio-model` | `boltz2` (default), `protenix-v2`, `opendde-abag`, `rf3` |
| `--ttbio-launcher` | path to the `tt-bio` launcher; empty uses `$TTBIO`, else `/home/bizon/ttbio` |
| `--ttbio-extra-args` | pass-through to `tt-bio predict` (e.g. `--msa_dir`, `--msa_cache_only`, `--single_sequence`) |

`--n_cycle`, `--n_step`, `--n_sample`, `--seed` map to tt-bio's
`--recycling_steps/--sampling_steps/--diffusion_samples/--seed`.

## MSA caching (load-bearing)

tt-bio models are MSA-dependent by default, so recomputing an MSA for every candidate is the main
avoidable cost. The target chains are fixed across the whole run, so compute their MSA **once** and
reuse it; the binder changes every generation and stays single-sequence:

- `--use_msa` lists the target chains (`msa_chain_ids` in the spec) for MSA, binder excluded.
- Point `--ttbio-extra-args --msa_dir <dir> --msa_cache_only` at a cache so the target MSA is built
  once and every later fold reads it from disk. Add `--single_sequence` to force the binder MSA-free.

Without this, each of `population x (generations + 1) x (1 on-target + k off-targets)` folds pays an
MSA search.

## Cost

Each fold on a Tenstorrent card is on the order of hundreds of seconds, and a GA is
`population x (generations + 1) x (1 + n_off_targets)` folds. Size the run accordingly: this is a
throughput-bound campaign, not an interactive one. Scalars only (ipTM, chain-pair ipTM, interface
PAE, pLDDT) drive selection; tt-bio `predict` does not surface distogram logits, so the
contact/distogram scoring terms are skipped automatically (the same as the other subprocess folders).

## Status

The oracle is wired to the documented `tt-bio predict` CLI and output layout
(`structures/<name>.cif` with pLDDT in B-factors, `structures/<name>_pae.npz['pae']`, and a
`results.json` carrying pTM / ipTM / chain-pair ipTM). The confidence reader falls back across the
likely chain-pair ipTM key names; confirm the exact key on your first fold.
