# proga on Tenstorrent (tt-bio oracle)

`oracle_backend=ttbio` folds each candidate on a Tenstorrent card via the
[`tt-bio`](https://github.com/moritztng/tt-bio) `predict` CLI. The GA + NSGA-II loop runs on the
host; only the fold runs on the card. Use it to optimise sequences against a non-differentiable
interface score, which gradient and generative design cannot.

## Install

    pip install proga                      # core (the oracle shells out to tt-bio)
    pip install 'tt-bio[tenstorrent]'      # on the Tenstorrent host
    tt-bio weights --download boltz2

Put the `tt-bio` launcher on `PATH` or at `$TTBIO`.

## Run

    proga --spec specs/example_ttbio_selectivity.yaml --output_dir ./out \
          --oracle-backend ttbio --ttbio-model boltz2 \
          --selection pareto --pareto_objectives iptm_global mean_ipae \
          --use_msa --ttbio-extra-args --msa_dir ~/.cache/ttbio_msa --msa_cache_only

`specs/example_kras_g12d.yaml` is a complete on/off (pMHC) spec; run it with the same flags.

## Flags

| flag | meaning |
|---|---|
| `--oracle-backend ttbio` | select this backend |
| `--ttbio-model` | `boltz2` (default), `protenix-v2`, `opendde-abag`, `rf3` |
| `--ttbio-launcher` | launcher path; empty uses `$TTBIO`, else `tt-bio` on PATH |
| `--ttbio-extra-args` | pass-through to `tt-bio predict` (`--msa_dir`, `--msa_cache_only`, `--single_sequence`) |

`--n_cycle/--n_step/--n_sample/--seed` map to tt-bio's
`--recycling_steps/--sampling_steps/--diffusion_samples/--seed`.

## MSA

tt-bio folds with an MSA by default. The target is fixed across the run, so cache its MSA once
(`--use_msa` lists the target chains; `--msa_dir`/`--msa_cache_only`); the binder stays
single-sequence (`--single_sequence`). Otherwise every candidate pays an MSA search.

## Notes

- Confidence is read from tt-bio's stable keys (README, "Confidence Scores"): `iptm`,
  `pair_chains_iptm`, `chains_ptm`, `status`. The Boltz-2-only `interface_score_distribution` is not
  used, so the oracle works across Boltz-2 / Protenix-v2 / OpenDDE.
- No distogram logits from tt-bio, so contact/distogram scoring terms are skipped.
- Scores are comparable within one model, not across models; do not mix oracles in one run.
- Each fold is hundreds of seconds and a GA is `population x (generations + 1) x (1 + n_off_targets)`
  folds: a throughput run, not interactive.

## Protein -> ligand design (Nesso-1 affinity)

`oracle_backend=nesso1` scores the GA-designed protein against a fixed small-molecule ligand with
`tt-bio affinity --model nesso1` — a scalar, no fold (Nesso-1 is protein-ligand only). It is the one
case where the oracle returns no structure: the scorer runs on `affinity` alone (its has-structure
guard zeroes the structure terms). Pair it with `--objective affinity`.

    proga --spec specs/example_nesso1_ligand.yaml --output_dir ./out \
          --oracle-backend nesso1 --objective affinity \
          --nesso1-ligand-smiles "N[C@@H](Cc1ccc(O)cc1)C(=O)O" --nesso1-ligand-id LIG \
          --population 16 --generations 10

| flag | meaning |
|---|---|
| `--nesso1-ligand-smiles` / `--nesso1-ligand-ccd` | the target ligand (one of the two) |
| `--nesso1-ligand-id` | ligand chain id in the affinity YAML (default `LIG`) | Use a non-CCD id; a 3-letter CCD code (e.g. `TYR`) makes tt-bio expect CCD atom names and fail on a SMILES ligand. |
| `--nesso1-affinity-key` | `affinity_probability_binary` (default, P(binder), higher better) or `affinity_pred_value` (lower = stronger) |
| `--nesso1-launcher` / `--nesso1-extra-args` | launcher path / pass-through to `tt-bio affinity` |

Each affinity call is ~20-30 s on a p150a (no fold), cheaper than a Boltz-2 affinity fold.
