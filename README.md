# proga

Gradient-free (genetic-algorithm) binder design. The GA runs on the host; a structure predictor
behind a one-method `Oracle` is the fitness. Selection is scalar or Pareto (NSGA-II). Runs on
NVIDIA GPUs or Tenstorrent cards — the hardware is just which oracle you pick.

## Oracles

Choose with `--oracle-backend`. Add your own with `register_oracle("name", "module:Class")` against
the `Oracle` protocol in `proga/oracle_base.py`; the scorer drops any term an oracle can't supply
(no distogram → contact terms skipped), never faked.

- **NVIDIA GPU** — `protenix` (Protenix-v2, in-process, default), `boltz2` (`boltz predict`), `opendde`
- **Tenstorrent** — `ttbio` (fold via `tt-bio predict`: Boltz-2 / Protenix-v2 / OpenDDE on-card),
  `nesso1` (protein–ligand affinity via `tt-bio affinity`, no fold)

## Install

    pip install proga                 # core: GA + the subprocess / Tenstorrent oracles
    pip install "proga[protenix]"     # + in-process Protenix-v2 (NVIDIA)

Core deps are CPU-only; each oracle pulls its own GPU/accelerator stack.

## Run — NVIDIA (A6000 etc.)

Needs a CUDA GPU plus the oracle's package: `[protenix]` for Protenix-v2; `boltz` on PATH for
`boltz2`; OpenDDE installed for `opendde`.

    proga --spec specs/example_kras_g12d.yaml --output_dir ./out \
          --oracle-backend protenix --objective iptm_ipae

## Run — Tenstorrent

Install tt-bio and put its launcher on PATH or at `$TTBIO`:

    pip install "tt-bio[tenstorrent]" && tt-bio weights --download boltz2

Fold on a card:

    proga --spec specs/example_ttbio_selectivity.yaml --output_dir ./out \
          --oracle-backend ttbio --ttbio-model boltz2 --selection pareto

Protein → ligand (Nesso-1 affinity, no fold):

    proga --spec specs/example_nesso1_ligand.yaml --output_dir ./out \
          --oracle-backend nesso1 --objective affinity \
          --nesso1-ligand-smiles "<SMILES>" --nesso1-ligand-id LIG

See [docs/tenstorrent.md](docs/tenstorrent.md) for the Tenstorrent flags, MSA caching and ligand-id notes.

## Specs & objectives

A spec (`specs/*.yaml`) defines the binder, target(s) and off-targets. `--objective` sets the score
weights (`iptm_ipae`, `affinity`, …); `--selection scalar|pareto` runs NSGA-II over
`--pareto_objectives`.

## License

Apache-2.0 (`LICENSE`, `NOTICE`). Protenix © ByteDance (Apache-2.0); ProteinMPNN weights MIT.
