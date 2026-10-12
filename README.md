# proga

Gradient-free (genetic-algorithm) binder design. The GA runs on the host CPU; a structure
predictor behind a one-method `Oracle` provides the fitness. Selection is scalar or Pareto
(NSGA-II). The same GA runs on NVIDIA GPUs or on Tenstorrent cards; the hardware is just which
oracle you pick.

## What it does

Give it a target (a protein, a pMHC, or a small-molecule ligand) and a binder length or scaffold.
Each generation mutates and recombines the binder sequences, folds or scores every candidate
through the chosen oracle, and keeps the best by weighted-sum fitness or by NSGA-II Pareto fronts
(so on-target vs off-target trade-offs are preserved, not collapsed to one ratio). A spec
(`specs/*.yaml`) defines the binder, the target chains, and any off-targets.

## Oracles

Pick one with `--oracle-backend`. Add your own with `register_oracle("name", "module:Class")`
against the `Oracle` protocol in `proga/oracle_base.py`; the scorer drops any term an oracle cannot
supply (no distogram, no contact terms) rather than faking it.

NVIDIA GPU:
* `protenix` (default): Protenix-v2 fold and metrics, in process.
* `boltz2`: Boltz-2 fold via the `boltz predict` CLI.
* `opendde`: OpenDDE fold, in process.

Tenstorrent:
* `ttbio`: fold on a card via `tt-bio predict` (Boltz-2 / Protenix-v2 / OpenDDE).
* `nesso1`: protein and ligand affinity via `tt-bio affinity` (a scalar, no fold).

## Install

    pip install proga                     # core: the GA plus the subprocess and Tenstorrent oracles

Then add the stack for the oracle you want:

    pip install "proga[protenix]"         # protenix: pulls Protenix-v2; needs a CUDA GPU
    pip install boltz                     # boltz2: puts `boltz` on PATH; needs a CUDA GPU
    # opendde: install OpenDDE and a checkpoint (set OPENDDE_ROOT_DIR); needs a CUDA GPU
    pip install "tt-bio[tenstorrent]" && tt-bio weights --download boltz2
                                          # ttbio / nesso1: launcher on PATH or $TTBIO; needs a card

Core deps are CPU-only; each oracle brings its own GPU or accelerator stack.

## Run

Protenix-v2 (NVIDIA, default):

    proga --spec specs/example_kras_g12d.yaml --output_dir ./out \
          --oracle-backend protenix --objective iptm_ipae

Boltz-2 (NVIDIA):

    proga --spec specs/example_kras_g12d.yaml --output_dir ./out \
          --oracle-backend boltz2 --objective iptm_ipae

OpenDDE (NVIDIA):

    OPENDDE_CHECKPOINT=/path/to/ckpt proga --spec specs/example_kras_g12d.yaml --output_dir ./out \
          --oracle-backend opendde --objective iptm_ipae

tt-bio fold (Tenstorrent):

    proga --spec specs/example_ttbio_selectivity.yaml --output_dir ./out \
          --oracle-backend ttbio --ttbio-model boltz2 --selection pareto

Nesso-1 protein to ligand (Tenstorrent):

    proga --spec specs/example_nesso1_ligand.yaml --output_dir ./out \
          --oracle-backend nesso1 --objective affinity \
          --nesso1-ligand-smiles "<SMILES>" --nesso1-ligand-id LIG

`--objective` picks the score weights (`iptm_ipae`, `affinity`, and others); `--selection
scalar|pareto` runs NSGA-II over `--pareto_objectives`. See
[docs/tenstorrent.md](docs/tenstorrent.md) for the Tenstorrent flags, MSA caching, and ligand-id
notes.

## References

* ProteinMPNN: Dauparas et al., Science 2022 (inverse-folding term).
* NSGA-II: Deb, Pratap, Agarwal, Meyarivan, IEEE Trans. Evol. Comput. 2002 (Pareto selection).
* Protenix: ByteDance, https://github.com/bytedance/Protenix
* Boltz: https://github.com/jwohlwend/boltz
* tt-bio: https://github.com/moritztng/tt-bio

## License

Apache-2.0 (see `LICENSE`, `NOTICE`). Protenix is (c) ByteDance (Apache-2.0); ProteinMPNN weights
are MIT.
