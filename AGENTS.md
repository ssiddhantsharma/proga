# AGENTS.md

Context for coding agents (Claude Code, Codex) working in this repo. Humans should start
with [README.md](README.md); this file points an agent at the same material and the
conventions worth knowing before changing or running anything.

## What this repo is

Gradient-free (genetic-algorithm) binder design. The GA runs on the host CPU; a structure
predictor behind a one-method `Oracle` supplies the fitness. Each generation mutates and
recombines binder sequences, scores every candidate through the chosen oracle, and keeps the
best by weighted-sum fitness or by NSGA-II Pareto fronts. A spec (`specs/*.yaml`) defines the
binder, the target chains, and any off-targets.

## Where to read (in order)

- [README.md](README.md) - oracles, install, run commands.
- [docs/extending.md](docs/extending.md) - authoring a new oracle against the `Oracle` protocol.
- [docs/tenstorrent.md](docs/tenstorrent.md) - the accelerator-card oracles.
- `specs/*.yaml` - worked examples (a pMHC target, a small-molecule ligand, a selectivity run).

## Run it

```bash
pip install proga                 # core: CPU-only GA + subprocess/accelerator oracles
pip install "proga[protenix]"     # add the stack for the oracle you want (needs a CUDA GPU)

proga --spec specs/example_kras_g12d.yaml --output_dir ./out \
      --oracle-backend protenix --objective iptm_ipae
```

- `--oracle-backend`: `protenix` (default), `boltz2`, `opendde` on an NVIDIA GPU; `ttbio`,
  `nesso1` on a Tenstorrent card. The core deps are CPU-only; each oracle brings its own stack.
- `--selection scalar` (weighted-sum) or `pareto` (NSGA-II, so on-target vs off-target trade-offs
  are preserved rather than collapsed to one ratio).
- `--objective` picks the fitness terms (e.g. `iptm_ipae`).

Requires Python 3.12; `uv.lock` pins the environment.

## Conventions that must not be broken

- **The scorer drops any term an oracle cannot supply** (no distogram, no contact terms) rather
  than faking it. Keep that contract when editing scoring or adding an oracle.
- Add an oracle with `register_oracle("name", "module:Class")` against the `Oracle` protocol in
  `proga/oracle_base.py`. Do not hard-wire a backend into the GA.
- Off-target selectivity rides on the Pareto path; do not collapse multi-objective fitness into a
  single ratio in the selection code.

## Tests

```bash
pytest tests/            # GA, scoring, and oracle-adapter tests (CPU)
```
