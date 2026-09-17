# proga

Gradient-free **genetic-algorithm binder design** with a **Protenix-v2** structure+metrics oracle.

`proga` evolves a population of binder sequences against a target, scoring each candidate by folding the complex with Protenix-v2 and reading interface metrics (iptm, inter-chain PAE, RMSD, and a ProteinMPNN term). It supports **multi-objective (Pareto / NSGA-II)** selection for competing goals — e.g. maximize on-target binding while minimizing off-target cross-reactivity — as well as the classic single scalar-fitness path.

It is *gradient-free*: the structure predictor is used only as a forward-pass oracle, so no backprop through the folder is required.

## Install

Needs a CUDA GPU (Protenix folds on-device).

```bash
uv sync          # or: pip install -e .
```

The Protenix engine is pulled as a dependency; ProteinMPNN weights ship in `proga/weights/`.

## Usage

```bash
proga --spec specs/example_kras_g12d.yaml --output_dir ./design_out \
      --population 128 --generations 60 --selection pareto
```

- `--selection scalar` (default) collapses all score terms into one weighted sum.
- `--selection pareto` keeps the non-dominated front across the objectives in `--pareto_objectives` (default: every non-zero-weight term). Axis direction is the sign of each term's weight, so maximized and minimized terms compose in one domination test.

Run `proga --help` for the full flag list (population, generations, mutation rate, score weights, …).

## Design spec

A target-agnostic YAML/JSON (see `specs/example_kras_g12d.yaml`):

```yaml
name: my_target
binder:
  id: A
  scaffold: EVQLVESGGG...XXXXXXXXX...WGQGTQVTVSS   # X = designable position
  mask_char: X
  alphabet: ADEFGHIKLMNPQRSTVWY
chains:
  - id: B
    sequence: <target sequence>
    role: target
    contact_residues: [6]        # 1-indexed; guide the binder toward these
    avoidance_residues: []
  - id: C
    sequence: <context/off-target sequence>
    role: context                # steric context or off-target to avoid
    avoidance_residues: [182, 183, ...]
```

Any chain layout works (nanobody/antibody, peptide, multi-chain context); the example is a public KRAS-G12D neoantigen for illustration.

## Roadmap — Tenstorrent

The oracle is the one pluggable seam (the GA core is pure-Python, no torch). A planned second backend runs the fold on **Tenstorrent** via **tt-bio** (forward-only) instead of in-process Protenix, so this gradient-free GA can run on TT hardware. Gradient-based hallucination still requires a backward pass and stays on GPU.

## Attribution & license

Apache-2.0 (see `LICENSE`, `NOTICE`). Derived from the internal `protenix-ga` project; the Protenix engine is © ByteDance (Apache-2.0); bundled ProteinMPNN weights are MIT.
