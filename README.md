# proga

Gradient-free genetic-algorithm binder design with a Protenix-v2 oracle (scalar or Pareto/NSGA-II selection).

## Install

```bash
uv sync   # needs a CUDA GPU
```

## Run

```bash
proga --spec specs/example_kras_g12d.yaml --output_dir ./design_out
```

`proga --help` for options.

## License

Apache-2.0 (see `LICENSE`, `NOTICE`). Protenix © ByteDance (Apache-2.0); bundled ProteinMPNN weights are MIT.
