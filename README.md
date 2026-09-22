# proga

Gradient-free genetic-algorithm binder design. Any structure predictor can be the fold + metrics
oracle behind a small interface; survivor selection is scalar or Pareto (NSGA-II).

## Oracles

Any structure predictor plugs in behind the `Oracle` protocol in `proga/oracle_base.py`;
register it with `register_oracle("name", "module:Class")`. The scorer degrades gracefully
when a backend omits an output (e.g. no distogram → contact terms are skipped, not faked).

Oracles: **Protenix-v2** (`protenix`, in-process, default), **Boltz-2**, **OpenDDE**.

## Install

    pip install proga[protenix]   # GPU (Protenix-v2) oracle
    pip install proga             # core only

## Run

    proga --spec specs/example_kras_g12d.yaml --output_dir ./out

## License

Apache-2.0 (see `LICENSE`, `NOTICE`). Protenix © ByteDance (Apache-2.0); ProteinMPNN weights MIT.
