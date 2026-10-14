# Writing a new oracle

proga's genetic algorithm is engine agnostic. It evolves sequences and asks an **oracle** to fold
each candidate and return interface metrics; selection, NSGA-II, mutation and the off-target
machinery never change. Adding a fold engine means writing one class and registering it.

## The contract

An oracle implements a single method:

```python
def predict_complex(
    self,
    chains: list[tuple[str, str]],   # [(chain_id, sequence), ...], binder first
    roles: dict[str, str],           # chain_id -> "binder" | "target" | "context"
    design_positions: list[int] | None = None,
    name: str = "design",
    chain_names: dict[str, str] | None = None,
    dump_dir: str | None = None,
) -> PredictionOutputs: ...
```

It returns a `PredictionOutputs` (see `proga/oracle_base.py`):

| field | type | meaning |
|---|---|---|
| `per_token_plddt` | `[N]` | per-token pLDDT (0 to 100) |
| `pae` | `[N, N]` | predicted aligned error, token by token |
| `distogram_logits` | `[N, N, bins]` or `None` | distogram logits; `None` disables contact terms |
| `ca_coords` | `[N, 3]` | Ca coordinates (empty for a fold-free oracle) |
| `chain_pair_iptm` | `[C, C]` | per chain-pair ipTM |
| `chain_index` | `ChainIndex` | token-to-chain map (below) |
| `distogram_bin_params` | `dict` | bin edges for the distogram, `{}` if none |
| `iptm_global` | `float` | complex ipTM |
| `backbone_coords` | `[N, 4, 3]` or `None` | N, CA, C, O per token, for the ProteinMPNN term |
| `sequence` | `str` | per-token one-letter sequence |
| `affinity` | `float` or `None` | scalar affinity for a fold-free oracle |

`ChainIndex` carries `chain_ids`, `chain_tokens` (chain_id to token indices), `chain_roles`,
`chain_asym`, `binder_id`, `residue_index`, `asym_id` and `design_tokens`. The scorer reads the
binder and target through it, so fill it from the folded structure's real chain and residue layout.

## Registering it

Add one line so `--oracle-backend <name>` resolves to the class:

```python
from proga.oracle_base import register_oracle

register_oracle("myengine", "proga.myengine_oracle:MyEngineOracle")
```

`make_oracle(name, config, work_dir)` imports and constructs it lazily, so a heavy dependency is
only imported when that backend is selected. Keep engine imports inside `__init__` or the fold
method, never at module top, so the registry still lists the backend on a box without it installed.

## Two things that commonly differ between engines

- **No distogram.** A subprocess engine often cannot surface distogram logits. Set
  `distogram_logits=None`; the scorer skips contact or distogram terms and warns if one is
  weighted, rather than scoring them as zero silently.
- **Fold-free oracles.** An engine that returns only a scalar (no coordinates) leaves `ca_coords`
  empty and sets `affinity`. `score_terms` detects the empty structure and scores on `affinity`
  alone instead of indexing empty tensors.

## Worked examples in this repo

- `proga/boltz_oracle.py` shells out to `boltz predict` and parses the CIF, PAE and confidence.
- `proga/ttbio_oracle.py` drives `tt-bio predict` on a Tenstorrent card and reads tt-bio's stable
  `results.json` keys plus the `*_pae.json` sidecar.
- `proga/opendde_oracle.py` runs OpenDDE in process and reuses the Boltz parsers.
- `proga/nesso1_oracle.py` is a fold-free affinity oracle.

Match one of these to your engine and swap the fold step plus the readers.
