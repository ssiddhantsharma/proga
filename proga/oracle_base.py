"""proga oracle contract — the interface every fold engine implements, and the light
data types the scorer consumes. Torch-only, NO protenix/runner import, so a lightweight
backend can implement it without pulling in the GPU protenix stack.
"""
import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

ROLE_BINDER = "binder"
ROLE_TARGET = "target"
ROLE_CONTEXT = "context"


def genome_struct_name(genome: str) -> str:
    """Stable, filesystem-safe name for a genome's persisted structure.

    Used as both the per-genome structure-cache key and the Protenix dumper
    ``name`` so the dump location, the prune keep-set, and the final rename all
    agree. The genome fully determines the on-target assembly within a run
    (targets are fixed), so it is a valid cache key.
    """
    return hashlib.sha1(genome.encode()).hexdigest()[:16]


class ChainIndex:
    def __init__(
        self,
        *,
        residue_index: torch.Tensor,
        asym_id: torch.Tensor,
        design_tokens: torch.Tensor,
        # --- general form ---
        chain_ids: list[str] | None = None,
        chain_tokens: dict[str, torch.Tensor] | None = None,
        chain_roles: dict[str, str] | None = None,
        chain_asym: dict[str, int] | None = None,
        binder_id: str | None = None,
        chain_names: dict[str, str] | None = None,
        # --- legacy pMHC form ---
        nanobody_tokens: torch.Tensor | None = None,
        peptide_tokens: torch.Tensor | None = None,
        mhc_tokens: torch.Tensor | None = None,
        nanobody_asym: int | None = None,
        peptide_asym: int | None = None,
        mhc_asym: int | None = None,
    ) -> None:
        self.residue_index = residue_index
        self.asym_id = asym_id
        self.design_tokens = design_tokens

        if nanobody_tokens is not None:
            binder_id = "A"
            chain_ids = ["A", "B", "C"]
            chain_tokens = {"A": nanobody_tokens, "B": peptide_tokens, "C": mhc_tokens}
            chain_roles = {"A": ROLE_BINDER, "B": ROLE_TARGET, "C": ROLE_CONTEXT}
            chain_asym = {"A": nanobody_asym, "B": peptide_asym, "C": mhc_asym}
            chain_names = {"peptide": "B", "mhc": "C"}

        if chain_ids is None or chain_tokens is None or binder_id is None:
            raise ValueError("ChainIndex needs either the general or legacy chain fields.")

        self.chain_ids = chain_ids
        self.chain_tokens = chain_tokens
        self.chain_roles = chain_roles or {}
        self.chain_asym = chain_asym or {}
        self.binder_id = binder_id
        self.chain_names = chain_names or {}

    @property
    def binder_tokens(self) -> torch.Tensor:
        return self.chain_tokens[self.binder_id]

    @property
    def binder_asym(self) -> int:
        return self.chain_asym[self.binder_id]

    def _cat(self, ids: list[str]) -> torch.Tensor:
        groups = [self.chain_tokens[c] for c in ids if self.chain_tokens[c].numel() > 0]
        if not groups:
            return self.design_tokens[:0]
        return torch.cat(groups)

    @property
    def target_tokens(self) -> torch.Tensor:
        return self._cat([c for c in self.chain_ids if self.chain_roles.get(c) == ROLE_TARGET])

    @property
    def rest_of_complex_tokens(self) -> torch.Tensor:
        return self._cat([c for c in self.chain_ids if c != self.binder_id])

    def chain_tokens_for_residues(self, chain_id: str, res_ids: list[int]) -> torch.Tensor:
        tokens = self.chain_tokens[chain_id]
        if not res_ids or tokens.numel() == 0:
            return tokens[:0]
        wanted = torch.as_tensor(list(res_ids), device=self.residue_index.device)
        mask = torch.isin(self.residue_index[tokens], wanted)
        return tokens[mask]

    def contact_tokens(self, contact_map: dict[str, list[int]]) -> torch.Tensor:
        groups = [self.chain_tokens_for_residues(cid, res) for cid, res in contact_map.items()]
        groups = [g for g in groups if g.numel() > 0]
        if not groups:
            return self.design_tokens[:0]
        return torch.cat(groups)

    def avoidance_tokens(self, avoid_map: dict[str, list[int]]) -> torch.Tensor:
        return self.contact_tokens(avoid_map)

    @property
    def nanobody_tokens(self) -> torch.Tensor:
        return self.binder_tokens

    @property
    def nanobody_asym(self) -> int:
        return self.binder_asym

    @property
    def peptide_tokens(self) -> torch.Tensor:
        return self.chain_tokens[self.chain_names["peptide"]]

    @property
    def peptide_asym(self) -> int:
        return self.chain_asym[self.chain_names["peptide"]]

    @property
    def mhc_tokens(self) -> torch.Tensor:
        return self.chain_tokens[self.chain_names["mhc"]]

    @property
    def mhc_asym(self) -> int:
        return self.chain_asym[self.chain_names["mhc"]]

    def mhc_tokens_for_residues(self, res_ids: list[int]) -> torch.Tensor:
        return self.chain_tokens_for_residues(self.chain_names["mhc"], res_ids)


@dataclass
class PredictionOutputs:
    per_token_plddt: torch.Tensor
    pae: torch.Tensor
    distogram_logits: torch.Tensor
    ca_coords: torch.Tensor
    chain_pair_iptm: torch.Tensor
    chain_index: ChainIndex
    distogram_bin_params: dict
    iptm_global: float = 0.0
    # Per-token backbone coords (N, CA, C, O), [N_token, 4, 3], NaN where an
    # atom is missing; and the per-token 1-letter sequence (length N_token).
    # Populated for full complexes; consumed by the ProteinMPNN score.
    backbone_coords: torch.Tensor | None = None
    sequence: str = ""



@runtime_checkable
class Oracle(Protocol):
    """Any structure predictor proga can drive (Protenix-v2, Boltz-2, OpenDDE, ...).
    `design_positions` (binder-local indices) is set by the CLI after construction and read
    inside predict_complex."""

    design_positions: list[int]

    def predict_complex(
        self,
        chains: list[tuple[str, str]],
        roles: dict[str, str],
        design_positions: list[int] | None = None,
        name: str = "design",
        chain_names: dict[str, str] | None = None,
        dump_dir: str | None = None,
    ) -> "PredictionOutputs":
        ...


# --- backend registry: name -> "module:Class"; make_oracle imports the chosen one lazily,
# so a box only needs the deps of the backend it actually uses.
_ORACLE_BACKENDS: dict[str, str] = {
    "protenix": "proga.oracle:ProtenixOracle",      # Protenix-v2, GPU, in-process torch
    "boltz2": "proga.boltz_oracle:BoltzOracle",     # Boltz-2, subprocess `boltz predict` (forward-only)
    "opendde": "proga.opendde_oracle:OpenDDEOracle",# OpenDDE, GPU, in-process (shared Protenix runner)
    "ttbio": "proga.ttbio_oracle:TTBioOracle",   # tt-bio (Tenstorrent) subprocess `tt-bio predict`
}


def register_oracle(name: str, target: str) -> None:
    """Register a new fold backend. `target` is "module.path:ClassName"; the class must satisfy
    the Oracle protocol. Lets third parties add oracles without editing proga."""
    _ORACLE_BACKENDS[name] = target


def make_oracle(name: str, config, work_dir: str | None = None) -> "Oracle":
    import importlib
    if name not in _ORACLE_BACKENDS:
        raise ValueError(f"unknown oracle backend {name!r}; known: {sorted(_ORACLE_BACKENDS)}")
    mod, cls = _ORACLE_BACKENDS[name].split(":")
    return getattr(importlib.import_module(mod), cls)(config, work_dir=work_dir)
