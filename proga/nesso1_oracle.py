"""Nesso-1 (Tenstorrent) affinity oracle for proga: protein -> ligand design.

Scores the GA-designed protein against a fixed small-molecule ligand with `tt-bio affinity
--model nesso1` (a scalar, no fold). The ligand is read from config; the binder protein is the
GA genome. Returns a PredictionOutputs carrying only `affinity` (structure fields empty), so the
scorer runs on affinity alone (its has-structure guard skips the structure terms).

Nesso-1 is protein-ligand only and returns no coordinates (docs/nesso1.md), so this is a distinct
oracle, not a fold backend. `<id>_affinity.json` keys (pinned on a p150a): affinity_pred_value
(lower = stronger), affinity_probability_binary (P(binder), 0-1). The scored key is configurable.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from proga.oracle_base import ChainIndex, PredictionOutputs


class NessoAffinityOracle:
    def __init__(self, config, work_dir: str | None = None):
        self.config = config
        self.work = Path(work_dir or tempfile.mkdtemp(prefix="proga_nesso1_"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.launcher = getattr(config, "nesso1_launcher", "") or os.environ.get("TTBIO", "tt-bio")
        self.ligand_id = getattr(config, "nesso1_ligand_id", "LIG")
        self.ligand_smiles = getattr(config, "nesso1_ligand_smiles", "")
        self.ligand_ccd = getattr(config, "nesso1_ligand_ccd", "")
        self.affinity_key = getattr(config, "nesso1_affinity_key", "affinity_probability_binary")
        self.extra = list(getattr(config, "nesso1_extra_args", None) or [])
        self.design_positions: list[int] = []
        if not (self.ligand_smiles or self.ligand_ccd):
            raise ValueError("nesso1 oracle needs nesso1_ligand_smiles or nesso1_ligand_ccd")

    def _write_yaml(self, binder_id: str, binder_seq: str, path: Path) -> None:
        lig = f'smiles: "{self.ligand_smiles}"' if self.ligand_smiles else f"ccd: {self.ligand_ccd}"
        path.write_text(
            "version: 1\n"
            "sequences:\n"
            f"  - protein:\n      id: {binder_id}\n      sequence: {binder_seq}\n"
            f"  - ligand:\n      id: {self.ligand_id}\n      {lig}\n"
            "properties:\n"
            f"  - affinity:\n      binder: {self.ligand_id}\n"
        )

    def predict_complex(
        self,
        chains: list[tuple[str, str]],
        roles: dict[str, str],
        design_positions: list[int] | None = None,
        name: str = "design",
        chain_names: dict[str, str] | None = None,
        dump_dir: str | None = None,
    ) -> PredictionOutputs:
        binder_id, binder_seq = chains[0]
        run_dir = (Path(dump_dir) if dump_dir else (self.work / "run")).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = run_dir / f"{name}.yaml"
        self._write_yaml(binder_id, binder_seq, yaml_path)

        cmd = [
            self.launcher, "affinity", str(yaml_path),
            "--out_dir", str(run_dir),
            "--model", "nesso1",
            *self.extra,
        ]
        subprocess.run(cmd, check=True, cwd=str(run_dir))
        affinity = self._parse_affinity(run_dir, name)

        empty = torch.zeros(0)
        ci = ChainIndex(
            chain_ids=[binder_id],
            chain_tokens={binder_id: torch.zeros(0, dtype=torch.long)},
            chain_roles={binder_id: roles.get(binder_id, "binder")},
            chain_asym={binder_id: 0},
            binder_id=binder_id,
            chain_names=chain_names,
            design_tokens=torch.zeros(0, dtype=torch.long),
            residue_index=torch.zeros(0, dtype=torch.long),
            asym_id=torch.zeros(0, dtype=torch.long),
        )
        return PredictionOutputs(
            per_token_plddt=empty,
            pae=torch.zeros(0, 0),
            distogram_logits=None,
            ca_coords=torch.zeros(0, 3),
            chain_pair_iptm=torch.zeros(0, 0),
            chain_index=ci,
            distogram_bin_params={},
            iptm_global=0.0,
            backbone_coords=None,
            sequence=binder_seq,
            affinity=affinity,
        )

    def _parse_affinity(self, run_dir: Path, name: str) -> float:
        p = (
            next(run_dir.glob(f"{name}*_affinity.json"), None)
            or next(run_dir.glob("*_affinity.json"), None)
            or next(run_dir.glob("**/*_affinity.json"), None)
        )
        if p is None:
            raise FileNotFoundError(f"no *_affinity.json under {run_dir}")
        with open(p) as fh:
            rec = json.load(fh)
        if self.affinity_key not in rec:
            raise KeyError(f"{self.affinity_key!r} not in {p.name}: keys {sorted(rec)}")
        return float(rec[self.affinity_key])
