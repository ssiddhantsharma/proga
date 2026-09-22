"""Boltz-2 forward-only oracle backend for proga.

Folds each complex by shelling out to `boltz predict` and parsing the results back into the
same `PredictionOutputs` contract the scorer consumes. The GA core is pure-Python, so this is
a drop-in oracle swap — only the fold engine changes.

Limits (same shape as any subprocess folder):
  * Forward-only — no gradients; only proga's gradient-free/MCMC path can use it.
  * Boltz does not emit distogram logits -> `distogram_logits=None`, so the distogram/contact
    scoring terms are skipped (scoring.py guards on `distogram_logits is None`).
  * Model is reloaded per `boltz predict` call -> throughput-bound; a pilot/offload path, not a
    large campaign. (In-process reuse is a future optimisation.)

Grounded against the installed `boltz` CLI (flags: --diffusion_samples/--sampling_steps/
--recycling_steps/--write_full_pae/--output_format mmcif/--seed/--override/--accelerator) and
Boltz-2's output layout: boltz_results_<stem>/predictions/<name>/{confidence_<name>_model_0.json
(iptm, pair_chains_iptm), pae_<name>_model_0.npz (key 'pae'/'predicted_aligned_error'),
<name>_model_0.cif (pLDDT 0-100 in the B-factor)}.
"""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from proga.oracle_base import ChainIndex, PredictionOutputs

AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


class BoltzOracle:
    def __init__(self, config, work_dir: Optional[str] = None):
        self.config = config
        self.work = Path(work_dir or tempfile.mkdtemp(prefix="proga_boltz_"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.msa_dir = self.work / "msa"
        self.msa_dir.mkdir(exist_ok=True)
        self.boltz_bin = getattr(config, "boltz_bin", "boltz")
        self.accelerator = getattr(config, "boltz_accelerator", "gpu")
        self.seed = int(config.seeds[0]) if getattr(config, "seeds", None) else 0
        self.n_step = getattr(config, "n_step", 200)
        self.n_cycle = getattr(config, "n_cycle", 10)
        self.n_sample = getattr(config, "n_sample", 1)
        # per-genome binder folds single-sequence; targets reuse a cached a3m when proga carries one
        self.target_msa = dict(getattr(config, "target_msa", None) or {})
        self.design_positions: list[int] = []

    def precompute_target_msa(self, target_seqs: list[str]) -> None:
        """Cache target-chain MSAs for reuse across folds (called by the CLI when --use_msa).
        Offline default: single-sequence a3m per target (Boltz folds the target MSA-free).
        Server MSA (mmseqs `--use_msa_server`, or a real a3m) is a follow-up — wire the a3m
        text into ``self.target_msa[seq]`` and _a3m_path picks it up."""
        for seq in target_seqs:
            self.target_msa.setdefault(seq, f">query\n{seq}\n")

    # ---------- MSA: content-addressed a3m; Boltz reads a path per protein ----------
    def _a3m_path(self, seq: str) -> Path:
        p = self.msa_dir / f"{_seq_hash(seq)}.a3m"
        if not p.exists():
            p.write_text(self.target_msa.get(seq, f">query\n{seq}\n"))
        return p

    def _write_yaml(self, chains: list[tuple[str, str]], path: Path) -> None:
        lines = ["version: 1", "sequences:"]
        for cid, seq in chains:
            lines += [
                "  - protein:",
                f"      id: {cid}",
                f"      sequence: {seq}",
                f"      msa: {self._a3m_path(seq)}",
            ]
        path.write_text("\n".join(lines) + "\n")

    # ---------- run ----------
    def predict_complex(
        self,
        chains: list[tuple[str, str]],
        roles: dict[str, str],
        design_positions: Optional[list[int]] = None,
        name: str = "design",
        chain_names: Optional[dict[str, str]] = None,
        dump_dir: Optional[str] = None,
    ) -> PredictionOutputs:
        assembly_ids = [cid for cid, _ in chains]
        run_dir = Path(dump_dir) if dump_dir else (self.work / "run")
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = run_dir / f"{name}.yaml"
        self._write_yaml(chains, yaml_path)

        cmd = [
            self.boltz_bin, "predict", str(yaml_path),
            "--out_dir", str(run_dir),
            "--diffusion_samples", str(self.n_sample),
            "--sampling_steps", str(self.n_step),
            "--recycling_steps", str(self.n_cycle),
            "--output_format", "mmcif",
            "--write_full_pae",
            "--accelerator", self.accelerator,
            "--devices", "1",
            "--num_workers", "0",
            "--seed", str(self.seed),
            "--override",
        ]
        subprocess.run(cmd, check=True, cwd=str(run_dir))

        pred_dir = self._find_pred_dir(run_dir, name)
        iptm_global, chain_pair_iptm = self._parse_confidence(pred_dir, name, len(chains))
        pae = self._parse_pae(pred_dir, name)
        plddt, ca, backbone, sequence, asym_id, residue_index, chain_tokens = self._parse_cif(
            pred_dir, name, assembly_ids
        )

        positions = design_positions if design_positions is not None else self.design_positions
        binder_id = assembly_ids[0]
        binder_tokens = chain_tokens[binder_id]
        if positions:
            design_tokens = binder_tokens[torch.as_tensor(positions).long()]
        else:
            design_tokens = binder_tokens[:0]

        chain_index = ChainIndex(
            chain_ids=list(assembly_ids),
            chain_tokens=chain_tokens,
            chain_roles=roles,
            chain_asym={cid: k for k, cid in enumerate(assembly_ids)},
            binder_id=binder_id,
            chain_names=chain_names,
            design_tokens=design_tokens,
            residue_index=residue_index,
            asym_id=asym_id,
        )
        return PredictionOutputs(
            per_token_plddt=plddt,
            pae=pae,
            distogram_logits=None,           # Boltz does not emit distogram logits
            ca_coords=ca,
            chain_pair_iptm=chain_pair_iptm,
            chain_index=chain_index,
            distogram_bin_params={},         # contact/distogram terms disabled (scoring guards)
            iptm_global=iptm_global,
            backbone_coords=backbone,
            sequence=sequence,
        )

    # ---------- parsers ----------
    def _find_pred_dir(self, run_dir: Path, name: str) -> Path:
        for c in run_dir.glob("**/predictions/*"):
            if c.is_dir() and any(c.glob("*.cif")):
                return c
        cif = next(run_dir.glob("**/*.cif"), None)
        if cif is None:
            raise FileNotFoundError(f"no CIF under {run_dir}")
        return cif.parent

    def _parse_confidence(self, pred_dir: Path, name: str, n_chain: int):
        cj = next(pred_dir.glob("confidence_*.json"), None) or next(pred_dir.glob("**/confidence_*.json"), None)
        if cj is None:
            raise FileNotFoundError(f"no confidence_*.json under {pred_dir}")
        data = json.load(open(cj))
        iptm_global = float(data["iptm"])
        cpi = torch.zeros(n_chain, n_chain)
        pc = data.get("pair_chains_iptm") or {}
        for i, row in pc.items():
            for j, v in (row or {}).items():
                cpi[int(i), int(j)] = float(v)
        return iptm_global, cpi

    def _parse_pae(self, pred_dir: Path, name: str) -> torch.Tensor:
        p = next(pred_dir.glob("pae_*.np[yz]"), None) or next(pred_dir.glob("**/pae_*.np[yz]"), None)
        if p is None:
            raise FileNotFoundError(f"no pae_*.npy/npz under {pred_dir}")
        z = np.load(p)
        if hasattr(z, "files"):
            key = "pae" if "pae" in z.files else ("predicted_aligned_error" if "predicted_aligned_error" in z.files else next(k for k in z.files if z[k].ndim >= 2))
            arr = z[key]
        else:
            arr = z
        arr = np.asarray(arr)
        if arr.ndim == 3:   # (samples, N, N) -> first sample
            arr = arr[0]
        return torch.as_tensor(arr, dtype=torch.float32)

    def _parse_cif(self, pred_dir: Path, name: str, assembly_ids: list[str]):
        import gemmi
        cif = next(pred_dir.glob("*.cif"))
        st = gemmi.read_structure(str(cif))
        model = st[0]
        by_id = {ch.name: ch for ch in model}
        chains = [by_id[c] for c in assembly_ids if c in by_id] or list(model)

        plddt, ca, bb, seq, asym, res_idx = [], [], [], [], [], []
        chain_tokens: dict[str, list[int]] = {cid: [] for cid in assembly_ids}
        tok = 0
        for k, ch in enumerate(chains):
            cid = assembly_ids[k] if k < len(assembly_ids) else ch.name
            for ri, res in enumerate(ch):
                atoms = {a.name: a for a in res}
                caa = atoms.get("CA")
                if caa is None:
                    continue
                plddt.append(caa.b_iso / 100.0)          # Boltz writes pLDDT (0-100) in B-factor
                ca.append([caa.pos.x, caa.pos.y, caa.pos.z])
                row = []
                for an in ("N", "CA", "C", "O"):
                    a = atoms.get(an)
                    row.append([a.pos.x, a.pos.y, a.pos.z] if a else [float("nan")] * 3)
                bb.append(row)
                seq.append(AA3to1.get(res.name, "X"))
                asym.append(k)
                res_idx.append(ri + 1)
                chain_tokens[cid].append(tok)
                tok += 1

        return (
            torch.tensor(plddt, dtype=torch.float32),
            torch.tensor(ca, dtype=torch.float32),
            torch.tensor(bb, dtype=torch.float32),
            "".join(seq),
            torch.tensor(asym, dtype=torch.long),
            torch.tensor(res_idx, dtype=torch.long),
            {cid: torch.tensor(t, dtype=torch.long) for cid, t in chain_tokens.items()},
        )
