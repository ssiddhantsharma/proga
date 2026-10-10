"""tt-bio (Tenstorrent) subprocess oracle for proga.

Folds each genome on a Tenstorrent card via the ``tt-bio predict`` CLI and parses tt-bio's
output back into proga's ``PredictionOutputs``. Subclasses :class:`BoltzOracle` to reuse the
boltz-compatible YAML writer, the MSA cache and the CIF parser (pLDDT in B-factors); only the run
command and the confidence/PAE readers are tt-bio specific.

Non-differentiable scalar oracle, so it pairs with the GA / NSGA-II path, not gradients. tt-bio
predict does not surface distogram logits, so ``distogram_logits=None`` and the contact/distogram
scoring terms are skipped (scoring.py guards on it), the same as the other subprocess folders.

Grounded against ``tt-bio predict --help`` (``--out_dir/--model/--write_pae/--output_format/
--recycling_steps/--sampling_steps/--diffusion_samples/--seed/--msa_dir/--msa_cache_only/
--single_sequence``) and the documented output layout (docs/confidence-outputs.md):
``structures/<name>.cif`` (pLDDT in B-factors), ``structures/<name>_pae.npz['pae']`` and a
``results.json`` carrying pTM / ipTM / chain-pair ipTM scalars.

STATUS: wired to the documented CLI and outputs; the confidence reader falls back across the
likely chain-pair ipTM key names. Validate the exact key on the first real p150a fold.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from proga.boltz_oracle import BoltzOracle
from proga.oracle_base import ChainIndex, PredictionOutputs


class TTBioOracle(BoltzOracle):
    def __init__(self, config, work_dir: str | None = None):
        super().__init__(config, work_dir=work_dir)
        self.launcher = getattr(config, "ttbio_launcher", "") or os.environ.get(
            "TTBIO", "/home/bizon/ttbio"
        )
        self.model = getattr(config, "ttbio_model", "boltz2")
        self.extra = list(getattr(config, "ttbio_extra_args", None) or [])

    def predict_complex(
        self,
        chains: list[tuple[str, str]],
        roles: dict[str, str],
        design_positions: list[int] | None = None,
        name: str = "design",
        chain_names: dict[str, str] | None = None,
        dump_dir: str | None = None,
    ) -> PredictionOutputs:
        assembly_ids = [cid for cid, _ in chains]
        run_dir = Path(dump_dir) if dump_dir else (self.work / "run")
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = run_dir / f"{name}.yaml"
        self._write_yaml(chains, yaml_path)

        cmd = [
            self.launcher, "predict", str(yaml_path),
            "--out_dir", str(run_dir),
            "--model", self.model,
            "--write_pae",
            "--output_format", "cif",
            "--recycling_steps", str(self.n_cycle),
            "--sampling_steps", str(self.n_step),
            "--diffusion_samples", str(self.n_sample),
            "--seed", str(self.seed),
            *self.extra,
        ]
        subprocess.run(cmd, check=True, cwd=str(run_dir))

        struct = run_dir / "structures"
        pred_dir = struct if struct.is_dir() else self._find_pred_dir(run_dir, name)
        plddt, ca, backbone, sequence, asym_id, residue_index, chain_tokens = self._parse_cif(
            pred_dir, name, assembly_ids
        )
        pae = self._parse_pae_ttbio(pred_dir, name)
        iptm_global, chain_pair_iptm = self._parse_confidence_ttbio(run_dir, pred_dir, assembly_ids, name)

        positions = design_positions if design_positions is not None else self.design_positions
        binder_id = assembly_ids[0]
        binder_tokens = chain_tokens[binder_id]
        design_tokens = (
            binder_tokens[torch.as_tensor(positions).long()] if positions else binder_tokens[:0]
        )
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
            distogram_logits=None,
            ca_coords=ca,
            chain_pair_iptm=chain_pair_iptm,
            chain_index=chain_index,
            distogram_bin_params={},
            iptm_global=iptm_global,
            backbone_coords=backbone,
            sequence=sequence,
        )

    def _parse_pae_ttbio(self, pred_dir: Path, name: str) -> torch.Tensor:
        p = (
            next(pred_dir.glob(f"{name}*_pae.npz"), None)
            or next(pred_dir.glob("*_pae.npz"), None)
            or next(pred_dir.glob("**/*_pae.npz"), None)
        )
        if p is None:
            raise FileNotFoundError(f"no *_pae.npz under {pred_dir} (pass --write_pae)")
        z = np.load(p)
        arr = z["pae"] if "pae" in getattr(z, "files", []) else np.asarray(z)
        if arr.ndim == 3:
            arr = arr[0]
        return torch.as_tensor(arr, dtype=torch.float32)

    def _parse_confidence_ttbio(self, run_dir, pred_dir, assembly_ids, name):
        rj = (
            next(pred_dir.glob("results.json"), None)
            or next(run_dir.glob("**/results.json"), None)
            or next(pred_dir.glob("*summary_confidences.json"), None)
        )
        if rj is None:
            raise FileNotFoundError(f"no results.json under {run_dir}")
        with open(rj) as fh:
            data = json.load(fh)
        recs = data if isinstance(data, list) else [data]
        rec = next((r for r in recs if r.get("id") == name), recs[0])
        idx = {cid: k for k, cid in enumerate(assembly_ids)}
        cpi = torch.zeros(len(assembly_ids), len(assembly_ids))
        for k, v in (rec.get("chains_ptm") or {}).items():
            if int(k) < len(assembly_ids):
                cpi[int(k), int(k)] = float(v)
        pairs = (rec.get("interface_score_distribution") or {}).get("pairs", {})
        iptms = []
        for key, blk in pairs.items():
            parts = key.split("-")
            if len(parts) == 2 and parts[0] in idx and parts[1] in idx:
                v = float((blk.get("iptm") or {}).get("mean", 0.0))
                i, j = idx[parts[0]], idx[parts[1]]
                cpi[i, j] = cpi[j, i] = v
                iptms.append(v)
        iptm_global = max(iptms) if iptms else float(rec.get("confidence_score", 0.0))
        return iptm_global, cpi
