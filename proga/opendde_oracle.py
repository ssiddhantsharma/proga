"""OpenDDE in-process oracle backend for proga.

OpenDDE is a Protenix-family fold engine (OpenDDE ~ Protenix-v2): it exposes the SAME
`runner.batch_inference.get_default_runner` API as `ProtenixOracle` and dumps to the same
layout `<dump_dir>/<name>/seed_<seed>/predictions/<name>_sample_*.cif`, so this oracle reuses
`BoltzOracle`'s CIF/PAE/confidence parsers and only the run step is OpenDDE-specific.

Run path (OpenDDE's public package API):
  build_inference_config(model_name="opendde_v1", fill_required_with_null=True)
  -> get_default_runner(..., load_checkpoint_path=<ckpt>, need_atom_confidence=True)
  -> per genome: write the AF3-schema input json, set configs.input_json_path, infer_predict(...)
  -> parse the dumped CIF (+ confidence/pae) into PredictionOutputs.

Requires the OpenDDE GPU env (the `opendde` package + a resolved checkpoint via OPENDDE_ROOT_DIR).
`opendde`/`runner` are imported lazily in __init__ so this module loads (and the backend registry
lists it) on a box without OpenDDE installed.

STATUS: validated end-to-end on bizon (iptm 0.421). OpenDDE writes the Protenix-family layout — validate
on the first fold in the OpenDDE env (confirm the dumped confidence/pae filenames the parsers glob,
and that build_sample_dict_n's json is accepted as-is). Like distogram-free subprocess folders,
distogram_logits is None (OpenDDE's low-level distogram tensor is not surfaced by infer_predict),
so contact/distogram scoring terms are skipped.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from proga.boltz_oracle import BoltzOracle
from proga.oracle_base import ChainIndex, PredictionOutputs


class OpenDDEOracle(BoltzOracle):
    def __init__(self, config, work_dir: str | None = None):
        # NB: do NOT call BoltzOracle.__init__ (boltz CLI setup); we only inherit its parsers.
        self.config = config
        self.work = Path(work_dir or tempfile.mkdtemp(prefix="proga_opendde_"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.dump_root = self.work / "predictions"
        self.dump_root.mkdir(parents=True, exist_ok=True)
        self.seeds = [int(s) for s in getattr(config, "seeds", None) or [101]]
        self.n_step = getattr(config, "n_step", 200)
        self.n_cycle = getattr(config, "n_cycle", 10)
        self.n_sample = getattr(config, "n_sample", 1)
        self.model_name = getattr(config, "opendde_model", "opendde_v1")
        self.checkpoint = getattr(config, "opendde_checkpoint", "") or os.environ.get(
            "OPENDDE_CHECKPOINT", ""
        )
        self.target_msa = dict(getattr(config, "target_msa", None) or {})
        self.design_positions: list[int] = []

        from opendde.config.inference import build_inference_config
        from opendde.utils.download import download_inference_cache
        from runner.batch_inference import get_default_runner

        configs = build_inference_config(model_name=self.model_name, fill_required_with_null=True)
        configs.load_checkpoint_path = self.checkpoint
        configs.use_template = False
        download_inference_cache(configs)

        self.runner = get_default_runner(
            seeds=self.seeds,
            dump_dir=str(self.dump_root),
            n_cycle=self.n_cycle,
            n_step=self.n_step,
            n_sample=self.n_sample,
            dtype="fp32",
            model_name=self.model_name,
            load_checkpoint_path=self.checkpoint,
            use_msa=True,
            use_template=False,
            need_atom_confidence=True,
        )

    def _write_input_json(self, chains: list[tuple[str, str]], name: str) -> Path:
        # AF3-schema input json (Protenix/OpenDDE shared); binder first.
        from proga.complex import build_sample_dict_n

        sample = build_sample_dict_n(chains, name=name)
        json_dir = self.work / "input_jsons"
        json_dir.mkdir(exist_ok=True)
        p = json_dir / f"{name}.json"
        with open(p, "w") as f:
            json.dump([sample], f)
        return p

    def predict_complex(
        self,
        chains: list[tuple[str, str]],
        roles: dict[str, str],
        design_positions: list[int] | None = None,
        name: str = "design",
        chain_names: dict[str, str] | None = None,
        dump_dir: str | None = None,
    ) -> PredictionOutputs:
        from runner.inference import infer_predict

        assembly_ids = [cid for cid, _ in chains]
        self.runner.configs.input_json_path = str(self._write_input_json(chains, name))
        infer_predict(self.runner, self.runner.configs)

        pred_dir = self._find_pred_dir(self.dump_root, name)
        iptm_global, chain_pair_iptm = self._parse_confidence(pred_dir, name, len(chains))
        pae = self._parse_pae(pred_dir, name)
        plddt, ca, backbone, sequence, asym_id, residue_index, chain_tokens = self._parse_cif(
            pred_dir, name, assembly_ids
        )

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

    def _parse_confidence(self, pred_dir, name, n_chain):
        cj = next(pred_dir.glob("*_summary_confidence_sample_*.json"), None) or next(
            pred_dir.glob("**/*_summary_confidence_sample_*.json"), None
        )
        if cj is None:
            raise FileNotFoundError(f"no *_summary_confidence_sample_*.json under {pred_dir}")
        with open(cj) as fh:
            d = json.load(fh)
        iptm_global = float(d.get("iptm", 0.0))
        cpi = torch.zeros(n_chain, n_chain)
        t = torch.as_tensor(d.get("chain_pair_iptm") or d.get("chain_pair_iptm_global") or [],
                            dtype=torch.float32)
        if t.ndim == 2:
            m = min(n_chain, t.shape[0])
            cpi[:m, :m] = t[:m, :m]
        return iptm_global, cpi

    def _parse_pae(self, pred_dir, name):
        fj = next(pred_dir.glob("*_full_data_sample_*.json"), None) or next(
            pred_dir.glob("**/*_full_data_sample_*.json"), None
        )
        if fj is None:
            raise FileNotFoundError(f"no *_full_data_sample_*.json under {pred_dir}")
        with open(fj) as fh:
            arr = np.asarray(json.load(fh)["token_pair_pae"], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return torch.as_tensor(arr, dtype=torch.float32)
