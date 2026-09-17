import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from proga.complex import (
    MHC_CHAIN_ID,
    NANOBODY_CHAIN_ID,
    PEPTIDE_CHAIN_ID,
    build_sample_dict,
    build_sample_dict_n,
)
from proga.config import DesignConfig
from protenix.model import sample_confidence
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import to_device
from runner.batch_inference import get_default_runner
from protenix.data.inference.infer_dataloader import InferenceDataset

logger = get_logger(__name__)


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
        chain_ids: Optional[list[str]] = None,
        chain_tokens: Optional[dict[str, torch.Tensor]] = None,
        chain_roles: Optional[dict[str, str]] = None,
        chain_asym: Optional[dict[str, int]] = None,
        binder_id: Optional[str] = None,
        chain_names: Optional[dict[str, str]] = None,
        # --- legacy pMHC form ---
        nanobody_tokens: Optional[torch.Tensor] = None,
        peptide_tokens: Optional[torch.Tensor] = None,
        mhc_tokens: Optional[torch.Tensor] = None,
        nanobody_asym: Optional[int] = None,
        peptide_asym: Optional[int] = None,
        mhc_asym: Optional[int] = None,
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
    backbone_coords: Optional[torch.Tensor] = None
    sequence: str = ""


class ProtenixOracle:
    def __init__(self, config: DesignConfig, work_dir: Optional[str] = None):

        self.config = config
        self.runner = get_default_runner(
            seeds=config.seeds,
            n_cycle=config.n_cycle,
            n_step=config.n_step,
            n_sample=config.n_sample,
            dtype=config.dtype,
            model_name=config.model_name,
            use_msa=config.use_msa,
            use_template=False,
            use_rna_msa=False,
        )
        self.configs = self.runner.configs

        self._work_dir = work_dir or tempfile.mkdtemp(prefix="protenix_design_")
        os.makedirs(self._work_dir, exist_ok=True)
        placeholder = os.path.join(self._work_dir, "_placeholder.json")
        with open(placeholder, "w") as f:
            json.dump([build_sample_dict("A", "A", "A", name="placeholder")], f)
        self.configs.input_json_path = placeholder
        self.configs.dump_dir = os.path.join(self._work_dir, "dump")
        os.makedirs(self.configs.dump_dir, exist_ok=True)
        self.dataset = InferenceDataset(configs=self.configs)

        # Local (within-nanobody) indices of the design mask. The genetic
        # runner sets this once (oracle.design_positions = mask.positions)
        # before any predict() call so design tokens can be located.
        self.design_positions: list[int] = []

        # Precomputed MSA paths for the fixed target chains, keyed by sequence.
        # Populated once via precompute_target_msa(); reused on every predict().
        self.target_msa: dict[str, dict[str, str]] = {}

        # Capture raw distogram logits each forward pass.
        self._captured_distogram: Optional[torch.Tensor] = None
        self.runner.model.distogram_head.register_forward_hook(self._distogram_hook)

    # ------------------------------------------------------------- target MSA
    def precompute_target_msa(self, target_seqs: list[str]) -> None:
        """Compute (or load from cache) MSAs for the fixed target chains once.

        Stored in ``self.target_msa`` and reused for every subsequent predict().
        Requires the runner to have been built with ``use_msa=True``.
        """
        from proga.msa import MSACache

        cache = MSACache(self.config.msa_cache_dir, mode=self.config.msa_mode)
        self.target_msa = cache.get_or_compute(target_seqs)
        for seq, paths in self.target_msa.items():
            logger.info(
                "Target MSA [%s len=%d]: %s",
                seq[:8], len(seq), ", ".join(sorted(paths)) or "none",
            )

    # ------------------------------------------------------------------ hooks
    def _distogram_hook(self, module, inputs, output):
        # The inference path calls distogram_head once (protenix.py:583). Keep
        # the most recent [N_token, N_token, n_bins] output.
        self._captured_distogram = output.detach()

    # ----------------------------------------------------------------- predict
    def predict(self, nanobody_seq: str, peptide_seq: str, mhc_seq: str) -> PredictionOutputs:
        """pMHC convenience wrapper: fold the nanobody/peptide/MHC (A/B/C) complex.

        Thin wrapper over :meth:`predict_complex` preserving the historical
        signature for the pMHC code path and the smoke test.
        """
        chains = [
            (NANOBODY_CHAIN_ID, nanobody_seq),
            (PEPTIDE_CHAIN_ID, peptide_seq),
            (MHC_CHAIN_ID, mhc_seq),
        ]
        roles = {
            NANOBODY_CHAIN_ID: ROLE_BINDER,
            PEPTIDE_CHAIN_ID: ROLE_TARGET,
            MHC_CHAIN_ID: ROLE_CONTEXT,
        }
        return self.predict_complex(
            chains,
            roles,
            chain_names={"peptide": PEPTIDE_CHAIN_ID, "mhc": MHC_CHAIN_ID},
        )

    @torch.no_grad()
    def predict_complex(
        self,
        chains: list[tuple[str, str]],
        roles: dict[str, str],
        design_positions: Optional[list[int]] = None,
        name: str = "design",
        chain_names: Optional[dict[str, str]] = None,
        dump_dir: Optional[str] = None,
    ) -> PredictionOutputs:
        """Fold an arbitrary N-chain complex (binder first) and decode outputs.

        ``chains`` is the ordered ``(chain_id, sequence)`` assembly with the
        designed binder first. ``roles`` maps each chain id to ``binder`` /
        ``target`` / ``context``. ``design_positions`` (binder-local indices)
        defaults to ``self.design_positions``. ``chain_names`` optionally records
        named chains (e.g. ``{"peptide": "B", "mhc": "C"}``) for pMHC shims.

        ``dump_dir``: when given, the predicted structure is written via Protenix's
        own dumper to ``<dump_dir>/<name>/seed_<seed>/predictions/<name>_sample_*.cif``.
        Left ``None`` during the GA (structures discarded); set only when saving
        the final designs, since it adds CIF-writing I/O per call.
        """
        from runner.inference import update_inference_configs

        sample_dict = build_sample_dict_n(chains, name=name, msa_by_seq=self.target_msa)
        data, atom_array, _ = self.dataset.process_one(sample_dict)

        n_token = int(data["N_token"].item())
        new_configs = update_inference_configs(self.configs, n_token)
        self.runner.update_model_configs(new_configs)

        self._captured_distogram = None
        prediction = self.runner.predict(data)
        if self._captured_distogram is None:
            raise RuntimeError("Distogram logits were not captured by the hook.")

        if dump_dir is not None:
            # Reuse Protenix's dumper to write the predicted structure (and
            # confidence) to dump_dir, temporarily retargeting its base_dir.
            prev_base = self.runner.dumper.base_dir
            prev_need = self.runner.dumper.need_atom_confidence
            self.runner.dumper.base_dir = dump_dir
            # full_data (incl. the PAE matrix) is already computed in pred_dict;
            # enabling this only writes it out (no extra compute) so the final
            # designs persist a *_full_data_sample_0.json alongside each CIF.
            self.runner.dumper.need_atom_confidence = True
            try:
                self.runner.dumper.dump(
                    dataset_name="",
                    pdb_id=name,
                    seed=self.config.seeds[0],
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type={
                        k: v
                        for k, v in data["entity_poly_type"].items()
                        if v != "non-polymer"
                    },
                )
            finally:
                self.runner.dumper.base_dir = prev_base
                self.runner.dumper.need_atom_confidence = prev_need
        # Defensively drop any leading singleton (batch) dims -> [Nt, Nt, bins].
        while self._captured_distogram.dim() > 3:
            self._captured_distogram = self._captured_distogram.squeeze(0)

        feat = data["input_feature_dict"]
        assembly_ids = [cid for cid, _ in chains]
        if design_positions is None:
            design_positions = self.design_positions
        # Per-token sequence: one token per residue for protein chains, in
        # assembly (asym) order -> concatenated chain sequences.
        sequence = "".join(seq for _, seq in chains)
        outputs = self._decode(
            prediction, feat, assembly_ids, roles, design_positions,
            chain_names, atom_array=atom_array, sequence=sequence,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return outputs

    def _decode(
        self,
        prediction,
        feat,
        assembly_ids: list[str],
        roles: dict[str, str],
        design_positions: list[int],
        chain_names: Optional[dict[str, str]] = None,
        atom_array=None,
        sequence: str = "",
    ) -> PredictionOutputs:
        loss_cfg = self.configs.loss
        device = self._captured_distogram.device

        asym_id = feat["asym_id"].to(device).long()  # [N_token]
        residue_index = feat["residue_index"].to(device).long()
        atom_to_token = feat["atom_to_token_idx"].to(device).long()
        n_token = asym_id.shape[0]

        plddt_logits = prediction["plddt"][0].to(device).float()  # [N_atom, bins]
        atom_plddt = sample_confidence.logits_to_score(
            plddt_logits, **sample_confidence.get_bin_params(loss_cfg.plddt)
        )  # [N_atom] in [0, 1]
        plddt_rep = feat["plddt_m_rep_atom_mask"].to(device).bool()
        per_token_plddt = torch.zeros(n_token, device=device)
        per_token_plddt[atom_to_token[plddt_rep]] = atom_plddt[plddt_rep]

        pae = sample_confidence.logits_to_score(
            prediction["pae"][0].to(device).float(),
            **sample_confidence.get_bin_params(loss_cfg.pae),
        )

        coord = prediction["coordinate"][0].to(device).float()  # [N_atom, 3]
        disto_rep = feat["distogram_rep_atom_mask"].to(device).bool()
        ca_coords = torch.zeros(n_token, 3, device=device)
        ca_coords[atom_to_token[disto_rep]] = coord[disto_rep]

        # Per-token backbone (N, CA, C, O) for the ProteinMPNN score. coord is
        # aligned 1:1 with atom_array (the dumper assigns coord -> atom_array
        # in order), so atom_array.atom_name selects the backbone atoms.
        backbone_coords = None
        if atom_array is not None:
            atom_names = np.asarray(atom_array.atom_name)
            backbone_coords = torch.full((n_token, 4, 3), float("nan"), device=device)
            for j, aname in enumerate(("N", "CA", "C", "O")):
                sel = torch.as_tensor(atom_names == aname, device=device, dtype=torch.bool)
                if sel.any():
                    backbone_coords[atom_to_token[sel], j, :] = coord[sel]

        unique_asym = torch.unique(asym_id).tolist()
        if unique_asym != list(range(len(unique_asym))):
            raise RuntimeError(f"Non-contiguous asym_id ordering: {unique_asym}")
        if len(unique_asym) != len(assembly_ids):
            raise RuntimeError(
                f"Decoded {len(unique_asym)} chains but assembled "
                f"{len(assembly_ids)} ({assembly_ids})."
            )
        chain_asym = {cid: k for k, cid in enumerate(assembly_ids)}
        chain_tokens = {cid: torch.where(asym_id == k)[0] for cid, k in chain_asym.items()}

        binder_id = assembly_ids[0]
        binder_tokens = chain_tokens[binder_id]
        if design_positions:
            design_local = torch.as_tensor(design_positions, device=device).long()
            design_tokens = binder_tokens[design_local]
        else:
            design_tokens = binder_tokens[:0]

        chain_index = ChainIndex(
            chain_ids=list(assembly_ids),
            chain_tokens=chain_tokens,
            chain_roles=roles,
            chain_asym=chain_asym,
            binder_id=binder_id,
            chain_names=chain_names,
            design_tokens=design_tokens,
            residue_index=residue_index,
            asym_id=asym_id,
        )

        summary = prediction["summary_confidence"][0]
        chain_pair_iptm = summary["chain_pair_iptm"].to(device).float()
        iptm_global = float(summary["iptm"])

        return PredictionOutputs(
            per_token_plddt=per_token_plddt,
            pae=pae,
            distogram_logits=self._captured_distogram.to(device).float(),
            ca_coords=ca_coords,
            chain_pair_iptm=chain_pair_iptm,
            iptm_global=iptm_global,
            chain_index=chain_index,
            distogram_bin_params=sample_confidence.get_bin_params(loss_cfg.distogram),
            backbone_coords=backbone_coords,
            sequence=sequence,
        )
