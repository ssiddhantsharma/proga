import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gemmi
import torch
import torch.nn.functional as F

from proga import rmsd
from proga.config import DesignConfig
from proga.oracle import PredictionOutputs
from proga.spec import ROLE_TARGET, TargetSpec
from protenix.model.sample_confidence import compute_contact_prob

from typing import Final

_EPS: Final[float] = 1e-8

_MPNN_ALPHABET: Final[str] = "ACDEFGHIKLMNPQRSTVWYX"


def min_k(values: torch.Tensor, k: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    k = min(k, values.numel())
    return torch.topk(values, k, largest=False).values.mean()


def max_k(values: torch.Tensor, k: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    k = min(k, values.numel())
    return torch.topk(values, k, largest=True).values.mean()


def _neg_log(p: torch.Tensor) -> torch.Tensor:
    return -torch.log(p.clamp_min(_EPS))


@dataclass
class _TabsRef:
    """Resolved reference for the target-aligned binder RMSD term.

    ``*_ca`` are reference Cα (gemmi.Position) in matched order; ``*_pred_idx``
    select the paired predicted Cα from the concatenated target/binder tokens.
    """

    target_ca: rmsd.Positions
    binder_ca: rmsd.Positions
    target_pred_idx: list[int]
    binder_pred_idx: list[int]


def build_tabs_ref(spec: TargetSpec, weight: float) -> Optional[_TabsRef]:
    """Load + align the reference once, or None when the term is unweighted.

    Raises ``ValueError`` with an actionable message when the term is weighted
    but the reference is missing or inconsistent with the design.
    """
    if not weight:
        return None
    ref = spec.reference
    if ref is None or not ref.path or not ref.target_chains or not ref.binder_chains:
        raise ValueError(
            "Objective weights 'target_aligned_binder_rmsd' but no complete "
            "reference is configured. Add a `reference` block to the spec with "
            "`path`, `target_chains`, and `binder_chains`."
        )
    if not Path(ref.path).exists():
        raise ValueError(f"reference structure not found: {ref.path!r}")

    design_targets = [c for c in spec.chains if c.role == ROLE_TARGET]
    if len(design_targets) != len(ref.target_chains):
        raise ValueError(
            f"reference target_chains {ref.target_chains} must match the "
            f"{len(design_targets)} design target chain(s) "
            f"{[c.id for c in design_targets]}."
        )

    try:
        target_ca: rmsd.Positions = []
        target_pred_idx: list[int] = []
        offset = 0
        for dchain, rchain in zip(design_targets, ref.target_chains):
            rseq, rca = rmsd.load_chain_ca(ref.path, rchain)
            pred_idx, ref_idx = rmsd.matched_columns(dchain.sequence, rseq)
            target_pred_idx += [offset + i for i in pred_idx]
            target_ca += [rca[i] for i in ref_idx]
            offset += len(dchain.sequence)

        scaffold = spec.binder.scaffold
        if scaffold is None:
            scaffold = spec.binder.mask_char * int(spec.binder.denovo_length)
        ref_binder_seq = ""
        ref_binder_ca: rmsd.Positions = []
        for rchain in ref.binder_chains:
            seq, ca = rmsd.load_chain_ca(ref.path, rchain)
            ref_binder_seq += seq
            ref_binder_ca += ca
        binder_pred_idx, ref_idx = rmsd.matched_columns(scaffold, ref_binder_seq)
        binder_ca = [ref_binder_ca[i] for i in ref_idx]
    except KeyError as exc:  # chain named in the spec absent from the file
        raise ValueError(f"reference chain missing: {exc}") from exc

    if not target_ca or not binder_ca:
        raise ValueError(
            "reference alignment produced no matched residues; check that the "
            "reference chains correspond to the design target and binder."
        )
    return _TabsRef(target_ca, binder_ca, target_pred_idx, binder_pred_idx)


@dataclass
class Scorer:
    config: DesignConfig
    discriminating_positions: list[int] = field(default_factory=list)
    contact_residues: dict[str, list[int]] = field(default_factory=dict)
    avoidance_residues: dict[str, list[int]] = field(default_factory=dict)
    iptm_target_id: Optional[str] = None
    mpnn_model: Optional[object] = None
    tabs: Optional[_TabsRef] = None

    def _iptm_target_asym(self, outputs: PredictionOutputs) -> int:
        """Resolve which chain's ipTM-vs-binder is reported by ``term_iptm``."""
        ci = outputs.chain_index
        if self.iptm_target_id is not None:
            return ci.chain_asym[self.iptm_target_id]
        if "peptide" in ci.chain_names:  # legacy pMHC default
            return ci.peptide_asym
        targets = [c for c in ci.chain_ids if ci.chain_roles.get(c) == "target"]
        if len(targets) == 1:
            return ci.chain_asym[targets[0]]
        if not targets:
            raise ValueError("term_iptm: no target chain; set iptm_target_id.")
        return max(targets, key=lambda c: float(outputs.chain_pair_iptm[ci.binder_asym, ci.chain_asym[c]]))  # noqa: E501

    def _contact_prob(self, outputs: PredictionOutputs, cutoff: float) -> torch.Tensor:
        return compute_contact_prob(
            outputs.distogram_logits,
            thres=cutoff,
            **outputs.distogram_bin_params,
        )

    def _epitope_contact_loss(
        self,
        contact: torch.Tensor,
        region_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        if region_tokens.numel() == 0 or query_tokens.numel() == 0:
            return contact.new_zeros(())
        sub = contact[query_tokens][:, region_tokens]  # [|P|, |region|]
        logp = _neg_log(sub)
        per_pep = torch.stack([min_k(logp[r], k) for r in range(logp.shape[0])])
        return per_pep.mean()

    def term_iptm(self, outputs: PredictionOutputs) -> float:
        ci = outputs.chain_index
        return float(outputs.chain_pair_iptm[ci.binder_asym, self._iptm_target_asym(outputs)])

    def term_iptm_global(self, outputs: PredictionOutputs) -> float:
        return float(outputs.iptm_global)

    def interface_pae_stats(self, outputs: PredictionOutputs) -> dict[str, float]:
        ci = outputs.chain_index
        binder = ci.binder_tokens
        target = ci.rest_of_complex_tokens
        if binder.numel() == 0 or target.numel() == 0:
            return {k: 0.0 for k in
                    ("min_ipae", "min_2_ipae", "min_3_ipae", "min_4_ipae", "mean_ipae")}
        pae1 = outputs.pae[target][:, binder].flatten()  # target rows, binder cols
        pae2 = outputs.pae[binder][:, target].flatten()  # binder rows, target cols
        vals = torch.cat([pae1, pae2])
        sorted_vals, _ = torch.sort(vals)

        def mink(k: int) -> float:
            return float(sorted_vals[: min(k, sorted_vals.numel())].mean())

        return {
            "min_ipae": float(sorted_vals[0]),
            "min_2_ipae": mink(2),
            "min_3_ipae": mink(3),
            "min_4_ipae": mink(4),
            "mean_ipae": float(vals.mean()),
        }

    def term_plddt(self, outputs: PredictionOutputs) -> float:  # eq 5b, L_pLDDT
        a = outputs.chain_index.binder_tokens
        return float(1.0 - outputs.per_token_plddt[a].mean())

    def term_i_plddt(self, outputs: PredictionOutputs) -> float:  # eq 5b, L_i-pLDDT
        d = outputs.chain_index.design_tokens
        if d.numel() == 0:
            return 0.0
        return float(1.0 - outputs.per_token_plddt[d].mean())

    def term_i_pae(self, outputs: PredictionOutputs) -> float:  # eq 5c, L_i-pAE
        d = outputs.chain_index.design_tokens
        p = outputs.chain_index.target_tokens
        if d.numel() == 0 or p.numel() == 0:
            return 0.0
        sub = outputs.pae[d][:, p] / self.config.pae_norm
        return float(sub.mean())

    def term_pae_intra(self, outputs: PredictionOutputs) -> float:  # eq 5c, L_pAE-intra
        a = outputs.chain_index.binder_tokens
        sub = outputs.pae[a][:, a] / self.config.pae_norm
        return float(sub.mean())

    def term_rg(self, outputs: PredictionOutputs) -> float:  # eqs 6-8, L_Rg
        a = outputs.chain_index.binder_tokens
        coords = outputs.ca_coords[a]
        n = coords.shape[0]
        center = coords.mean(dim=0)
        rg = torch.sqrt(((coords - center) ** 2).sum(dim=-1).mean())
        rg_expected = self.config.rg_a * (n ** self.config.rg_b)
        return float(F.elu(rg - rg_expected))

    def term_con_intra(self, outputs: PredictionOutputs) -> float:  # eq 9, L_con-intra
        a = outputs.chain_index.binder_tokens
        contact = self._contact_prob(outputs, self.config.intra_contact_cutoff)
        sub = contact[a][:, a]  # [|A|, |A|]
        n = sub.shape[0]
        idx = torch.arange(n, device=sub.device)
        sep = (idx[:, None] - idx[None, :]).abs()
        mask = (sep > self.config.intra_seq_sep) & (idx[:, None] < idx[None, :])
        values = _neg_log(sub[mask])
        return float(min_k(values, self.config.k_con_intra))

    def term_paratope(self, outputs: PredictionOutputs) -> float:  # eq 10, L_paratope
        ci = outputs.chain_index
        contact = self._contact_prob(outputs, self.config.contact_cutoff)
        framework = ci.binder_tokens[
            ~torch.isin(ci.binder_tokens, ci.design_tokens)
        ]
        l_cdr = self._epitope_contact_loss(
            contact, ci.design_tokens, ci.target_tokens, self.config.k_peptide_contact
        )
        l_fw = self._epitope_contact_loss(
            contact, framework, ci.target_tokens, self.config.k_peptide_contact
        )
        denom = max(float(l_fw) - self.config.paratope_lambda, self.config.paratope_eps)
        return float(l_cdr) ** 2 / denom

    def term_target_contact(self, outputs: PredictionOutputs) -> float:  # eq 11
        ci = outputs.chain_index
        contact = self._contact_prob(outputs, self.config.contact_cutoff)
        if self.contact_residues:
            # General path: contact set T given as {chain_id: [res_ids]}.
            t_tokens = ci.contact_tokens(self.contact_residues)
        elif self.discriminating_positions:
            # Legacy pMHC path: local indices into the peptide chain.
            t_tokens = ci.peptide_tokens[
                torch.as_tensor(self.discriminating_positions, device=ci.peptide_tokens.device)
            ]
        else:
            return 0.0
        if t_tokens.numel() == 0:
            return 0.0
        return float(
            self._epitope_contact_loss(
                contact, ci.design_tokens, t_tokens, self.config.k_target_contact
            )
        )

    def term_peptide_contact(self, outputs: PredictionOutputs) -> float:  # eq 12
        ci = outputs.chain_index
        contact = self._contact_prob(outputs, self.config.contact_cutoff)
        return float(
            self._epitope_contact_loss(
                contact, ci.design_tokens, ci.target_tokens, self.config.k_peptide_contact
            )
        )

    def term_repulsion(self, outputs: PredictionOutputs) -> float:  # eq 13
        ci = outputs.chain_index
        if self.avoidance_residues:
            # General path: avoidance set M given as {chain_id: [res_ids]}.
            m_tokens = ci.avoidance_tokens(self.avoidance_residues)
        elif "mhc" in ci.chain_names and self.config.mhc_avoidance_residues:
            # Legacy pMHC path: res_ids on the MHC chain.
            m_tokens = ci.mhc_tokens_for_residues(self.config.mhc_avoidance_residues)
        else:
            return 0.0
        d = ci.design_tokens
        if m_tokens.numel() == 0 or d.numel() == 0:
            return 0.0
        contact = self._contact_prob(outputs, self.config.contact_cutoff)
        sub = contact[m_tokens][:, d]  # [|M|, |D|]
        logp = _neg_log((1.0 - sub))  # -log(1 - P(contact))
        per_m = torch.stack([max_k(logp[r], self.config.k_repulsion) for r in range(logp.shape[0])])
        return float(per_m.mean())

    @staticmethod
    def sigmoid_ddg(ddg_pred: float) -> float:
        return 1.0 / (1.0 + math.exp(-ddg_pred))

    def _mpnn_scored_tokens(self, outputs: PredictionOutputs) -> torch.Tensor:
        ci = outputs.chain_index
        sel = self.config.mpnn_score_chain
        if sel == "target":
            tokens = ci.peptide_tokens if "peptide" in ci.chain_names else ci.target_tokens
        elif sel == "binder":
            tokens = ci.binder_tokens
        elif sel == "design":
            tokens = ci.design_tokens
        else:
            tokens = ci.chain_tokens[sel]
        local = self.config.mpnn_local_indices
        if local is not None and tokens.numel() > 0:
            idx = torch.as_tensor(local, device=tokens.device).long()
            tokens = tokens[idx]
        return tokens

    def term_mpnn(self, outputs: PredictionOutputs) -> float:
        model = self.mpnn_model
        if model is None or outputs.backbone_coords is None or not outputs.sequence:
            return 0.0
        tokens = self._mpnn_scored_tokens(outputs)
        if tokens.numel() == 0:
            return 0.0

        ci = outputs.chain_index
        device = next(model.parameters()).device
        bb = outputs.backbone_coords.to(device).float()  # [N, 4, 3]
        n_token = bb.shape[0]
        if len(outputs.sequence) != n_token:
            return 0.0

        tokens = tokens.to(device).long()
        aa_to_idx = {aa: i for i, aa in enumerate(_MPNN_ALPHABET)}
        S = torch.tensor(
            [aa_to_idx.get(a, 20) for a in outputs.sequence], device=device
        ).long().view(1, -1)
        mask = torch.isfinite(bb).flatten(1).all(dim=1).float().view(1, -1)
        valid = mask[0, tokens] > 0
        tokens = tokens[valid]
        if tokens.numel() == 0:
            return 0.0
        residue_idx = torch.zeros(n_token, device=device, dtype=torch.long)
        chain_encoding = torch.zeros(n_token, device=device, dtype=torch.long)
        for cid in ci.chain_ids:
            tok = ci.chain_tokens[cid].to(device).long()
            if tok.numel() == 0:
                continue
            k = ci.chain_asym[cid]  # true asym ordinal
            residue_idx[tok] = 100 * k + torch.arange(tok.numel(), device=device)
            chain_encoding[tok] = k + 1
        residue_idx = residue_idx.view(1, -1)
        chain_encoding = chain_encoding.view(1, -1)

        chain_M = torch.zeros(n_token, device=device)
        chain_M[tokens] = 1.0
        chain_M = chain_M.view(1, -1)

        X = torch.nan_to_num(bb.unsqueeze(0), nan=0.0)  # [1, N, 4, 3]
        randn = torch.ones(1, n_token, device=device)  # decode-order tiebreak only
        with torch.no_grad():
            log_probs = model.conditional_probs(
                X, S, mask, chain_M, residue_idx, chain_encoding, randn,
                backbone_only=False,
            )

        native = S[0, tokens]
        per_pos = log_probs[0, tokens].gather(1, native.view(-1, 1)).squeeze(1)  # [P]
        if self.config.mpnn_reduction == "mean":
            return float(per_pos.mean())
        return float(per_pos.min())

    def term_target_aligned_binder_rmsd(self, outputs: PredictionOutputs) -> float:
        """Binder Cα RMSD vs a reference pose, target-superposed; ELU-hinged."""
        if self.tabs is None:
            return 0.0
        ci = outputs.chain_index
        pred_target = outputs.ca_coords[ci.target_tokens].tolist()
        pred_binder = outputs.ca_coords[ci.binder_tokens].tolist()
        pt = [gemmi.Position(*pred_target[i]) for i in self.tabs.target_pred_idx]
        pb = [gemmi.Position(*pred_binder[i]) for i in self.tabs.binder_pred_idx]
        value = rmsd.target_aligned_binder_rmsd(
            self.tabs.target_ca, pt, self.tabs.binder_ca, pb
        )
        return float(F.elu(torch.tensor(value - self.config.tabs_tol)))

    def score_terms(
        self, outputs: PredictionOutputs, ddg_pred: Optional[float] = None
    ) -> dict[str, float]:
        """Compute every term. ``ddg_pred`` (raw) is included only if given."""
        terms = {
            "iptm": self.term_iptm(outputs),
            "iptm_global": self.term_iptm_global(outputs),
            "plddt": self.term_plddt(outputs),
            "i_plddt": self.term_i_plddt(outputs),
            "i_pae": self.term_i_pae(outputs),
            "pae_intra": self.term_pae_intra(outputs),
            "con_intra": self.term_con_intra(outputs),
            "rg": self.term_rg(outputs),
            "paratope": self.term_paratope(outputs),
            "target_contact": self.term_target_contact(outputs),
            "peptide_contact": self.term_peptide_contact(outputs),
            "repulsion": self.term_repulsion(outputs),
            "mpnn": self.term_mpnn(outputs),
            "target_aligned_binder_rmsd": self.term_target_aligned_binder_rmsd(outputs),
        }
        terms.update(self.interface_pae_stats(outputs))
        if ddg_pred is not None:
            terms["ddg"] = self.sigmoid_ddg(ddg_pred)
        return terms

    def fitness(self, terms: dict[str, float]) -> float:
        w = self.config.weights
        return sum(
            getattr(w, name) * value
            for name, value in terms.items()
            if hasattr(w, name)
        )
