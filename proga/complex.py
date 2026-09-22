from dataclasses import dataclass
from typing import Optional

from proga.config import NANOBODY_SCAFFOLD

from typing import Final


NANOBODY_CHAIN_ID: Final[str] = "A"
PEPTIDE_CHAIN_ID: Final[str] = "B"
MHC_CHAIN_ID: Final[str] = "C"


@dataclass
class DesignMask:
    scaffold: str
    positions: list[int]
    cdr_spans: list[tuple[int, int]]

    @property
    def length(self) -> int:
        return len(self.scaffold)

    @property
    def span(self) -> tuple[int, int]:
        return (self.positions[0], self.positions[-1] + 1)

    def framework_positions(self) -> list[int]:
        design = set(self.positions)
        return [i for i in range(self.length) if i not in design]


def parse_design_mask(scaffold: str = NANOBODY_SCAFFOLD, mask_char: str = "X") -> DesignMask:
    positions = [i for i, ch in enumerate(scaffold) if ch == mask_char]
    if not positions:
        raise ValueError(f"Scaffold contains no designable ({mask_char!r}) positions.")

    spans: list[tuple[int, int]] = []
    start = prev = positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
            continue
        spans.append((start, prev + 1))
        start = prev = p
    spans.append((start, prev + 1))

    return DesignMask(scaffold=scaffold, positions=positions, cdr_spans=spans)


def apply_sequence(mask: DesignMask, cdr_residues: str) -> str:
    if len(cdr_residues) != len(mask.positions):
        raise ValueError(
            f"Expected {len(mask.positions)} CDR residues, got {len(cdr_residues)}."
        )
    chars = list(mask.scaffold)
    for pos, aa in zip(mask.positions, cdr_residues):
        chars[pos] = aa
    return "".join(chars)


def extract_cdr_residues(mask: DesignMask, nanobody_seq: str) -> str:
    """Inverse of :func:`apply_sequence`: pull the residues at mask positions."""
    return "".join(nanobody_seq[p] for p in mask.positions)


@dataclass
class PMHCTarget:
    """A peptide-MHC pair loaded from a structure / config."""

    name: str
    peptide_seq: str
    mhc_seq: str


def load_pmhc_from_cif(cif_path: str, name: Optional[str] = None) -> PMHCTarget:
    """Load peptide and MHC sequences from a pMHC mmCIF file.

    The shortest protein chain is treated as the peptide and the longest as the
    MHC heavy chain. Uses Protenix's own CIF parser for consistency.
    """
    # Imported lazily so unit tests that don't touch CIFs avoid heavy deps.
    from protenix.data.inference.json_maker import cif_to_input_json

    json_dict = cif_to_input_json(cif_path)
    protein_seqs: list[str] = []
    for entity in json_dict["sequences"]:
        for entity_type, body in entity.items():
            if entity_type == "proteinChain":
                protein_seqs.extend([body["sequence"]] * int(body.get("count", 1)))

    if len(protein_seqs) < 2:
        raise ValueError(
            f"Expected >=2 protein chains (peptide + MHC) in {cif_path}, "
            f"found {len(protein_seqs)}."
        )
    protein_seqs.sort(key=len)
    peptide_seq = protein_seqs[0]
    mhc_seq = protein_seqs[-1]
    if name is None:
        import os

        name = os.path.basename(cif_path).split(".")[0]
    return PMHCTarget(name=name, peptide_seq=peptide_seq, mhc_seq=mhc_seq)


def discriminating_indices(target_peptide: str, off_target_peptide: str) -> list[int]:
    """0-based positions where the on-target and off-target peptides differ.

    These are the residues ``T`` rewarded by the targeted-contact term (eq 11).
    """
    if len(target_peptide) != len(off_target_peptide):
        raise ValueError(
            "Target and off-target peptides must have equal length to align "
            f"({len(target_peptide)} != {len(off_target_peptide)})."
        )
    return [
        i
        for i, (a, b) in enumerate(zip(target_peptide, off_target_peptide))
        if a != b
    ]


def build_sample_dict_n(
    chains: list[tuple[str, str]],
    name: str = "design",
    msa_by_seq: Optional[dict[str, dict]] = None,
) -> dict:
    """Build a Protenix inference input dict for an N-chain complex.

    ``chains`` is an ordered list of ``(chain_id, sequence)`` pairs; the binder
    is conventionally first. Protenix preserves input chain order as ascending
    asym ids, so the assembled order is exactly the order given here.

    ``msa_by_seq`` maps a chain sequence to a dict of precomputed-MSA fields
    (``pairedMsaPath`` / ``unpairedMsaPath``); any chain whose sequence is a key
    gets those fields attached. Chains without an entry (e.g. the varying binder)
    are left MSA-free, which Protenix handles as a self-only MSA.
    """
    msa_by_seq = msa_by_seq or {}

    def chain(seq: str, chain_id: str) -> dict:
        body: dict = {"sequence": seq, "count": 1, "id": [chain_id]}
        for key, value in msa_by_seq.get(seq, {}).items():
            body[key] = value
        return {"proteinChain": body}

    return {
        "name": name,
        "sequences": [chain(seq, cid) for cid, seq in chains],
    }


def build_sample_dict(
    nanobody_seq: str,
    peptide_seq: str,
    mhc_seq: str,
    name: str = "design",
    msa_by_seq: Optional[dict[str, dict]] = None,
) -> dict:
    """Build a Protenix inference input dict for the 3-chain pMHC design complex.

    Thin wrapper over :func:`build_sample_dict_n` preserving the historical
    nanobody/peptide/MHC chain order (A/B/C). Kept for the pMHC code path,
    the oracle placeholder, and existing tests.
    """
    return build_sample_dict_n(
        [
            (NANOBODY_CHAIN_ID, nanobody_seq),
            (PEPTIDE_CHAIN_ID, peptide_seq),
            (MHC_CHAIN_ID, mhc_seq),
        ],
        name=name,
        msa_by_seq=msa_by_seq,
    )
