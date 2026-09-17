"""gemmi-based superposition helpers (no hand-rolled Kabsch)."""

import math
from pathlib import Path

import gemmi

Positions = list[gemmi.Position]


def load_chain_ca(path: str | Path, chain_id: str) -> tuple[str, Positions]:
    """One-letter sequence and ordered Cα positions for a chain (1:1)."""
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"no models in {path}")
    for chain in structure[0]:
        if chain.name != chain_id:
            continue
        seq: list[str] = []
        cas: Positions = []
        for res in chain:
            atom = res.find_atom("CA", "*")
            if atom is None:
                continue
            code = gemmi.find_tabulated_residue(res.name)
            seq.append(code.one_letter_code.upper() if code else "X")
            cas.append(atom.pos)
        if not cas:
            raise ValueError(f"chain {chain_id!r} in {path} has no Cα atoms")
        return "".join(seq), cas
    raise KeyError(f"chain {chain_id!r} not found in {path}")


def matched_columns(seq_a: str, seq_b: str) -> tuple[list[int], list[int]]:
    """Aligned-column index pairs of two sequences (gap columns dropped)."""
    alignment = gemmi.align_string_sequences(list(seq_a), list(seq_b), [])
    gapped_a = alignment.add_gaps(seq_a, 1)
    gapped_b = alignment.add_gaps(seq_b, 2)
    idx_a: list[int] = []
    idx_b: list[int] = []
    ia = ib = 0
    for ca, cb in zip(gapped_a, gapped_b):
        if ca != "-" and cb != "-":
            idx_a.append(ia)
            idx_b.append(ib)
        ia += ca != "-"
        ib += cb != "-"
    return idx_a, idx_b


def target_aligned_binder_rmsd(
    ref_target_ca: Positions,
    pred_target_ca: Positions,
    ref_binder_ca: Positions,
    pred_binder_ca: Positions,
) -> float:
    """Binder Cα RMSD after superposing the predicted target onto the reference.

    All four lists must already be index-matched (reference/predicted paired
    residue-for-residue).
    """
    if not ref_target_ca or not ref_binder_ca:
        return 0.0
    transform = gemmi.superpose_positions(ref_target_ca, pred_target_ca).transform
    total = 0.0
    for pred, ref in zip(pred_binder_ca, ref_binder_ca):
        moved = transform.apply(pred)
        total += (moved.x - ref.x) ** 2 + (moved.y - ref.y) ** 2 + (moved.z - ref.z) ** 2
    return math.sqrt(total / len(ref_binder_ca))
