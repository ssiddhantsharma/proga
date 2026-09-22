import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from proga.complex import (
    DesignMask,
    discriminating_indices,
    load_pmhc_from_cif,
    parse_design_mask,
)
from proga.config import DESIGN_ALPHABET, NANOBODY_SCAFFOLD

from typing import Self

ROLE_TARGET = "target"
ROLE_CONTEXT = "context"
ROLE_BINDER = "binder"


@dataclass
class FixedChain:
    id: str
    sequence: str
    role: str = ROLE_TARGET
    name: str = ""
    contact_residues: list[int] = field(default_factory=list)
    avoidance_residues: list[int] = field(default_factory=list)


@dataclass
class BinderSpec:
    id: str = "A"
    scaffold: Optional[str] = None
    mask_char: str = "X"
    denovo_length: Optional[int] = None
    alphabet: str = DESIGN_ALPHABET
    name: str = "binder"

    def __post_init__(self) -> None:
        if (self.scaffold is None) == (self.denovo_length is None):
            raise ValueError("BinderSpec needs exactly one of scaffold or denovo_length.")

    def mask(self) -> DesignMask:
        """The :class:`DesignMask` for this binder (all positions for de-novo)."""
        if self.scaffold is not None:
            return parse_design_mask(self.scaffold, mask_char=self.mask_char)
        scaffold = self.mask_char * int(self.denovo_length)
        return parse_design_mask(scaffold, mask_char=self.mask_char)


@dataclass
class OffTarget:
    name: str
    chains: list[FixedChain]
    aligned_to: dict[str, str] = field(default_factory=dict)


@dataclass
class ReferenceSpec:
    """A reference complex whose binding conformation designs should reproduce.

    ``path`` is a PDB/CIF file; ``target_chains``/``binder_chains`` name the
    chains *within that file* that play the target and binder roles. They are
    paired positionally with the design's own target chains (role == target)
    and binder chain, so a design target superposed onto the reference target
    carries its binder into the reference frame for the RMSD (see
    :mod:`proga.rmsd`).
    """

    path: str
    target_chains: list[str] = field(default_factory=list)
    binder_chains: list[str] = field(default_factory=list)


@dataclass
class TargetSpec:
    name: str
    binder: BinderSpec
    chains: list[FixedChain]
    off_targets: list[OffTarget] = field(default_factory=list)
    msa_chain_ids: list[str] = field(default_factory=list)
    iptm_target_id: Optional[str] = None
    discriminating_positions: list[int] = field(default_factory=list)
    reference: Optional[ReferenceSpec] = None

    def roles(self) -> dict[str, str]:
        roles = {self.binder.id: ROLE_BINDER}
        roles.update({c.id: c.role for c in self.chains})
        return roles

    def chain_names(self) -> dict[str, str]:
        return {c.name: c.id for c in self.chains if c.name}

    def assembly_chains(self, binder_seq: str) -> list[tuple[str, str]]:
        return [(self.binder.id, binder_seq)] + [(c.id, c.sequence) for c in self.chains]

    def offtarget_assembly_chains(
        self, off_target: OffTarget, binder_seq: str
    ) -> list[tuple[str, str]]:
        return [(self.binder.id, binder_seq)] + [(c.id, c.sequence) for c in off_target.chains]

    def offtarget_roles(self, off_target: OffTarget) -> dict[str, str]:
        roles = {self.binder.id: ROLE_BINDER}
        roles.update({c.id: c.role for c in off_target.chains})
        return roles

    def contact_residues_map(self) -> dict[str, list[int]]:
        return {c.id: list(c.contact_residues) for c in self.chains if c.contact_residues}

    def avoidance_residues_map(self) -> dict[str, list[int]]:
        return {c.id: list(c.avoidance_residues) for c in self.chains if c.avoidance_residues}

    def target_chain_ids(self) -> list[str]:
        return [c.id for c in self.chains if c.role == ROLE_TARGET]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        binder = BinderSpec(**d["binder"])
        chains = [FixedChain(**c) for c in d["chains"]]
        off_targets = [
            OffTarget(
                name=o["name"],
                chains=[FixedChain(**c) for c in o["chains"]],
                aligned_to=o.get("aligned_to", {}),
            )
            for o in d.get("off_targets", [])
        ]
        ref = d.get("reference")
        reference = ReferenceSpec(**ref) if ref else None
        return cls(
            name=d["name"],
            binder=binder,
            chains=chains,
            off_targets=off_targets,
            msa_chain_ids=d.get("msa_chain_ids", []),
            iptm_target_id=d.get("iptm_target_id"),
            discriminating_positions=d.get("discriminating_positions", []),
            reference=reference,
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> Self:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_file(cls, path: str) -> Self:
        if path.endswith((".yaml", ".yml")):
            import yaml

            with open(path) as f:
                return cls.from_dict(yaml.safe_load(f))
        return cls.from_json(path)


def pmhc_spec_from_cifs(
    target_cif: str,
    off_target_cif: str,
    scaffold: str = NANOBODY_SCAFFOLD,
    mhc_avoidance_residues: Optional[list[int]] = None,
) -> TargetSpec:
    if mhc_avoidance_residues is None:
        mhc_avoidance_residues = list(range(182, 275))
    target = load_pmhc_from_cif(target_cif, name="target")
    off = load_pmhc_from_cif(off_target_cif, name="off_target")
    disc = discriminating_indices(target.peptide_seq, off.peptide_seq)  # 0-based local
    binder = BinderSpec(id="A", scaffold=scaffold, name="nanobody")
    chains = [
        FixedChain(
            id="B",
            sequence=target.peptide_seq,
            role=ROLE_TARGET,
            name="peptide",
            contact_residues=[i + 1 for i in disc],  # local 0-based -> 1-based res_id
        ),
        FixedChain(
            id="C",
            sequence=target.mhc_seq,
            role=ROLE_CONTEXT,
            name="mhc",
            avoidance_residues=list(mhc_avoidance_residues),
        ),
    ]
    off_targets = [
        OffTarget(
            name="off_target",
            chains=[
                FixedChain(id="B", sequence=off.peptide_seq, role=ROLE_TARGET, name="peptide"),
                FixedChain(id="C", sequence=off.mhc_seq, role=ROLE_CONTEXT, name="mhc"),
            ],
            aligned_to={"B": "B"},
        )
    ]
    return TargetSpec(
        name=target.name,
        binder=binder,
        chains=chains,
        off_targets=off_targets,
        msa_chain_ids=["C"],
        iptm_target_id="B",
        discriminating_positions=disc,
    )
