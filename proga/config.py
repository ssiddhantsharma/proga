from dataclasses import dataclass, field, fields
from typing import Final, Self

DESIGN_ALPHABET: Final[str] = "ADEFGHIKLMNPQRSTVWY"

NANOBODY_SCAFFOLD: Final[str] = (
    "EVQLVESGGGLVQPGGSLRLSCAAS"
    # CDR1 (9 positions)
    "XXXXXXXXX"
    "MGWFRQAPGKGRELVAA"
    # CDR2 (9 positions)
    "XXXXXXXXXX"
    "YYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYC"
    # CDR3 (33 positions)
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    "WGQGTQVTVSS"
)


@dataclass
class ScoreWeights:
    iptm: float = 0.7
    iptm_global: float = 0.0
    # ipSAE (Dunbar d0res) between binder and target: the interface metric that separates true
    # de-novo binders best. Inert by default (0.0); the `iptm_ipsae` objective turns it on.
    ipsae: float = 0.0
    plddt: float = -1.0
    i_plddt: float = -1.0
    i_pae: float = -0.5
    pae_intra: float = -0.1
    con_intra: float = -0.1
    rg: float = -0.1
    paratope: float = -0.2
    target_contact: float = -0.5
    peptide_contact: float = -0.5
    repulsion: float = -0.5
    ddg: float = 0.7
    min_ipae: float = 0.0
    min_2_ipae: float = 0.0
    min_3_ipae: float = 0.0
    min_4_ipae: float = 0.0
    mean_ipae: float = 0.0
    # ProteinMPNN peptide log-likelihood (higher = more native-like). Positive
    # weight rewards better peptide recognition. Inert by default (0.0); only
    # the dedicated `iptm_ipae_target_peptide_mpnn` objective turns it on.
    mpnn: float = 0.0
    # Target-aligned binder RMSD to a reference pose. Negative weight drives the
    # binder toward the reference conformation. Inert by default (0.0); a
    # non-zero weight requires a `reference` block in the spec.
    target_aligned_binder_rmsd: float = 0.0
    # Nesso-1 scalar affinity (fold-free oracle). Maximised with a positive weight.
    affinity: float = 0.0

    @classmethod
    def affinity_only(cls, affinity: float = 1.0) -> Self:
        kwargs = {f.name: 0.0 for f in fields(cls)}
        kwargs["affinity"] = affinity
        return cls(**kwargs)

    @classmethod
    def iptm_ipae(cls, target_contact: float = -0.5) -> Self:
        kwargs = {f.name: 0.0 for f in fields(cls)}
        kwargs["iptm_global"] = 1.0
        kwargs["mean_ipae"] = -1.0 / 31.0
        kwargs["target_contact"] = target_contact
        return cls(**kwargs)

    @classmethod
    def iptm_ipsae(cls, ipsae: float = 1.0, target_contact: float = -0.5) -> Self:
        """Rank primarily on ipSAE (Dunbar d0res), with the usual ipae penalty and target-contact
        steer. ipSAE separates true de-novo binders better than ipTM alone."""
        kwargs = {f.name: 0.0 for f in fields(cls)}
        kwargs["ipsae"] = ipsae
        kwargs["mean_ipae"] = -1.0 / 31.0
        kwargs["target_contact"] = target_contact
        return cls(**kwargs)

    @classmethod
    def mpnn_only(cls, mpnn: float = 1.0) -> Self:
        kwargs = {f.name: 0.0 for f in fields(cls)}
        kwargs["mpnn"] = mpnn
        return cls(**kwargs)

    @classmethod
    def iptm_ipae_mpnn(cls, target_contact: float = -0.5, mpnn: float = 0.1) -> Self:
        kwargs = {f.name: 0.0 for f in fields(cls)}
        kwargs["iptm_global"] = 1.0
        kwargs["mean_ipae"] = -1.0 / 31.0
        kwargs["target_contact"] = target_contact
        kwargs["mpnn"] = mpnn
        return cls(**kwargs)

    @classmethod
    def iptm_ipae_target_peptide(cls, target_contact: float = -0.5, peptide_contact: float = -0.1) -> Self:
        kwargs = {f.name: 0.0 for f in fields(cls)}
        kwargs["iptm_global"] = 1.0
        kwargs["mean_ipae"] = -1.0 / 31.0
        kwargs["target_contact"] = target_contact
        kwargs["peptide_contact"] = peptide_contact
        return cls(**kwargs)

    @classmethod
    def iptm_ipae_target_peptide_mpnn(cls, target_contact: float = -0.5, peptide_contact: float = -0.1, mpnn: float = 0.1) -> Self:
        kwargs = {f.name: 0.0 for f in fields(cls)}
        kwargs["iptm_global"] = 1.0
        kwargs["mean_ipae"] = -1.0 / 31.0
        kwargs["target_contact"] = target_contact
        kwargs["peptide_contact"] = peptide_contact
        kwargs["mpnn"] = mpnn
        return cls(**kwargs)

    @classmethod
    def iptm_ipae_repulsion(
        cls, target_contact: float = -0.5, repulsion: float = -0.5
    ) -> Self:
        w = cls.iptm_ipae(target_contact=target_contact)
        w.repulsion = repulsion
        return w

    @classmethod
    def iptm_ipae_paratope(
        cls, target_contact: float = -0.5, paratope: float = -0.2
    ) -> Self:
        w = cls.iptm_ipae(target_contact=target_contact)
        w.paratope = paratope
        return w

    @classmethod
    def iptm_ipae_tabs(
        cls, target_contact: float = -0.5, target_aligned_binder_rmsd: float = -0.1
    ) -> Self:
        w = cls.iptm_ipae(target_contact=target_contact)
        w.target_aligned_binder_rmsd = target_aligned_binder_rmsd
        return w


@dataclass
class DesignConfig:
    # Should be set near 100 for a real run
    population: int = 8
    # Should be set near 80 for a real run
    generations: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    top_n: int = 10
    t_ddg: int = 40
    seed: int = 0

    weights: ScoreWeights = field(default_factory=ScoreWeights)

    # Selection strategy for survivors and final output.
    #   "scalar" (default): rank by the weighted-sum ``ScoreWeights`` fitness
    #     (existing behaviour — byte-identical when left unset).
    #   "pareto": NSGA-II non-dominated sorting + crowding distance over
    #     ``pareto_objectives``, so trade-offs between competing objectives
    #     (e.g. on-target iptm vs off-target repulsion) are preserved on a
    #     Pareto front instead of collapsed by a pre-committed weight ratio.
    selection: str = "scalar"
    # Term names used as independent Pareto axes when ``selection == "pareto"``.
    # Empty -> every term with a non-zero configured weight. The preference
    # DIRECTION of each axis is taken from the SIGN of that term's weight (so a
    # design is "better" on an axis in exactly the direction the weighted sum
    # already prefers); the weight MAGNITUDE is irrelevant to domination.
    pareto_objectives: list[str] = field(default_factory=list)

    contact_cutoff: float = 8.0
    intra_contact_cutoff: float = 14.0
    intra_seq_sep: int = 6
    ipsae_pae_cutoff: float = 10.0  # PAE cutoff (A) for the ipSAE term (Dunbar default)

    k_target_contact: int = 2
    k_peptide_contact: int = 2
    k_con_intra: int = 2
    k_repulsion: int = 2

    pae_norm: float = 31.0

    rg_a: float = 2.38
    rg_b: float = 0.365

    # Target-aligned binder RMSD (term "target_aligned_binder_rmsd"): ELU hinge
    # tolerance in Å (no penalty below this).
    tabs_tol: float = 2.5

    paratope_lambda: float = 0.0
    paratope_eps: float = 1e-3

    mhc_avoidance_residues: list[int] = field(
        default_factory=lambda: list(range(182, 275))
    )

    model_name: str = "protenix-v2"
    n_cycle: int = 10
    n_step: int = 200
    oracle_backend: str = "protenix"   # registered oracle (see proga.oracle_base); add more via register_oracle
    boltz_accelerator: str = "gpu"  # boltz2 oracle: gpu|cpu
    # tt-bio (Tenstorrent) oracle knobs (oracle_backend="ttbio"):
    ttbio_model: str = "boltz2"   # boltz2|protenix-v2|opendde-abag|rf3|af2ig
    ttbio_launcher: str = ""      # launcher path; "" -> $TTBIO or /home/bizon/ttbio
    ttbio_extra_args: list[str] = field(default_factory=list)  # extra `tt-bio predict` flags
    # Nesso-1 protein-ligand affinity oracle (oracle_backend="nesso1"):
    nesso1_launcher: str = ""      # launcher path; "" -> $TTBIO or tt-bio
    nesso1_ligand_id: str = "LIG"
    nesso1_ligand_smiles: str = ""
    nesso1_ligand_ccd: str = ""
    nesso1_affinity_key: str = "affinity_probability_binary"  # or affinity_pred_value
    nesso1_extra_args: list[str] = field(default_factory=list)
    n_sample: int = 1
    dtype: str = "bf16"
    use_msa: bool = True # Only for the target
    seeds: list[int] = field(default_factory=lambda: [101])

    msa_cache_dir: str = "~/.cache/protenix_design_msa"
    msa_mode: str = "protenix"
    msa_for_peptide: bool = False

    # ProteinMPNN score (term "mpnn"). Scored chain selection, reduction over
    # positions, optional within-chain index restriction, and the checkpoint.
    mpnn_score_chain: str = "target"  # "target" | "binder" | "design" | chain id
    mpnn_reduction: str = "min"  # "min" | "mean" over scored positions
    mpnn_local_indices: list[int] | None = None  # within-chain indices, e.g. discriminating positions
    mpnn_weights_name: str = "v_48_020"

    def num_oracle_calls(self) -> int:
        return self.population * (self.generations + 1)
