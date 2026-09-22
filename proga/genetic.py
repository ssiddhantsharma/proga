import random
from dataclasses import dataclass, field, fields
from typing import Callable

from proga.complex import DesignMask
from proga.config import DESIGN_ALPHABET, DesignConfig, ScoreWeights

type EvaluateFn = Callable[[str, bool], tuple[float, dict[str, float]]]
type EvaluateBatchFn = Callable[[list[str], bool], list[tuple[str, float, dict[str, float]]]]


@dataclass
class Individual:
    genome: str
    fitness: float
    terms: dict[str, float] = field(default_factory=dict)
    generation: int = -1


def random_genome(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(DESIGN_ALPHABET) for _ in range(length))


def initialize_population(k: int, length: int, rng: random.Random) -> list[str]:
    return [random_genome(length, rng) for _ in range(k)]


def mutate(genome: str, mutation_rate: float, rng: random.Random) -> str:
    chars = list(genome)
    for i in range(len(chars)):
        if rng.random() < mutation_rate:
            chars[i] = rng.choice(DESIGN_ALPHABET)
    return "".join(chars)


def crossover(g1: str, g2: str, mask: DesignMask, rng: random.Random) -> str:
    start, end = mask.span
    breakpoint_ = rng.randint(start, end)  # inclusive of both ends
    return "".join(
        g1[i] if pos < breakpoint_ else g2[i]
        for i, pos in enumerate(mask.positions)
    )


def top_k(population: list[Individual], k: int) -> list[Individual]:
    return sorted(population, key=lambda ind: ind.fitness, reverse=True)[:k]


# --------------------------------------------------------------------------- #
# Multi-objective (NSGA-II) selection                                         #
#                                                                             #
# All objective axes are expressed as "higher is better" (the axis value is   #
# ``signed_weight * term``, so the weight's sign fixes the direction and its  #
# magnitude is irrelevant to domination). Selection is deterministic — no RNG #
# — so ``run_resumable`` reproduces exactly across resume, same as scalar.    #
# --------------------------------------------------------------------------- #
def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """True iff ``a`` Pareto-dominates ``b`` (>= on every axis, > on at least one)."""
    better_any = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            better_any = True
    return better_any


def non_dominated_fronts(objs: list[tuple[float, ...]]) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al. 2002). Returns fronts of indices."""
    n = len(objs)
    dominated: list[list[int]] = [[] for _ in range(n)]  # who each i dominates
    ndom_count = [0] * n  # how many dominate i
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                dominated[p].append(q)
            elif _dominates(objs[q], objs[p]):
                ndom_count[p] += 1
        if ndom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                ndom_count[q] -= 1
                if ndom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()  # trailing empty front
    return fronts


def crowding_distance(
    objs: list[tuple[float, ...]], front: list[int]
) -> dict[int, float]:
    """Per-axis-normalised crowding distance; boundary points get +inf."""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    m = len(objs[0])
    for k in range(m):
        order = sorted(front, key=lambda i: objs[i][k])
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        lo, hi = objs[order[0]][k], objs[order[-1]][k]
        span = hi - lo
        if span <= 0:
            continue
        for j in range(1, len(order) - 1):
            dist[order[j]] += (objs[order[j + 1]][k] - objs[order[j - 1]][k]) / span
    return dist


def pareto_select(
    population: list[Individual],
    k: int,
    objective: Callable[[Individual], tuple[float, ...]],
) -> list[Individual]:
    """NSGA-II environmental selection: fill by front, then crowding distance.

    Within a front, ties break on scalar ``fitness`` (desc) so the choice is
    deterministic and never regresses the aggregate objective. Returns up to
    ``k`` individuals in rank-major order (usable for both culling and output).
    """
    objs = [objective(ind) for ind in population]
    ordered: list[int] = []
    for front in non_dominated_fronts(objs):
        dist = crowding_distance(objs, front)
        ordered.extend(
            sorted(front, key=lambda i: (dist[i], population[i].fitness), reverse=True)
        )
    return [population[i] for i in ordered[:k]]


@dataclass
class GAResult:
    top: list[Individual]
    best_per_generation: list[float]
    num_evaluations: int


class GeneticOptimizer:
    def __init__(self, config: DesignConfig, mask: DesignMask):
        self.config = config
        self.mask = mask
        self.rng = random.Random(config.seed)
        self._seen: dict[str, Individual] = {}
        self._pareto_axes: list[str] = (
            self._resolve_pareto_axes() if config.selection == "pareto" else []
        )

    def _resolve_pareto_axes(self) -> list[str]:
        w = self.config.weights
        names = self.config.pareto_objectives or [
            f.name for f in fields(ScoreWeights) if getattr(w, f.name) != 0.0
        ]
        axes = [n for n in names if getattr(w, n) != 0.0]  # need a defined direction
        if not axes:
            raise ValueError(
                "selection='pareto' requires >=1 objective with a non-zero weight; "
                "set config.pareto_objectives or give the terms non-zero weights."
            )
        return axes

    def _pareto_objective(self, ind: Individual) -> tuple[float, ...]:
        """Signed-weight * term for each axis, so every axis is 'higher is better'."""
        w = self.config.weights
        return tuple(
            getattr(w, name) * ind.terms.get(name, 0.0) for name in self._pareto_axes
        )

    def select(self, population: list[Individual], k: int) -> list[Individual]:
        """Pick ``k`` survivors: Pareto (NSGA-II) or scalar top-k per config."""
        if self.config.selection == "pareto":
            return pareto_select(population, k, self._pareto_objective)
        return top_k(population, k)

    def _record(self, genome: str, fitness: float, terms: dict, generation: int):
        prev = self._seen.get(genome)
        if prev is None or fitness > prev.fitness:
            self._seen[genome] = Individual(genome, fitness, terms, generation)

    def run(self, evaluate: EvaluateFn) -> GAResult:
        def evaluate_batch(genomes: list[str], use_ddg: bool):
            return [(g, *evaluate(g, use_ddg)) for g in genomes]
        return self.run_resumable(evaluate_batch)

    def _eval_batch_into(
        self,
        evaluate_batch: EvaluateBatchFn,
        genomes: list[str],
        generation: int,
        use_ddg: bool,
        num_eval: int,
    ) -> tuple[list[Individual], int]:
        to_eval: list[str] = []
        queued: set[str] = set()
        for g in genomes:
            cached = self._seen.get(g)
            if cached is not None and ("ddg" in cached.terms) == use_ddg:
                continue
            if g in queued:
                continue
            queued.add(g)
            to_eval.append(g)

        if to_eval:
            for genome, fitness, terms in evaluate_batch(to_eval, use_ddg):
                num_eval += 1
                self._record(genome, fitness, terms, generation)

        return [self._seen[g] for g in genomes], num_eval

    def snapshot(
        self,
        generation: int,
        population: list[Individual],
        best_per_gen: list[float],
        num_eval: int,
    ) -> dict:
        return {
            "generation": generation,
            "rng_state": self.rng.getstate(),
            "population_genomes": [ind.genome for ind in population],
            "seen": {g: [i.fitness, i.terms, i.generation] for g, i in self._seen.items()},
            "best_per_generation": list(best_per_gen),
            "num_evaluations": num_eval,
        }

    def _restore(self, state: dict) -> None:
        self._seen = {
            g: Individual(g, f, terms, gen) for g, (f, terms, gen) in state["seen"].items()
        }
        rs = state["rng_state"]
        self.rng.setstate((rs[0], tuple(rs[1]), rs[2]))

    def run_resumable(
        self,
        evaluate_batch: EvaluateBatchFn,
        checkpoint_cb: Callable[[dict], None] | None = None,
        resume_state: dict | None = None,
    ) -> GAResult:
        cfg = self.config
        length = len(self.mask.positions)

        if resume_state is not None:
            self._restore(resume_state)
            best_per_gen = list(resume_state["best_per_generation"])
            num_eval = resume_state["num_evaluations"]
            population = [self._seen[g] for g in resume_state["population_genomes"]]
            start_gen = resume_state["generation"] + 1
        else:
            # --- initialise P_0 (Algorithm 1, line 1) ----------------------
            init_genomes = initialize_population(cfg.population, length, self.rng)
            population, num_eval = self._eval_batch_into(
                evaluate_batch, init_genomes, 0, False, 0
            )
            best_per_gen = [max(ind.fitness for ind in population)]
            start_gen = 1
            if checkpoint_cb is not None:
                checkpoint_cb(self.snapshot(0, population, best_per_gen, num_eval))

        # --- evolutionary loop (lines 2-13) --------------------------------
        for t in range(start_gen, cfg.generations + 1):
            use_ddg = t >= cfg.t_ddg
            parents = top_k(population, cfg.population)  # P <- TopK(P_{t-1})

            offspring_genomes: list[str] = []
            # Mutation: one mutant per parent (lines 4-6).
            for p in parents:
                offspring_genomes.append(mutate(p.genome, cfg.mutation_rate, self.rng))
            # Crossover: floor(K * p_c) parent pairs (lines 7-10).
            n_cross = int(cfg.population * cfg.crossover_rate)
            for _ in range(n_cross):
                p1, p2 = self.rng.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
                offspring_genomes.append(crossover(p1.genome, p2.genome, self.mask, self.rng))

            # Subsample O to |O| = K (line 11).
            if len(offspring_genomes) > cfg.population:
                offspring_genomes = self.rng.sample(offspring_genomes, cfg.population)

            offspring, num_eval = self._eval_batch_into(
                evaluate_batch, offspring_genomes, t, use_ddg, num_eval
            )

            # P_t <- Select(P ∪ O) (line 12) — scalar top-k or Pareto (NSGA-II).
            population = self.select(parents + offspring, cfg.population)
            best_per_gen.append(max(ind.fitness for ind in population))
            if checkpoint_cb is not None:
                checkpoint_cb(self.snapshot(t, population, best_per_gen, num_eval))

        # Output top-n unique sequences across the full trajectory (Sec. 3).
        if cfg.selection == "pareto":
            top = pareto_select(
                list(self._seen.values()), cfg.top_n, self._pareto_objective
            )
        else:
            top = sorted(
                self._seen.values(), key=lambda ind: ind.fitness, reverse=True
            )[: cfg.top_n]
        return GAResult(top=top, best_per_generation=best_per_gen, num_evaluations=num_eval)
