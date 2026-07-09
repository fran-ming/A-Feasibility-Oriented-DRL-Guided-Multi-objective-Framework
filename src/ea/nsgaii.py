import copy
import math
import random

@dataclass
class Individual:
    """
    实数编码个体：
    genome[i] = task i 的优先级分数（越小越先执行）
    """
    genome: List[float]
    objectives: Optional[Tuple[float, float, float,float]] = None
    eval_result: Optional[EvalResult] = None

    rank: Optional[int] = None
    crowding_distance: float = 0.0

    # 可追踪
    meta: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "Individual":
        return Individual(
            genome=self.genome[:],
            objectives=self.objectives,
            eval_result=self.eval_result,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            meta=copy.deepcopy(self.meta),
        )

def dominates(a: Individual, b: Individual) -> bool:
    assert a.objectives is not None and b.objectives is not None
    le_all = True
    lt_any = False
    for x, y in zip(a.objectives, b.objectives):  # 自动适应长度
        if x > y:
            le_all = False
            break
        if x < y:
            lt_any = True
    return le_all and lt_any


def fast_non_dominated_sort(pop: List[Individual]) -> List[List[int]]:
    """
    返回 fronts: 每个 front 是个体索引列表
    """
    n = len(pop)
    S = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    pop[q].rank = i + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts


def assign_crowding_distance(pop: List[Individual], front: List[int]) -> None:
    """
    NSGA-II crowding distance
    """
    if not front:
        return

    m = len(pop[front[0]].objectives)
    for idx in front:
        pop[idx].crowding_distance = 0.0

    if len(front) <= 2:
        for idx in front:
            pop[idx].crowding_distance = INF
        return

    for obj_i in range(m):
        sorted_front = sorted(front, key=lambda idx: pop[idx].objectives[obj_i])
        pop[sorted_front[0]].crowding_distance = INF
        pop[sorted_front[-1]].crowding_distance = INF

        fmin = pop[sorted_front[0]].objectives[obj_i]
        fmax = pop[sorted_front[-1]].objectives[obj_i]

        if abs(fmax - fmin) < 1e-12:
            continue

        for k in range(1, len(sorted_front) - 1):
            prev_val = pop[sorted_front[k - 1]].objectives[obj_i]
            next_val = pop[sorted_front[k + 1]].objectives[obj_i]
            pop[sorted_front[k]].crowding_distance += (next_val - prev_val) / (fmax - fmin)


def crowded_compare(a: Individual, b: Individual) -> bool:
    """
    比较拥挤度
    """
    if a.rank < b.rank:
        return True
    if a.rank > b.rank:
        return False
    return a.crowding_distance > b.crowding_distance


def binary_tournament(pop: List[Individual], rng: random.Random, k: int = 2) -> Individual:
    """
    k-way tournament (默认2)
    """
    cand = rng.sample(pop, min(k, len(pop)))
    best = cand[0]
    for c in cand[1:]:
        if crowded_compare(c, best):
            best = c
    return best


def evaluate_individual(
    ind: Individual,
    scenario: Scenario,
    spm: ShortestPathManager,
) -> None:
    """
    评估个体
    """
    ev = evaluate_priority_scores(scenario, spm, ind.genome)
    ind.eval_result = ev
    ind.objectives = ev.objectives


def evaluate_population(
    pop: List[Individual],
    scenario: Scenario,
    spm: ShortestPathManager,
) -> None:
    for ind in pop:
        if ind.objectives is None:
            evaluate_individual(ind, scenario, spm)


def sbx_crossover(
    p1: Individual,
    p2: Individual,
    rng: random.Random,
    eta_c: float = 15.0,
) -> Tuple[Individual, Individual]:
    """
    实数编码常用，适合priority vector
    """
    n = len(p1.genome)
    c1 = p1.genome[:]
    c2 = p2.genome[:]

    for i in range(n):
        x1 = p1.genome[i]
        x2 = p2.genome[i]

        if rng.random() <= 0.5 and abs(x1 - x2) > 1e-14:
            xl = -1.0
            xu = 1.0
            if x1 > x2:
                x1, x2 = x2, x1

            rand = rng.random()

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - pow(beta, -(eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = pow(rand * alpha, 1.0 / (eta_c + 1.0))
            else:
                betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (eta_c + 1.0))
            child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - pow(beta, -(eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = pow(rand * alpha, 1.0 / (eta_c + 1.0))
            else:
                betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (eta_c + 1.0))
            child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

            child1 = min(max(child1, xl), xu)
            child2 = min(max(child2, xu if False else xl), xu)

            if rng.random() <= 0.5:
                c1[i] = child2
                c2[i] = child1
            else:
                c1[i] = child1
                c2[i] = child2

    return Individual(genome=c1), Individual(genome=c2)


def gaussian_mutation(
    ind: Individual,
    rng: random.Random,
    mutation_rate: float,
    sigma: float,
) -> Individual:
    """
    高斯变异 + 截断到 [-1,1]
    """
    g = ind.genome[:]
    for i in range(len(g)):
        if rng.random() < mutation_rate:
            g[i] += rng.gauss(0.0, sigma)
            g[i] = min(1.0, max(-1.0, g[i]))
    return Individual(genome=g)


def create_random_individual(num_tasks: int, rng: random.Random) -> Individual:
    genome = [rng.uniform(-1.0, 1.0) for _ in range(num_tasks)]
    return Individual(genome=genome, meta={"source": "random"})


def environmental_selection(pop_union: List[Individual], pop_size: int) -> List[Individual]:
    """
    NSGA-II 环境选择
    """
    fronts = fast_non_dominated_sort(pop_union)
    next_pop: List[Individual] = []

    for front in fronts:
        assign_crowding_distance(pop_union, front)
        if len(next_pop) + len(front) <= pop_size:
            next_pop.extend([pop_union[idx].clone() for idx in front])
        else:
            sorted_front = sorted(
                front,
                key=lambda idx: (
                    pop_union[idx].rank,
                    -pop_union[idx].crowding_distance
                )
            )
            remain = pop_size - len(next_pop)
            next_pop.extend([pop_union[idx].clone() for idx in sorted_front[:remain]])
            break

    return next_pop


def get_pareto_front(pop: List[Individual]) -> List[Individual]:
    """
    返回 rank=0 前沿
    """
    tmp = [p.clone() for p in pop]
    fronts = fast_non_dominated_sort(tmp)
    if not fronts:
        return []
    front0 = fronts[0]
    assign_crowding_distance(tmp, front0)
    res = [tmp[i].clone() for i in front0]
    return res


def select_best_by_scalarization(
    pop: List[Individual],
    weights: Tuple[float, float, float,float] = (100, 10, 1,0.05),
) -> Optional[Individual]:
    """
    从一组解中挑一个“代表最好解”
    仅用于日志/单解展示，不影响多目标流程
    """
    if not pop:
        return None

    # 先做简单 min-max normalize
    objs = list(zip(*[ind.objectives for ind in pop]))
    mins = [min(x) for x in objs]
    maxs = [max(x) for x in objs]

    def score(ind: Individual) -> float:
        s = 0.0
        for i, w in enumerate(weights):
            lo, hi = mins[i], maxs[i]
            val = ind.objectives[i]
            norm = 0.0 if abs(hi - lo) < 1e-12 else (val - lo) / (hi - lo)
            s += w * norm
        return s

    return min(pop, key=score)


class NSGAII:
    """
    标准 NSGA-II 基线
    """
    def __init__(
        self,
        pop_size: int = NSGA2_POP_SIZE,
        num_generations: int = NSGA2_NUM_GENERATIONS,
        crossover_rate: float = NSGA2_CROSSOVER_RATE,
        mutation_rate: float = NSGA2_MUTATION_RATE,
        mutation_sigma: float = NSGA2_MUTATION_SIGMA,
        tournament_k: int = NSGA2_TOURNAMENT_K,
        seed: int = 42,
    ):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_k = tournament_k
        self.seed = seed
        self.rng = random.Random(seed)

    def initialize_population(self, num_tasks: int) -> List[Individual]:
        return [create_random_individual(num_tasks, self.rng) for _ in range(self.pop_size)]

    def make_offspring(
        self,
        population: List[Individual],
        scenario: Scenario,
        spm: ShortestPathManager,
    ) -> List[Individual]:
        """
        生成子代
        """
        # 先确保 parent 已经有 rank/crowding
        tmp = [p.clone() for p in population]
        fronts = fast_non_dominated_sort(tmp)
        for front in fronts:
            assign_crowding_distance(tmp, front)

        # 把 rank/crowding 回写到原对象（便于 tournament）
        for src, dst in zip(tmp, population):
            dst.rank = src.rank
            dst.crowding_distance = src.crowding_distance

        offspring: List[Individual] = []
        while len(offspring) < self.pop_size:
            p1 = binary_tournament(population, self.rng, self.tournament_k)
            p2 = binary_tournament(population, self.rng, self.tournament_k)

            if self.rng.random() < self.crossover_rate:
                c1, c2 = sbx_crossover(p1, p2, self.rng)
                c1.meta["source"] = "crossover"
                c2.meta["source"] = "crossover"
            else:
                c1 = p1.clone()
                c2 = p2.clone()
                c1.objectives = None
                c1.eval_result = None
                c2.objectives = None
                c2.eval_result = None
                c1.meta["source"] = "clone"
                c2.meta["source"] = "clone"

            c1 = gaussian_mutation(c1, self.rng, self.mutation_rate, self.mutation_sigma)
            c2 = gaussian_mutation(c2, self.rng, self.mutation_rate, self.mutation_sigma)

            c1.meta["mutated"] = True
            c2.meta["mutated"] = True

            offspring.append(c1)
            if len(offspring) < self.pop_size:
                offspring.append(c2)

        evaluate_population(offspring, scenario, spm)
        return offspring

    def run(
        self,
        scenario: Scenario,
        spm: ShortestPathManager,
        init_population: Optional[List[Individual]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        运行 NSGA-II
        返回：
        {
            "final_population": ...,
            "pareto_front": ...,
            "best": ...,
            "history": [...]
        }
        """
        num_tasks = len(scenario.tasks)
        if num_tasks == 0:
            return {
                "final_population": [],
                "pareto_front": [],
                "best": None,
                "history": [],
            }

        if init_population is None:
            population = self.initialize_population(num_tasks)
        else:
            # 补齐/裁剪到 pop_size
            population = [ind.clone() for ind in init_population[:self.pop_size]]
            while len(population) < self.pop_size:
                population.append(create_random_individual(num_tasks, self.rng))

        evaluate_population(population, scenario, spm)

        history = []

        for gen in range(self.num_generations):
            offspring = self.make_offspring(population, scenario, spm)
            union = population + offspring
            population = environmental_selection(union, self.pop_size)

            front = get_pareto_front(population)
            best = select_best_by_scalarization(front if front else population)

            if best is not None:
                history.append({
                    "gen": gen,
                    "best_obj": best.objectives,
                    "best_success_rate": best.eval_result.success_rate if best.eval_result else None,
                    "front_size": len(front),
                })
                if verbose and (gen % 10 == 0 or gen == self.num_generations - 1):
                    print(
                        f"[NSGAII] gen={gen:03d} front={len(front):03d} "
                        f"best_obj={best.objectives} "
                        f"succ={best.eval_result.success_rate:.3f}"
                    )

        pareto_front = get_pareto_front(population)
        best = select_best_by_scalarization(pareto_front if pareto_front else population)

        return {
            "final_population": population,
            "pareto_front": pareto_front,
            "best": best,
            "history": history,
        }