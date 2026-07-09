import random
import copy

@dataclass
class Individual:
    chrom: Chromosome
    metrics: Dict[str, float]
    fitness: float
    rank: int = 0
    crowding: float = 0.0

def dominates(a: Individual, b: Individual) -> bool:
    """
    当前单目标执行：fitness 越小越好
    为兼容 NSGA-II 框架，仍使用 dominance 形式
    """
    return a.fitness < b.fitness

def fast_non_dominated_sort(pop: List[Individual]) -> List[List[int]]:
    S = [[] for _ in range(len(pop))]
    n = [0 for _ in range(len(pop))]
    fronts: List[List[int]] = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    pop[q].rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)

    return fronts

def compute_crowding_distance(pop: List[Individual], front: List[int]):
    if not front:
        return
    for idx in front:
        pop[idx].crowding = 0.0

    # 单目标下 crowding 简化按 fitness
    front_sorted = sorted(front, key=lambda i: pop[i].fitness)
    pop[front_sorted[0]].crowding = float("inf")
    pop[front_sorted[-1]].crowding = float("inf")

    fmin = pop[front_sorted[0]].fitness
    fmax = pop[front_sorted[-1]].fitness
    if abs(fmax - fmin) < 1e-12:
        return

    for k in range(1, len(front_sorted) - 1):
        prev_f = pop[front_sorted[k - 1]].fitness
        next_f = pop[front_sorted[k + 1]].fitness
        pop[front_sorted[k]].crowding += (next_f - prev_f) / (fmax - fmin + 1e-12)

def tournament_select(pop: List[Individual], rng: random.Random) -> Individual:
    a, b = rng.sample(pop, 2)
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding > b.crowding:
        return a
    if b.crowding > a.crowding:
        return b
    return a if rng.random() < 0.5 else b

class NSGA2Solver:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)

    # algorithms/nsga2_solver.py
    def _evaluate(self, chrom: Chromosome, instance: ProblemInstance) -> Individual:
        schedule = decode_chromosome(instance, chrom)
        # 传入 cfg
        obj_values = evaluate_decoded_result(schedule, instance, self.cfg)
        metrics = {
            'J_d': obj_values.J_d,
            'J_m': obj_values.J_m,
            'J_b': obj_values.J_b,
            'J_t': obj_values.J_t,
            'success_rate': obj_values.success_rate,
            'finished_tasks': obj_values.num_success,
            'failed_tasks': obj_values.num_failed,
            'C_max': obj_values.J_m,
            'n_max': obj_values.load_max,
            'n_min': obj_values.load_min,
            'total_distance': obj_values.J_t,
        }
        fit = obj_values.J
        return Individual(chrom=chrom, metrics=metrics, fitness=fit)
    
    def set_initial_population(self, init_pop: List[Chromosome]):
        """设置外部生成的初始种群（仅用于一次 solve）"""
        self._custom_initial_pop = init_pop

    def _initialize(self, instance: ProblemInstance) -> List[Individual]:
        if hasattr(self, '_custom_initial_pop') and self._custom_initial_pop:
            pop = [self._evaluate(chrom, instance) for chrom in self._custom_initial_pop]
        # 补齐到 pop_size
        while len(pop) < self.cfg.pop_size:
            chrom = random_solution(
                len(instance.tasks), self.cfg.num_vehicles,
                seed=self.rng.randint(0, 10**9)
                    )
            pop.append(self._evaluate(chrom, instance))
            delattr(self, '_custom_initial_pop')
            return pop[:self.cfg.pop_size]
        else:
        # 原始随机初始化
            pop = []
            for _ in range(self.cfg.pop_size):
                chrom = random_solution(
                    len(instance.tasks), self.cfg.num_vehicles,
                    seed=self.rng.randint(0, 10**9)
            )
                pop.append(self._evaluate(chrom, instance))
            return pop
        
    def _initialize(self, instance: ProblemInstance) -> List[Individual]:
        pop = []
        for _ in range(self.cfg.pop_size):
            seed = self.rng.randint(0, 10 ** 9)
            chrom = random_solution(
                num_tasks=len(instance.tasks),
                num_vehicles=self.cfg.num_vehicles,
                seed=seed 
            )
            pop.append(self._evaluate(chrom, instance))
        return pop

    def _assign_rank_and_crowding(self, pop: List[Individual]):
        fronts = fast_non_dominated_sort(pop)
        for front in fronts:
            compute_crowding_distance(pop, front)

    def _environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        self._assign_rank_and_crowding(combined)
        fronts = fast_non_dominated_sort(combined)

        new_pop: List[Individual] = []
        for front in fronts:
            if len(new_pop) + len(front) <= self.cfg.pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                sorted_front = sorted(
                    [combined[i] for i in front],
                    key=lambda x: (x.rank, -x.crowding)
                )
                remain = self.cfg.pop_size - len(new_pop)
                new_pop.extend(sorted_front[:remain])
                break
        return new_pop

    def solve(self, instance: ProblemInstance, return_pop: bool = False):
        pop = self._initialize(instance)
        self._assign_rank_and_crowding(pop)

        best = min(pop, key=lambda x: x.fitness)

        # ========== 第1步：创建空的history列表 ==========
        history = []

        for gen in range(self.cfg.n_gen):
            offspring: List[Individual] = []

            while len(offspring) < self.cfg.pop_size:
                p1 = tournament_select(pop, self.rng)
                p2 = tournament_select(pop, self.rng)

                c1, c2 = crossover_solution(
                    p1.chrom, p2.chrom,
                    num_tasks=len(instance.tasks),
                    num_vehicles=self.cfg.num_vehicles,
                    seed=self.rng.randint(0, 10 ** 9)
                )
                c1 = mutate_solution(
                    c1,
                    num_tasks=len(instance.tasks),
                    num_vehicles=self.cfg.num_vehicles,
                    seed=self.rng.randint(0, 10 ** 9)
                )
                c2 = mutate_solution(
                    c2,
                    num_tasks=len(instance.tasks),
                    num_vehicles=self.cfg.num_vehicles,
                    seed=self.rng.randint(0, 10 ** 9)
                )

                offspring.append(self._evaluate(c1, instance))
                if len(offspring) < self.cfg.pop_size:
                    offspring.append(self._evaluate(c2, instance))

            combined = pop + offspring
            pop = self._environmental_selection(combined)

            cur_best = min(pop, key=lambda x: x.fitness)

            # ========== 第2步：每代结束后，记录当前代的最优fitness ==========
            history.append({
                "gen": gen,
                "fitness_J": cur_best.fitness
            })

            if cur_best.fitness < best.fitness:
                best = copy.deepcopy(cur_best)

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"[NSGA-II] Gen {gen + 1}/{self.cfg.n_gen} | Best J={best.fitness:.4f}")

        # ========== 第3步：返回best, history, 和最终种群 ==========
        if return_pop:
            return best, history, pop
        return best, history