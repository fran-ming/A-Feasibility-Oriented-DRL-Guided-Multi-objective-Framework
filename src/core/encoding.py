import random

"""
Chromosome encoding for CMAPP:
chi = (pi, v)
- pi: permutation order (PO layer), a permutation of task indices [0, ..., N-1]
- v : vehicle selection (VS layer), v[j] is the assigned vehicle for task pi[j]
"""

def validate_solution(sol: Solution, num_tasks: int, num_vehicles: int) -> bool:
    """
    检查二维染色体是否合法
    """
    if len(sol.pi) != num_tasks:
        return False
    if len(sol.v) != num_tasks:
        return False

    if sorted(sol.pi) != list(range(num_tasks)):
        return False

    for veh in sol.v:
        if not (0 <= veh < num_vehicles):
            return False

    return True


def clone_solution(sol: Solution) -> Solution:
    """深拷贝解"""
    return Solution(pi=sol.pi[:], v=sol.v[:])

def random_solution(
    num_tasks: int,
    num_vehicles: int,
    seed: Optional[int] = None
) -> Solution:
    """随机生成一个合法二维染色体"""
    rng = random.Random(seed)

    pi = list(range(num_tasks))
    rng.shuffle(pi)

    v = [rng.randrange(num_vehicles) for _ in range(num_tasks)]

    sol = Solution(pi=pi, v=v)
    if not validate_solution(sol, num_tasks, num_vehicles):
        raise RuntimeError("random_solution generated invalid chromosome.")
    return sol


def init_population(
    pop_size: int,
    num_tasks: int,
    num_vehicles: int,
    seed: Optional[int] = None
) -> List[Solution]:
    """初始化种群（随机）"""
    rng = random.Random(seed)
    population = []

    for _ in range(pop_size):
        s = random_solution(
            num_tasks=num_tasks,
            num_vehicles=num_vehicles,
            seed=rng.randint(0, 10**9)
        )
        population.append(s)

    return population


def order_crossover(
    p1: List[int],
    p2: List[int],
    seed: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """
    OX (Order Crossover) for permutation layer pi
    PO 层交叉算子。
    """
    rng = random.Random(seed)
    n = len(p1)
    if n != len(p2):
        raise ValueError("order_crossover: parent lengths mismatch")
    if n <= 1:
        return p1[:], p2[:]

    a, b = sorted(rng.sample(range(n), 2))

    def ox(parent_a: List[int], parent_b: List[int]) -> List[int]:
        child = [-1] * n
        child[a:b + 1] = parent_a[a:b + 1]

        fill_vals = [x for x in parent_b if x not in child]
        idxs = [i for i in range(n) if child[i] == -1]

        for i, val in zip(idxs, fill_vals):
            child[i] = val

        return child

    return ox(p1, p2), ox(p2, p1)

def uniform_vehicle_crossover(
    v1: List[int],
    v2: List[int],
    num_vehicles: int,
    seed: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """
    VS层均匀交叉
    """
    rng = random.Random(seed)
    n = len(v1)
    if n != len(v2):
        raise ValueError("uniform_vehicle_crossover: parent lengths mismatch")

    c1, c2 = [], []

    for i in range(n):
        if rng.random() < 0.5:
            c1.append(v1[i])
            c2.append(v2[i])
        else:
            c1.append(v2[i])
            c2.append(v1[i])

    # 保险修正
    c1 = [min(max(x, 0), num_vehicles - 1) for x in c1]
    c2 = [min(max(x, 0), num_vehicles - 1) for x in c2]

    return c1, c2

def insertion_mutation(
    perm: List[int],
    seed: Optional[int] = None
) -> List[int]:
    """
    PO层插入变异
    """
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm[:]

    child = perm[:]
    i, j = rng.sample(range(n), 2)
    val = child.pop(i)
    child.insert(j, val)
    return child

def vehicle_reset_mutation(
    v: List[int],
    num_vehicles: int,
    seed: Optional[int] = None
) -> List[int]:
    """
    VS层 Reset Mutation
    随机选择一个位置，把该任务重新分配给另一辆车
    """
    rng = random.Random(seed)
    n = len(v)
    if n == 0:
        return v[:]

    child = v[:]
    idx = rng.randrange(n)

    old = child[idx]
    candidates = [k for k in range(num_vehicles) if k != old]
    if not candidates:
        return child

    child[idx] = rng.choice(candidates)
    return child

def crossover_solution(
    parent1: Solution,
    parent2: Solution,
    num_tasks: int,
    num_vehicles: int,
    seed: Optional[int] = None
) -> Tuple[Solution, Solution]:
    """
    二维染色体交叉：
    - PO层：OX (order_crossover)
    - VS层：uniform crossover (uniform_vehicle_crossover)
    """
    rng = random.Random(seed)

    c1_pi, c2_pi = order_crossover(
        parent1.pi, parent2.pi, seed=rng.randint(0, 10**9)
    )

    c1_v, c2_v = uniform_vehicle_crossover(
        parent1.v, parent2.v, num_vehicles=num_vehicles, seed=rng.randint(0, 10**9)
    )

    child1 = Solution(pi=c1_pi, v=c1_v)
    child2 = Solution(pi=c2_pi, v=c2_v)

    if not validate_solution(child1, num_tasks, num_vehicles):
        raise RuntimeError("Invalid child1 after crossover_solution.")
    if not validate_solution(child2, num_tasks, num_vehicles):
        raise RuntimeError("Invalid child2 after crossover_solution.")

    return child1, child2


def mutate_solution(
    sol: Solution,
    num_tasks: int,
    num_vehicles: int,
    seed: Optional[int] = None,
    p_pi_mut: float = 0.7,
    p_v_mut: float = 0.7,
) -> Solution:
    """
    二维染色体变异
    - PO层：Insertion Mutation
    - VS层：Reset Mutation
    """
    rng = random.Random(seed)

    child = clone_solution(sol)

    if rng.random() < p_pi_mut:
        child.pi = insertion_mutation(child.pi, seed=rng.randint(0, 10**9))

    if rng.random() < p_v_mut:
        child.v = vehicle_reset_mutation(
            child.v,
            num_vehicles=num_vehicles,
            seed=rng.randint(0, 10**9),
        )

    if not validate_solution(child, num_tasks, num_vehicles):
        raise RuntimeError("Invalid child after mutate_solution.")

    return child