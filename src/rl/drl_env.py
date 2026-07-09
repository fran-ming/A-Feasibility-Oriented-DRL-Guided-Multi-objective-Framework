import math
import numpy as np
import torch
import networkx as nx

def build_task_features(instance: ProblemInstance, cfg: ExperimentConfig) -> np.ndarray:
    """
    每个任务构造简单特征:
    [tw_start, tw_end, service, dist_to_nearest_depot, dist_u_to_v]
    """
    # 从 vehicles 中提取所有 depot
    depots = [v.depot for v in instance.vehicles]

    # 确保有距离矩阵
    if not hasattr(instance, 'distance_matrix') or not instance.distance_matrix:
        instance.distance_matrix = {}
        for depot in depots:
            for task in instance.tasks:
                try:
                    dist = nx.shortest_path_length(instance.G, depot, task.u, weight='length')
                except:
                    dist = 1e9
                instance.distance_matrix[(depot, task.u)] = dist

                try:
                    dist = nx.shortest_path_length(instance.G, task.u, task.v, weight='length')
                except:
                    dist = 1e9
                instance.distance_matrix[(task.u, task.v)] = dist

    feats = []
    for task in instance.tasks:
        # 计算到最近depot的距离
        nearest = min(
            instance.distance_matrix.get((depot, task.u), 1e9)
            for depot in depots
        )

        # 获取任务边 u->v 的距离
        uv = instance.distance_matrix.get((task.u, task.v), 1e9)

        feats.append([
            float(task.tw[0]),
            float(task.tw[1]),
            float(task.service),
            float(nearest),
            float(uv),
        ])

    x = np.array(feats, dtype=np.float32)

    # 简单标准化（按列）
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std
    return x


def _safe_shortest_path_length(
    G: nx.DiGraph,
    src: str,
    dst: str,
    weight: str = 'length'
) -> float:
    if src == dst:
        return 0.0
    try:
        return float(nx.shortest_path_length(G, source=src, target=dst, weight=weight))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float('inf')


def init_drl_state(instance: ProblemInstance) -> dict:
    return {
        'remaining': list(range(len(instance.tasks))),
        'vehicles': [
            {
                'node': v.depot,
                'time': 0.0,
                'load': 0,
                'distance': 0.0,
                'success': 0,
                'assigned': 0,
            }
            for v in instance.vehicles
        ],
        'records': [],
        'task_seq': [],
        'veh_seq': [],
        'previous_J': None,  # For differential rewards
    }


def build_task_state_features(
    instance: ProblemInstance,
    state: dict,
    cfg: ExperimentConfig
) -> np.ndarray:
    feats = []
    for task_idx in state['remaining']:
        task = instance.tasks[task_idx]
        min_dist = float('inf')
        min_time = 0.0
        for veh in state['vehicles']:
            dist = _safe_shortest_path_length(instance.G, veh['node'], task.u)
            if dist < min_dist:
                min_dist = dist
                min_time = veh['time']

        if not math.isfinite(min_dist):
            min_dist = 1e6

        feats.append([
            float(task.tw[0]),
            float(task.tw[1]),
            float(task.service),
            float(min_dist),
            float(min_time),
        ])

    if len(feats) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    x = np.array(feats, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std
    return x

#rl进化奖励函数设计
def compute_reward(
    state: dict,
    cfg: ExperimentConfig,
    instance: ProblemInstance = None
) -> float:
    """
    核心奖励函数设计
    R = R_tw + R_sync
    R_tw = - (1/|I|) * Σ_i e_i^tw
         e_i^tw = max(0, a_i - t_i) + max(0, t_i - b_i)
    R_sync = - (1/|C|) * Σ_s e_s^sync
         e_s^sync = (2/(|I_s|*(|I_s|-1))) * Σ_{i<j} |t_i - t_j|
    """
    records = state['records']
    if not records:
        return 0.0
    #时间窗奖励
    total_tw_violation = 0.0
    num_tasks = 0

    for rec in records:
        arrive = rec['arrive']
        if not math.isfinite(arrive):
            # 不可达任务视为最大惩罚
            tw_len = rec['tw'][1] - rec['tw'][0]
            total_tw_violation += tw_len if tw_len > 0 else 10.0
            num_tasks += 1
            continue

        a, b = rec['tw']
        left = max(0.0, a - arrive)   # 早到惩罚
        right = max(0.0, arrive - b)  # 晚到惩罚
        total_tw_violation += left + right
        num_tasks += 1
    R_tw = -(total_tw_violation / max(1, num_tasks))
    #同步奖励
    R_sync = 0.0
    if instance is not None and hasattr(instance, 'tasks'):
        # 从 records 中提取每个任务的 trigger
        task_trigger = {}
        for idx, rec in enumerate(records):
            if rec['task_idx'] < len(instance.tasks):
                trigger = getattr(instance.tasks[rec['task_idx']], 'vut_trigger', None)
                if trigger:
                    task_trigger[rec['task_idx']] = trigger
        # 按 trigger 分组
        groups = {}
        for rec in records:
            tidx = rec['task_idx']
            if tidx in task_trigger:
                trigger = task_trigger[tidx]
                groups.setdefault(trigger, []).append(rec)
        # 只处理包含 ≥2 个任务的协同组
        sync_errors = []
        for trigger, recs in groups.items():
            if len(recs) < 2:
                continue
            # 收集该组内所有任务的到达时间
            times = []
            for rec in recs:
                if math.isfinite(rec['arrive']):
                    times.append(rec['arrive'])
            if len(times) < 2:
                continue
            pair_sum = 0.0
            pair_count = 0
            for i in range(len(times)):
                for j in range(i + 1, len(times)):
                    pair_sum += abs(times[i] - times[j])
                    pair_count += 1

            e_sync = pair_sum / max(1, pair_count)
            sync_errors.append(e_sync)

        if sync_errors:
            R_sync = -(sum(sync_errors) / len(sync_errors))

    return R_tw + R_sync

def compute_partial_reward(state: dict, cfg: ExperimentConfig, previous_J: float = None) -> float:
    current_penalty = compute_reward(state, cfg)

    if previous_J is None:
        return 0.0  # First step, no reward
    else:
        # Differential reward: improvement is positive
        reward = previous_J - current_penalty
        return reward


def apply_drl_action(
    instance: ProblemInstance,
    state: dict,
    task_idx: int,
    vehicle_idx: int,
    cfg: ExperimentConfig
) -> tuple[dict, float]:
    task = instance.tasks[task_idx]
    vehicle = state['vehicles'][vehicle_idx]
    speed = instance.vehicles[vehicle_idx].speed_mps

    dist_to_u = _safe_shortest_path_length(instance.G, vehicle['node'], task.u)
    arrive = float('inf')
    success = False
    travel_dist = 0.0

    if math.isfinite(dist_to_u):
        arrive = vehicle['time'] + dist_to_u / speed
        travel_dist += dist_to_u

        if arrive <= task.tw[1]:
            dist_uv = _safe_shortest_path_length(instance.G, task.u, task.v)
            if math.isfinite(dist_uv):
                service_start = max(arrive, task.tw[0])
                finish = service_start + dist_uv / speed + task.service
                vehicle['node'] = task.v
                vehicle['time'] = finish
                vehicle['distance'] += travel_dist + dist_uv
                success = True
                vehicle['load'] += 1
                vehicle['success'] += 1
                travel_dist += dist_uv
            else:
                vehicle['node'] = task.u
                vehicle['time'] = arrive
                vehicle['distance'] += travel_dist
        else:
            vehicle['node'] = task.u
            vehicle['time'] = arrive
            vehicle['distance'] += travel_dist
    else:
        # 无法到达任务起点，记录失败但不改变车辆位置/时间
        pass

    vehicle['assigned'] += 1
    state['records'].append({
        'task_idx': task_idx,
        'vehicle_id': vehicle_idx,
        'arrive': arrive,
        'tw': task.tw,
        'success': success,
    })
    state['task_seq'].append(task_idx)
    state['veh_seq'].append(vehicle_idx)
    state['remaining'].remove(task_idx)

    reward = compute_partial_reward(state, cfg, state['previous_J'])
    state['previous_J'] = compute_reward(state, cfg)  # Update previous_J for next step
    return state, reward


def build_solution_from_state(state: dict) -> 'Solution':
    return Solution(pi=list(state['task_seq']), v=list(state['veh_seq']))


def policy_output_to_chromosome(
    priority_scores: torch.Tensor,
    vehicle_logits: torch.Tensor
) -> Chromosome:
    """
    priority 排序 -> order
    argmax vehicle -> vehicle_assign
    """
    order = torch.argsort(priority_scores, descending=True).cpu().numpy().tolist()
    veh = torch.argmax(vehicle_logits, dim=-1).cpu().numpy().tolist()

    # vehicle_assign 必须与 order 同步
    veh_ordered = [int(veh[i]) for i in order]

    return Chromosome(order=[int(i) for i in order], vehicle_assign=veh_ordered)

#评估解的优劣
def evaluate_policy_chromosome(
    chrom: Chromosome,
    instance: ProblemInstance,
    cfg: ExperimentConfig
) -> Tuple[float, dict]:
    schedule = decode_chromosome(instance, chrom)
    obj_values = evaluate_decoded_result(schedule, instance, cfg)  # 传入 cfg
    metrics = {
        'J': obj_values.J,
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
    beta = 100.0
    alpha = 1.0
    window_penalty = 0.0
    for rec in schedule.ordered_records:
        arrive = rec.arrive_u_time
        if not math.isfinite(arrive):
            continue
        left = max(0.0, rec.tw[0] - arrive)
        right = max(0.0, arrive - rec.tw[1])
        window_penalty += left + right
    reward = -(beta * window_penalty + alpha * obj_values.J_t)
    return reward, metrics
