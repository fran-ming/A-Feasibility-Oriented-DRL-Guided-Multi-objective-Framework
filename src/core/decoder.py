# core/decoder.py

from __future__ import annotations
from core.graph_utils import ShortestTimeOracle
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from core.entities import Task, Vehicle, Scenario
from core.graph_utils import ShortestPathManager


INF = 1e18


@dataclass
class TaskExecRecord:
    task_idx: int
    vehicle_id: str
    arrive_u: float
    start_service: float
    finish_service: float
    success: bool
    lateness: float
    travel_to_u: float
    exec_edge_time: float


@dataclass
class VehicleState:
    vehicle_id: str
    cur_node: str
    cur_time: float
    route_task_indices: List[int]


@dataclass
class DecodeResult:
    records: List[TaskExecRecord]
    per_vehicle_routes: Dict[str, List[int]]
    success_flags: List[int]
    finish_times: List[float]
    lateness: List[float]
    total_wait: float
    total_travel: float
    total_exec_edge: float
    failed_tasks: List[int]


# core/decoder.py



def _task_exec_edge_time(task: Task, oracle: ShortestTimeOracle) -> float:
    """
    一个任务 = 在边 (u -> v) 上执行
    执行耗时 = 最短路径时间 + service
    """
    # 使用 oracle 计算从 u 到 v 的最短时间
    travel_time = oracle.time(task.u, task.v)
    if travel_time >= 1e17:  # 不可达
        travel_time = 1e9  # 返回一个大数
    return travel_time + float(task.service)


def _evaluate_vehicle_for_task(
        vstate: VehicleState,
        task: Task,
        oracle: ShortestTimeOracle  # 改为 oracle
) -> Tuple[float, float, float, float, float]:
    """
    给定某车当前状态，评估执行 task 的结果。
    """
    # 使用 oracle 计算旅行时间
    travel_to_u = oracle.time(vstate.cur_node, task.u)
    if travel_to_u >= 1e17:  # 不可达
        return INF, INF, INF, INF, INF

    arrive_u = vstate.cur_time + travel_to_u

    tw_a, tw_b = task.tw
    start_service = max(arrive_u, tw_a)  # 允许等待
    wait = max(0.0, tw_a - arrive_u)

    exec_edge_time = _task_exec_edge_time(task, oracle)
    finish_service = start_service + exec_edge_time

    # 时间窗成功判定：执行开始不晚于 b
    lateness = max(0.0, start_service - tw_b)
    success = 1 if start_service <= tw_b + 1e-9 else 0

    # greedy added cost
    penalty = 1e6 * (1 - success) + 1000.0 * lateness
    added_cost = penalty + travel_to_u + 0.1 * wait + 0.1 * exec_edge_time

    return arrive_u, start_service, finish_service, lateness, added_cost


def decode_priority_solution(
        scenario: Scenario,
        oracle: ShortestTimeOracle,  # 改为 oracle
        priority_order: List[int]
) -> DecodeResult:
    """
    输入：
      priority_order: 任务索引排列，例如 [5, 2, 0, 1, ...]
    """
    tasks = scenario.tasks
    vehicles = scenario.vehicles

    # 创建车辆状态
    vstates: Dict[int, VehicleState] = {}
    for v in vehicles:
        # 获取车辆ID
        if hasattr(v, 'vehicle_id'):
            vid = v.vehicle_id
        elif hasattr(v, 'id'):
            vid = v.id
        elif hasattr(v, 'veh_id'):
            vid = v.veh_id
        else:
            vid = len(vstates)

        # 获取起始节点
        start_node = ''
        if hasattr(v, 'depot_node'):
            start_node = v.depot_node
        elif hasattr(v, 'start_node'):
            start_node = v.start_node
        elif hasattr(v, 'depot'):
            start_node = v.depot

        vstates[vid] = VehicleState(
            vehicle_id=vid,
            cur_node=start_node,
            cur_time=float(getattr(v, 'start_time', 0.0)),
            route_task_indices=[]
        )

    records: List[TaskExecRecord] = []
    success_flags = [0] * len(tasks)
    finish_times = [INF] * len(tasks)
    lateness_arr = [INF] * len(tasks)
    total_wait = 0.0
    total_travel = 0.0
    total_exec_edge = 0.0
    failed_tasks: List[int] = []

    assigned = set()

    # 去重 + 补齐
    clean_order = []
    for x in priority_order:
        if 0 <= x < len(tasks) and x not in assigned:
            clean_order.append(x)
            assigned.add(x)
    for i in range(len(tasks)):
        if i not in assigned:
            clean_order.append(i)

    for tidx in clean_order:
        task = tasks[tidx]

        best_vid = None
        best_tuple = None
        best_cost = INF

        for vid, vs in vstates.items():
            arrive_u, start_service, finish_service, lateness, added_cost = _evaluate_vehicle_for_task(
                vs, task, oracle  # 传入 oracle
            )
            if added_cost < best_cost:
                best_cost = added_cost
                best_vid = vid
                best_tuple = (arrive_u, start_service, finish_service, lateness)

        if best_vid is None or best_tuple is None or best_cost >= INF:
            failed_tasks.append(tidx)
            records.append(
                TaskExecRecord(
                    task_idx=tidx,
                    vehicle_id=-1,
                    arrive_u=INF,
                    start_service=INF,
                    finish_service=INF,
                    success=False,
                    lateness=INF,
                    travel_to_u=INF,
                    exec_edge_time=INF
                )
            )
            success_flags[tidx] = 0
            finish_times[tidx] = INF
            lateness_arr[tidx] = INF
            continue

        vs = vstates[best_vid]
        arrive_u, start_service, finish_service, lateness = best_tuple
        travel_to_u = oracle.time(vs.cur_node, task.u)  # 使用 oracle
        exec_edge_time = _task_exec_edge_time(task, oracle)

        wait = max(0.0, task.tw[0] - arrive_u)
        success = start_service <= task.tw[1] + 1e-9

        # 更新车辆状态
        vs.cur_node = task.v
        vs.cur_time = finish_service
        vs.route_task_indices.append(tidx)

        records.append(
            TaskExecRecord(
                task_idx=tidx,
                vehicle_id=best_vid,
                arrive_u=arrive_u,
                start_service=start_service,
                finish_service=finish_service,
                success=bool(success),
                lateness=lateness,
                travel_to_u=travel_to_u,
                exec_edge_time=exec_edge_time
            )
        )

        success_flags[tidx] = 1 if success else 0
        finish_times[tidx] = finish_service
        lateness_arr[tidx] = lateness

        total_wait += wait
        total_travel += travel_to_u
        total_exec_edge += exec_edge_time

        if not success:
            failed_tasks.append(tidx)

    per_vehicle_routes = {vid: vs.route_task_indices[:] for vid, vs in vstates.items()}

    return DecodeResult(
        records=records,
        per_vehicle_routes=per_vehicle_routes,
        success_flags=success_flags,
        finish_times=finish_times,
        lateness=lateness_arr,
        total_wait=total_wait,
        total_travel=total_travel,
        total_exec_edge=total_exec_edge,
        failed_tasks=failed_tasks
    )
def decode_priority_scores(
    scenario: Scenario,
    spm: ShortestPathManager,
    priority_scores: List[float]
) -> DecodeResult:
    """
    输入为每个任务的优先级分数，分数越小优先级越高。
    """
    n = len(scenario.tasks)
    if len(priority_scores) != n:
        raise ValueError(f"priority_scores len mismatch: got {len(priority_scores)} expected {n}")

    order = sorted(range(n), key=lambda i: (priority_scores[i], i))
    return decode_priority_solution(scenario, spm, order)