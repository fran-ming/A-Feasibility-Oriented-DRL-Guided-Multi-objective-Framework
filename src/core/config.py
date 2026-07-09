import os
import random
import numpy as np

"""
本文件定义了项目需要用到的参数
读者可根据实际情况修改参数值
"""
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

@dataclass
class Task:
    """单执行点任务"""
    task_id: str
    u: str
    v: str
    tw: Tuple[float, float]
    service: float
    vut_trigger: str

@dataclass
class Solution:
    """染色体解（二维编码）"""
    pi: List[int]
    v: List[int]

@dataclass
class Vehicle:
    """测试车辆"""
    veh_id: int
    depot: str
    speed_mps: float

@dataclass
class RouteStep:
    """解码后的调度步骤"""
    task_idx: int
    veh_id: int
    arrive_time: float
    start_time: float
    finish_time: float
    travel_dist: float
    success: bool

@dataclass
class DecodedSolution:
    """解码结果"""
    steps: List[RouteStep]
    veh_last_node: Dict[int, str]
    veh_last_time: Dict[int, float]
    veh_task_count: Dict[int, int]
    success_flags: Dict[int, int]
    total_distance: float
    c_max: float

@dataclass
class ObjectiveResult:
    """目标函数结果"""
    J_d: float  # 任务失败数
    J_m: float  # 全局完成时间
    J_b: float  # 负载均衡差值
    J_t: float  # 总路径长度
    J: float  # 加权单目标值


@dataclass
class ExperimentConfig:
    """实验总配置"""

    # Scenario
    num_vehicles: int = NUM_VEHICLES
    vehicle_speed_mps: float = VEHICLE_SPEED

    # VUT轨迹生成
    vut_walk_steps: int = VUT_WALK_STEPS
    vut_speed_mps: float = VUT_SPEED_MPS

    # 任务生成
    num_triggers: int = NUM_TRIGGERS
    tasks_per_trigger: Tuple[int, int] = (TASKS_PER_TRIGGER_MIN, TASKS_PER_TRIGGER_MAX)

    # 时间窗/服务时间
    pre_slack_s: float = PRE_SLACK_S
    tw_len_s: float = TW_LEN_S
    service_min_s: float = SERVICE_MIN_S
    service_max_s: float = SERVICE_MAX_S

    # 在VUT触发点附近找任务点
    u_pool_hops: int = U_POOL_HOPS
    u_pool_max: int = U_POOL_MAX

    # Logging / save
    results_dir: str = RESULTS_DIR
    models_dir: str = MODELS_DIR
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.task is not None:
            self.target_tasks = self.task
        elif self.tasks is not None:
            self.target_tasks = self.tasks

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("task", "tasks") and value is not None:
            super().__setattr__("target_tasks", value)
        super().__setattr__(name, value)


CFG = ExperimentConfig()


def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def update_cfg_from_dict(cfg: ExperimentConfig, kwargs: dict) -> ExperimentConfig:
    for k, v in kwargs.items():
        if k in ("task", "tasks"):
            k = "target_tasks"

        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise AttributeError(f"Unknown config key: {k}")
    return cfg
Config = ExperimentConfig