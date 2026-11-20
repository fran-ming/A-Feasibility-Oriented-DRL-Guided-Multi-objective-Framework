# Constraint-Aware Deep Reinforcement Learning-Guided Cooperative Scheduling Optimization Framework for Closed-Track Testing

---

## abstract

This study investigates the cooperative scheduling problem of controllable object targets (OTs) in continuous scenario-flow testing for closed-track automated driving. To address the difficulty of generating executable schedules under coupled constraints, including task assignment, constrained network routing, and strict temporal synchronization with limited OTs, a constraint-aware deep reinforcement learning-guided NSGA-II hybrid optimization framework is proposed. The framework learns feasibility-oriented scheduling distributions through deep reinforcement learning and integrates NSGA-II for multi-objective refinement, achieving a balance between scheduling feasibility and optimization performance.

![](./config/overview_of_algorithm.png)

**Figure 1:** Overview of the proposed constraint-aware DRL-guided NSGA-II framework for cooperative OT scheduling in continuous scenario-flow testing.

---

## 1. Source Code

### 1.1 Code Structure

src/
├── core/
│   ├── config.py              # Experiment configuration and data classes (Task/Vehicle/Solution)
│   ├── decode.py              # Chromosome-to-schedule decoding
│   └── encoding.py            # Chromosome encoding and genetic operators
│
├── rl/
│   ├── drl_policy.py          # Policy network (TaskPolicyNet)
│   ├── drl_env.py             # Environment interaction and reward function 
│   └── drl_trainer.py         # REINFORCE trainer
│
├── ea/
│   ├── nsga2_solver.py        # NSGA-II solver
│   ├── nsgaii.py              # NSGA-II baseline 
│   └── drl_nsgaii_solver.py   # DRL-NSGA-II hybrid solver
│
└── sumo/                      # SUMO execution of a schedule and cooperative task handling
    ├── cooperative_schedule.py  # Cooperative groups, shared window, release clock, theoretical delay
    ├── retime.py                # VUT retiming and OT dwell re-anchoring to the trigger crossings
    ├── build_fcd.py             # Recorded schedule to FCD replay file
    ├── replay_utils.py           # Lane-centre geometry, id mapping, FCD writing
    ├── scenario_replay.sumocfg   # SUMO configuration
    ├── types.add.xml             # Vehicle types of the VUT and the OTs
    │
    └── data/
        ├── tasks.csv             # Task table: trigger node, OT, service time, task point
        ├── route.csv             # Per-edge route of every vehicle
        └── speed.csv             # Speed of every vehicle on a 0.2 s grid

### 1.2 Prerequisites

- Python 3.8+
- PyTorch
- NetworkX
- NumPy / Pandas
- Matplotlib
- [SUMO](https://eclipse.dev/sumo/) 1.28.0 with [sumolib](https://sumo.dlr.de/docs/Tools/Sumolib.html)

### 1.3 Key Modules

- `core/encoding.py`: Implements two-layer chromosome encoding and the corresponding genetic operators: Order Crossover, Uniform Crossover, Insertion Mutation, and Reset Mutation.
- `rl/drl_env.py`: Simulates the scheduling environment and computes the constraint-aware reward, defined as the sum of time-window reward and synchronization reward.
- `rl/drl_trainer.py`: Trains the policy network using the REINFORCE algorithm and generates elite individuals for population initialization.
- `ea/nsga2_solver.py`: Implements the main NSGA-II loop, including non-dominated sorting, crowding distance computation, and tournament selection.
- `ea/drl_nsgaii_solver.py`: Hybrid solver that first trains the DRL policy and then uses the sampled solutions to initialize the NSGA-II population.
- `sumo/cooperative_schedule.py`: Implements the cooperative-task rules and reports the theoretical delay of every task.
- `sumo/retime.py`: Builds the executed scenario with a continuous VUT trajectory and OT dwell times re-anchored to the VUT trigger crossings.
- `sumo/build_fcd.py`: Replays the recorded schedule unchanged as an FCD file.
- `sumo/replay_utils.py`: Shared helpers for lane-centre geometry and FCD output.

### 1.4 Reward Function

The constraint-aware reward is defined as:

$R = R_{tw} + R_{sync}$

where:

- $R_{tw} = -\frac{1}{|I|}\sum_{i \in I} e_i^{tw}$ penalizes both early and late arrivals with respect to task-specific time windows.
- $R_{sync} = -\frac{1}{|C|}\sum_{s \in C} e_s^{sync}$ penalizes synchronization deviations among OTs assigned to the same cooperative event.

This reward design ensures that the DRL module learns scheduling patterns that satisfy both task-level temporal constraints and group-level coordination requirements.

### 1.5 SUMO Execution and Cooperative Task Handling

The `sumo/` folder executes a schedule in SUMO with one VUT and four OTs. It contains the scenario logic only. The road network and all geographic data are not part of the repository.

#### Simulation workflow

1. **Inputs.** `data/` holds the routes, the speed profiles on a 0.2 s grid and the task table. Nodes and edges use neutral ids and no coordinates are stored.
2. **Cooperative analysis.** `cooperative_schedule.py` reads the time at which the VUT reaches each trigger node and derives the cooperative groups, time windows, release instants and theoretical delays.
3. **Replay generation.** `build_fcd.py` maps the recorded schedule onto the lane centre lines. `retime.py` additionally gives the VUT a continuous trajectory and re-anchors every OT dwell to the instant the VUT crosses its trigger node.
4. **Execution.** `scenario_replay.sumocfg` loads the network and replays `output/replay.fcd.xml` with the FCD replay device. The VUT speed control and the OT releases are already contained in the replay file.

#### Cooperative task handling

Seven task points of five tasks are served by four OTs while the VUT never stops. Coordination is the difficult part, not running SUMO. Two situations need care. Tasks gated by the same trigger node, such as T3#0 and T3#1, must be served at the same instant by different OTs. The two halves of a task gated by different trigger nodes, such as T1#0 and T1#1, are served in sequence, so the second half carries the VUT travel time between the two nodes as its theoretical delay.

Four rules handle both. Tasks sharing a trigger node form one cooperative group, which corresponds to `Task.vut_trigger` in `src/core/config.py` and to `group_by_trigger()` in `sumo/cooperative_schedule.py`. Each group has one time window anchored at the VUT crossing of its node. Each node has one release clock that is computed once and read by all members, so all OTs of a group leave their task points in the same simulation step. A group needs distinct OTs, which `group_is_feasible()` checks.

The VUT never stops because the release is triggered by its arrival and not by the end of the service. The schedule only has to guarantee that every OT is in position before the VUT reaches the trigger, and `retime.py` reports every OT that is late.

---

## 2. Usage

This section describes the complete workflow for solving the CMAPP problem using this repository. The whole procedure includes problem instance preparation, algorithm configuration, solver selection, optimization execution, and solution decoding.

## 2.1 Problem Instance Preparation

Before running the optimization algorithms, users need to construct a complete CMAPP problem instance, which consists of three fundamental components:

- Road network graph
- Vehicle set
- Task set

These components should be encapsulated into a unified `ProblemInstance` or `Scenario` object, which serves as the input of the optimization framework.

In addition, a shortest path manager is required to provide travel time estimation between different nodes. The manager should implement the following interface:

`time(src, dst)`

where `src` and `dst` represent the source and destination nodes, respectively. The output should be the shortest travel time between two nodes in seconds.

The shortest path information can be pre-computed using graph-based algorithms such as `networkx` to improve query efficiency during optimization.

## 2.2 Algorithm Configuration

All algorithm parameters are managed in `config.py` through the `ExperimentConfig` class.

Users can configure the optimization process in two ways:

- Modify the default parameter values directly in `config.py`.
- Dynamically override parameters when initializing the solver.

This configuration mechanism enables flexible adjustment of experimental settings for different optimization algorithms.

## 2.3 Solver Selection and Execution

This repository provides three optimization approaches for solving the CMAPP problem. Users can select different solvers according to their experimental requirements.

### (A) Standard NSGA-II

The standard NSGA-II solver adopts real-value chromosome encoding and directly performs evolutionary multi-objective optimization.

Implementation:

`nsgaii.py`

Main class:

`NSGAII`

The optimization procedure follows:

Problem Instance → Population Initialization → Evolutionary Optimization → Pareto Solutions

The obtained individuals contain chromosome information that can be further decoded into scheduling solutions.

### (B) Simplified NSGA-II Solver

The simplified NSGA-II solver converts multiple optimization objectives into a single weighted fitness value.

Implementation:

`nsga2_solver.py`

Main class:

`NSGA2Solver`

Before execution, users need to define the objective weights in `evaluate_decoded_result` to specify the contribution of different optimization objectives.

This solver provides a simplified optimization framework for scenarios where a single fitness measurement is preferred.

### (C) DRL-guided NSGA-II Hybrid Solver

The hybrid solver combines feasibility-oriented deep reinforcement learning with NSGA-II evolutionary optimization.

Implementation:

`drl_nsgaii_solver.py`

Main class:

`DRLNSGA2Solver`

The optimization process consists of two stages:

Stage 1:  
DRL policy learning generates high-quality feasible initial solutions.

Stage 2:  
NSGA-II performs evolutionary refinement to obtain optimized Pareto solutions.

The DRL module improves the quality of the initial population, while NSGA-II further optimizes multiple objectives.

## 2.4 Solution Decoding

After optimization, each solver returns an optimized individual containing chromosome information.

The encoded chromosome needs to be decoded into an executable scheduling plan.

For the standard NSGA-II solver:

`best.genome`

represents the priority value of each task. The task execution sequence can be obtained by sorting tasks according to their priority values.

For the simplified NSGA-II solver:

`best.chrom`

contains a `Solution(pi, v)` structure, where:

- `pi` represents the task execution sequence.
- `v` represents the vehicle assignment.

## 2.5 Solution Decoder

The repository provides decoding functions in:

`decoder.py`

The function:

`decode_priority_solution()`

can be used to convert priority-based chromosome representations into executable scheduling solutions.

Users can further extend the decoder module to support additional chromosome encoding schemes.

## 2.6 Complete Execution Pipeline

The complete workflow of this repository is:

Create Problem Instance  
↓  
Configure Algorithm Parameters  
↓  
Select Optimization Solver  
↓  
Run Optimization Algorithm  
↓  
Decode Scheduling Solution  
↓  
Analyze Optimization Results  
↓  
Execute the schedule in SUMO, optional, see Section 2.7

## 2.7 Running the SUMO Simulation

This optional step executes a schedule in SUMO. All commands are run inside `sumo/`.

1. Install the dependencies in `requirements.txt` and set `SUMO_HOME`.
2. Put your own road network at `network/map.net.xml`. The edge and junction ids in `data/route.csv` and `data/tasks.csv` must match it.
3. Run `cooperative_schedule.py` for the cooperative analysis, then `retime.py` or `build_fcd.py` to generate `output/replay.fcd.xml`, then open `scenario_replay.sumocfg` in `sumo-gui`.

After changing the network, copy the trigger distances printed by `cooperative_schedule.py` into the column `trigger_s` of `data/tasks.csv`.

---

## 3. Results and Demonstration

https://github.com/user-attachments/assets/75ea5618-04bc-4859-957e-5b2b40cf36a6

To validate the executability of the generated schedule, we implement it in SUMO (version 1.28.0), an open-source microscopic traffic simulator, using the road network of an actual closed-track proving ground. A representative VUT straight route is selected, covering five typical testing scenarios in sequence: roundabout, car following, intersection, U-turn, and boxed-in. These scenarios are modeled as five testing events (T1–T5) served by four OTs. The VUT travels at a nominal speed of 6 m/s, with deceleration permitted during task execution and interaction, while OT speeds are limited to 0–10 m/s.

The generated schedule is successfully executed without violating road-network or vehicle-motion constraints. All OTs reach their assigned task locations, with some reused across multiple tasks to accommodate the task distribution.

All seven task executions start before the corresponding VUT arrival times, including both same-trigger and different-trigger cooperative tasks. No additional departure delays are introduced by the OTs, indicating effective temporal coordination.

The VUT maintains its scheduled progression and decelerates only during designated interaction events. No waiting caused by late OT arrivals is observed, demonstrating that the generated schedule supports continuous test execution.

Further validation with more cooperative tasks is needed to assess the robustness of these observations under higher coordination demands.

---

##### 4. Summary and Future Work

- **Constraint-aware Reward Design**: The proposed reward function integrates time-window and synchronization constraints into a unified scale, improving training stability and convergence.
- **DRL-EA Hybrid Optimization**: The framework combines DRL-based feasibility learning with NSGA-II Pareto refinement, reducing search difficulty in sparse feasible regions while maintaining multi-objective optimization capability.
- **Flexible Framework Design**: The DRL module is replaceable and can be extended with advanced algorithms such as PPO or A2C. A shared decoder ensures consistent evaluation between DRL-generated and evolutionary solutions.
- **Future Improvements**: Current modeling mainly considers network-level travel constraints. Future work will incorporate vehicle dynamics, trajectory-level feasibility, and online scheduling strategies for real-time disturbances.

Welcome for questions, discussions, or potential collaborations, please contact:

**Email:** murphy1051792848@gmail.com
