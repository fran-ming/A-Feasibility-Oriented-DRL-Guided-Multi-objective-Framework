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
│ ├── config.py # Experiment configuration and data classes (Task/Vehicle/Solution)  
│ ├── decode.py # Chromosome-to-schedule decoding  
│ ├── encoding.py # Chromosome encoding and genetic operators 
├── rl/  
│ ├── drl_policy.py # Policy network (TaskPolicyNet)  
│ ├── drl_env.py # Environment interaction and reward function (R_tw + R_sync)  
│ └── drl_trainer.py # REINFORCE trainer  
└── ea/  
├── nsga2_solver.py # NSGA-II solver  
├── nsgaii.py # NSGA-II baseline (real-valued encoding)  
└── drl_nsgaii_solver.py # DRL-NSGA-II hybrid solver

### 1.2 Prerequisites

- Python 3.8+
- PyTorch
- NetworkX
- NumPy / Pandas
- Matplotlib

### 1.3 Key Modules

- `core/encoding.py`: Implements two-layer chromosome encoding and the corresponding genetic operators: Order Crossover, Uniform Crossover, Insertion Mutation, and Reset Mutation.
- `rl/drl_env.py`: Simulates the scheduling environment and computes the constraint-aware reward, defined as the sum of time-window reward and synchronization reward.
- `rl/drl_trainer.py`: Trains the policy network using the REINFORCE algorithm and generates elite individuals for population initialization.
- `ea/nsga2_solver.py`: Implements the main NSGA-II loop, including non-dominated sorting, crowding distance computation, and tournament selection.
- `ea/drl_nsgaii_solver.py`: Hybrid solver that first trains the DRL policy and then uses the sampled solutions to initialize the NSGA-II population.

### 1.4 Reward Function

The constraint-aware reward is defined as:

$R = R_{tw} + R_{sync}$

where:

- $R_{tw} = -\frac{1}{|I|}\sum_{i \in I} e_i^{tw}$ penalizes both early and late arrivals with respect to task-specific time windows.
- $R_{sync} = -\frac{1}{|C|}\sum_{s \in C} e_s^{sync}$ penalizes synchronization deviations among OTs assigned to the same cooperative event.

This reward design ensures that the DRL module learns scheduling patterns that satisfy both task-level temporal constraints and group-level coordination requirements.

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
