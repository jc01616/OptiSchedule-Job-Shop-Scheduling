# OptiSchedule — Job Shop Scheduling Decision Intelligence System

> **MBA Operations Management Capstone Project**  
> **Operations Research, Constraint Programming, and Flow-Shop Optimization applied to a CNC Manufacturing Workshop**

---

## 1. Executive Summary

OptiSchedule models and solves industrial manufacturing scheduling problems across two core Operations Research paradigms:
1. **General Job Shop Scheduling Problem (JSSP)** ($J_m || C_{\max}$): An NP-hard combinatorial problem with 8 customer jobs, 4 machines, and 24 operations in non-uniform machine routings.
2. **Dedicated Flow Shop Benchmarks** ($F_m || C_{\max}$): Pure permutation flow shops where all jobs visit machines in the exact same sequence, evaluating **Johnson's Algorithm** for 2-machine and 3-machine problems.

The primary objective is to minimize **Makespan ($C_{\max}$)**—total elapsed time to complete all orders—while maximizing machine utilization and minimizing delivery tardiness.

---

## 2. Dataset Assessment: Starjob Repository

- **Repository Inspected**: [https://github.com/starjob42/Starjob](https://github.com/starjob42/Starjob)  
- **License**: MIT (HuggingFace mirror)  
- **Assessment Verdict**: ❌ **Cannot be used directly by OR-Tools**

### Findings
The Starjob dataset contains 130,000 instruction-tuning pairs for Large Language Models (LLMs) formatted as natural-language text. Deterministic solvers like Google OR-Tools CP-SAT require structured, deterministic numeric arrays of `(machine_id, duration)`. 

Per Capstone Requirement 4, realistic, documented synthetic benchmark datasets were engineered as the single source of truth.

---

## 3. Benchmark Datasets

### 3.1 Main Job Shop Dataset (8 Jobs, 4 Machines, 24 Operations)
Defined in [`data/synthetic_jobs.py`](file:///C:/Users/jairc/.gemini/antigravity-ide/scratch/opti_schedule/data/synthetic_jobs.py).

*Machines:* **M0:** Lathe, **M1:** Mill, **M2:** Drill Press, **M3:** Grinder.

| Job ID | Part Name | Op 0: [Mach, Dur] | Op 1: [Mach, Dur] | Op 2: [Mach, Dur] | Total Work ($P_j$) | Customer Due Date ($d_j$) | Tightness Ratio ($d_j / P_j$) | Routing Sequence |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **J0** | Shaft | M0, 3 min | M1, 2 min | M2, 2 min | 7 min | 12 min | 1.71 | M0(3) $\to$ M1(2) $\to$ M2(2) |
| **J1** | Bracket | M1, 4 min | M0, 3 min | M3, 2 min | 9 min | 15 min | 1.67 | M1(4) $\to$ M0(3) $\to$ M3(2) |
| **J2** | Flange | M2, 3 min | M1, 2 min | M0, 4 min | 9 min | 18 min | 2.00 | M2(3) $\to$ M1(2) $\to$ M0(4) |
| **J3** | Gear | M0, 2 min | M3, 5 min | M1, 2 min | 9 min | 20 min | 2.22 | M0(2) $\to$ M3(5) $\to$ M1(2) |
| **J4** | Pin | M1, 3 min | M2, 1 min | M3, 4 min | 8 min | 14 min | 1.75 | M1(3) $\to$ M2(1) $\to$ M3(4) |
| **J5** | Housing | M3, 2 min | M0, 3 min | M2, 3 min | 8 min | 22 min | 2.75 | M3(2) $\to$ M0(3) $\to$ M2(3) |
| **J6** | Cam | M2, 4 min | M3, 2 min | M1, 3 min | 9 min | 25 min | 2.78 | M2(4) $\to$ M3(2) $\to$ M1(3) |
| **J7** | Bushing | M0, 1 min | M1, 2 min | M3, 3 min | 6 min | 16 min | 2.67 | M0(1) $\to$ M1(2) $\to$ M3(3) |
| **Total** | — | — | — | — | **65 min** | **Mean: 17.75 min** | **Mean: 2.20** | **24 Operations** |

- **Theoretical Lower Bound**:
  $$LB = \max\left(\max_j \sum_{m} p_{j,m}, \max_m \sum_{j} p_{j,m}\right) = \max(9, 18) = 18 \text{ minutes}$$

### 3.2 Standalone Flow-Shop Benchmark Datasets
Defined in [`data/flowshop_datasets.py`](file:///C:/Users/jairc/.gemini/antigravity-ide/scratch/opti_schedule/data/flowshop_datasets.py).

1. **Two-Machine Flow Shop (`FLOWSHOP_2M_DATA`)**:
   - 6 Jobs visiting Machine A (Lathe) $\to$ Machine B (Mill).
   - Machine A Load: 29 min, Machine B Load: 27 min. Total work: 56 min.
2. **Three-Machine Flow Shop (`FLOWSHOP_3M_DATA`)**:
   - 5 Jobs visiting Machine A (Turning) $\to$ Machine B (Milling) $\to$ Machine C (Grinding).
   - Machine A Load: 35 min, Machine B Load: 11 min, Machine C Load: 30 min. Total work: 76 min.
   - Middle machine is strictly dominated: $\min(A) = 5 \ge \max(B) = 3$ and $\min(C) = 4 \ge \max(B) = 3$.

---

## 4. Algorithms Implemented & Methodological Documentation

### 4.1 Priority Dispatching Heuristics: Sequential List Scheduling
- **Mechanism**: The heuristics in `src/dispatching_rules.py` (FCFS, SPT, LPT, EDD) sort jobs into a priority permutation. The simulation commits all operations of Job $A$ across its route before scheduling Job $B$.
- **Contrast with Dynamic Dispatching**: Unlike dynamic event-driven dispatching (which re-evaluates physical machine queues at each completion event $t=\tau$), sequential list scheduling cannot insert subsequent jobs into earlier idle gaps claimed by a previous job. This creates larger machine idle blocks, resulting in higher makespans (FCFS: 55 min).
- **Feasibility**: All heuristic schedules are verified 100% mathematically feasible.

### 4.2 Johnson's Algorithm & Flow-Shop Optimization
- **Two-Machine Flow Shop ($F2 || C_{\max}$)**: Johnson's algorithm partitions jobs into Set U ($A_i \le B_i$, sorted ascending $A$) and Set V ($A_i > B_i$, sorted descending $B$). Proven mathematically optimal by Johnson (1954).
- **Three-Machine Flow Shop ($F3 || C_{\max}$)**: If the middle machine satisfies dominance ($\min A \ge \max B$ or $\min C \ge \max B$), reducing to virtual 2-machine processing times $A'_i = A_i + B_i$ and $B'_i = B_i + C_i$ yields a provably optimal sequence. If dominance is violated, it functions as an approximate heuristic (Palmer / CDS).
- **Job Shop Exploratory Heuristic**: On the 4-machine job shop, virtual times are used purely as a heuristic ordering tool (`Palmer Heuristic (Job-Shop)`).

### 4.3 Google OR-Tools CP-SAT Optimization
- **Formulation**: Formulated in `src/cpsat_solver.py` using `NewIntervalVar`, `AddNoOverlap` for machine exclusivity, and linear routing precedence inequalities.
### 4.4 Comprehensive Key Performance Indicator (KPI) Definitions

All performance indicators are computed deterministically from scheduled operation start/end times and customer due dates.

1. **Makespan ($C_{\max}$)**:
   $$C_{\max} = \max_{j, k} \left(\text{End}_{j,k}\right)$$
   Total elapsed time from commencement of the first operation to completion of the final operation. Primary combinatorial objective to minimize.

2. **Job Completion Time ($C_j$)**:
   $$C_j = \max_{k \in \mathcal{O}_j} \left(\text{End}_{j,k}\right)$$
   The exact time at which the final operation of Job $j$ finishes.

3. **Signed Job Lateness ($L_j$) & Average Lateness ($\bar{L}$)**:
   $$L_j = C_j - d_j, \quad \bar{L} = \frac{1}{N} \sum_{j=1}^N L_j$$
   Signed metric representing delivery deviation from customer due date $d_j$. Negative values indicate early completion ($C_j < d_j$), whereas positive values indicate late completion ($C_j > d_j$).

4. **Job Tardiness ($T_j$) & Average Tardiness ($\bar{T}$)**:
   $$T_j = \max(0, C_j - d_j) = \max(0, L_j), \quad \bar{T} = \frac{1}{N} \sum_{j=1}^N T_j$$
   Separately isolates delivery delay penalties without allowing early jobs to cancel out late delivery penalties. Key customer service metrics include:
   - **Maximum Tardiness**: $T_{\max} = \max_j T_j$
   - **Percentage of Tardy Jobs**: $\% \text{Tardy} = \frac{1}{N} \sum_{j=1}^N \mathbb{I}(T_j > 0) \times 100\%$

5. **Machine Utilization ($U_m$) & Shop Average Utilization ($\bar{U}$)**:
   $$U_m = \frac{\sum_{j \in \mathcal{J}_m} p_{j,m}}{C_{\max}} \times 100\%, \quad \bar{U} = \frac{1}{M} \sum_{m=1}^M U_m = \frac{\sum_{j,m} p_{j,m}}{M \times C_{\max}} \times 100\%$$
   *Assumption:* Standard non-preemptive work content ratio across active machines. Because total shop processing work is constant ($\sum p_{j,m} = 65\text{ min}$ across 4 machines), average utilization is strictly inversely proportional to makespan: $\bar{U} = \frac{65}{4 \times C_{\max}} \times 100\%$.

6. **Benchmark Production Throughput (jobs/hr)**:
   $$\text{Throughput (jobs/hr)} = \frac{N_{\text{jobs}}}{C_{\max} / 60} = \frac{60 \times N_{\text{jobs}}}{C_{\max}}$$
   *Operational Context:* This represents the makespan-based benchmark completion rate across the 8-job schedule planning horizon. It measures scheduling efficiency and does not assume sustained, infinite-horizon real-world factory production throughput.

---

## 5. Benchmark Results & Verification Tables

### 5.1 Main Job Shop Benchmark Table (`outputs/comparison_table.csv`)
*Full 8-Job, 4-Machine Instance (24 Operations, 65 minutes work, Base FCFS makespan = 55 min).*

| Method | Makespan | vs Baseline (%) | Avg Lateness | Avg Tardiness | Max Tardiness | % Tardy Jobs | Avg Util (%) | Throughput (jobs/hr) | Feasibility Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FCFS (Baseline)** | 55 min | 0.00% | +12.38 min | 13.38 min | 39.0 min | 62.5% | 29.55% | 8.73 jobs/hr | Verified Feasible |
| **EDD** | 48 min | 12.73% | +7.50 min | 8.25 min | 23.0 min | 75.0% | 33.85% | 10.00 jobs/hr | Verified Feasible |
| **Palmer Heuristic** | 46 min | 16.36% | +8.25 min | 10.12 min | 34.0 min | 50.0% | 35.33% | 10.43 jobs/hr | Verified Feasible |
| **SPT** | 43 min | 21.82% | +5.75 min | 7.50 min | 20.0 min | 50.0% | 37.79% | 11.16 jobs/hr | Verified Feasible |
| **LPT** | 40 min | 27.27% | +8.50 min | 11.38 min | 24.0 min | 75.0% | 40.62% | 12.00 jobs/hr | Verified Feasible |
| **CP-SAT Optimal** | **18 min** | **67.27%** | **-2.25 min** | **0.75 min** | **3.0 min** | **25.0%** | **90.28%** | **26.67 jobs/hr** | **Proven Optimal** |

### 5.2 Gantt Chart Evidence: Baseline FCFS vs. Optimized CP-SAT
Visual evidence confirms the operational mechanism driving CP-SAT's performance:
- **Baseline FCFS Schedule** ([`outputs/gantt_fcfs.png`](file:///C:/Users/jairc/.gemini/antigravity-ide/scratch/opti_schedule/outputs/gantt_fcfs.png) & [`outputs/gantt_fcfs.html`](file:///C:/Users/jairc/.gemini/antigravity-ide/scratch/opti_schedule/outputs/gantt_fcfs.html)): Sequential list commitment prevents later jobs from occupying earlier idle windows. Machines experience extensive idle starvation periods (e.g., Lathe M0 sits idle between minutes 18 and 35), resulting in a bloated 55-minute makespan.
- **Optimized CP-SAT Schedule** ([`outputs/gantt_cpsat_optimal.png`](file:///C:/Users/jairc/.gemini/antigravity-ide/scratch/opti_schedule/outputs/gantt_cpsat_optimal.png) & [`outputs/gantt_cpsat_optimal.html`](file:///C:/Users/jairc/.gemini/antigravity-ide/scratch/opti_schedule/outputs/gantt_cpsat_optimal.html)): CP-SAT concurrently interleaves operations across all 4 machines. Operations are explicitly labeled with job and operation IDs (`J{id}.O{op}`). Mill (M1) and Grinder (M3) operate continuously with zero idle time from minute 0 to 18, reaching 100% saturation and compressing shop makespan to the theoretical minimum of 18 minutes.
- **Constraint Verification**: All schedules independently verified: 0 machine overlaps ($\text{End}_{i} \le \text{Start}_{i+1}$ on all machines), 0 job precedence violations ($\text{End}_{j,k} \le \text{Start}_{j,k+1}$ across all routes).

### 5.3 Standalone Flow-Shop Benchmark Tables
Saved to `outputs/flowshop_2machine_comparison.csv` and `outputs/flowshop_3machine_comparison.csv`:

#### Two-Machine Flow Shop ($F2 || C_{\max}$)
| Sequence Method | Sequence Permutation | Makespan | vs FCFS Baseline (%) | Machine A Util | Machine B Util | Optimality Classification |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Johnson's Algorithm** | `[J5, J0, J2, J3, J4, J1]` | **31 min** | **8.82%** | **93.55%** | **87.10%** | **PROVEN OPTIMAL (Johnson 1954)** |
| **FCFS (Arrival Order)** | `[J0, J1, J2, J3, J4, J5]` | 34 min | 0.00% | 85.29% | 79.41% | Suboptimal Baseline |
| **Reverse Order** | `[J5, J4, J3, J2, J1, J0]` | 35 min | -2.94% | 82.86% | 77.14% | Suboptimal Sequence |

#### Three-Machine Flow Shop ($F3 || C_{\max}$)
*Dominance Status: Satisfied ($\min A = 5 \ge \max B = 3$ and $\min C = 4 \ge \max B = 3$).*

| Sequence Method | Sequence Permutation | Makespan | vs FCFS Baseline (%) | Machine A Util | Machine B Util | Machine C Util | Optimality Classification |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Johnson 3M Reduction** | `[J3, J2, J4, J0, J1]` | **42 min** | **6.67%** | **83.33%** | **26.19%** | **71.43%** | **PROVEN OPTIMAL (Dominance Verified)** |
| **FCFS (Arrival Order)** | `[J0, J1, J2, J3, J4]` | 45 min | 0.00% | 77.78% | 24.44% | 66.67% | Suboptimal Baseline |
| **Reverse Order** | `[J4, J3, J2, J1, J0]` | 42 min | 6.67% | 83.33% | 26.19% | 71.43% | Optimal Alternative for This Instance |

---

## 6. What-If Scenario Analyses & Bottleneck Dynamics

### 6.1 Scenario A: Before-and-After Resource Addition (Parallel Lathe M4)
To test whether expanding Lathe capacity improves performance, Scenario A introduces an identical parallel Lathe (M4). Operations requiring Lathe can run on either M0 or M4 via optional interval variables.

#### Before vs. After Capacity Addition Comparison

| Metric / Resource | Baseline CP-SAT (4 Machines) | Scenario A: +Lathe M4 (5 Machines) | Variance / Impact | Operational Insight |
|:---|:---:|:---:|:---:|:---|
| **Makespan ($C_{\max}$)** | **18 minutes** | **18 minutes** | **0.00% (0 min gain)** | Non-critical resource addition yields zero throughput gain |
| **Throughput (jobs/hr)** | 26.67 jobs/hr | 26.67 jobs/hr | 0.00% | Bottleneck constraints remain unchanged |
| **M0 (Lathe 1) Workload** | 16 min (88.89% util) | 0 min (0.00% util) | -16 min | Solver shifted Lathe tasks to M4; combined load remains 16 min |
| **M4 (Lathe 2) Workload** | Non-existent | 16 min (88.89% util) | +16 min | Lathes combined have 20 idle min across 36 min capacity (44.4% util) |
| **M1 (Mill) Workload** | **18 min (100.0% util)** | **18 min (100.0% util)** | **0 min** | **Binding Bottleneck Constraint 1 (Fully Saturated)** |
| **M2 (Drill) Workload** | 13 min (72.22% util) | 13 min (72.22% util) | 0 min | Non-bottleneck machine with 5 min idle slack |
| **M3 (Grinder) Workload**| **18 min (100.0% util)** | **18 min (100.0% util)** | **0 min** | **Binding Bottleneck Constraint 2 (Fully Saturated)** |
| **Shop Total Workload** | 65 minutes | 65 minutes | 0 min | Total work content is identical |
| **Average Shop Utilization** | **90.28%** | **72.22%** | **-18.06%** | Utilization diluted due to redundant machine capital capacity |

#### Theory of Constraints (ToC) Interpretation:
According to Goldratt's Theory of Constraints:
1. *System throughput is governed strictly by the bottleneck resources.* Here, M1 (Mill) and M3 (Grinder) require 18 minutes of processing each. Because neither can be bypassed, the theoretical lower bound cannot fall below $\max(18, 18) = 18$ minutes.
2. Lathe M0 had only 16 minutes of work (88.89% utilization) and 2 minutes of buffer slack. Adding capacity to a non-bottleneck does not relieve the binding constraints on Mill and Grinder.
3. **Managerial Takeaway:** Investing capital expenditure ($CapEx$) into duplicating a Lathe produces 0% return on investment ($ROI$). Capital should instead be directed toward debottlenecking the Mill or Grinder.

---

### 6.2 Scenario B: Bottleneck Disruption (+15% Mill Processing Inflation)
Scenario B models process deterioration (tool wear, harder alloy stock, or feed rate reduction) on the primary bottleneck, Machine 1 (Mill).

- **Discrete Rounding Rule**:
  In this integer discrete-time formulation, processing durations must be integer minutes. Applying `math.ceil(duration * 1.15)` increases:
  - 2 min $\to$ $\lceil 2.30 \rceil$ = 3 min (+50.0%)
  - 3 min $\to$ $\lceil 3.45 \rceil$ = 4 min (+33.3%)
  - 4 min $\to$ $\lceil 4.60 \rceil$ = 5 min (+25.0%)
  Each of the 7 operations on Machine 1 receives exactly +1 minute.
- **Aggregate Workload Expansion**:
  Total Mill workload grows from 18 min to 25 minutes, representing an aggregate **+38.89% workload increase**.
- **Makespan Deterioration**:
  Because Mill is a 100% utilized binding constraint, this +7 minute workload translates 1:1 into a makespan expansion from 18 min to **25 minutes (+38.89%)**.
- **Operations Governance Decision Rule (>10% Threshold)**:
  Under standard manufacturing governance rules, any operational disruption that increases schedule makespan by **greater than 10%** triggers **mandatory Operations Manager escalation and formal sign-off**. At +38.89%, Scenario B severely breaches this threshold.
  **Managerial Mitigation Actions Required:**
  1. *Immediate Tooling Inspection:* Tool re-grinding and spindle calibration to restore nominal cutting speeds.
  2. *Overtime Allocation:* Authorize 1.5 hours of weekend overtime on the Mill to buffer against order delivery tardiness.
  3. *Selective Subcontracting:* Outsource rough milling on Flanges (J2) and Pins (J4) to external machine shops.

---

## 7. Execution Instructions

### Setup
```powershell
pip install -r requirements.txt
```

### Step 1: Run Manual Textbook Validation
Validates CP-SAT and FCFS against a textbook 3-job, 2-machine problem:
```powershell
$env:PYTHONIOENCODING = "utf-8"
python validate_example.py
```

### Step 2: Run Full Pipeline
Executes main job shop, flow shop benchmarks, scenario analyses, and exports all outputs:
```powershell
python main.py
```

### Step 3: Run Comprehensive Technical Audit
Performs independent constraint validation, recomputes all KPIs, and verifies zero violations:
```powershell
python audit_runner.py
```

---

## 8. Remaining Limitations

1. **Deterministic Processing Times**: Machine breakdowns, tool wear variability, and operator stochasticity are not modeled.
2. **Zero Setup Times**: Machine changeover and tooling times between consecutive jobs on the same machine are assumed to be negligible or incorporated into processing times. Sequence-dependent setup times ($SDST$) are not included.
3. **Absence of Financial/ROI Model**: The repository models operational metrics (makespan, machine utilization, tardiness, throughput). No cost accounting parameters (hourly machine rates, labor wages, inventory holding costs, capital amortization) exist in the code. Financial ROI claims must not be asserted.
4. **Non-Preemptive Scheduling**: Operations cannot be paused or resumed once started.
5. **Synthetic Benchmark Scope & Real-World Realization**: The 67.27% makespan reduction achieved by Google OR-Tools CP-SAT over FCFS is a mathematical optimization proof on the engineered 8-job, 4-machine benchmark instance. While it demonstrates the tremendous power of constraint programming over naive sequential list scheduling, it does **not** guarantee an equivalent 67.27% makespan reduction in real-world factory operations due to unmodeled shop-floor stochasticity (dynamic order arrivals, material staging delays, operator absenteeism, and machine breakdowns).
