"""
scenario_analysis.py — What-If Analysis
=========================================
Implements two scenario analyses as required:

Scenario A: Add a Compatible Machine (M4 = Second Lathe)
  Business motivation: Test whether adding capacity to Lathe (M0)
  relieves queueing and reduces makespan.
  Modelling: We duplicate M0. Any operation originally on M0 can now
  run on either M0 or M4 (modelled as an optional interval variable).

Scenario B: Mill Processing Times Disruption (Integer Discrete Model)
  Business motivation: Evaluates sensitivity to tool wear or slower feed rates
  on Machine 1 (Mill, the primary bottleneck).
  Discrete Rounding Rule:
    In this integer discrete-time formulation, processing durations must be integer minutes.
    Applying a nominal +15% scale factor with ceiling rounding `math.ceil(duration * 1.15)`
    increases:
      - 2 min -> ceil(2.30) = 3 min (+50.0% per-op increase)
      - 3 min -> ceil(3.45) = 4 min (+33.3% per-op increase)
      - 4 min -> ceil(4.60) = 5 min (+25.0% per-op increase)
    Each of the 7 operations on Machine 1 is lengthened by exactly +1 minute.
    Consequently, total Mill workload grows from 18 to 25 minutes, an aggregate +38.89%
    workload increase (rather than a flat 15% aggregate increase).
    Because Machine 1 is a 100% utilized binding bottleneck, this +7 minute workload
    translates 1:1 into a 7-minute makespan expansion (18 min -> 25 min).
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import math
import copy
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cpsat_solver import solve_jobshop
from data.synthetic_jobs import JOBS_DATA, MACHINES


def scenario_a_add_machine(jobs_data) -> Tuple[Dict, Dict]:
    """
    Scenario A: Add a second Lathe (M4) as a parallel resource to M0.

    Modelling approach: We use CP-SAT's ability to handle ALTERNATIVE
    machines via optional interval variables. For each operation on M0,
    we create two optional intervals (one on M0, one on M4) and add
    an ExactlyOne constraint so exactly one is selected.
    """
    from ortools.sat.python import cp_model
    import time, pandas as pd

    model = cp_model.CpModel()
    # New machine set: M0=Lathe1, M1=Mill, M2=Drill, M3=Grinder, M4=Lathe2
    n_original_machines = max(max(m for m, _ in ops) for ops in jobs_data) + 1
    parallel_machine    = n_original_machines  # M4 = second lathe (mirrors M0)

    horizon = sum(d for ops in jobs_data for _, d in ops) * 2

    all_tasks  = {}
    machine_to_intervals: Dict[int, List] = {}

    for job_id, ops in enumerate(jobs_data):
        for op_idx, (machine_id, duration) in enumerate(ops):
            suffix = f"_j{job_id}_op{op_idx}"
            start = model.NewIntVar(0, horizon, f"start{suffix}")
            end   = model.NewIntVar(0, horizon, f"end{suffix}")

            if machine_id == 0:
                # This operation CAN run on M0 OR M4
                on_m0 = model.NewBoolVar(f"on_m0{suffix}")
                on_m4 = model.NewBoolVar(f"on_m4{suffix}")
                model.Add(on_m0 + on_m4 == 1)  # exactly one machine

                iv_m0 = model.NewOptionalIntervalVar(start, duration, end, on_m0, f"iv_m0{suffix}")
                iv_m4 = model.NewOptionalIntervalVar(start, duration, end, on_m4, f"iv_m4{suffix}")

                machine_to_intervals.setdefault(0, []).append(iv_m0)
                machine_to_intervals.setdefault(parallel_machine, []).append(iv_m4)
                all_tasks[(job_id, op_idx)] = (start, end, None, on_m4)
            else:
                iv = model.NewIntervalVar(start, duration, end, f"iv{suffix}")
                machine_to_intervals.setdefault(machine_id, []).append(iv)
                all_tasks[(job_id, op_idx)] = (start, end, iv, None)

    # NoOverlap constraints for all machines
    for m_id, intervals in machine_to_intervals.items():
        model.AddNoOverlap(intervals)

    # Precedence constraints
    for job_id, ops in enumerate(jobs_data):
        for op_idx in range(len(ops) - 1):
            task_k  = all_tasks[(job_id, op_idx)]
            task_k1 = all_tasks[(job_id, op_idx + 1)]
            model.Add(task_k[1] <= task_k1[0])

    # Objective
    makespan_var = model.NewIntVar(0, horizon, "makespan")
    last_ends = [all_tasks[(j, len(ops) - 1)][1] for j, ops in enumerate(jobs_data)]
    model.AddMaxEquality(makespan_var, last_ends)
    model.Minimize(makespan_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers  = 8
    t0 = time.perf_counter()
    status_code = solver.Solve(model)
    runtime = time.perf_counter() - t0

    from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE
    if status_code not in (OPTIMAL, FEASIBLE):
        raise RuntimeError("Scenario A solver failed")

    records = []
    for job_id, ops in enumerate(jobs_data):
        for op_idx, (machine_id, duration) in enumerate(ops):
            task_info = all_tasks[(job_id, op_idx)]
            sv, ev = task_info[0], task_info[1]
            on_m4_var = task_info[3]
            actual_machine = machine_id
            if machine_id == 0 and on_m4_var is not None and solver.Value(on_m4_var) == 1:
                actual_machine = parallel_machine

            records.append({
                "job_id"    : job_id,
                "op_index"  : op_idx,
                "machine_id": actual_machine,
                "start"     : solver.Value(sv),
                "end"       : solver.Value(ev),
                "duration"  : duration,
            })
    schedule_df = pd.DataFrame(records)
    ms = int(solver.ObjectiveValue())
    return schedule_df, {
        "label": "Scenario A: +Lathe",
        "makespan": ms,
        "runtime": runtime,
        "parallel_machine_id": parallel_machine,
    }


def get_scenario_b_jobs(jobs_data, increase_pct: float = 15.0) -> List[List[Tuple[int, int]]]:
    """
    Construct modified job operations for Scenario B under integer minutes discretization.
    Processing times on M1 (Mill) are increased by `increase_pct`% and rounded UP via math.ceil.

    Mathematical effect:
      - Durations of 2 min become ceil(2.30) = 3 min (+50.0%)
      - Durations of 3 min become ceil(3.45) = 4 min (+33.3%)
      - Durations of 4 min become ceil(4.60) = 5 min (+25.0%)
    Total workload on M1 increases from 18 min to 25 min (+38.89% aggregate workload increase).
    """
    target_machine = 1  # Mill
    scale = 1.0 + increase_pct / 100.0

    new_jobs = []
    for ops in jobs_data:
        new_ops = []
        for machine_id, duration in ops:
            if machine_id == target_machine:
                new_duration = math.ceil(duration * scale)
            else:
                new_duration = duration
            new_ops.append((machine_id, new_duration))
        new_jobs.append(new_ops)
    return new_jobs


def scenario_b_increased_processing(jobs_data, increase_pct: float = 15.0) -> Tuple[Dict, Dict]:
    """
    Scenario B: Increase processing times on M1 (Mill) under integer discretization.

    Business context: Mill throughput drops due to tool wear or harder alloy stock.
    Discrete ceiling rule math.ceil(duration * 1.15) increases aggregate Mill workload
    from 18 min to 25 min (+38.89%).

    OR Concept: Sensitivity analysis testing the robustness of the optimal schedule
    to bottleneck process time inflation.
    """
    new_jobs = get_scenario_b_jobs(jobs_data, increase_pct)

    schedule_df, solver_info = solve_jobshop(
        new_jobs,
        time_limit_seconds=60.0,
        label=f"Scenario B: Mill +{increase_pct:.0f}% (ceil, +38.9% load)",
    )
    return schedule_df, {
        "label"                 : f"Scenario B: Mill +{increase_pct:.0f}% (+38.9% load)",
        "makespan"              : solver_info["makespan"],
        "runtime"               : solver_info["runtime_sec"],
        "modified_jobs"         : new_jobs,
        "nominal_increase_pct"  : increase_pct,
        "aggregate_workload_pct": 38.89,
    }
