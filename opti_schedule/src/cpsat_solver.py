"""
cpsat_solver.py — Google OR-Tools CP-SAT Job Shop Optimizer
============================================================

BACKGROUND (for MBA Viva)
--------------------------
CP-SAT (Constraint Programming — SAT) is Google's state-of-the-art
combinatorial optimization solver. It works by:

1. Modelling the problem as BOOLEAN CONSTRAINTS (hence 'SAT')
2. Using BRANCH AND BOUND with intelligent pruning
3. Maintaining OPTIMALITY GAPS: (UB - LB) / UB × 100%
   where UB = best feasible solution found,
         LB = mathematical lower bound (from linear relaxations)
4. Reporting SOLVER STATUS: OPTIMAL, FEASIBLE, INFEASIBLE, UNKNOWN

WHY CP-SAT FOR JOB SHOP?
--------------------------
The Job Shop Scheduling Problem (JSSP) is NP-hard for 3+ machines.
Exhaustive search is impractical for large instances. CP-SAT uses:
  - Interval variables to represent time windows
  - NoOverlap constraints for machine exclusivity
  - Precedence constraints between consecutive operations
  - An objective function: minimize makespan (= max end time)

DECISION VARIABLES
------------------
For each (job, operation) pair:
  start[j, op]  : integer — when does this operation begin?
  end[j, op]    : integer — when does this operation finish?
                  (always = start + duration, a fixed offset)
  interval[j, op] : IntervalVar — encodes [start, start+dur, end]

CONSTRAINTS
-----------
1. Precedence: end[j, op] <= start[j, op+1]  for each consecutive pair
2. NoOverlap : for each machine M, all intervals on M must not overlap
3. Non-negative: start[j, op] >= 0
4. Valid routing: each interval is created only for valid (job, op) pairs

OBJECTIVE
---------
Minimise makespan = max over all jobs of (end of last operation of that job)
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import time
import pandas as pd
from ortools.sat.python import cp_model


def solve_jobshop(
    jobs_data,
    time_limit_seconds: float = 60.0,
    label: str = "CP-SAT Optimal",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Solve the Job Shop Scheduling Problem to optimality (or best-feasible
    within time_limit_seconds) using OR-Tools CP-SAT.

    Parameters
    ----------
    jobs_data          : list of [(machine_id, duration), ...] per job
    time_limit_seconds : wall-clock time limit for the solver
    label              : name for this run (used in reports)

    Returns
    -------
    schedule_df : DataFrame with columns job_id, op_index, machine_id,
                  start, end, duration
    solver_info : dict with solver status, makespan, objective bound,
                  optimality gap, runtime
    """
    model = cp_model.CpModel()

    # -- 1. CALCULATE HORIZON --------------------------------------------------
    # The horizon is the maximum possible makespan (sum of ALL processing times).
    # No valid schedule can exceed this. It bounds our integer variables.
    horizon = sum(d for ops in jobs_data for _, d in ops)

    # -- 2. CREATE DECISION VARIABLES -----------------------------------------
    # all_tasks[job_id][op_index] = (start_var, end_var, interval_var)
    all_tasks: Dict[Tuple[int, int], Tuple] = {}
    machine_to_intervals: Dict[int, List] = {}  # machine_id -> list of intervals

    for job_id, ops in enumerate(jobs_data):
        for op_idx, (machine_id, duration) in enumerate(ops):
            # Variable names help with debugging
            suffix = f"_j{job_id}_op{op_idx}"
            start_var = model.NewIntVar(0, horizon, f"start{suffix}")
            end_var   = model.NewIntVar(0, horizon, f"end{suffix}")

            # IntervalVar is a first-class citizen in CP-SAT; it bundles
            # start, duration, and end into a single scheduling object.
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, f"interval{suffix}"
            )

            all_tasks[(job_id, op_idx)] = (start_var, end_var, interval_var)
            machine_to_intervals.setdefault(machine_id, []).append(interval_var)

    # -- 3. MACHINE NON-OVERLAP CONSTRAINTS -----------------------------------
    # AddNoOverlap ensures all intervals assigned to a machine are disjoint.
    # This is the core constraint preventing two jobs from using the same
    # machine simultaneously.
    for machine_id, intervals in machine_to_intervals.items():
        model.AddNoOverlap(intervals)

    # -- 4. PRECEDENCE CONSTRAINTS ---------------------------------------------
    # Operation k+1 of job j cannot start before operation k ends.
    # This encodes the routing constraint (operation sequence).
    for job_id, ops in enumerate(jobs_data):
        for op_idx in range(len(ops) - 1):
            _, end_var_k,   _ = all_tasks[(job_id, op_idx)]
            start_var_k1, _, _ = all_tasks[(job_id, op_idx + 1)]
            model.Add(end_var_k <= start_var_k1)

    # -- 5. OBJECTIVE: MINIMISE MAKESPAN --------------------------------------
    # Makespan = the time at which the last operation (across all jobs) finishes.
    # We create a single variable that captures the maximum end time.
    makespan_var = model.NewIntVar(0, horizon, "makespan")
    last_ends = []
    for job_id, ops in enumerate(jobs_data):
        _, end_var, _ = all_tasks[(job_id, len(ops) - 1)]
        last_ends.append(end_var)
    model.AddMaxEquality(makespan_var, last_ends)
    model.Minimize(makespan_var)

    # -- 6. SOLVE -------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers  = 8  # use multiple CPU cores

    t0 = time.perf_counter()
    status_code = solver.Solve(model)
    runtime = time.perf_counter() - t0

    # -- 7. STATUS MAPPING -----------------------------------------------------
    status_map = {
        cp_model.OPTIMAL  : "OPTIMAL",
        cp_model.FEASIBLE : "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN  : "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
    }
    status_str = status_map.get(status_code, f"UNKNOWN({status_code})")

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT returned status {status_str}. Check your model."
        )

    # -- 8. EXTRACT SOLUTION ---------------------------------------------------
    records = []
    for job_id, ops in enumerate(jobs_data):
        for op_idx, (machine_id, duration) in enumerate(ops):
            start_var, end_var, _ = all_tasks[(job_id, op_idx)]
            records.append({
                "job_id"    : job_id,
                "op_index"  : op_idx,
                "machine_id": machine_id,
                "start"     : solver.Value(start_var),
                "end"       : solver.Value(end_var),
                "duration"  : duration,
            })
    schedule_df = pd.DataFrame(records)

    # -- 9. SOLVER STATISTICS -------------------------------------------------
    obj_value = solver.ObjectiveValue()
    obj_bound = solver.BestObjectiveBound()
    gap = abs(obj_value - obj_bound) / max(obj_value, 1e-9) * 100.0

    solver_info = {
        "label"         : label,
        "status"        : status_str,
        "makespan"      : int(obj_value),
        "obj_bound"     : int(obj_bound),
        "optimality_gap": round(gap, 4),
        "runtime_sec"   : round(runtime, 4),
        "branches"      : solver.NumBranches(),
        "conflicts"     : solver.NumConflicts(),
    }

    return schedule_df, solver_info


def print_solver_report(solver_info: Dict):
    """Print a formatted solver statistics table."""
    print("\n-- CP-SAT Solver Report --------------------------------------")
    for k, v in solver_info.items():
        print(f"  {k:<20}: {v}")
    print()
