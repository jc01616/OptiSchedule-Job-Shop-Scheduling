# -*- coding: utf-8 -*-
"""
validate_example.py -- Manual Verification of a Small 3-Job, 2-Machine Example
================================================================================
Validates the CP-SAT solver and dispatching rules against a MANUALLY
COMPUTED example that can be verified by hand in under 5 minutes.

EXAMPLE (Baker 1974, textbook 3x2 job shop):
  Machine 0 (M0) then Machine 1 (M1):
    Job 0: (M0, 2), (M1, 3)
    Job 1: (M0, 4), (M1, 1)
    Job 2: (M0, 1), (M1, 2)

Manual computation (FCFS order J0, J1, J2):
  M0: J0[0-2], J1[2-6], J2[6-7]
  M1: J0[2-5], J1[6-7], J2[7-9]
  Makespan = 9

Known optimal (sequence J2, J0, J1):
  M0: J2[0-1], J0[1-3], J1[3-7]
  M1: J2[1-3], J0[3-6], J1[7-8]
  Makespan = 8

VALIDATION CHECKS:
  1. No machine processes two jobs simultaneously (no overlap)
  2. Each job's operations satisfy precedence order
  3. CP-SAT finds makespan = 8
  4. FCFS finds makespan = 9
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.dispatching_rules import _simulate_schedule, fcfs
from src.cpsat_solver import solve_jobshop

# ---- Validation Dataset -----------------------------------------------------
VAL_JOBS = [
    [(0, 2), (1, 3)],   # J0
    [(0, 4), (1, 1)],   # J1
    [(0, 1), (1, 2)],   # J2
]
KNOWN_OPTIMAL_MAKESPAN = 8
KNOWN_FCFS_MAKESPAN    = 9


def check_no_machine_overlap(schedule_df: pd.DataFrame, label: str) -> bool:
    """Verify no two operations on the same machine overlap in time."""
    passed = True
    for machine_id, group in schedule_df.groupby("machine_id"):
        ops = group.sort_values("start")[["job_id", "op_index", "start", "end"]].values
        for i in range(len(ops) - 1):
            if ops[i][3] > ops[i + 1][2]:  # end_i > start_{i+1}
                print(
                    f"  [OVERLAP] M{machine_id}: "
                    f"J{ops[i][0]}_op{ops[i][1]} ends at {ops[i][3]} but "
                    f"J{ops[i+1][0]}_op{ops[i+1][1]} starts at {ops[i+1][2]}"
                )
                passed = False
    return passed


def check_job_precedence(schedule_df: pd.DataFrame, jobs_data, label: str) -> bool:
    """Verify that within each job, operation k ends before operation k+1 starts."""
    passed = True
    for job_id, ops in enumerate(jobs_data):
        job_rows = schedule_df[schedule_df["job_id"] == job_id].sort_values("op_index")
        ends   = job_rows["end"].values
        starts = job_rows["start"].values
        for op_idx in range(len(ops) - 1):
            if ends[op_idx] > starts[op_idx + 1]:
                print(
                    f"  [PREC VIOLATION] J{job_id} op{op_idx} ends {ends[op_idx]} "
                    f"> op{op_idx+1} starts {starts[op_idx+1]}"
                )
                passed = False
    return passed


def run_validation():
    print("=" * 65)
    print("OptiSchedule -- Manual Validation Test")
    print("=" * 65)
    print(f"  Test dataset: 3 jobs, 2 machines")
    print(f"  Known FCFS makespan   : {KNOWN_FCFS_MAKESPAN}")
    print(f"  Known optimal makespan: {KNOWN_OPTIMAL_MAKESPAN}")
    print()

    all_passed = True

    # ---- TEST 1: FCFS --------------------------------------------------------
    print("-- Test 1: FCFS -------------------------------------------------")
    fcfs_sched, _ = fcfs(VAL_JOBS, ["J0", "J1", "J2"])
    fcfs_makespan  = int(fcfs_sched["end"].max())
    print("  FCFS schedule:")
    print(fcfs_sched[["job_id", "op_index", "machine_id", "start", "end"]].to_string(index=False))

    ok1 = check_no_machine_overlap(fcfs_sched, "FCFS")
    ok2 = check_job_precedence(fcfs_sched, VAL_JOBS, "FCFS")
    ok3 = (fcfs_makespan == KNOWN_FCFS_MAKESPAN)
    if not ok3:
        print(f"  [FAIL] Expected FCFS makespan {KNOWN_FCFS_MAKESPAN}, got {fcfs_makespan}")
    else:
        print(f"  [PASS] FCFS makespan = {fcfs_makespan} (matches expected {KNOWN_FCFS_MAKESPAN})")
    if ok1:
        print("  [PASS] No machine overlaps detected")
    if ok2:
        print("  [PASS] All job precedence constraints satisfied")

    all_passed = all_passed and ok1 and ok2 and ok3

    # ---- TEST 2: CP-SAT ------------------------------------------------------
    print("\n-- Test 2: CP-SAT Optimal ---------------------------------------")
    cpsat_sched, info = solve_jobshop(VAL_JOBS, time_limit_seconds=10.0)
    cpsat_makespan = info["makespan"]
    print("  CP-SAT schedule:")
    print(cpsat_sched[["job_id", "op_index", "machine_id", "start", "end"]].to_string(index=False))
    print(f"  Solver status     : {info['status']}")
    print(f"  Objective value   : {info['makespan']}")
    print(f"  Objective bound   : {info['obj_bound']}")
    print(f"  Optimality gap    : {info['optimality_gap']}%")
    print(f"  Runtime           : {info['runtime_sec']}s")

    ok4 = check_no_machine_overlap(cpsat_sched, "CP-SAT")
    ok5 = check_job_precedence(cpsat_sched, VAL_JOBS, "CP-SAT")
    ok6 = (cpsat_makespan == KNOWN_OPTIMAL_MAKESPAN)
    if not ok6:
        print(f"  [FAIL] Expected optimal makespan {KNOWN_OPTIMAL_MAKESPAN}, got {cpsat_makespan}")
    else:
        print(f"  [PASS] CP-SAT makespan = {cpsat_makespan} (matches known optimal)")
    if ok4:
        print("  [PASS] No machine overlaps detected")
    if ok5:
        print("  [PASS] All job precedence constraints satisfied")

    # Note: only report as PROVEN OPTIMAL if solver says OPTIMAL
    if info["status"] == "OPTIMAL":
        print("  [NOTE] Solver status = OPTIMAL: this is a proven globally optimal schedule")
    else:
        print(f"  [NOTE] Solver status = {info['status']}: solution is feasible but not proven optimal")

    all_passed = all_passed and ok4 and ok5 and ok6

    # ---- SUMMARY -------------------------------------------------------------
    print()
    print("=" * 65)
    if all_passed:
        print("  [ALL TESTS PASSED]")
    else:
        print("  [SOME TESTS FAILED] -- review output above")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
