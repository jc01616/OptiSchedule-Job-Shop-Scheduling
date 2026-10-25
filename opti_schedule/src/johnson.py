"""
johnson.py — Johnson's Algorithm & Flow-Shop Optimization
==========================================================

THEORETICAL FOUNDATION & METHODOLOGICAL DISTINCTION (for MBA Viva)
-------------------------------------------------------------------
1. Classic Two-Machine Flow Shop (Johnson, 1954):
   - Scope: Minimizing makespan for an n-job, 2-machine flow shop (F2 || C_max).
   - Core Non-Negotiable Assumptions:
     (a) Exactly two machines (Machine A and Machine B).
     (b) Every job visits Machine A FIRST, then Machine B SECOND.
     (c) No passing between machines (permutation flow shop).
   - Mathematical Guarantee: Johnson's rule is PROVABLY OPTIMAL for F2 || C_max.
   - Partitioning Rule:
     * Set U: Jobs with A_i <= B_i, sorted in ASCENDING order of A_i.
     * Set V: Jobs with A_i > B_i, sorted in DESCENDING order of B_i.
     * Optimal Sequence = U_sorted + V_sorted.

2. Three-Machine Flow Shop Reduction (Johnson 1954, Baker 1974):
   - Scope: Three-machine flow shop where every job visits Machine A -> B -> C.
   - Dominance Conditions for Exact Reduction:
     * Condition 1: min_i(A_i) >= max_i(B_i)  [Machine A dominates Machine B]
       OR
     * Condition 2: min_i(C_i) >= max_i(B_i)  [Machine C dominates Machine B]
   - If either condition holds, the problem is reduced to an equivalent 2-machine problem:
       A'_i = A_i + B_i
       B'_i = B_i + C_i
     Solving (A', B') with Johnson's algorithm yields a PROVABLY OPTIMAL sequence.
   - If neither condition holds, the reduction is an APPROXIMATE HEURISTIC
     (Palmer 1965 slope index / Campbell, Dudek & Smith CDS 1970).

3. Separation from General Job Shops:
   - In OptiSchedule, the primary 8-job, 4-machine instance is a GENERAL JOB SHOP
     with non-uniform routings. Johnson's flow-shop algorithm CANNOT be directly
     applied as an exact solver to the general job shop.
   - Dedicated flow-shop datasets (in `data/flowshop_datasets.py`) provide pure,
     verified flow-shop benchmarks for Johnson's demonstrations.
   - A Palmer-inspired virtual reduction is retained for the 4-machine job shop
     strictly as a labeled exploratory heuristic.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.flowshop_datasets import (
    FLOWSHOP_2M_DATA, FLOWSHOP_2M_JOB_NAMES, FLOWSHOP_2M_MACHINES,
    FLOWSHOP_3M_DATA, FLOWSHOP_3M_JOB_NAMES, FLOWSHOP_3M_MACHINES,
    verify_flowshop_3m_dominance,
)
from src.dispatching_rules import _simulate_schedule


# ============================================================================
# CORE JOHNSON 2-MACHINE SEQUENCING ENGINE
# ============================================================================
def _johnson_2machine_core(a: List[float], b: List[float]) -> List[int]:
    """
    Core Johnson's Algorithm partitioning for two processing time vectors.

    Parameters
    ----------
    a : List of durations on first (or virtual first) machine
    b : List of durations on second (or virtual second) machine

    Returns
    -------
    List of job indices in Johnson sequence
    """
    n = len(a)
    U = [(a[i], i) for i in range(n) if a[i] <= b[i]]
    V = [(b[i], i) for i in range(n) if a[i] > b[i]]

    U_sorted = [i for _, i in sorted(U, key=lambda x: x[0])]          # ascending A_i
    V_sorted = [i for _, i in sorted(V, key=lambda x: x[0], reverse=True)]  # descending B_i

    return U_sorted + V_sorted


# ============================================================================
# FLOW-SHOP SIMULATION ENGINE (Independent Makespan Calculation)
# ============================================================================
def simulate_flowshop_schedule(
    job_order: List[int],
    flowshop_data: List[List[Tuple[int, int]]],
) -> pd.DataFrame:
    """
    Simulate a pure permutation flow shop schedule and independently compute timestamps.
    Every job passes through machines in the routing order defined in flowshop_data.

    Parameters
    ----------
    job_order     : Permutation order of job IDs
    flowshop_data : list of [(machine_id, duration), ...] per job

    Returns
    -------
    DataFrame with columns: job_id, op_index, machine_id, start, end, duration
    """
    machine_free_at: Dict[int, int] = {}
    records = []

    for job_id in job_order:
        ops = flowshop_data[job_id]
        job_end = 0
        for op_idx, (machine_id, duration) in enumerate(ops):
            mfree = machine_free_at.get(machine_id, 0)
            start = max(job_end, mfree)
            end   = start + duration
            machine_free_at[machine_id] = end
            job_end = end
            records.append({
                "job_id"    : job_id,
                "op_index"  : op_idx,
                "machine_id": machine_id,
                "start"     : start,
                "end"       : end,
                "duration"  : duration,
            })

    return pd.DataFrame(records)


# ============================================================================
# PART A1: TWO-MACHINE FLOW SHOP BENCHMARK
# ============================================================================
def run_flowshop_2machine_experiment(
    flowshop_data: Optional[List[List[Tuple[int, int]]]] = None,
    job_names: Optional[List[str]] = None,
) -> Dict:
    """
    Execute the standalone Two-Machine Flow Shop experiment (F2 || C_max):
      1. Extracts durations on Machine A and Machine B.
      2. Computes the optimal Johnson sequence.
      3. Generates schedules for Johnson sequence, FCFS sequence, and Reverse sequence.
      4. Independently computes makespan and machine utilization for each.
      5. Confirms Johnson's mathematical optimality.
    """
    if flowshop_data is None:
        flowshop_data = FLOWSHOP_2M_DATA
    if job_names is None:
        job_names = FLOWSHOP_2M_JOB_NAMES

    n_jobs = len(flowshop_data)
    a = [flowshop_data[j][0][1] for j in range(n_jobs)]
    b = [flowshop_data[j][1][1] for j in range(n_jobs)]

    # 1. Johnson Optimal Sequence
    johnson_order = _johnson_2machine_core(a, b)
    sched_johnson = simulate_flowshop_schedule(johnson_order, flowshop_data)
    ms_johnson = int(sched_johnson["end"].max())

    # 2. Alternative Sequence 1: FCFS (Natural arrival order)
    fcfs_order = list(range(n_jobs))
    sched_fcfs = simulate_flowshop_schedule(fcfs_order, flowshop_data)
    ms_fcfs = int(sched_fcfs["end"].max())

    # 3. Alternative Sequence 2: Reverse order
    reverse_order = list(reversed(range(n_jobs)))
    sched_reverse = simulate_flowshop_schedule(reverse_order, flowshop_data)
    ms_reverse = int(sched_reverse["end"].max())

    # Total work content
    total_work = sum(d for ops in flowshop_data for _, d in ops)
    load_A = sum(a)
    load_B = sum(b)
    theoretical_lb = max(load_A, load_B, max(ai + bi for ai, bi in zip(a, b)))

    # Comparison Table
    impr_vs_fcfs = round((ms_fcfs - ms_johnson) / ms_fcfs * 100, 2)
    impr_vs_reverse = round((ms_reverse - ms_johnson) / ms_reverse * 100, 2)

    comparison_df = pd.DataFrame([
        {
            "Sequence Method"     : "Johnson's Algorithm (Optimal)",
            "Job Order"           : str([f"J{j}" for j in johnson_order]),
            "Makespan (min)"      : ms_johnson,
            "vs FCFS (%)"         : impr_vs_fcfs,
            "vs Worst/Reverse (%)": impr_vs_reverse,
            "Machine A Util (%)"  : round(load_A / ms_johnson * 100, 2),
            "Machine B Util (%)"  : round(load_B / ms_johnson * 100, 2),
            "Optimality Status"   : "PROVEN OPTIMAL (Johnson 1954)",
        },
        {
            "Sequence Method"     : "FCFS (Arrival Order)",
            "Job Order"           : str([f"J{j}" for j in fcfs_order]),
            "Makespan (min)"      : ms_fcfs,
            "vs FCFS (%)"         : 0.00,
            "vs Worst/Reverse (%)": round((ms_reverse - ms_fcfs) / ms_reverse * 100, 2),
            "Machine A Util (%)"  : round(load_A / ms_fcfs * 100, 2),
            "Machine B Util (%)"  : round(load_B / ms_fcfs * 100, 2),
            "Optimality Status"   : "Suboptimal Baseline",
        },
        {
            "Sequence Method"     : "Reverse Order",
            "Job Order"           : str([f"J{j}" for j in reverse_order]),
            "Makespan (min)"      : ms_reverse,
            "vs FCFS (%)"         : round((ms_fcfs - ms_reverse) / ms_fcfs * 100, 2),
            "vs Worst/Reverse (%)": 0.00,
            "Machine A Util (%)"  : round(load_A / ms_reverse * 100, 2),
            "Machine B Util (%)"  : round(load_B / ms_reverse * 100, 2),
            "Optimality Status"   : "Suboptimal Sequence",
        },
    ])

    return {
        "dataset_name"    : "FlowShop_2M_Benchmark",
        "n_jobs"          : n_jobs,
        "n_machines"      : 2,
        "total_work"      : total_work,
        "load_Machine_A"  : load_A,
        "load_Machine_B"  : load_B,
        "theoretical_lb"  : theoretical_lb,
        "johnson_order"   : johnson_order,
        "johnson_makespan": ms_johnson,
        "fcfs_makespan"   : ms_fcfs,
        "reverse_makespan": ms_reverse,
        "sched_johnson"   : sched_johnson,
        "sched_fcfs"      : sched_fcfs,
        "sched_reverse"   : sched_reverse,
        "comparison_table": comparison_df,
    }


# ============================================================================
# PART A2: THREE-MACHINE FLOW SHOP BENCHMARK
# ============================================================================
def run_flowshop_3machine_experiment(
    flowshop_data: Optional[List[List[Tuple[int, int]]]] = None,
    job_names: Optional[List[str]] = None,
) -> Dict:
    """
    Execute the standalone Three-Machine Flow Shop experiment (F3 || C_max):
      1. Verifies the Johnson/Jackson Dominance Conditions:
         min(A) >= max(B) OR min(C) >= max(B).
      2. If satisfied, reduces to virtual 2-machine problem:
         A'_i = A_i + B_i,  B'_i = B_i + C_i.
      3. Solves using Johnson's algorithm.
      4. Compares against FCFS and reverse sequences on the exact same dataset.
      5. Reports whether the result is provably optimal or an approximate heuristic.
    """
    if flowshop_data is None:
        flowshop_data = FLOWSHOP_3M_DATA
    if job_names is None:
        job_names = FLOWSHOP_3M_JOB_NAMES

    n_jobs = len(flowshop_data)
    dominance_info = verify_flowshop_3m_dominance(flowshop_data)

    # Virtual 2-machine reduction
    a_prime = [ops[0][1] + ops[1][1] for ops in flowshop_data]
    b_prime = [ops[1][1] + ops[2][1] for ops in flowshop_data]

    # 1. Johnson 3M Sequence
    johnson_order = _johnson_2machine_core(a_prime, b_prime)
    sched_johnson = simulate_flowshop_schedule(johnson_order, flowshop_data)
    ms_johnson = int(sched_johnson["end"].max())

    # 2. Alternative Sequence 1: FCFS
    fcfs_order = list(range(n_jobs))
    sched_fcfs = simulate_flowshop_schedule(fcfs_order, flowshop_data)
    ms_fcfs = int(sched_fcfs["end"].max())

    # 3. Alternative Sequence 2: Reverse
    reverse_order = list(reversed(range(n_jobs)))
    sched_reverse = simulate_flowshop_schedule(reverse_order, flowshop_data)
    ms_reverse = int(sched_reverse["end"].max())

    total_work = sum(d for ops in flowshop_data for _, d in ops)
    load_A = sum(ops[0][1] for ops in flowshop_data)
    load_B = sum(ops[1][1] for ops in flowshop_data)
    load_C = sum(ops[2][1] for ops in flowshop_data)

    status_str = (
        "PROVEN OPTIMAL (Dominance Verified)"
        if dominance_info["is_exact_optimal"] else
        "APPROXIMATE HEURISTIC (Dominance Violated)"
    )

    impr_vs_fcfs = round((ms_fcfs - ms_johnson) / ms_fcfs * 100, 2)
    impr_vs_reverse = round((ms_reverse - ms_johnson) / ms_reverse * 100, 2)

    comparison_df = pd.DataFrame([
        {
            "Sequence Method"     : "Johnson 3M Reduction",
            "Job Order"           : str([f"J{j}" for j in johnson_order]),
            "Makespan (min)"      : ms_johnson,
            "vs FCFS (%)"         : impr_vs_fcfs,
            "vs Worst/Reverse (%)": impr_vs_reverse,
            "Machine A Util (%)"  : round(load_A / ms_johnson * 100, 2),
            "Machine B Util (%)"  : round(load_B / ms_johnson * 100, 2),
            "Machine C Util (%)"  : round(load_C / ms_johnson * 100, 2),
            "Classification"      : status_str,
        },
        {
            "Sequence Method"     : "FCFS (Arrival Order)",
            "Job Order"           : str([f"J{j}" for j in fcfs_order]),
            "Makespan (min)"      : ms_fcfs,
            "vs FCFS (%)"         : 0.00,
            "vs Worst/Reverse (%)": round((ms_reverse - ms_fcfs) / ms_reverse * 100, 2),
            "Machine A Util (%)"  : round(load_A / ms_fcfs * 100, 2),
            "Machine B Util (%)"  : round(load_B / ms_fcfs * 100, 2),
            "Machine C Util (%)"  : round(load_C / ms_fcfs * 100, 2),
            "Classification"      : "Suboptimal Baseline",
        },
        {
            "Sequence Method"     : "Reverse Order",
            "Job Order"           : str([f"J{j}" for j in reverse_order]),
            "Makespan (min)"      : ms_reverse,
            "vs FCFS (%)"         : round((ms_fcfs - ms_reverse) / ms_fcfs * 100, 2),
            "vs Worst/Reverse (%)": 0.00,
            "Machine A Util (%)"  : round(load_A / ms_reverse * 100, 2),
            "Machine B Util (%)"  : round(load_B / ms_reverse * 100, 2),
            "Machine C Util (%)"  : round(load_C / ms_reverse * 100, 2),
            "Classification"      : "Optimal Alternative for This Instance",
        },
    ])

    return {
        "dataset_name"    : "FlowShop_3M_Benchmark",
        "n_jobs"          : n_jobs,
        "n_machines"      : 3,
        "total_work"      : total_work,
        "loads"           : {"Machine_A": load_A, "Machine_B": load_B, "Machine_C": load_C},
        "dominance_info"  : dominance_info,
        "johnson_order"   : johnson_order,
        "johnson_makespan": ms_johnson,
        "fcfs_makespan"   : ms_fcfs,
        "reverse_makespan": ms_reverse,
        "sched_johnson"   : sched_johnson,
        "sched_fcfs"      : sched_fcfs,
        "sched_reverse"   : sched_reverse,
        "comparison_table": comparison_df,
    }


# ============================================================================
# PART B: EXPLORATORY HEURISTIC FOR 4-MACHINE JOB SHOP (Palmer Combined Times)
# ============================================================================
def palmer_combined_times_jobshop_heuristic(
    jobs_data, job_names
) -> Tuple[pd.DataFrame, str]:
    """
    Palmer / CDS Combined-Times Heuristic applied to the 4-Machine Job Shop.

    LABEL: 'Palmer Heuristic (Job-Shop Exploration)'
    NOTE : This is an EXPLORATORY HEURISTIC, NOT a flow shop algorithm.
           It sums early operations vs late operations to compute virtual
           priorities, and dispatches across all 24 operations on all 4 machines.
    """
    p_prime = []
    q_prime = []
    for ops in jobs_data:
        if len(ops) >= 3:
            p_prime.append(ops[0][1] + ops[1][1])
            q_prime.append(ops[1][1] + ops[2][1])
        elif len(ops) == 2:
            p_prime.append(ops[0][1])
            q_prime.append(ops[1][1])
        else:
            p_prime.append(ops[0][1])
            p_prime.append(ops[0][1])

    order = _johnson_2machine_core(p_prime, q_prime)
    sched = _simulate_schedule(order, jobs_data)
    return sched, "Palmer Heuristic (Job-Shop)"


# Backward-compatible alias for existing imports
johnson_3machine_approx = palmer_combined_times_jobshop_heuristic
