"""
dispatching_rules.py — Priority Dispatching Heuristics
=======================================================
Implements FCFS, SPT, LPT, and EDD priority dispatching heuristics.

METHODOLOGY & ARCHITECTURAL DOCUMENTATION (for MBA Viva / Capstone Review)
--------------------------------------------------------------------------
1. Implementation Mechanism: Sequential List Scheduling with Job-Level Commitment
   - The heuristics sort jobs into a static permutation order based on their priority key
     (e.g., arrival order for FCFS, total processing time for SPT/LPT, due date for EDD).
   - The simulation engine (`_simulate_schedule`) iterates through jobs one by one in that
     order. For each job, all of its operations are sequentially scheduled on their required
     machines at the earliest valid start time.
   - Machine non-overlap is guaranteed by tracking `machine_free_at[m]`.
   - Precedence constraints are guaranteed by tracking `job_end`.

2. Contrast with True Dynamic (Event-Driven) Dispatching:
   - Dynamic Event-Driven Dispatching: When a machine finishes a task at time t, it inspects
     the queue of all jobs currently waiting at that machine and chooses the next job by priority.
     This allows jobs to interleave freely and fill downstream idle gaps.
   - Sequential List Scheduling: Commits all operations of Job A before examining Job B.
     If Job A requires Machine 2 starting at t=7, and Machine 2 is idle from t=0 to t=7,
     Job B cannot be placed in the [0, 7] slot if it appears later in the job sequence.
   - Operational Consequence: Sequential list scheduling creates larger machine idle blocks,
     resulting in longer makespans (e.g., FCFS = 55 min). This transparently illustrates
     why basic queue heuristics without lookahead or dynamic interleaving fall far short
     of mathematical optimization.

3. Feasibility & Optimality:
   - Feasibility Guarantee: Every generated schedule is verified to be 100% FEASIBLE,
     strictly satisfying precedence (op k+1 starts after op k finishes) and machine
     exclusivity (no two jobs on the same machine simultaneously).
   - Non-Optimality: These heuristics do NOT guarantee optimality for job shops (which is NP-hard).
     They serve as operational baselines against which exact optimization (OR-Tools CP-SAT)
     is rigorously evaluated.
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np


def _simulate_schedule(job_order: List[int], jobs_data) -> pd.DataFrame:
    """
    Core simulation engine for all dispatching rules.

    Given a job ordering (permutation of job IDs), simulate the shop using
    sequential list scheduling with job-level commitment:
      - Jobs are processed in the specified permutation order.
      - For each job, operations are committed sequentially across its route.
      - A machine processes at most one operation at any point in time.
      - An operation on machine M cannot start before:
          (a) the preceding operation of the same job completes (precedence constraint), AND
          (b) machine M becomes free (machine capacity constraint).

    Parameters
    ----------
    job_order  : List of job IDs in the priority sequence
    jobs_data  : JOBS_DATA list of [(machine_id, duration), ...]

    Returns
    -------
    DataFrame with columns:
        job_id, op_index, machine_id, start, end, duration
    """
    # Track the earliest availability timestamp for each machine
    machine_free_at: Dict[int, int] = {}

    records = []
    for job_id in job_order:
        ops = jobs_data[job_id]
        job_end = 0  # Earliest start time for the next operation of this job (precedence)
        for op_idx, (machine_id, duration) in enumerate(ops):
            mfree = machine_free_at.get(machine_id, 0)
            # Earliest start satisfies BOTH precedence and machine exclusivity
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


def fcfs(jobs_data, job_names) -> Tuple[pd.DataFrame, str]:
    """
    First Come, First Served (FCFS / FIFO)
    ----------------------------------------
    Jobs are scheduled in their natural arrival order (job_id 0, 1, 2, ...).

    OR Concept: FCFS is the standard industrial baseline representing a simple
    unprioritized queue. It is intuitive and perceived as fair, but typically produces
    poor makespan and high work-in-process (WIP) inventory in job shops.
    """
    order = list(range(len(jobs_data)))
    sched = _simulate_schedule(order, jobs_data)
    return sched, "FCFS"


def spt(jobs_data, job_names) -> Tuple[pd.DataFrame, str]:
    """
    Shortest Processing Time (SPT)
    --------------------------------
    Jobs are ordered by ASCENDING total processing time.

    OR Concept: SPT is mathematically proven to minimize mean flow time and mean
    waiting time on a single machine (Smith, 1956; Baker, 1974). In multi-machine
    job shops, it accelerates quick jobs through the shop, reducing average work-in-process,
    though it can delay long jobs.
    """
    totals = [(sum(d for _, d in ops), j) for j, ops in enumerate(jobs_data)]
    order = [j for _, j in sorted(totals)]
    sched = _simulate_schedule(order, jobs_data)
    return sched, "SPT"


def lpt(jobs_data, job_names) -> Tuple[pd.DataFrame, str]:
    """
    Longest Processing Time (LPT)
    --------------------------------
    Jobs are ordered by DESCENDING total processing time.

    OR Concept: LPT places the heaviest jobs on machines first. While counterproductive
    for mean flow time on single machines, it is a classic heuristic in parallel machine
    scheduling (Graham, 1969) because early scheduling of large tasks can balance workload.
    """
    totals = [(sum(d for _, d in ops), j) for j, ops in enumerate(jobs_data)]
    order = [j for _, j in sorted(totals, reverse=True)]
    sched = _simulate_schedule(order, jobs_data)
    return sched, "LPT"


def edd(jobs_data, job_names, due_dates) -> Tuple[pd.DataFrame, str]:
    """
    Earliest Due Date (EDD)
    -------------------------
    Jobs are ordered by ASCENDING due date.

    OR Concept: EDD is proven optimal for minimizing MAXIMUM LATENESS (L_max)
    on a single machine (Jackson's Rule, 1955). In job shops, it prioritizes orders
    with imminent customer commitments to protect on-time delivery service levels.

    Parameters
    ----------
    due_dates : list of due dates, one per job_id
    """
    order = [j for j, _ in sorted(enumerate(due_dates), key=lambda x: x[1])]
    sched = _simulate_schedule(order, jobs_data)
    return sched, "EDD"
