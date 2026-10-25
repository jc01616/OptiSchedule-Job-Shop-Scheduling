"""
metrics.py — KPI Calculation Module
=====================================
Computes all key performance indicators from a completed schedule.

METRICS EXPLAINED (for MBA Viva)
----------------------------------
Makespan (C_max):
  Total elapsed time from the start of the first operation to the completion
  of the last. The PRIMARY objective to minimise in job shop scheduling.

Machine Utilization (U_m):
  U_m = (Total processing time on machine m / Makespan) * 100%
  Idle time = Makespan - Total processing time on machine m.
  Average utilization is the unweighted mean across all active machines.

Job Completion Times (C_j):
  The time at which the final operation of job j finishes.

Tardiness (T_j):
  T_j = max(0, C_j - d_j), where d_j is the due date.
  T_j = 0 indicates on-time delivery; positive values represent late delivery.
  Key metrics: Mean Tardiness, Maximum Tardiness, % Jobs Tardy.

Throughput:
  Jobs per unit time = n_jobs / Makespan.

Makespan Improvement vs Baseline (%):
  ((Baseline_makespan - Improved_makespan) / Baseline_makespan) * 100%
  A positive value represents a percentage reduction in makespan.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.synthetic_jobs import MACHINES, JOB_NAMES, DUE_DATES


def compute_metrics(
    schedule_df: pd.DataFrame,
    method_name: str,
    due_dates: Optional[List[int]] = None,
    machines: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Compute all KPIs from a finished schedule.

    Parameters
    ----------
    schedule_df : output of any dispatcher or CP-SAT solver
                  Columns: job_id, op_index, machine_id, start, end, duration
    method_name : label for the method (e.g. 'FCFS', 'CP-SAT Optimal')
    due_dates   : list of due dates (minutes), one per job_id
    machines    : machine dictionary mapping ID to name

    Returns
    -------
    dict of named KPIs
    """
    if due_dates is None:
        due_dates = DUE_DATES
    if machines is None:
        machines = MACHINES

    makespan = int(schedule_df["end"].max())

    # -- Job completion times --------------------------------------------------
    completion_times = (
        schedule_df.groupby("job_id")["end"].max().sort_index().tolist()
    )

    # -- Signed Lateness (L_j = C_j - d_j) -------------------------------------
    latenesses = [
        ct - dd
        for ct, dd in zip(completion_times, due_dates)
    ]
    avg_lateness = float(np.mean(latenesses))

    # -- Tardiness (T_j = max(0, C_j - d_j)) -----------------------------------
    tardinesses = [
        max(0, l)
        for l in latenesses
    ]
    avg_tardiness  = float(np.mean(tardinesses))
    max_tardiness  = float(np.max(tardinesses))
    pct_tardy      = sum(1 for t in tardinesses if t > 0) / len(tardinesses) * 100

    # -- Machine utilization ----------------------------------------------------
    machine_loads = (
        schedule_df.groupby("machine_id")["duration"].sum().to_dict()
    )
    machine_util = {
        f"M{m} {machines.get(m, f'Machine {m}')}": round(machine_loads.get(m, 0) / makespan * 100, 2)
        for m in sorted(machines.keys())
    }
    avg_utilization = float(np.mean(list(machine_util.values())))

    # -- Throughput (jobs/hr benchmark rate & jobs/min) ------------------------
    n_jobs        = schedule_df["job_id"].nunique()
    throughput_hr = round(n_jobs / (makespan / 60.0), 2)  # jobs per hour
    throughput    = round(n_jobs / makespan, 4)            # jobs per minute

    # -- Mean flow time --------------------------------------------------------
    mean_flow_time = float(np.mean(completion_times))

    return {
        "method"             : method_name,
        "makespan"           : makespan,
        "mean_completion"    : round(mean_flow_time, 2),
        "avg_lateness"       : round(avg_lateness, 2),
        "avg_tardiness"      : round(avg_tardiness, 2),
        "max_tardiness"      : round(max_tardiness, 2),
        "pct_jobs_tardy"     : round(pct_tardy, 1),
        "avg_utilization_pct": round(avg_utilization, 2),
        "throughput"         : throughput,
        "throughput_hr"      : throughput_hr,
        "machine_util"       : machine_util,
        "completion_times"   : completion_times,
        "latenesses"         : latenesses,
        "tardinesses"        : tardinesses,
    }


def improvement_vs_baseline(baseline_makespan: int, improved_makespan: int) -> float:
    """Calculate makespan improvement percentage (relative reduction)."""
    if baseline_makespan == 0:
        return 0.0
    return round((baseline_makespan - improved_makespan) / baseline_makespan * 100, 2)


def print_metrics_report(metrics: Dict):
    """Pretty-print a single method's KPI report."""
    print(f"\n-- {metrics['method']} KPIs --------------------------------------")
    print(f"  Makespan             : {metrics['makespan']} min")
    print(f"  Mean Completion Time : {metrics['mean_completion']} min")
    print(f"  Avg Lateness         : {metrics['avg_lateness']:+.2f} min")
    print(f"  Avg Tardiness        : {metrics['avg_tardiness']} min")
    print(f"  Max Tardiness        : {metrics['max_tardiness']} min")
    print(f"  % Jobs Tardy         : {metrics['pct_jobs_tardy']}%")
    print(f"  Avg Machine Util     : {metrics['avg_utilization_pct']}%")
    print(f"  Throughput           : {metrics['throughput_hr']} jobs/hr ({metrics['throughput']} jobs/min)")
    print(f"  Machine Utilizations :")
    for machine, util in metrics["machine_util"].items():
        print(f"    {machine:<20}: {util}%")
    print(f"  Completion Times     : {metrics['completion_times']}")
    print(f"  Signed Latenesses    : {metrics['latenesses']}")
    print(f"  Tardinesses          : {metrics['tardinesses']}")
    print()


def build_comparison_table(all_metrics: List[Dict], baseline_method: str = "FCFS") -> pd.DataFrame:
    """
    Build a clean summary DataFrame comparing full job-shop methods.
    Excludes partial subproblem schedules (e.g., Johnson 2M) to maintain
    mathematical comparability across methods.
    """
    # Filter for full job-shop methods only
    full_metrics = [
        m for m in all_metrics
        if "subproblem" not in m["method"].lower() and "2m" not in m["method"].lower()
    ]
    # Find baseline
    baseline_entry = next((m for m in full_metrics if m["method"] == baseline_method), full_metrics[0])
    baseline_ms = baseline_entry["makespan"]

    rows = []
    for m in full_metrics:
        impr = improvement_vs_baseline(baseline_ms, m["makespan"])
        rows.append({
            "Method"              : m["method"],
            "Makespan (min)"      : m["makespan"],
            "vs Baseline (%)"     : impr,
            "Avg Lateness (min)"  : m["avg_lateness"],
            "Avg Tardiness (min)" : m["avg_tardiness"],
            "Max Tardiness (min)" : m["max_tardiness"],
            "% Tardy Jobs"        : m["pct_jobs_tardy"],
            "Avg Util (%)"        : m["avg_utilization_pct"],
            "Throughput (jobs/hr)": m["throughput_hr"],
        })
    return pd.DataFrame(rows)


def build_subproblem_table(j2m_metrics: Dict, subproblem_baseline_ms: int = 27) -> pd.DataFrame:
    """
    Build a dedicated summary table for the Johnson 2-Machine subproblem.
    States scope, operation coverage, and compares against the 2-op subproblem baseline.
    """
    impr = improvement_vs_baseline(subproblem_baseline_ms, j2m_metrics["makespan"])
    row = {
        "Subproblem"             : "Johnson 2M Flow-Shop Subproblem",
        "Scope / Coverage"       : "Ops 0 & 1 only (16 of 24 ops, 42 of 65 min work)",
        "Makespan (min)"         : j2m_metrics["makespan"],
        "Subproblem Baseline (min)": subproblem_baseline_ms,
        "vs Subproblem Baseline (%)": impr,
        "Avg Tardiness (min)"    : j2m_metrics["avg_tardiness"],
        "Max Tardiness (min)"    : j2m_metrics["max_tardiness"],
        "% Tardy"                : j2m_metrics["pct_jobs_tardy"],
        "Avg Util (%)"           : j2m_metrics["avg_utilization_pct"],
        "Throughput"             : j2m_metrics["throughput"],
    }
    return pd.DataFrame([row])
