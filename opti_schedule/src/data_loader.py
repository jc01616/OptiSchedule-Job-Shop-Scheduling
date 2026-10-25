"""
data_loader.py — Dataset Loading & Validation
==============================================
Converts the raw synthetic dataset into validated Pandas DataFrames
that all downstream modules consume. This is the single interface
between data and algorithms.
"""
import pandas as pd
import numpy as np
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.synthetic_jobs import JOBS_DATA, JOB_NAMES, MACHINES, DUE_DATES, validate_dataset


def load_operations() -> pd.DataFrame:
    """
    Build a flat operations table. Each row is ONE operation.

    Columns
    -------
    job_id       : int   — index into JOBS_DATA
    job_name     : str   — human-readable job label
    op_index     : int   — 0-based operation sequence number within the job
    machine_id   : int   — which machine processes this operation
    machine_name : str   — human-readable machine label
    duration     : int   — processing time in minutes
    due_date     : int   — job-level due date (minutes from horizon start)
    total_job_pt : int   — sum of all processing times for this job (for SPT/LPT)
    """
    validate_dataset()
    rows = []
    for job_id, (name, ops, due) in enumerate(zip(JOB_NAMES, JOBS_DATA, DUE_DATES)):
        total_pt = sum(d for _, d in ops)
        for op_idx, (machine_id, duration) in enumerate(ops):
            rows.append({
                "job_id"      : job_id,
                "job_name"    : name,
                "op_index"    : op_idx,
                "machine_id"  : machine_id,
                "machine_name": MACHINES[machine_id],
                "duration"    : duration,
                "due_date"    : due,
                "total_job_pt": total_pt,
            })
    df = pd.DataFrame(rows)
    return df


def load_jobs_summary() -> pd.DataFrame:
    """
    One row per job: aggregated job-level information.

    Columns: job_id, job_name, n_operations, total_pt, due_date
    """
    validate_dataset()
    rows = []
    for job_id, (name, ops, due) in enumerate(zip(JOB_NAMES, JOBS_DATA, DUE_DATES)):
        rows.append({
            "job_id"      : job_id,
            "job_name"    : name,
            "n_operations": len(ops),
            "total_pt"    : sum(d for _, d in ops),
            "due_date"    : due,
        })
    return pd.DataFrame(rows)


def machine_load_summary(ops_df: pd.DataFrame) -> pd.DataFrame:
    """
    Total processing time per machine across all jobs.
    Useful for computing machine utilization %.
    """
    return (
        ops_df.groupby(["machine_id", "machine_name"])["duration"]
        .sum()
        .reset_index()
        .rename(columns={"duration": "total_load_min"})
        .sort_values("machine_id")
    )


def print_ops_table(ops_df: pd.DataFrame):
    """Print a nicely formatted operations table."""
    print("\n-- Operations Table ------------------------------------------")
    print(ops_df[["job_id", "job_name", "op_index", "machine_name",
                  "duration", "due_date", "total_job_pt"]].to_string(index=False))
    print()
