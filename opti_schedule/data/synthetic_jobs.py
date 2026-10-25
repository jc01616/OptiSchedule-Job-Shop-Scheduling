"""
OptiSchedule — Synthetic CNC Machining Dataset
================================================
Source of Truth for all scheduling experiments.

WHY SYNTHETIC?
--------------
The Starjob dataset (https://github.com/starjob42/Starjob) is a 130,000-instance
dataset designed for fine-tuning Large Language Models on JSSP problems. Its fields
('input', 'output') are free-form natural-language text, NOT structured arrays.
OR-Tools CP-SAT requires deterministic tuples: (machine_id, duration). Parsing
natural language at scale is error-prone and unverifiable, so a documented synthetic
dataset is used instead (per project Requirement 4).

DATASET DESCRIPTION
-------------------
Scenario : A CNC machining workshop receives 8 customer orders (jobs).
           Each job must visit a subset of 4 machines in a fixed sequence.

Machines:
  M0 = Lathe       (turning and facing)
  M1 = Mill        (milling, slotting, profiling)
  M2 = Drill Press (drilling and reaming)
  M3 = Grinder     (surface finishing)

Jobs:
  Each job is a list of (machine_id, processing_time_minutes) tuples.
  The ORDER of tuples defines the mandatory operation sequence (routing).
  A job's next operation cannot start until the previous one is COMPLETE.

Due Dates:
  Due dates represent the customer-promised completion time in minutes
  from the start of the planning horizon.
"""

# --- MACHINE DEFINITIONS -------------------------------------------------------
MACHINES = {
    0: "Lathe",
    1: "Mill",
    2: "Drill Press",
    3: "Grinder",
}

# --- JOB NAMES (product types) -------------------------------------------------
JOB_NAMES = [
    "Shaft",     # J0
    "Bracket",   # J1
    "Flange",    # J2
    "Gear",      # J3
    "Pin",       # J4
    "Housing",   # J5
    "Cam",       # J6
    "Bushing",   # J7
]

# --- JOBS DATA ------------------------------------------------------------------
# Format: jobs_data[job_id] = [(machine_id, duration_minutes), ...]
# Each tuple is ONE OPERATION. Operations must be executed in the listed order.
JOBS_DATA = [
    [(0, 3), (1, 2), (2, 2)],   # J0 Shaft:   Lathe -> Mill -> Drill
    [(1, 4), (0, 3), (3, 2)],   # J1 Bracket: Mill -> Lathe -> Grinder
    [(2, 3), (1, 2), (0, 4)],   # J2 Flange:  Drill -> Mill -> Lathe
    [(0, 2), (3, 5), (1, 2)],   # J3 Gear:    Lathe -> Grinder -> Mill
    [(1, 3), (2, 1), (3, 4)],   # J4 Pin:     Mill -> Drill -> Grinder
    [(3, 2), (0, 3), (2, 3)],   # J5 Housing: Grinder -> Lathe -> Drill
    [(2, 4), (3, 2), (1, 3)],   # J6 Cam:     Drill -> Grinder -> Mill
    [(0, 1), (1, 2), (3, 3)],   # J7 Bushing: Lathe -> Mill -> Grinder
]

# --- DUE DATES (minutes from planning start) -----------------------------------
DUE_DATES = [12, 15, 18, 20, 14, 22, 25, 16]  # one per job

# --- VALIDATION -----------------------------------------------------------------
def validate_dataset():
    """Sanity-check the dataset for structural correctness."""
    assert len(JOBS_DATA) == len(JOB_NAMES), "Job count mismatch"
    assert len(JOBS_DATA) == len(DUE_DATES), "Due date count mismatch"
    machine_ids = set(MACHINES.keys())
    for j, ops in enumerate(JOBS_DATA):
        for op_idx, (m, d) in enumerate(ops):
            assert m in machine_ids, f"Job {j} op {op_idx}: unknown machine {m}"
            assert d > 0, f"Job {j} op {op_idx}: non-positive duration {d}"
    return True


# --- QUICK STATS ----------------------------------------------------------------
def dataset_summary():
    """Print a human-readable summary of the dataset."""
    validate_dataset()
    n_jobs = len(JOBS_DATA)
    n_machines = len(MACHINES)
    total_processing = sum(d for ops in JOBS_DATA for _, d in ops)

    machine_loads = {m: 0 for m in MACHINES}
    for ops in JOBS_DATA:
        for m, d in ops:
            machine_loads[m] += d

    print("=" * 60)
    print("OptiSchedule — CNC Machining Synthetic Dataset")
    print("=" * 60)
    print(f"  Jobs      : {n_jobs}")
    print(f"  Machines  : {n_machines}")
    print(f"  Operations: {sum(len(ops) for ops in JOBS_DATA)}")
    print(f"  Total work: {total_processing} minutes")
    print()
    print("  Machine loads (sum of all ops on each machine):")
    for m, load in machine_loads.items():
        print(f"    {MACHINES[m]:12s} (M{m}): {load} min")
    print()
    print("  Jobs:")
    print(f"  {'ID':<5} {'Name':<10} {'Route':<35} {'Due':>5}")
    print("  " + "-" * 60)
    for j, (name, ops, due) in enumerate(zip(JOB_NAMES, JOBS_DATA, DUE_DATES)):
        route = " -> ".join(f"M{m}({d})" for m, d in ops)
        print(f"  J{j:<4} {name:<10} {route:<35} {due:>5}")
    print("=" * 60)


if __name__ == "__main__":
    dataset_summary()
