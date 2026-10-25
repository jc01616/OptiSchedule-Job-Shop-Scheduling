"""
flowshop_datasets.py -- Dedicated Flow Shop Benchmark Datasets
=============================================================
Provides pure, verified flow-shop instances where every job visits
the identical sequence of machines in the exact same order.

Used exclusively for Johnson's Algorithm demonstrations to ensure
methodological validity and avoid conflation with the 4-machine job shop.
"""
from typing import List, Tuple, Dict

# ============================================================================
# 1. TWO-MACHINE FLOW SHOP DATASET (F2 || C_max)
# ============================================================================
# Pure 2-machine flow shop: Machine A (Lathe) -> Machine B (Mill)
# Every job visits Machine 0 then Machine 1.

FLOWSHOP_2M_MACHINES: Dict[int, str] = {
    0: "Machine A (Lathe)",
    1: "Machine B (Mill)",
}

FLOWSHOP_2M_JOB_NAMES: List[str] = [
    "Part 1 (Axle)",
    "Part 2 (Spindle)",
    "Part 3 (Collar)",
    "Part 4 (Rotor)",
    "Part 5 (Sleeve)",
    "Part 6 (Hub)",
]

# Format: [(mach_id, dur), (mach_id, dur)]
# Processing times:
#   Job 0: A=3, B=6 -> Set U (A <= B, A=3)
#   Job 1: A=7, B=2 -> Set V (A > B,  B=2)
#   Job 2: A=4, B=5 -> Set U (A <= B, A=4)
#   Job 3: A=5, B=7 -> Set U (A <= B, A=5)
#   Job 4: A=8, B=3 -> Set V (A > B,  B=3)
#   Job 5: A=2, B=4 -> Set U (A <= B, A=2)
#
# Johnson sequence:
#   Set U sorted ascending A: J5(2), J0(3), J2(4), J3(5)
#   Set V sorted descending B: J4(3), J1(2)
#   Optimal Sequence: [J5, J0, J2, J3, J4, J1] (Makespan = 31 min)
#   Baseline FCFS [J0..J5] Makespan = 34 min
#   Reverse Sequence [J5, J4, J3, J2, J1, J0] Makespan = 35 min
FLOWSHOP_2M_DATA: List[List[Tuple[int, int]]] = [
    [(0, 3), (1, 6)],  # J0
    [(0, 7), (1, 2)],  # J1
    [(0, 4), (1, 5)],  # J2
    [(0, 5), (1, 7)],  # J3
    [(0, 8), (1, 3)],  # J4
    [(0, 2), (1, 4)],  # J5
]


# ============================================================================
# 2. THREE-MACHINE FLOW SHOP DATASET (F3 || C_max)
# ============================================================================
# Pure 3-machine flow shop: Machine A (Turning) -> Machine B (Milling) -> Machine C (Grinding)
# Every job visits Machine 0 -> Machine 1 -> Machine 2.

FLOWSHOP_3M_MACHINES: Dict[int, str] = {
    0: "Machine A (Turning)",
    1: "Machine B (Milling)",
    2: "Machine C (Grinding)",
}

FLOWSHOP_3M_JOB_NAMES: List[str] = [
    "Order 1 (Drive Shaft)",
    "Order 2 (Coupling)",
    "Order 3 (Ring Gear)",
    "Order 4 (Pinion)",
    "Order 5 (Impeller)",
]

# Processing times:
#   Job 0: A=6, B=2, C=5
#   Job 1: A=8, B=3, C=4
#   Job 2: A=7, B=1, C=8
#   Job 3: A=5, B=2, C=6
#   Job 4: A=9, B=3, C=7
#
# Dominance Condition Check:
#   min(A) = 5, max(B) = 3, min(C) = 4
#   Condition 1: min(A) >= max(B) (5 >= 3) -> TRUE (SATISFIED)
#   Condition 2: min(C) >= max(B) (4 >= 3) -> TRUE (SATISFIED)
#
# Because the middle machine (B) is dominated, Johnson's reduction:
#   A'_i = A_i + B_i
#   B'_i = B_i + C_i
# produces a PROVEN OPTIMAL sequence: [J3, J2, J4, J0, J1] (Makespan = 42 min)
# FCFS Sequence [J0..J4] Makespan = 45 min
# Reverse Sequence [J4, J3, J2, J1, J0] Makespan = 42 min (Optimal Alternative for This Instance)
FLOWSHOP_3M_DATA: List[List[Tuple[int, int]]] = [
    [(0, 6), (1, 2), (2, 5)],  # J0
    [(0, 8), (1, 3), (2, 4)],  # J1
    [(0, 7), (1, 1), (2, 8)],  # J2
    [(0, 5), (1, 2), (2, 6)],  # J3
    [(0, 9), (1, 3), (2, 7)],  # J4
]


def verify_flowshop_3m_dominance(flowshop_3m_data: List[List[Tuple[int, int]]]) -> Dict:
    """
    Check Johnson / Jackson dominance conditions for 3-machine flow shop reduction:
      Condition 1: min_i(A_i) >= max_i(B_i)
      Condition 2: min_i(C_i) >= max_i(B_i)
    If at least one condition holds, Johnson's algorithm reduction is guaranteed optimal.
    If neither holds, the reduction is an approximate heuristic (Palmer / CDS slope index).
    """
    a_vals = [ops[0][1] for ops in flowshop_3m_data]
    b_vals = [ops[1][1] for ops in flowshop_3m_data]
    c_vals = [ops[2][1] for ops in flowshop_3m_data]

    min_a = min(a_vals)
    max_b = max(b_vals)
    min_c = min(c_vals)

    c1 = (min_a >= max_b)
    c2 = (min_c >= max_b)
    is_exact_optimal = c1 or c2

    return {
        "min_A": min_a,
        "max_B": max_b,
        "min_C": min_c,
        "condition_1_satisfied": c1,
        "condition_2_satisfied": c2,
        "is_exact_optimal": is_exact_optimal,
        "classification": "EXACT_OPTIMUM" if is_exact_optimal else "APPROXIMATE_HEURISTIC",
        "explanation": (
            f"Condition 1 (min A >= max B: {min_a} >= {max_b}) is {c1}; "
            f"Condition 2 (min C >= max B: {min_c} >= {max_b}) is {c2}. "
            + ("Dominance holds: Johnson reduction is mathematically proven optimal."
               if is_exact_optimal else
               "Dominance does not hold: Johnson reduction is an approximate heuristic.")
        )
    }
