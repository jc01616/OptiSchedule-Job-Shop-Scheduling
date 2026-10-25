"""
visualizer.py — Gantt Charts & Comparison Plots
================================================
All charts are produced using both Matplotlib (static PNG) and Plotly
(interactive HTML). Charts are saved to outputs/.
"""
from __future__ import annotations
from typing import List, Dict, Optional
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.synthetic_jobs import MACHINES, JOB_NAMES


# -- Color palette for jobs ----------------------------------------------------
JOB_COLORS_PLOTLY = px.colors.qualitative.Bold
JOB_COLORS_MPL    = list(plt.get_cmap("tab10").colors)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _job_color_mpl(job_id: int) -> str:
    return JOB_COLORS_MPL[job_id % len(JOB_COLORS_MPL)]


def _job_color_plotly(job_id: int) -> str:
    return JOB_COLORS_PLOTLY[job_id % len(JOB_COLORS_PLOTLY)]


# -- 1. GANTT CHART (Matplotlib) -----------------------------------------------
def gantt_matplotlib(
    schedule_df: pd.DataFrame,
    title: str,
    filename: str,
    job_names: List[str] = JOB_NAMES,
    machines: Dict[int, str] = MACHINES,
):
    """
    Static Gantt chart using Matplotlib.
    Y-axis = machines, X-axis = time, each bar = one operation.
    """
    n_machines = len(machines)
    fig, ax = plt.subplots(figsize=(14, 5))

    for _, row in schedule_df.iterrows():
        job_id     = int(row["job_id"])
        machine_id = int(row["machine_id"])
        start      = float(row["start"])
        duration   = float(row["duration"])

        bar = ax.barh(
            y=machine_id,
            width=duration,
            left=start,
            height=0.6,
            color=_job_color_mpl(job_id),
            edgecolor="white",
            linewidth=0.8,
        )
        # Label each bar with job and operation
        op_idx = int(row["op_index"]) if "op_index" in row else 0
        label_text = f"J{job_id}.O{op_idx}" if duration >= 2.0 else f"J{job_id}"
        ax.text(
            start + duration / 2,
            machine_id,
            label_text,
            ha="center", va="center",
            fontsize=7, color="white", fontweight="bold",
        )

    ax.set_yticks(list(machines.keys()))
    ax.set_yticklabels([f"M{m}: {machines[m]}" for m in sorted(machines.keys())])
    ax.set_xlabel("Time (minutes)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(color=_job_color_mpl(j), label=f"J{j}: {job_names[j]}")
        for j in range(len(job_names))
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, framealpha=0.9)

    makespan = int(schedule_df["end"].max())
    ax.axvline(makespan, color="red", linestyle="--", linewidth=1.5, label=f"Makespan={makespan}")
    ax.set_xlim(0, makespan + 2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")
    return path


# -- 2. GANTT CHART (Plotly) ---------------------------------------------------
def gantt_plotly(
    schedule_df: pd.DataFrame,
    title: str,
    filename: str,
    job_names: List[str] = JOB_NAMES,
    machines: Dict[int, str] = MACHINES,
):
    """
    Interactive Gantt chart using Plotly.
    """
    fig = go.Figure()
    makespan = int(schedule_df["end"].max())

    added_jobs = set()
    for _, row in schedule_df.iterrows():
        job_id     = int(row["job_id"])
        machine_id = int(row["machine_id"])
        start      = float(row["start"])
        duration   = float(row["duration"])
        end        = float(row["end"])
        op_idx     = int(row["op_index"]) if "op_index" in row else 0
        machine_name = machines.get(machine_id, f"Machine {machine_id}")
        job_name   = job_names[job_id] if job_id < len(job_names) else f"Job {job_id}"
        show_legend = job_id not in added_jobs
        added_jobs.add(job_id)

        fig.add_trace(go.Bar(
            x=[duration],
            base=[start],
            y=[f"M{machine_id}: {machine_name}"],
            orientation="h",
            name=f"J{job_id}: {job_name}",
            marker_color=_job_color_plotly(job_id),
            showlegend=show_legend,
            hovertemplate=(
                f"<b>J{job_id}: {job_name} (Op {op_idx})</b><br>"
                f"Machine: M{machine_id} {machine_name}<br>"
                f"Start: {start:.0f} min | End: {end:.0f} min<br>"
                f"Duration: {duration:.0f} min"
                "<extra></extra>"
            ),
        ))

    fig.add_vline(
        x=makespan, line_dash="dash", line_color="red",
        annotation_text=f"Makespan = {makespan}", annotation_position="top right"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        barmode="stack",
        xaxis_title="Time (minutes)",
        yaxis_title="Machine",
        xaxis=dict(range=[0, makespan + 2]),
        legend=dict(orientation="v", x=1.01),
        height=350,
        plot_bgcolor="#f9f9f9",
    )

    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(path)
    print(f"  [Saved] {path}")
    return path


# -- 3. MAKESPAN COMPARISON BAR CHART -----------------------------------------
def makespan_comparison(all_metrics: List[Dict], filename_prefix: str):
    """Bar chart comparing makespan across all methods."""
    methods   = [m["method"] for m in all_metrics]
    makespans = [m["makespan"] for m in all_metrics]
    colors_plotly = [_job_color_plotly(i) for i in range(len(methods))]

    # Matplotlib version
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(methods, makespans,
                  color=[_job_color_mpl(i) for i in range(len(methods))],
                  edgecolor="white")
    for bar, ms in zip(bars, makespans):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(ms), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Makespan (minutes)")
    ax.set_title("Makespan Comparison — All Methods", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    mpl_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    fig.savefig(mpl_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {mpl_path}")

    # Plotly version
    fig_p = go.Figure(go.Bar(
        x=methods, y=makespans, marker_color=colors_plotly,
        text=makespans, textposition="outside"
    ))
    fig_p.update_layout(
        title="Makespan Comparison — All Methods",
        xaxis_title="Method", yaxis_title="Makespan (minutes)",
        plot_bgcolor="#f9f9f9", height=450,
    )
    html_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.html")
    fig_p.write_html(html_path)
    print(f"  [Saved] {html_path}")


# -- 4. MACHINE UTILIZATION CHART ---------------------------------------------
def machine_utilization_chart(metrics_dict: Dict, filename_prefix: str):
    """Grouped bar chart of machine utilization for a given method."""
    util = metrics_dict["machine_util"]
    machines = list(util.keys())
    utilizations = list(util.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(machines, utilizations,
                  color=[_job_color_mpl(i) for i in range(len(machines))],
                  edgecolor="white")
    for bar, val in zip(bars, utilizations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Machine Utilization — {metrics_dict['method']}", fontsize=13,
                 fontweight="bold")
    ax.axhline(100, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# -- 5. SCENARIO COMPARISON CHART ---------------------------------------------
def scenario_comparison_chart(results: List[Dict], filename_prefix: str):
    """
    Compare baseline vs Scenario A (extra machine) vs Scenario B (longer times).
    results: list of dicts with keys 'label' and 'makespan'.
    """
    labels    = [r["label"] for r in results]
    makespans = [r["makespan"] for r in results]

    fig = go.Figure()
    baseline_ms = makespans[0]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, (label, ms) in enumerate(zip(labels, makespans)):
        impr = (baseline_ms - ms) / baseline_ms * 100
        fig.add_trace(go.Bar(
            name=label, x=[label], y=[ms],
            marker_color=colors[i % len(colors)],
            text=f"{ms} min<br>({impr:+.1f}% vs baseline)",
            textposition="outside",
        ))
    fig.update_layout(
        title="Scenario Analysis — Makespan Comparison",
        yaxis_title="Makespan (minutes)", plot_bgcolor="#f9f9f9", height=450,
        showlegend=False,
    )
    path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.html")
    fig.write_html(path)
    print(f"  [Saved] {path}")

    # Matplotlib
    fig2, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, makespans,
                  color=colors[:len(labels)], edgecolor="white")
    for bar, ms in zip(bars, makespans):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(ms), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Makespan (minutes)")
    ax.set_title("Scenario Analysis — Makespan Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    fig2.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [Saved] {png_path}")


# -- 6. TARDINESS CHART ---------------------------------------------------------
def tardiness_chart(all_metrics: List[Dict], job_names: List[str], filename: str):
    """Grouped bar chart of per-job tardiness across methods."""
    n_jobs = len(job_names)
    x = np.arange(n_jobs)
    width = 0.8 / len(all_metrics)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, m in enumerate(all_metrics):
        tards = m["tardinesses"]
        offset = (i - len(all_metrics) / 2 + 0.5) * width
        ax.bar(x + offset, tards, width=width * 0.9,
               label=m["method"], color=_job_color_mpl(i), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"J{j}: {job_names[j]}" for j in range(n_jobs)], rotation=20)
    ax.set_ylabel("Tardiness (minutes)")
    ax.set_title("Per-Job Tardiness by Scheduling Method", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")
