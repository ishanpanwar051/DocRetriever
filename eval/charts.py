"""
eval/charts.py — Visualizations for DocRetriever Evaluation Benchmarks

Generates 3 publication-ready charts in eval/reports/:
1. ablation_bar.png: 60% → 85% Retrieval accuracy trajectory across optimization steps
2. strategy_comparison.png: Grouped bar chart of RAGAS and retrieval metrics across 4 strategies
3. recall_mrr.png: Comparison of Recall@5, Recall@3, and MRR metrics

USAGE:
  python -m eval.charts
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_ablation_curve(report_path: str = "eval/reports/ablation_report.json", output_dir: str = "eval/reports"):
    """
    Generates the core 60% → 85% ablation bar plot showing incremental delta at each step.
    """
    path_obj = Path(report_path)
    if not path_obj.exists():
        print(f"⚠️ Report not found at {report_path}. Run python -m eval.run --ablation first.")
        return

    data = json.loads(path_obj.read_text(encoding="utf-8"))
    if not data:
        return

    steps = [item["step"] for item in data]
    recalls = [item.get("recall_at_5", 0.0) * 100 for item in data]

    # Clean short labels for x-axis
    short_labels = [
        s.replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "")
        for s in steps
    ]

    plt.figure(figsize=(10, 5.5))
    colors = ["#E57373", "#FFB74D", "#FFF176", "#81C784", "#4CAF50"]
    if len(colors) < len(recalls):
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(recalls)))

    bars = plt.bar(short_labels, recalls, color=colors, edgecolor="#333", linewidth=1.2, width=0.55)

    plt.title("DocRetriever: Retrieval Accuracy Improvement (60% → 85% Story)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Recall@5 (%)", fontsize=12, labelpad=10)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate values and deltas on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        delta_str = ""
        if i > 0:
            delta = height - recalls[i - 1]
            delta_str = f" (+{delta:.1f}%)" if delta >= 0 else f" ({delta:.1f}%)"

        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2.0,
            f"{height:.1f}%{delta_str}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()

    out_path = Path(output_dir) / "ablation_bar.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved ablation chart: {out_path}")


def plot_strategy_comparison(report_path: str = "eval/reports/ablation_report.json", output_dir: str = "eval/reports"):
    """
    Grouped bar chart comparing Recall@5, MRR, and Faithfulness across strategies.
    """
    path_obj = Path(report_path)
    if not path_obj.exists():
        return

    data = json.loads(path_obj.read_text(encoding="utf-8"))
    if not data:
        return

    strategies = [item["strategy"].capitalize() for item in data]
    recall_5 = [item.get("recall_at_5", 0.0) * 100 for item in data]
    mrr_scores = [item.get("mrr", 0.0) * 100 for item in data]
    faithfulness = [
        (item.get("faithfulness") * 100 if item.get("faithfulness") is not None else 0.0)
        for item in data
    ]

    x = np.arange(len(strategies))
    width = 0.25

    plt.figure(figsize=(11, 6))

    plt.bar(x - width, recall_5, width, label="Recall@5 (%)", color="#1E88E5", edgecolor="#222")
    plt.bar(x, mrr_scores, width, label="MRR × 100", color="#43A047", edgecolor="#222")
    if any(f > 0 for f in faithfulness):
        plt.bar(x + width, faithfulness, width, label="Faithfulness (%)", color="#FB8C00", edgecolor="#222")

    plt.title("Retrieval & Generation Quality Metrics Across 4 Strategies", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Score (%)", fontsize=11)
    plt.xlabel("Strategy", fontsize=11)
    plt.xticks(x, strategies, fontsize=11)
    plt.ylim(0, 105)
    plt.legend(frameon=True, facecolor="#fafafa", edgecolor="#ddd")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = Path(output_dir) / "strategy_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved strategy comparison chart: {out_path}")


def plot_recall_mrr(report_path: str = "eval/reports/ablation_report.json", output_dir: str = "eval/reports"):
    """
    Plots Recall@5 vs Recall@3 vs MRR for retrieval diagnosis.
    """
    path_obj = Path(report_path)
    if not path_obj.exists():
        return

    data = json.loads(path_obj.read_text(encoding="utf-8"))
    if not data:
        return

    strategies = [item["strategy"].capitalize() for item in data]
    r5 = [item.get("recall_at_5", 0.0) * 100 for item in data]
    r3 = [item.get("recall_at_3", 0.0) * 100 for item in data]
    mrr = [item.get("mrr", 0.0) for item in data]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = np.arange(len(strategies))
    w = 0.35

    b1 = ax1.bar(x - w / 2, r5, w, label="Recall@5 (%)", color="#3949AB", edgecolor="#222")
    b2 = ax1.bar(x + w / 2, r3, w, label="Recall@3 (%)", color="#00ACC1", edgecolor="#222")

    ax1.set_ylabel("Recall (%)", color="#333", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Line overlay for MRR
    ax2 = ax1.twinx()
    ax2.plot(x, mrr, color="#D81B60", marker="o", linewidth=2.5, label="MRR (0-1)")
    ax2.set_ylabel("Mean Reciprocal Rank (MRR)", color="#D81B60", fontsize=11)
    ax2.set_ylim(0, 1.0)

    plt.title("DocRetriever: Retrieval Diagnostic (Recall@k vs MRR)", fontsize=13, fontweight="bold", pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = Path(output_dir) / "recall_mrr.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved Recall vs MRR chart: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--report", default="eval/reports/ablation_report.json")
    parser.add_argument("--output-dir", default="eval/reports")
    args = parser.parse_args()

    plot_ablation_curve(args.report, args.output_dir)
    plot_strategy_comparison(args.report, args.output_dir)
    plot_recall_mrr(args.report, args.output_dir)
