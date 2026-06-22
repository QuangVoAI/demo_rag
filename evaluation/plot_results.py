import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(csv_path=None):
    if not csv_path:
        files = glob.glob("evaluation/results/summary_*.csv")
        if not files:
            print("No summary CSV found in evaluation/results/")
            return
        csv_path = max(files, key=os.path.getctime)

    print(f"Reading summary data from: {csv_path}")
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")

    metrics = ["BLEU", "ROUGE-L", "BERTScore", "Recall@5"]
    df_metrics = df.melt(id_vars="Architecture", value_vars=metrics, var_name="Metric", value_name="Score")
    df_metrics["Score"] = pd.to_numeric(df_metrics["Score"], errors="coerce")

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_metrics, x="Metric", y="Score", hue="Architecture", palette="viridis")
    plt.title("EmpathAI quality and retrieval metrics", fontsize=14, pad=12, fontweight="bold")
    plt.ylim(0, max(float(df_metrics["Score"].max() or 0) + 10, 100))
    plt.ylabel("Score (0-100)")
    plt.xlabel("Metric")
    plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc="upper left")
    for container in ax.containers:
        labels = [f"{v.get_height():.1f}" if pd.notna(v.get_height()) and v.get_height() > 0 else "" for v in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)
    plt.tight_layout()
    out_path1 = "evaluation/results/plot_quality.png"
    plt.savefig(out_path1, dpi=300)
    print(f"Saved: {out_path1}")

    plt.figure(figsize=(7, 4))
    ax2 = sns.barplot(data=df, x="Architecture", y="Avg latency (s)", hue="Architecture", palette="rocket", legend=False)
    plt.title("EmpathAI average latency", fontsize=14, pad=12, fontweight="bold")
    plt.ylabel("Seconds")
    plt.xlabel("Architecture")
    plt.xticks(rotation=10, ha="right")
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1fs", padding=3, fontsize=10)
    plt.tight_layout()
    out_path2 = "evaluation/results/plot_latency.png"
    plt.savefig(out_path2, dpi=300)
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to summary CSV. Defaults to latest summary file.")
    args = parser.parse_args()
    plot_results(args.csv)
