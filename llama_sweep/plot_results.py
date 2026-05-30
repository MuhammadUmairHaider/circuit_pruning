"""
Plot Llama lambda_sparsity sweep results.

Generates per-model and cross-model plots:
  - Accuracy, KL, pruning, compression vs lambda (per model, lines per task)
  - Accuracy vs pruning trade-off
  - Cross-model comparison
  - Training curves
  - LaTeX table

Usage:
  python -m llama_sweep.plot_results
  python -m llama_sweep.plot_results --format pdf
"""

import os
import argparse
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

TASK_COLORS = {"ioi": "#1f77b4", "gp": "#ff7f0e"}
TASK_LABELS = {"ioi": "IOI", "gp": "GP"}
TASK_MARKERS = {"ioi": "o", "gp": "s"}
MODEL_LINESTYLES = {"Llama-3.2-1B": "-", "Llama-3.1-8B": "--"}


def load_results(results_dir):
    csv_path = os.path.join(results_dir, "llama_sweep_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No results at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    return df


def plot_metric_per_model(df, metric, ylabel, title_suffix, save_prefix, fmt, baseline_metric=None):
    """One plot per model: metric vs lambda, one line per task."""
    for model_short, mdf in df.groupby("model_short"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for task in mdf["task"].unique():
            tdf = mdf[mdf["task"] == task].sort_values("lambda_sparsity")
            ax.plot(
                tdf["lambda_sparsity"], tdf[metric],
                marker=TASK_MARKERS.get(task, "o"),
                color=TASK_COLORS.get(task),
                label=TASK_LABELS.get(task, task),
                linewidth=2, markersize=8,
            )
            if baseline_metric and baseline_metric in tdf.columns:
                ax.axhline(y=tdf[baseline_metric].iloc[0], color=TASK_COLORS.get(task), linestyle="--", alpha=0.4)

        ax.set_xlabel(r"$\lambda_{\mathrm{sparsity}}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_short}: {title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_{model_short}.{fmt}", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_prefix}_{model_short}.{fmt}")


def plot_cross_model_comparison(df, save_path, fmt):
    """Compare both models side by side: accuracy vs lambda, one subplot per task."""
    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        tdf = df[df["task"] == task]
        for model_short, mdf in tdf.groupby("model_short"):
            mdf = mdf.sort_values("lambda_sparsity")
            ax.plot(
                mdf["lambda_sparsity"], mdf["final_accuracy"],
                marker="o", linewidth=2, markersize=7,
                linestyle=MODEL_LINESTYLES.get(model_short, "-"),
                label=model_short,
            )
            baseline = mdf["baseline_accuracy"].iloc[0]
            ax.axhline(y=baseline, linestyle=":", alpha=0.3, color="gray")

        ax.set_xlabel(r"$\lambda_{\mathrm{sparsity}}$")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{TASK_LABELS.get(task, task)}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Model Comparison: Accuracy vs. $\\lambda_{\\mathrm{sparsity}}$", fontsize=15)
    fig.tight_layout()
    path = f"{save_path}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy_vs_pruning(df, save_path, fmt):
    """Pareto-style: accuracy vs reduction %, annotated with lambda values."""
    for model_short, mdf in df.groupby("model_short"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for task in mdf["task"].unique():
            tdf = mdf[mdf["task"] == task].sort_values("reduction_percentage")
            ax.scatter(
                tdf["reduction_percentage"], tdf["final_accuracy"],
                marker=TASK_MARKERS.get(task, "o"),
                color=TASK_COLORS.get(task),
                label=TASK_LABELS.get(task, task), s=80, zorder=5,
            )
            ax.plot(tdf["reduction_percentage"], tdf["final_accuracy"], color=TASK_COLORS.get(task), alpha=0.4)
            for _, row in tdf.iterrows():
                ax.annotate(
                    f'{row["lambda_sparsity"]:.2f}',
                    (row["reduction_percentage"], row["final_accuracy"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7,
                )

        ax.set_xlabel("Parameter Reduction (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_short}: Accuracy vs. Pruning")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = f"{save_path}_{model_short}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_training_curves(results_dir, save_prefix, fmt):
    epoch_dir = os.path.join(results_dir, "per_epoch")
    if not os.path.exists(epoch_dir):
        print("  No per-epoch data, skipping training curves.")
        return

    csv_files = sorted(glob.glob(os.path.join(epoch_dir, "*_epochs.csv")))
    if not csv_files:
        return

    # Group by model+task
    groups = {}
    for f in csv_files:
        basename = os.path.basename(f)
        # format: ModelName_task_lambda_0.XX_epochs.csv
        parts = basename.split("_lambda_")
        key = parts[0]  # e.g. "Llama-3.2-1B_ioi"
        if key not in groups:
            groups[key] = []
        groups[key].append(f)

    for key, files in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Curves - {key}", fontsize=16)

        for f in files:
            edf = pd.read_csv(f)
            basename = os.path.basename(f)
            lam = float(basename.split("lambda_")[1].split("_")[0])
            label = f"$\\lambda$={lam:.2f}"
            axes[0].plot(edf["epoch"], edf["avg_loss"], label=label, alpha=0.8)
            axes[1].plot(edf["epoch"], edf["avg_kl_loss"], label=label, alpha=0.8)
            axes[2].plot(edf["epoch"], edf["avg_sparsity_loss"], label=label, alpha=0.8)

        for ax, title in zip(axes, ["Total Loss", "KL Divergence Loss", "Sparsity Loss"]):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = f"{save_prefix}_{key}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def generate_latex_table(df, save_path):
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of $\lambda_{\mathrm{sparsity}}$ on Llama circuit discovery.}")
    lines.append(r"\label{tab:llama_lambda_sweep}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lll|cccc|cc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Task & $\lambda_s$ & Accuracy & Acc. Drop & KL Div. & Logit Diff & Pruned (\%) & Compression \\")
    lines.append(r"\midrule")

    for model_short in sorted(df["model_short"].unique()):
        mdf = df[df["model_short"] == model_short]
        first_model = True
        for task in ["ioi", "gp"]:
            tdf = mdf[mdf["task"] == task].sort_values("lambda_sparsity")
            if tdf.empty:
                continue
            first_task = True
            for _, row in tdf.iterrows():
                model_label = model_short if first_model else ""
                task_label = TASK_LABELS.get(task, task) if first_task else ""
                lines.append(
                    f"{model_label} & {task_label} & {row['lambda_sparsity']:.2f} & "
                    f"{row['final_accuracy']:.4f} & {row['accuracy_drop']:.4f} & "
                    f"{row['final_kl_div']:.4f} & {row['final_logit_diff']:.4f} & "
                    f"{row['reduction_percentage']:.1f} & {row['compression_ratio']:.2f}$\\times$ \\\\"
                )
                first_model = False
                first_task = False
            lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved LaTeX: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Llama sweep results")
    parser.add_argument(
        "--results_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    df = load_results(args.results_dir)
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fmt = args.format

    print("\nGenerating Llama sweep plots...")

    plot_metric_per_model(df, "final_accuracy", "Accuracy",
        "Accuracy vs. $\\lambda_{\\mathrm{sparsity}}$",
        os.path.join(plots_dir, "accuracy_vs_lambda"), fmt, baseline_metric="baseline_accuracy")

    plot_metric_per_model(df, "reduction_percentage", "Parameter Reduction (%)",
        "Pruning vs. $\\lambda_{\\mathrm{sparsity}}$",
        os.path.join(plots_dir, "pruning_vs_lambda"), fmt)

    plot_metric_per_model(df, "final_kl_div", "KL Divergence",
        "KL Divergence vs. $\\lambda_{\\mathrm{sparsity}}$",
        os.path.join(plots_dir, "kl_vs_lambda"), fmt)

    plot_metric_per_model(df, "compression_ratio", "Compression Ratio",
        "Compression vs. $\\lambda_{\\mathrm{sparsity}}$",
        os.path.join(plots_dir, "compression_vs_lambda"), fmt)

    plot_cross_model_comparison(df, os.path.join(plots_dir, "cross_model_comparison"), fmt)

    plot_accuracy_vs_pruning(df, os.path.join(plots_dir, "accuracy_vs_pruning"), fmt)

    plot_training_curves(args.results_dir, os.path.join(plots_dir, "curves"), fmt)

    generate_latex_table(df, os.path.join(plots_dir, "llama_sweep_table.tex"))

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
