"""
Plot hyperparameter sweep results from sweep_results.csv.

Generates publication-quality plots:
  1. Accuracy vs lambda_sparsity (one line per task)
  2. Pruning reduction % vs lambda_sparsity
  3. KL divergence vs lambda_sparsity
  4. Accuracy vs Pruning (Pareto front)
  5. Training curves per task (from per_epoch CSVs)
  6. Generates a LaTeX-ready summary table

Usage:
  python -m hyperparameter_sweep.plot_results
  python -m hyperparameter_sweep.plot_results --results_dir hyperparameter_sweep/results --format pdf
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

TASK_LABELS = {"ioi": "IOI", "gp": "GP", "gt": "GT"}
TASK_COLORS = {"ioi": "#1f77b4", "gp": "#ff7f0e", "gt": "#2ca02c"}
TASK_MARKERS = {"ioi": "o", "gp": "s", "gt": "^"}


def load_results(results_dir):
    csv_path = os.path.join(results_dir, "sweep_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No results found at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiment results from {csv_path}")
    return df


def plot_metric_vs_lambda(df, metric, ylabel, title, save_path, baseline_metric=None):
    """Generic plot of a metric vs lambda_sparsity, one line per task."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for task in df["task"].unique():
        task_df = df[df["task"] == task].sort_values("lambda_sparsity")
        ax.plot(
            task_df["lambda_sparsity"],
            task_df[metric],
            marker=TASK_MARKERS.get(task, "o"),
            color=TASK_COLORS.get(task, None),
            label=TASK_LABELS.get(task, task),
            linewidth=2,
            markersize=8,
        )
        # Plot baseline as dashed horizontal line
        if baseline_metric and baseline_metric in task_df.columns:
            baseline_val = task_df[baseline_metric].iloc[0]
            ax.axhline(
                y=baseline_val,
                color=TASK_COLORS.get(task, "gray"),
                linestyle="--",
                alpha=0.4,
                linewidth=1,
            )

    ax.set_xlabel(r"$\lambda_{\mathrm{sparsity}}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_accuracy_vs_pruning(df, save_path):
    """Pareto-style plot: accuracy vs reduction percentage."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for task in df["task"].unique():
        task_df = df[df["task"] == task].sort_values("reduction_percentage")
        ax.scatter(
            task_df["reduction_percentage"],
            task_df["final_accuracy"],
            marker=TASK_MARKERS.get(task, "o"),
            color=TASK_COLORS.get(task, None),
            label=TASK_LABELS.get(task, task),
            s=80,
            zorder=5,
        )
        # Connect points with lines
        ax.plot(
            task_df["reduction_percentage"],
            task_df["final_accuracy"],
            color=TASK_COLORS.get(task, None),
            alpha=0.4,
            linewidth=1,
        )
        # Annotate each point with lambda value
        for _, row in task_df.iterrows():
            ax.annotate(
                f'{row["lambda_sparsity"]:.2f}',
                (row["reduction_percentage"], row["final_accuracy"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xlabel("Parameter Reduction (%)")
    ax.set_ylabel("Task Accuracy")
    ax.set_title("Accuracy vs. Pruning Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(results_dir, save_path_prefix, fmt):
    """Plot training curves (loss, KL, sparsity) from per-epoch CSVs."""
    epoch_dir = os.path.join(results_dir, "per_epoch")
    if not os.path.exists(epoch_dir):
        print("  No per-epoch data found, skipping training curves.")
        return

    csv_files = sorted(glob.glob(os.path.join(epoch_dir, "*_epochs.csv")))
    if not csv_files:
        print("  No epoch CSV files found.")
        return

    # Group by task
    tasks = {}
    for f in csv_files:
        basename = os.path.basename(f)
        task = basename.split("_lambda_")[0]
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(f)

    for task, files in tasks.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Curves - {TASK_LABELS.get(task, task)}", fontsize=16)

        for f in files:
            df = pd.read_csv(f)
            basename = os.path.basename(f)
            lam = float(basename.split("lambda_")[1].split("_")[0])
            label = f"$\\lambda$={lam:.2f}"

            axes[0].plot(df["epoch"], df["avg_loss"], label=label, alpha=0.8)
            axes[1].plot(df["epoch"], df["avg_kl_loss"], label=label, alpha=0.8)
            axes[2].plot(df["epoch"], df["avg_sparsity_loss"], label=label, alpha=0.8)

        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")

        axes[1].set_title("KL Divergence Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("KL Loss")

        axes[2].set_title("Sparsity Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Sparsity Loss")

        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = f"{save_path_prefix}_{task}_training_curves.{fmt}"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


def generate_latex_table(df, save_path):
    """Generate a LaTeX table summarizing the sweep results."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Hyperparameter sensitivity: effect of $\lambda_{\mathrm{sparsity}}$ on circuit discovery across tasks.}")
    lines.append(r"\label{tab:lambda_sweep}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll|cccc|cc}")
    lines.append(r"\toprule")
    lines.append(r"Task & $\lambda_s$ & Accuracy & Acc. Drop & KL Div. & Logit Diff & Pruned (\%) & Compression \\")
    lines.append(r"\midrule")

    for task in ["ioi", "gp", "gt"]:
        task_df = df[df["task"] == task].sort_values("lambda_sparsity")
        if task_df.empty:
            continue
        first = True
        for _, row in task_df.iterrows():
            task_label = TASK_LABELS.get(task, task) if first else ""
            lines.append(
                f"{task_label} & {row['lambda_sparsity']:.2f} & "
                f"{row['final_accuracy']:.4f} & {row['accuracy_drop']:.4f} & "
                f"{row['final_kl_div']:.4f} & {row['final_logit_diff']:.4f} & "
                f"{row['reduction_percentage']:.1f} & {row['compression_ratio']:.2f}$\\times$ \\\\"
            )
            first = False
        lines.append(r"\midrule")

    # Remove last midrule and replace with bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved LaTeX table: {save_path}")


def plot_component_pruning(df, save_path):
    """Bar chart showing per-component pruning percentages across lambdas for each task."""
    components = [
        ("pruned_pct_attention_heads", "Attn Heads"),
        ("pruned_pct_attention_blocks", "Attn Blocks"),
        ("pruned_pct_mlp_blocks", "MLP Blocks"),
        ("pruned_pct_mlp_hidden", "MLP Hidden"),
        ("pruned_pct_mlp_output", "MLP Output"),
        ("pruned_pct_attention_neurons", "Attn Neurons"),
    ]

    for task in df["task"].unique():
        task_df = df[df["task"] == task].sort_values("lambda_sparsity")
        if task_df.empty:
            continue

        # Filter to components that exist in the data
        available = [(col, label) for col, label in components if col in task_df.columns]
        if not available:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        lambdas = task_df["lambda_sparsity"].values
        x = range(len(lambdas))
        width = 0.12
        n_components = len(available)

        for idx, (col, label) in enumerate(available):
            offset = (idx - n_components / 2) * width + width / 2
            vals = task_df[col].fillna(0).values
            ax.bar([xi + offset for xi in x], vals, width=width, label=label, alpha=0.85)

        ax.set_xlabel(r"$\lambda_{\mathrm{sparsity}}$")
        ax.set_ylabel("Pruned (%)")
        ax.set_title(f"Component-level Pruning - {TASK_LABELS.get(task, task)}")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{l:.2f}" for l in lambdas])
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2, axis="y")

        fig.tight_layout()
        component_path = save_path.replace(".png", f"_{task}.png").replace(".pdf", f"_{task}.pdf")
        fig.savefig(component_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {component_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot lambda_sparsity sweep results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    df = load_results(args.results_dir)
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fmt = args.format

    print("\nGenerating plots...")

    # 1. Accuracy vs lambda
    plot_metric_vs_lambda(
        df, "final_accuracy", "Task Performance",
        r"Task Performance vs. $\lambda_{\mathrm{sparsity}}$",
        os.path.join(plots_dir, f"accuracy_vs_lambda.{fmt}"),
        baseline_metric="baseline_accuracy",
    )

    # 2. Reduction % vs lambda
    plot_metric_vs_lambda(
        df, "reduction_percentage", "Parameter Reduction (%)",
        r"Pruning Reduction vs. $\lambda_{\mathrm{sparsity}}$",
        os.path.join(plots_dir, f"pruning_vs_lambda.{fmt}"),
    )

    # 3. KL divergence vs lambda
    plot_metric_vs_lambda(
        df, "final_kl_div", "KL Divergence",
        r"KL Divergence vs. $\lambda_{\mathrm{sparsity}}$",
        os.path.join(plots_dir, f"kl_vs_lambda.{fmt}"),
    )

    # 4. Compression ratio vs lambda
    plot_metric_vs_lambda(
        df, "compression_ratio", "Compression Ratio",
        r"Compression Ratio vs. $\lambda_{\mathrm{sparsity}}$",
        os.path.join(plots_dir, f"compression_vs_lambda.{fmt}"),
    )

    # 5. Accuracy vs Pruning (Pareto)
    plot_accuracy_vs_pruning(df, os.path.join(plots_dir, f"accuracy_vs_pruning.{fmt}"))

    # 6. Component-level pruning
    plot_component_pruning(df, os.path.join(plots_dir, f"component_pruning.{fmt}"))

    # 7. Training curves
    plot_training_curves(
        args.results_dir,
        os.path.join(plots_dir, "curves"),
        fmt,
    )

    # 8. LaTeX table
    generate_latex_table(df, os.path.join(plots_dir, "lambda_sweep_table.tex"))

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
