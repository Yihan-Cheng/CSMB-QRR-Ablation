from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "ablation_outputs" / "ablation_results.csv"
FULL_VARIANT_NAME = "full_model"


def resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()

    search_roots = (
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
    )
    for root in search_roots:
        merged = (root / candidate).resolve()
        if merged.exists():
            return merged
    raise FileNotFoundError(f"Could not find file: {path}")


def load_results(results_path: str | Path) -> pd.DataFrame:
    path = resolve_existing_path(results_path)
    df = pd.read_csv(path)
    if "variant_order" in df.columns:
        df = df.sort_values("variant_order")
    else:
        df = df.sort_values("eval_score", ascending=False)
    return df.reset_index(drop=True)


def prepare_output_dir(results_path: str | Path, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        target = Path(output_dir).resolve()
    else:
        target = resolve_existing_path(results_path).resolve().parent
    target.mkdir(parents=True, exist_ok=True)
    return target


def plot_eval_scores(df: pd.DataFrame, output_dir: Path) -> Path:
    ordered = df.sort_values("eval_score", ascending=True).reset_index(drop=True)
    labels = ordered["variant_name"].tolist()
    scores = ordered["eval_score"].to_numpy(dtype=float)
    colors = ["#d97706" if name == FULL_VARIANT_NAME else "#2563eb" for name in labels]

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    bars = ax.barh(labels, scores, color=colors, edgecolor="#1f2937", linewidth=0.8)
    ax.set_title("Ablation Eval Score Comparison")
    ax.set_xlabel("Eval Score")
    ax.set_ylabel("Variant")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, score in zip(bars, scores):
        ax.text(
            float(bar.get_width()) + 0.08,
            float(bar.get_y() + bar.get_height() / 2.0),
            f"{score:.3f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    save_path = output_dir / "01_ablation_eval_score.png"
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_delta_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    ordered = df.sort_values("variant_order").reset_index(drop=True)
    row_labels = ordered["variant_name"].tolist()
    metric_labels = [
        "Eval Score delta",
        "Eval MAE delta",
        "Train Score delta",
    ]
    matrix = np.column_stack(
        [
            ordered["eval_score_delta_vs_full"].to_numpy(dtype=float),
            ordered["eval_mae_delta_vs_full"].to_numpy(dtype=float),
            ordered["train_score_delta_vs_full"].to_numpy(dtype=float),
        ]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    vmax = float(np.max(np.abs(matrix))) if np.any(matrix) else 1.0
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Delta vs Full Model")
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            ax.text(
                col,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path = output_dir / "02_ablation_delta_heatmap.png"
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def load_prediction_frame(results_path: str | Path, file_name: str) -> pd.DataFrame:
    results_parent = resolve_existing_path(results_path).resolve().parent
    pred_path = (results_parent / file_name).resolve()
    return pd.read_csv(pred_path)


def select_scatter_rows(df: pd.DataFrame) -> pd.DataFrame:
    if FULL_VARIANT_NAME in set(df["variant_name"]):
        reference = df.loc[df["variant_name"] == FULL_VARIANT_NAME].iloc[[0]]
        alternatives = df.loc[df["variant_name"] != FULL_VARIANT_NAME].sort_values(
            "eval_score",
            ascending=False,
        )
        if alternatives.empty:
            return reference
        return pd.concat([reference, alternatives.iloc[[0]]], ignore_index=True)
    return df.sort_values("eval_score", ascending=False).head(2).reset_index(drop=True)


def plot_prediction_scatter(df: pd.DataFrame, results_path: str | Path, output_dir: Path) -> Path | None:
    if "prediction_file" not in df.columns:
        return None

    chosen = select_scatter_rows(df)
    if chosen.empty:
        return None

    fig, axes = plt.subplots(1, len(chosen), figsize=(6.2 * len(chosen), 5.2), squeeze=False)
    axes_list = axes.ravel().tolist()

    for ax, (_, row) in zip(axes_list, chosen.iterrows()):
        pred_df = load_prediction_frame(results_path, row["prediction_file"])
        actual = pred_df["actual"].to_numpy(dtype=float)
        predicted = pred_df["predicted"].to_numpy(dtype=float)

        ax.scatter(actual, predicted, s=28, alpha=0.72, color="#0f766e", edgecolor="white", linewidth=0.4)
        low = float(min(np.min(actual), np.min(predicted)))
        high = float(max(np.max(actual), np.max(predicted)))
        ax.plot([low, high], [low, high], linestyle="--", color="#b91c1c", linewidth=1.2)
        ax.set_title(
            f"{row['variant_name']}\nMAE={float(row['eval_mae']):.3f}, Score={float(row['eval_score']):.3f}"
        )
        ax.set_xlabel("Actual Stress")
        ax.set_ylabel("Predicted Stress")
        ax.grid(alpha=0.28, linestyle="--")

    fig.suptitle("Prediction Scatter: Full Model vs Best Alternative", fontsize=13)
    fig.tight_layout()
    save_path = output_dir / "03_prediction_scatter.png"
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_plots(results_path: str | Path, output_dir: str | Path | None = None) -> List[Path]:
    df = load_results(results_path)
    target_dir = prepare_output_dir(results_path, output_dir)
    generated = [
        plot_eval_scores(df, target_dir),
        plot_delta_heatmap(df, target_dir),
    ]
    scatter_path = plot_prediction_scatter(df, results_path, target_dir)
    if scatter_path is not None:
        generated.append(scatter_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation figures from ablation_results.csv")
    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS),
        help="CSV file produced by ablation_study.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional figure output directory. Defaults to the results directory.",
    )
    args = parser.parse_args()

    generated = generate_plots(args.results, output_dir=args.output_dir)
    for path in generated:
        print(f"[Done] plot saved to: {path}")


if __name__ == "__main__":
    main()
