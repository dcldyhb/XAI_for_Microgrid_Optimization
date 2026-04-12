import argparse
import math
import os
from pathlib import Path
from typing import Iterable, List

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

tmp_root = os.path.join(project_root, "tmp", "plot_runtime")
mpl_cache_dir = os.path.join(tmp_root, "mplconfig")
xdg_cache_dir = os.path.join(tmp_root, "xdg_cache")
os.makedirs(mpl_cache_dir, exist_ok=True)
os.makedirs(xdg_cache_dir, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_dir)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from config import PYSR_OUTPUT_DIR, FONT_SANS_SERIF
except ImportError:
    PYSR_OUTPUT_DIR = os.path.join(current_dir, "output_vector")
    FONT_SANS_SERIF = ["sans-serif"]


plt.rcParams["font.sans-serif"] = FONT_SANS_SERIF
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PySR Pareto curves from exported equation candidate CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=PYSR_OUTPUT_DIR,
        help="Directory containing pareto_*.csv or hall_of_fame*.csv files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit CSV files to plot. If omitted, the script auto-discovers files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saved PNGs and summary CSV. Defaults to <input-dir>/pareto_plots.",
    )
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=2,
        help="Number of low-loss points to annotate per target.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of log-scale loss.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def discover_csv_files(input_dir: Path) -> List[Path]:
    patterns = [
        "pareto_*.csv",
        "hall_of_fame*.csv",
        "**/pareto_*.csv",
        "**/hall_of_fame*.csv",
    ]
    seen = set()
    files: List[Path] = []
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return files


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Complexity": "complexity",
        "Loss": "loss",
        "Equation": "equation",
        "Score": "score",
    }
    df = df.rename(columns=rename_map).copy()

    required_cols = {"complexity", "loss"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required_cols - set(df.columns))}")

    for col in ("complexity", "loss", "score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "pick" in df.columns:
        df["pick"] = df["pick"].astype(str).str.lower().isin(["1", "true", "t", "yes"])
    else:
        df["pick"] = False

    if "equation" not in df.columns:
        df["equation"] = ""

    df = df.dropna(subset=["complexity", "loss"]).copy()
    df = df.sort_values(["complexity", "loss"]).reset_index(drop=True)
    return df


def infer_target_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("pareto_"):
        return stem[len("pareto_") :]
    if stem.startswith("hall_of_fame_output"):
        return stem
    if stem == "hall_of_fame":
        parent_name = path.parent.name
        if parent_name.startswith("pysr_"):
            return parent_name[len("pysr_") :]
        return parent_name
    return stem


def choose_selected_row(df: pd.DataFrame) -> pd.Series:
    picked = df[df["pick"]]
    if not picked.empty:
        return picked.iloc[0]
    if "score" in df.columns and df["score"].notna().any():
        return df.loc[df["score"].idxmax()]
    return df.loc[df["loss"].idxmin()]


def short_equation(equation: str, limit: int = 60) -> str:
    equation = str(equation).replace("\n", " ").strip()
    if len(equation) <= limit:
        return equation
    return equation[: limit - 3] + "..."


def annotate_points(ax, df: pd.DataFrame, top_n: int) -> None:
    if top_n <= 0 or df.empty:
        return
    annotate_df = df.nsmallest(top_n, "loss")
    for _, row in annotate_df.iterrows():
        label = f"C={int(row['complexity'])}, L={row['loss']:.2e}"
        ax.annotate(
            label,
            xy=(row["complexity"], row["loss"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )


def plot_single_target(
    df: pd.DataFrame,
    target_name: str,
    output_path: Path,
    annotate_top: int,
    linear_y: bool,
    dpi: int,
) -> pd.Series:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(df["complexity"], df["loss"], color="#3b82f6", alpha=0.75, linewidth=1.6)
    ax.scatter(df["complexity"], df["loss"], color="#1d4ed8", s=36, alpha=0.9)

    chosen = choose_selected_row(df)
    ax.scatter(
        [chosen["complexity"]],
        [chosen["loss"]],
        color="#dc2626",
        s=84,
        label="Selected",
        zorder=5,
    )
    ax.annotate(
        f"selected\nC={int(chosen['complexity'])}, L={chosen['loss']:.2e}",
        xy=(chosen["complexity"], chosen["loss"]),
        xytext=(10, -14),
        textcoords="offset points",
        fontsize=9,
        color="#991b1b",
    )

    annotate_points(ax, df, annotate_top)

    ax.set_title(f"PySR Pareto Curve: {target_name}")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25, linestyle="--")
    if not linear_y:
        positive_mask = df["loss"] > 0
        if positive_mask.any():
            ax.set_yscale("log")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return pd.Series(
        {
            "target": target_name,
            "selected_complexity": int(chosen["complexity"]),
            "selected_loss": float(chosen["loss"]),
            "selected_equation": str(chosen.get("equation", "")),
            "min_loss": float(df["loss"].min()),
            "max_complexity": int(df["complexity"].max()),
            "num_candidates": int(len(df)),
        }
    )


def plot_combined(
    datasets: List[pd.DataFrame],
    target_names: List[str],
    output_path: Path,
    linear_y: bool,
    dpi: int,
) -> None:
    n = len(datasets)
    if n == 0:
        return

    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.8, rows * 4.8))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax, df, target_name in zip(axes.flat, datasets, target_names):
        ax.plot(df["complexity"], df["loss"], color="#3b82f6", alpha=0.75, linewidth=1.4)
        ax.scatter(df["complexity"], df["loss"], color="#1d4ed8", s=28, alpha=0.9)
        chosen = choose_selected_row(df)
        ax.scatter([chosen["complexity"]], [chosen["loss"]], color="#dc2626", s=60, zorder=5)
        ax.set_title(target_name)
        ax.set_xlabel("Complexity")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25, linestyle="--")
        if not linear_y and (df["loss"] > 0).any():
            ax.set_yscale("log")

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle("PySR Pareto Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_dir / "pareto_plots").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        csv_files = [Path(path).expanduser().resolve() for path in args.files]
    else:
        csv_files = discover_csv_files(input_dir)

    if not csv_files:
        print(
            "[ERROR] No Pareto CSV files found. "
            "Please rerun PySR analysis after enabling pareto_*.csv export."
        )
        return 1

    summaries = []
    datasets: List[pd.DataFrame] = []
    target_names: List[str] = []

    for csv_path in csv_files:
        try:
            df = normalize_dataframe(pd.read_csv(csv_path))
        except Exception as exc:
            print(f"[WARNING] Skip {csv_path}: {exc}")
            continue

        target_name = infer_target_name(csv_path)
        target_names.append(target_name)
        datasets.append(df)

        figure_path = output_dir / f"pareto_{target_name}.png"
        summary_row = plot_single_target(
            df=df,
            target_name=target_name,
            output_path=figure_path,
            annotate_top=args.annotate_top,
            linear_y=args.linear_y,
            dpi=args.dpi,
        )
        summary_row["source_csv"] = str(csv_path)
        summaries.append(summary_row)
        print(f"[OK] Saved {figure_path}")

    if not summaries:
        print("[ERROR] No valid Pareto tables were loaded.")
        return 1

    combined_path = output_dir / "pareto_overview.png"
    plot_combined(
        datasets=datasets,
        target_names=target_names,
        output_path=combined_path,
        linear_y=args.linear_y,
        dpi=args.dpi,
    )
    print(f"[OK] Saved {combined_path}")

    summary_df = pd.DataFrame(summaries)
    summary_csv = output_dir / "pareto_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Saved {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
