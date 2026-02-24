from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from thund_avoider.schemas.interpretation.config import PlotType, ReportConfig
from thund_avoider.schemas.interpretation.results import OptimalWindowResult


class PlotGenerator:
    """
    Generates statistical visualization plots with Russian labels.

    This class creates all plots for the statistical analysis report
    with consistent styling and Russian language labels.
    """

    def __init__(self, config: ReportConfig) -> None:
        self._config = config
        self._setup_style()

    def _setup_style(self) -> None:
        """Setup matplotlib/seaborn style for consistent appearance."""
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.dpi": self._config.figure_dpi,
                "savefig.dpi": self._config.figure_dpi,
                "figure.figsize": self._config.figure_size,
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )
        sns.set_palette(self._config.palette)

    def _get_color_map(self, algorithm_names: list[str]) -> dict[str, str]:
        """Create color mapping for algorithms."""
        return {
            name: self._config.palette[i % len(self._config.palette)]
            for i, name in enumerate(algorithm_names)
        }

    def generate_boxplot(
        self,
        dfs: dict[str, pd.DataFrame],
        output_path: Path,
        algorithm_names: list[str] | None = None,
    ) -> Path:
        """
        Generate boxplot of path length distributions.

        Args:
            dfs: Dictionary of DataFrames with pathfinding results.
            output_path: Path to save the plot.
            algorithm_names: Ordered list of algorithm names.

        Returns:
            Path to saved plot.
        """
        if algorithm_names is None:
            algorithm_names = list(dfs.keys())

        # Prepare data for plotting
        plot_data = []
        for name in algorithm_names:
            if name in dfs:
                df = dfs[name].copy()
                df["algorithm"] = name
                plot_data.append(df[["length", "window_size", "algorithm"]])

        combined = pd.concat(plot_data, ignore_index=True)

        # Create plot
        fig, ax = plt.subplots(figsize=self._config.figure_size)

        color_map = self._get_color_map(algorithm_names)
        palette = [color_map[name] for name in algorithm_names]

        sns.boxplot(
            data=combined,
            x="window_size",
            y="length",
            hue="algorithm",
            ax=ax,
            palette=palette,
        )

        ax.set_xlabel("Размер окна $w$")
        ax.set_ylabel("Длина финального маршрута обхода, м")
        ax.set_title("Распределение длин маршрутов обхода")
        ax.legend(title="Алгоритм", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_success_lineplot(
        self,
        success_rates: dict[str, dict[int, float]],
        output_path: Path,
        algorithm_names: list[str] | None = None,
    ) -> Path:
        """
        Generate lineplot of success rates by window size.

        Args:
            success_rates: Dictionary of success rates by window size.
            output_path: Path to save the plot.
            algorithm_names: Ordered list of algorithm names.

        Returns:
            Path to saved plot.
        """
        if algorithm_names is None:
            algorithm_names = list(success_rates.keys())

        # Prepare data
        plot_data = []
        for name in algorithm_names:
            if name in success_rates:
                for ws, rate in success_rates[name].items():
                    plot_data.append(
                        {"window_size": ws, "success_rate": rate, "algorithm": name}
                    )

        df = pd.DataFrame(plot_data)

        # Create plot
        fig, ax = plt.subplots(figsize=self._config.figure_size)

        color_map = self._get_color_map(algorithm_names)

        for name in algorithm_names:
            if name in success_rates:
                data = df[df["algorithm"] == name].sort_values("window_size")
                ax.plot(
                    data["window_size"],
                    data["success_rate"],
                    marker="o",
                    label=name,
                    color=color_map[name],
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("Размер окна $w$")
        ax.set_ylabel("Доля успешных маршрутов")
        ax.set_title("Зависимость доли успешных маршрутов от размера окна")
        ax.legend(title="Алгоритм", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_optimal_window_plot(
        self,
        optimal_results: list[OptimalWindowResult],
        output_path: Path,
    ) -> Path:
        """
        Generate bar plot of optimal window η scores.

        Args:
            optimal_results: List of optimal window results.
            output_path: Path to save the plot.

        Returns:
            Path to saved plot.
        """
        if not optimal_results:
            return output_path

        # Sort by eta score
        sorted_results = sorted(optimal_results, key=lambda x: x.eta_score, reverse=True)

        names = [r.algorithm for r in sorted_results]
        eta_scores = [r.eta_score for r in sorted_results]
        windows = [r.optimal_window for r in sorted_results]

        # Create short labels for x-axis
        short_names = [f"A{i + 1}" for i in range(len(names))]

        # Create plot with reasonable size
        n_items = len(names)
        fig_height = max(6, n_items * 0.5)
        fig_width = max(8, max(len(n) for n in names) * 0.1)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        color_map = self._get_color_map(names)
        colors = [color_map[name] for name in names]

        x_pos = range(len(names))
        bars = ax.bar(x_pos, eta_scores, color=colors)

        # Add window size labels on bars
        for bar, window in zip(bars, windows):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.00001,
                f"w={window}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_names, rotation=0)
        ax.set_ylabel("Метрика $\\eta$ (доля успехов / средняя длина)")
        ax.set_title("Оптимальные размеры окна по метрике $\\eta$")

        # Add legend with full names
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(names))]
        ax.legend(
            handles,
            names,
            title="Алгоритм",
            loc="upper right",
            fontsize=8,
        )

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=self._config.figure_dpi)
        plt.close(fig)

        return output_path

    def generate_dual_axis_marginal_plots(
        self,
        pairwise_results: list[Any],
        dfs: dict[str, pd.DataFrame],
        output_path: Path,
    ) -> Path:
        """
        Generate dual-axis plots showing ΔS vs ΔL for pairwise comparisons.

        Args:
            pairwise_results: List of pairwise comparison results.
            dfs: Dictionary of DataFrames.
            output_path: Path to save the plot.

        Returns:
            Path to saved plot.
        """
        if not pairwise_results:
            return output_path

        n_comparisons = len(pairwise_results)
        fig, axes = plt.subplots(
            1,
            n_comparisons,
            figsize=(5 * n_comparisons, 6),
        )

        if n_comparisons == 1:
            axes = [axes]

        for idx, result in enumerate(pairwise_results):
            ax = axes[idx]

            # Parse comparison names
            parts = result.comparison.split(" vs ")
            if len(parts) != 2:
                continue

            name1, name2 = parts

            if name1 not in dfs or name2 not in dfs:
                continue

            df1 = dfs[name1]
            df2 = dfs[name2]

            # Merge and compute deltas
            merged = pd.merge(
                df1,
                df2,
                on=["subject_id", "window_size"],
                suffixes=("_1", "_2"),
            )

            if len(merged) == 0:
                continue

            # Group by window
            grouped = merged.groupby("window_size").agg(
                success_1=("success_1", "mean"),
                success_2=("success_2", "mean"),
                length_1=("length_1", "mean"),
                length_2=("length_2", "mean"),
            )

            # Compute deltas
            delta_s = (grouped["success_1"] - grouped["success_2"]) / (
                (grouped["success_1"] + grouped["success_2"]) / 2 + 1e-10
            )
            delta_l = (grouped["length_1"] - grouped["length_2"]) / (
                (grouped["length_1"] + grouped["length_2"]) / 2 + 1e-10
            )

            # Plot
            color = self._config.palette[idx % len(self._config.palette)]
            ax.scatter(delta_l, delta_s, c=color, s=100, alpha=0.7, edgecolors="black")

            # Add reference lines
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

            ax.set_xlabel("Относительный прирост средней длины маршрута $\\Delta L$")
            ax.set_ylabel("Относительное изменение доли успешных маршрутов $\\Delta S$")
            ax.set_title(f"{name1}\nvs {name2}")

        fig.suptitle("Сравнение алгоритмов: успех vs длина", fontsize=14)

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_all_plots(
        self,
        dfs: dict[str, pd.DataFrame],
        results: Any,
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Generate all plots and return their paths.

        Args:
            dfs: Dictionary of preprocessed DataFrames.
            results: ReportResults with analysis results.
            output_dir: Directory to save plots.

        Returns:
            Dictionary mapping plot types to file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        algorithm_names = list(dfs.keys())

        plot_paths = {}

        # Boxplot
        boxplot_path = output_dir / PlotType.BOXPLOT.filename
        self.generate_boxplot(dfs, boxplot_path, algorithm_names)
        plot_paths[PlotType.BOXPLOT.value] = boxplot_path

        # Success lineplot
        lineplot_path = output_dir / PlotType.LINEPLOT_SUCCESS.filename
        self.generate_success_lineplot(
            results.success_rates, lineplot_path, algorithm_names
        )
        plot_paths[PlotType.LINEPLOT_SUCCESS.value] = lineplot_path

        # Optimal window plot
        optimal_path = output_dir / PlotType.OPTIMAL_WINDOW.filename
        self.generate_optimal_window_plot(
            results.optimal_window_results, optimal_path
        )
        plot_paths[PlotType.OPTIMAL_WINDOW.value] = optimal_path

        # Dual axis marginal plots
        dual_axis_path = output_dir / PlotType.DUAL_AXIS_MARGINAL.filename
        self.generate_dual_axis_marginal_plots(
            results.pairwise_results, dfs, dual_axis_path
        )
        plot_paths[PlotType.DUAL_AXIS_MARGINAL.value] = dual_axis_path

        return plot_paths
