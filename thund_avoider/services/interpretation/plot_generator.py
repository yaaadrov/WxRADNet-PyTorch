from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from thund_avoider.schemas.interpretation.config import PlotType, ReportConfig


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
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )
        sns.set_palette(self._config.palette)

    def _set_figure_size(self, size: tuple[float, float]) -> None:
        """Set figure size via rcParams."""
        plt.rcParams["figure.figsize"] = size

    def _get_color_map(self, algorithm_names: list[str]) -> dict[str, str]:
        """Create color mapping for algorithms."""
        return {
            name: self._config.palette[i % len(self._config.palette)]
            for i, name in enumerate(algorithm_names)
        }

    @staticmethod
    def _save_figure_both_formats(fig: plt.Figure, output_path: Path) -> list[Path]:
        """
        Save figure as both SVG and PDF.

        Args:
            fig: Matplotlib figure to save.
            output_path: Base path (without extension).

        Returns:
            List of paths to saved files (SVG and PDF).
        """
        svg_path = output_path.with_suffix(".svg")
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(svg_path, bbox_inches="tight", format="svg")
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
        return [svg_path, pdf_path]

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

        # Set figure size and create plot
        # self._set_figure_size(self._config.boxplot_size)
        fig, ax = plt.subplots(figsize=self._config.boxplot_size)

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

        ax.set_ylim(-150_000, None)
        ax.set_xlabel("Размер окна $w$")
        ax.set_ylabel("Длина маршрута обхода, м")
        ax.set_title("Распределение длин маршрутов обхода")
        ax.legend(title="Постановка эксперимента", loc="lower left")

        plt.tight_layout()
        self._save_figure_both_formats(fig, output_path)
        plt.close(fig)

        return output_path.with_suffix(".svg")

    def generate_success_lineplot(
        self,
        pred_valid_rates: dict[str, dict[int, float]],
        output_path: Path,
        algorithm_names: list[str] | None = None,
        color_map: dict[str, str] | None = None,
        marker_map: dict[str, str] | None = None,
    ) -> Path:
        """
        Generate lineplot of pred_valid rates by window size.

        Args:
            pred_valid_rates: Dictionary of pred_valid rates by window size.
            output_path: Path to save the plot.
            algorithm_names: Ordered list of algorithm names (only those with plot_pred_valid_rate=True).
            color_map: Mapping of algorithm names to colors (from DataSourceConfig).
            marker_map: Mapping of algorithm names to markers (from DataSourceConfig).

        Returns:
            Path to saved plot.
        """
        if algorithm_names is None:
            algorithm_names = list(pred_valid_rates.keys())

        # Prepare data
        plot_data = []
        for name in algorithm_names:
            if name not in pred_valid_rates:
                continue
            for ws, rate in pred_valid_rates[name].items():
                plot_data.append(
                    {"window_size": ws, "success_rate": rate, "algorithm": name}
                )

        df = pd.DataFrame(plot_data)

        if len(df) == 0:
            return output_path

        # Set figure size and create plot
        # self._set_figure_size(self._config.lineplot_size)
        fig, ax = plt.subplots(figsize=self._config.lineplot_size)

        # Use provided color_map or fall back to palette-based
        if color_map is None:
            color_map = self._get_color_map(algorithm_names)

        for name in algorithm_names:
            if name not in pred_valid_rates:
                continue
            data = df[df["algorithm"] == name].sort_values("window_size")
            if len(data) == 0:
                continue
            ax.plot(
                data["window_size"],
                data["success_rate"],
                marker=marker_map.get(name, "o") if marker_map else "o",
                label=name,
                color=color_map[name],
                linewidth=2.5,
                markersize=10,
            )

        ax.set_xlabel("Размер окна $w$")
        ax.set_ylabel("Доля валидных маршрутов")
        ax.set_title("Валидные маршруты")
        ax.legend(title="Постановка эксперимента", loc="lower left")
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        self._save_figure_both_formats(fig, output_path)
        plt.close(fig)

        return output_path.with_suffix(".svg")

    def generate_optimal_window_plot(
        self,
        dfs: dict[str, pd.DataFrame],
        pred_valid_rates: dict[str, dict[int, float]],
        output_path: Path,
        algorithm_names: list[str],
        color_map: dict[str, str],
    ) -> Path:
        """
        Generate multi-panel plot showing relative changes in success rate and path length.

        Creates a horizontal row of subplots, one per algorithm, showing:
        - delta_s: relative change in success rate vs window 0
        - delta_l: relative change in path length vs window 0
        - ratio: delta_s / delta_l

        Args:
            dfs: Dictionary of DataFrames with pathfinding results.
            pred_valid_rates: Dictionary of pred_valid rates by window size.
            output_path: Path to save the plot.
            algorithm_names: List of algorithm names (only those with plot_pred_valid_rate=True).
            color_map: Mapping of algorithm names to colors (from DataSourceConfig.color).

        Returns:
            Path to saved plot.
        """
        n_algorithms = len(algorithm_names)
        if n_algorithms == 0:
            return output_path

        # Collect all data for computing shared y-limits
        all_delta_s: list[float] = []
        all_delta_l: list[float] = []

        # Compute delta values for each algorithm (relative to previous window)
        plot_data: dict[str, dict[str, dict[int, float]]] = {}
        for name in algorithm_names:
            if name not in pred_valid_rates or name not in dfs:
                continue

            rates = pred_valid_rates[name]
            df = dfs[name]

            if 0 not in rates:
                continue

            # Get sorted window sizes
            window_sizes = sorted(rates.keys())

            # Compute mean lengths for each window
            mean_lengths = {}
            for ws in window_sizes:
                ml = df[df["window_size"] == ws]["length"].mean()
                if not pd.isna(ml):
                    mean_lengths[ws] = ml

            delta_s: dict[int, float] = {}
            delta_l: dict[int, float] = {}
            ratio: dict[int, float] = {}

            for i, ws in enumerate(window_sizes):
                if i == 0:
                    # No previous window for window 0
                    continue

                prev_ws = window_sizes[i - 1]
                rate = rates[ws]
                prev_rate = rates[prev_ws]

                if ws not in mean_lengths or prev_ws not in mean_lengths:
                    continue

                mean_length = mean_lengths[ws]
                prev_mean_length = mean_lengths[prev_ws]

                # Relative delta rate: (rate[w] - rate[w-1]) / rate[w-1]
                ds = (rate - prev_rate) / (prev_rate + 1e-10)
                # Relative delta length: (length[w] - length[w-1]) / length[w-1]
                dl = (mean_length - prev_mean_length) / (prev_mean_length + 1e-10)

                delta_s[ws] = ds
                delta_l[ws] = dl
                ratio[ws] = ds / dl if abs(dl) > 1e-10 else float("inf")

                all_delta_s.append(ds)
                all_delta_l.append(dl)

            plot_data[name] = {"delta_s": delta_s, "delta_l": delta_l, "ratio": ratio}

        if not plot_data:
            return output_path

        # Compute shared y-limits with padding
        def compute_limits(values: list[float], padding: float = 0.1) -> tuple[float, float]:
            if not values:
                return (-0.1, 0.1)
            vmin, vmax = min(values), max(values)
            span = vmax - vmin
            if span == 0:
                span = 0.1
            return (vmin - padding * span, vmax + padding * span)

        y_limits_s = compute_limits(all_delta_s)
        y_limits_l = compute_limits(all_delta_l)

        # Create figure with horizontal subplots
        fig_width = self._config.optimal_window_size[0] * n_algorithms
        fig_height = self._config.optimal_window_size[1]
        fig, axes = plt.subplots(1, n_algorithms, figsize=(fig_width, fig_height))

        # Ensure axes is always iterable
        if n_algorithms == 1:
            axes = [axes]

        # Colors for delta_l and ratio
        delta_l_color = "#FF5B17"
        ratio_color = "#89A1AE"
        border_color = "#1D2336"

        for ax, name in zip(axes, algorithm_names):
            if name not in plot_data:
                ax.set_visible(False)
                continue

            data = plot_data[name]
            delta_s_vals = data["delta_s"]
            delta_l_vals = data["delta_l"]
            ratio_vals = data["ratio"]

            window_sizes = sorted(delta_s_vals.keys())

            # Create twinx axis
            ax2 = ax.twinx()

            # Plot delta_s (primary y-axis)
            ds_values = [delta_s_vals[w] for w in window_sizes]
            line1 = ax.plot(
                window_sizes,
                ds_values,
                marker="o",
                color=color_map.get(name, "#89A1AE"),
                linewidth=2.5,
                markersize=10,
                label="$\\Delta S$",
            )

            # Plot delta_l (secondary y-axis)
            dl_values = [delta_l_vals[w] for w in window_sizes]
            line2 = ax2.plot(
                window_sizes,
                dl_values,
                marker="s",
                color=delta_l_color,
                linewidth=2.5,
                markersize=10,
                label="$\\Delta L$",
            )

            # Plot ratio (secondary y-axis, dashed) - rescaled to fit y_limits_l
            r_values = [ratio_vals[w] if w in ratio_vals and not pd.isna(ratio_vals[w]) else None for w in window_sizes]
            valid_ws = [w for w, r in zip(window_sizes, r_values) if r is not None and abs(r) < 100]
            valid_r = [r for r in r_values if r is not None and abs(r) < 100]

            # Rescale ratio values to fit within y_limits_l
            if valid_r:
                r_min, r_max = min(valid_r), max(valid_r)
                r_span = r_max - r_min if r_max != r_min else 1.0
                target_min, target_max = y_limits_l
                target_min = target_min + (target_max - target_min) * 0.1
                target_max = target_max - (target_max - target_min) * 0.1
                target_span = target_max - target_min
                rescaled_r = [target_min + (r - r_min) / r_span * target_span for r in valid_r]
            else:
                rescaled_r = []

            line3 = ax2.plot(
                valid_ws,
                rescaled_r,
                marker="D",
                color=ratio_color,
                linewidth=2.5,
                markersize=10,
                linestyle="--",
                label="$\\Delta S / \\Delta L$",
            )

            # Set y-limits (shared across all subplots)
            ax.set_ylim(y_limits_s)
            ax2.set_ylim(y_limits_l)

            # Style axes
            ax.set_xlabel("Размер окна $w$")
            ax.set_ylabel("$\\Delta S$ (Относительное изменение успеха)")
            ax2.set_ylabel("$\\Delta L$ (Относительное изменение длины)")

            ax.set_title(name)

            # Set border color
            for spine in ax.spines.values():
                spine.set_color(border_color)
            for spine in ax2.spines.values():
                spine.set_color(border_color)

            # Show y-ticks on both axes
            ax.tick_params(axis="y", which="both", left=True, right=False)
            ax2.tick_params(axis="y", which="both", left=False, right=True)

            # Add legend
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="best")

        plt.tight_layout()
        self._save_figure_both_formats(fig, output_path)
        plt.close(fig)

        return output_path.with_suffix(".svg")

    def generate_combined_plot(
        self,
        dfs: dict[str, pd.DataFrame],
        pred_valid_rates: dict[str, dict[int, float]],
        output_path: Path,
        boxplot_names: list[str],
        lineplot_names: list[str],
        lineplot_colors: dict[str, str],
        lineplot_markers: dict[str, str] | None = None,
    ) -> Path:
        """
        Generate combined plot with boxplot and success lineplot side by side.

        Args:
            dfs: Dictionary of DataFrames with pathfinding results.
            pred_valid_rates: Dictionary of pred_valid rates by window size.
            output_path: Path to save the plot.
            boxplot_names: List of algorithm names for boxplot.
            lineplot_names: List of algorithm names for lineplot.
            lineplot_colors: Mapping of algorithm names to colors for lineplot.
            lineplot_markers: Mapping of algorithm names to markers for lineplot.

        Returns:
            Path to saved plot.
        """
        # Create figure with two subplots side by side
        boxplot_width = self._config.boxplot_size[0]
        lineplot_width = self._config.lineplot_size[0]
        fig_height = max(self._config.boxplot_size[1], self._config.lineplot_size[1])
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(boxplot_width + lineplot_width, fig_height),
            gridspec_kw={"width_ratios": [boxplot_width, lineplot_width]},
        )

        # --- Left subplot: Boxplot ---
        plot_data = []
        for name in boxplot_names:
            if name in dfs:
                df = dfs[name].copy()
                df["algorithm"] = name
                plot_data.append(df[["length", "window_size", "algorithm"]])

        if plot_data:
            combined = pd.concat(plot_data, ignore_index=True)
            color_map = self._get_color_map(boxplot_names)
            palette = [color_map[name] for name in boxplot_names]

            sns.boxplot(
                data=combined,
                x="window_size",
                y="length",
                hue="algorithm",
                ax=ax1,
                palette=palette,
            )

            ax1.set_ylim(-150_000, None)
            ax1.set_xlabel("Размер окна $w$")
            ax1.set_ylabel("Длина маршрута обхода, м")
            ax1.set_title("Распределение длин маршрутов обхода")
            ax1.legend(title="Постановка эксперимента", loc="lower left")

        # --- Right subplot: Success lineplot ---
        lineplot_df_data = []
        for name in lineplot_names:
            if name not in pred_valid_rates:
                continue
            for ws, rate in pred_valid_rates[name].items():
                lineplot_df_data.append(
                    {"window_size": ws, "success_rate": rate, "algorithm": name}
                )

        df_lineplot = pd.DataFrame(lineplot_df_data)

        if len(df_lineplot) > 0:
            for name in lineplot_names:
                if name not in pred_valid_rates:
                    continue
                data = df_lineplot[df_lineplot["algorithm"] == name].sort_values("window_size")
                if len(data) == 0:
                    continue
                ax2.plot(
                    data["window_size"],
                    data["success_rate"],
                    marker=lineplot_markers.get(name, "o") if lineplot_markers else "o",
                    label=name,
                    color=lineplot_colors.get(name, "#89A1AE"),
                    linewidth=2.5,
                    markersize=10,
                )

            ax2.set_xlabel("Размер окна $w$")
            ax2.set_ylabel("Доля валидных маршрутов")
            ax2.set_title("Доля валидных маршрутов")
            ax2.legend(title="Постановка эксперимента", loc="lower left")
            ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        self._save_figure_both_formats(fig, output_path)
        plt.close(fig)

        return output_path.with_suffix(".svg")

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

        # Extract names and colors for each plot type
        lineplot_names: list[str] = []
        lineplot_colors: dict[str, str] = {}
        lineplot_markers: dict[str, str] = {}
        optimal_names: list[str] = []
        optimal_colors: dict[str, str] = {}

        for ds in self._config.data_sources:
            if ds.draw_success_lineplot:
                lineplot_names.append(ds.russian_name)
                lineplot_colors[ds.russian_name] = ds.color
                lineplot_markers[ds.russian_name] = ds.lineplot_marker
            if ds.draw_optimal_window_plot:
                optimal_names.append(ds.russian_name)
                optimal_colors[ds.russian_name] = ds.color

        plot_paths = {}

        # Boxplot
        boxplot_path = output_dir / PlotType.BOXPLOT.filename_base
        self.generate_boxplot(dfs, boxplot_path, algorithm_names)
        plot_paths[PlotType.BOXPLOT.value] = boxplot_path.with_suffix(".svg")

        # Success lineplot (only for algorithms with draw_success_lineplot=True)
        lineplot_path = output_dir / PlotType.LINEPLOT_SUCCESS.filename_base
        self.generate_success_lineplot(
            results.pred_valid_rates,
            lineplot_path,
            lineplot_names,
            color_map=lineplot_colors,
            marker_map=lineplot_markers,
        )
        plot_paths[PlotType.LINEPLOT_SUCCESS.value] = lineplot_path.with_suffix(".svg")

        # Optimal window plot (only for algorithms with draw_optimal_window_plot=True)
        optimal_path = output_dir / PlotType.OPTIMAL_WINDOW.filename_base
        self.generate_optimal_window_plot(
            dfs,
            results.pred_valid_rates,
            optimal_path,
            optimal_names,
            color_map=optimal_colors,
        )
        plot_paths[PlotType.OPTIMAL_WINDOW.value] = optimal_path.with_suffix(".svg")

        # Combined plot (boxplot + success lineplot side by side)
        combined_path = output_dir / PlotType.COMBINED.filename_base
        self.generate_combined_plot(
            dfs,
            results.pred_valid_rates,
            combined_path,
            boxplot_names=algorithm_names,
            lineplot_names=lineplot_names,
            lineplot_colors=lineplot_colors,
            lineplot_markers=lineplot_markers,
        )
        plot_paths[PlotType.COMBINED.value] = combined_path.with_suffix(".svg")

        return plot_paths
