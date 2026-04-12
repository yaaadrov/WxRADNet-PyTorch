from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from thund_avoider.settings import RESULT_PATH


class PlotType(StrEnum):
    """Types of plots for statistical analysis."""

    BOXPLOT = "boxplot"
    LINEPLOT_SUCCESS = "lineplot_success"
    OPTIMAL_WINDOW = "optimal_window"
    COMBINED = "combined"

    @property
    def filename_base(self) -> str:
        """Get the base filename (without extension) for this plot type."""
        match self:
            case PlotType.BOXPLOT:
                return "boxplot_path_lengths"
            case PlotType.LINEPLOT_SUCCESS:
                return "lineplot_success_rate"
            case PlotType.OPTIMAL_WINDOW:
                return "optimal_window_eta"
            case PlotType.COMBINED:
                return "combined_boxplot_lineplot"


class DataSourceConfig(BaseModel):
    """Configuration for a single parquet data source."""

    file_path: Path
    russian_name: str
    color: str = "#89A1AE"
    plot_pred_valid_rate: bool = False
    draw_success_lineplot: bool = False
    draw_optimal_window_plot: bool = False
    lineplot_marker: str = "o"

    def model_post_init(self, __context: Any) -> None:
        """Ensure file_path is absolute."""
        if not self.file_path.is_absolute():
            self.file_path = RESULT_PATH / self.file_path


class OutlierConfig(BaseModel):
    """Configuration for outlier filtering."""

    min_length: float = 0.0  # 300_000.0  # Minimum path length in meters
    max_length: float = 10_000_000.0  #  1_500_000.0  # Maximum path length in meters
    suspicious_timestamps: list[str] = Field(default_factory=list)


class ReportConfig(BaseModel):
    """Main configuration for report generation."""

    reports_path: Path = RESULT_PATH / "reports"
    data_sources: list[DataSourceConfig] = Field(default_factory=list)
    outlier_config: OutlierConfig = OutlierConfig()
    confidence_level: float = 0.95
    alpha: float = 0.05
    palette: list[str] = Field(
        default_factory=lambda: [
            "#89A1AE",
            "#FF5B17",
            "#4042EE",
            "#00DA72",
            "#FF92B7",
        ]
    )
    figure_dpi: int = 150
    figure_size: tuple[float, float] = (12, 7.5)
    # Per-plot figure sizes
    boxplot_size: tuple[float, float] = (12, 7.5)
    lineplot_size: tuple[float, float] = (5, 5)
    optimal_window_size: tuple[float, float] = (5, 5)

    def get_timestamped_report_path(self, timestamp: str) -> Path:
        """Get path for timestamped report directory."""
        return self.reports_path / timestamp

    def get_plots_path(self, timestamp: str) -> Path:
        """Get path for plots directory within report."""
        return self.get_timestamped_report_path(timestamp) / "plots"
