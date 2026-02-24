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
    DUAL_AXIS_MARGINAL = "dual_axis_marginal"

    @property
    def filename(self) -> str:
        """Get the filename for this plot type."""
        match self:
            case PlotType.BOXPLOT:
                return "boxplot_path_lengths.png"
            case PlotType.LINEPLOT_SUCCESS:
                return "lineplot_success_rate.png"
            case PlotType.OPTIMAL_WINDOW:
                return "optimal_window_eta.png"
            case PlotType.DUAL_AXIS_MARGINAL:
                return "dual_axis_marginal_plots.png"


class DataSourceConfig(BaseModel):
    """Configuration for a single parquet data source."""

    file_path: Path
    russian_name: str
    color: str = "#89A1AE"

    def model_post_init(self, __context: Any) -> None:
        """Ensure file_path is absolute."""
        if not self.file_path.is_absolute():
            self.file_path = RESULT_PATH / self.file_path


class OutlierConfig(BaseModel):
    """Configuration for outlier filtering."""

    min_length: float = 300_000.0  # Minimum path length in meters
    max_length: float = 1_500_000.0  # Maximum path length in meters
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
    figure_size: tuple[int, int] = (12, 8)

    def get_timestamped_report_path(self, timestamp: str) -> Path:
        """Get path for timestamped report directory."""
        return self.reports_path / timestamp

    def get_plots_path(self, timestamp: str) -> Path:
        """Get path for plots directory within report."""
        return self.get_timestamped_report_path(timestamp) / "plots"
