from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class DescriptiveStats(BaseModel):
    """Descriptive statistics for a dataset."""

    mean: float
    median: float
    std: float
    variance: float
    n: int
    ci_lower: float
    ci_upper: float
    min_val: float
    max_val: float

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_markdown_table(self) -> str:
        """Convert to markdown table row."""
        return (
            f"{self.mean:.2f} | {self.median:.2f} | {self.std:.2f} | "
            f"{self.n} | [{self.ci_lower:.2f}, {self.ci_upper:.2f}]"
        )


class NormalityTestResult(BaseModel):
    """Result of Shapiro-Wilk normality test."""

    statistic: float
    p_value: float
    is_normal: bool
    alpha: float = 0.05

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PairwiseTestResult(BaseModel):
    """Result of pairwise statistical test (t-test or Wilcoxon)."""

    comparison: str
    test_name: str
    statistic: float
    p_value: float
    p_value_corrected: float | None = None
    is_significant: bool
    effect_size: float  # Cohen's d or rank-biserial correlation
    effect_size_interpretation: str
    alpha: float = 0.05

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_markdown_row(self) -> str:
        """Convert to markdown table row."""
        p_str = (
            f"{self.p_value_corrected:.4f}"
            if self.p_value_corrected is not None
            else f"{self.p_value:.4f}"
        )
        return (
            f"| {self.comparison} | {self.test_name} | {self.statistic:.4f} | "
            f"{p_str} | {'Да' if self.is_significant else 'Нет'} | "
            f"{self.effect_size:.3f} ({self.effect_size_interpretation}) |"
        )


class McNemarResult(BaseModel):
    """Result of McNemar's test for binary outcomes."""

    comparison: str
    statistic: float
    p_value: float
    p_value_corrected: float | None = None
    is_significant: bool
    discordant_pairs: int
    alpha: float = 0.05

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_markdown_row(self) -> str:
        """Convert to markdown table row."""
        p_str = (
            f"{self.p_value_corrected:.4f}"
            if self.p_value_corrected is not None
            else f"{self.p_value:.4f}"
        )
        return (
            f"| {self.comparison} | {self.statistic:.4f} | {p_str} | "
            f"{'Да' if self.is_significant else 'Нет'} | {self.discordant_pairs} |"
        )


class AnovaResult(BaseModel):
    """Result of repeated measures ANOVA."""

    factor: str
    f_statistic: float
    p_value: float
    degrees_of_freedom: tuple[int, int]
    effect_size_eta_squared: float
    is_significant: bool
    alpha: float = 0.05

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_markdown_row(self) -> str:
        """Convert to markdown table row."""
        return (
            f"| {self.factor} | {self.f_statistic:.4f} | {self.p_value:.4f} | "
            f"({self.degrees_of_freedom[0]}, {self.degrees_of_freedom[1]}) | "
            f"{self.effect_size_eta_squared:.3f} | "
            f"{'Да' if self.is_significant else 'Нет'} |"
        )


class LeveneResult(BaseModel):
    """Result of Levene's test for equality of variances."""

    comparison: str
    statistic: float
    p_value: float
    is_homogeneous: bool
    alpha: float = 0.05

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_markdown_row(self) -> str:
        """Convert to markdown table row."""
        return (
            f"| {self.comparison} | {self.statistic:.4f} | {self.p_value:.4f} | "
            f"{'Да' if self.is_homogeneous else 'Нет'} |"
        )


class OptimalWindowResult(BaseModel):
    """Result of optimal window analysis."""

    algorithm: str
    optimal_window: int
    eta_score: float
    success_rate: float
    mean_length: float
    n_success: int
    n_total: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_markdown_row(self) -> str:
        """Convert to markdown table row."""
        return (
            f"| {self.algorithm} | {self.optimal_window} | {self.eta_score:.6f} | "
            f"{self.success_rate:.2%} | {self.mean_length:.0f} м | "
            f"{self.n_success}/{self.n_total} |"
        )


class ReportResults(BaseModel):
    """Container for all statistical analysis results."""

    descriptive_stats: dict[str, DescriptiveStats] = {}
    normality_results: dict[str, NormalityTestResult] = {}
    pairwise_results: list[PairwiseTestResult] = []
    mcnemar_results: list[McNemarResult] = []
    anova_results: list[AnovaResult] = []
    levene_results: list[LeveneResult] = []
    optimal_window_results: list[OptimalWindowResult] = []
    success_rates: dict[str, dict[int, float]] = {}
    plot_paths: dict[str, Path] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_descriptive_stats(self, name: str, stats: DescriptiveStats) -> None:
        """Add descriptive statistics for a dataset."""
        self.descriptive_stats[name] = stats

    def add_normality_result(self, name: str, result: NormalityTestResult) -> None:
        """Add normality test result."""
        self.normality_results[name] = result

    def add_pairwise_result(self, result: PairwiseTestResult) -> None:
        """Add pairwise test result."""
        self.pairwise_results.append(result)

    def add_mcnemar_result(self, result: McNemarResult) -> None:
        """Add McNemar test result."""
        self.mcnemar_results.append(result)

    def add_anova_result(self, result: AnovaResult) -> None:
        """Add ANOVA result."""
        self.anova_results.append(result)

    def add_levene_result(self, result: LeveneResult) -> None:
        """Add Levene test result."""
        self.levene_results.append(result)

    def add_optimal_window_result(self, result: OptimalWindowResult) -> None:
        """Add optimal window result."""
        self.optimal_window_results.append(result)

    def add_plot_path(self, plot_type: str, path: Path) -> None:
        """Add plot file path."""
        self.plot_paths[plot_type] = path
