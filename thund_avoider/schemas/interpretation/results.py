from pathlib import Path

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
        """Convert to Markdown table row."""
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
    """
    Result of a paired comparison test operating on relative differences δ.
    effect_size is Cohen's d when a t-test was used, rank-biserial r otherwise.
    """

    comparison: str
    test_name: str
    statistic: float
    p_value: float
    p_value_corrected: float | None = None
    is_significant: bool
    effect_size: float
    effect_size_interpretation: str
    alpha: float = 0.05
    correction_method: str = "none"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class McNemarResult(BaseModel):
    """Result of McNemar's test for paired binary outcomes."""

    comparison: str
    statistic: float
    p_value: float
    p_value_corrected: float | None = None
    is_significant: bool
    discordant_pairs: int
    alpha: float = 0.05
    correction_method: str = "none"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FriedmanResult(BaseModel):
    """
    Result of a non-parametric Friedman repeated-measures omnibus test.

    kendalls_w is the effect size: W = χ² / (n · (k − 1)).
    condition_medians holds the per-condition median of the tested quantity.
    posthoc_conover is populated after run_posthoc_conover(); it maps
    'CondA vs CondB' → p-value (corrected).
    """

    factor: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    kendalls_w: float
    conditions: list[str]
    condition_medians: dict[str, float]
    posthoc_conover: dict[str, float] = {}   # filled in separately

    model_config = ConfigDict(arbitrary_types_allowed=True)


class JonckheereTerpstraResult(BaseModel):
    """
    Result of the Jonckheere-Terpstra test for ordered alternatives.

    Tests H₁: θ₁ ≤ θ₂ ≤ … ≤ θₖ (at least one strict) across ordered groups
    (window sizes 0 → 6).  A significant result confirms a monotone trend in
    median path length as the prediction window grows.
    """

    factor: str                  # e.g. "window_size (ConvLSTM, base)"
    statistic: float             # standardised JT statistic
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    trend_direction: str         # "decreasing", "increasing", or "none"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MannWhitneyResult(BaseModel):
    """
    Result of a Mann-Whitney U test comparing two independent distributions.

    Used in Phase 2 to test whether the masking effect on δ differs between
    the two algorithm variants.
    """

    comparison: str
    statistic: float
    p_value: float
    p_value_corrected: float | None = None
    is_significant: bool
    effect_size_r: float          # r = Z / √N
    effect_size_interpretation: str
    alpha: float = 0.05
    correction_method: str = "none"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SpearmanResult(BaseModel):
    """
    Result of a Spearman rank correlation.

    Used in Phase 5 to correlate optimisation gain with situation difficulty.
    """

    label: str
    rho: float
    p_value: float
    n: int
    is_significant: bool
    alpha: float = 0.05

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DegradationResult(BaseModel):
    """
    Descriptive summary of (predicted_config − ground_truth_config) / mid-point.

    Quantifies the cost of replacing oracle knowledge with a predictive model.
    """

    predicted_config: str
    ground_truth_config: str
    stats: DescriptiveStats       # over the per-situation δ values

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OptimalWindowResult(BaseModel):
    """
    Optimal window size identified by two complementary criteria.

    Primary (bootstrap): smallest window whose 95 % CI on the bootstrap
    median δ overlaps the CI of the window with the minimum bootstrap median.

    Secondary (heuristic): window maximising η = success_rate / mean_length.
    """

    algorithm: str

    # Primary bootstrap criterion
    optimal_window: int
    bootstrap_median: float
    ci_lower: float
    ci_upper: float

    # Secondary heuristic
    eta_optimal_window: int
    eta_score: float

    # Summaries at the bootstrap-optimal window
    success_rate: float
    mean_length: float
    n_success: int
    n_total: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnovaResult(BaseModel):
    """Result of repeated-measures ANOVA (parametric fallback only)."""

    factor: str
    f_statistic: float
    p_value: float
    degrees_of_freedom: tuple[int, int]
    effect_size_eta_squared: float
    is_significant: bool
    alpha: float = 0.05

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LeveneResult(BaseModel):
    """Result of Levene's test for equality of variances."""

    comparison: str
    statistic: float
    p_value: float
    is_homogeneous: bool
    alpha: float = 0.05

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DiagnosticsResult(BaseModel):
    """
    Per-configuration Phase-1 diagnostics.

    Aggregates normality test results and zero-inflation rates; the
    use_nonparametric flag is set True when *any* configuration is non-normal,
    and is propagated to all subsequent test-selection logic.
    """

    normality_by_config: dict[str, NormalityTestResult] = {}
    zero_inflation_by_config: dict[str, float] = {}   # fraction of δ == 0
    use_nonparametric: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PlotRegistry(BaseModel):
    """Typed registry of generated plot file paths."""

    boxplot: Path | None = None
    lineplot_success: Path | None = None
    optimal_window: Path | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def items(self) -> list[tuple[str, Path]]:
        """Return (name, path) pairs for all non-None plots."""
        return [
            (k, v)
            for k, v in self.model_dump().items()
            if v is not None
        ]


class ReportResults(BaseModel):
    """Container for all statistical analysis results across all phases."""

    # Phase 1
    diagnostics: DiagnosticsResult = DiagnosticsResult()

    # Phase 2 & 5 — pairwise length comparisons and McNemar success tests
    pairwise_results: list[PairwiseTestResult] = []
    mcnemar_results: list[McNemarResult] = []
    mcnemar_pred_valid_results: list[McNemarResult] = []

    # Phase 2 — masking interaction (Mann-Whitney on δ between algo groups)
    masking_interaction_results: list[MannWhitneyResult] = []

    # Phase 3 — Friedman omnibus + Conover post-hoc (stored inside FriedmanResult)
    friedman_results: list[FriedmanResult] = []

    # Phase 4 — trend tests and optimal windows
    trend_results: list[JonckheereTerpstraResult] = []
    optimal_window_results: list[OptimalWindowResult] = []

    # Phase 5 — Spearman gain vs difficulty
    spearman_results: list[SpearmanResult] = []

    # Phase 6 — degradation from ground-truth
    degradation_results: list[DegradationResult] = []

    # Supporting / descriptive
    descriptive_stats: dict[str, DescriptiveStats] = {}
    normality_results: dict[str, NormalityTestResult] = {}   # kept for back-compat
    anova_results: list[AnovaResult] = []                    # parametric fallback
    levene_results: list[LeveneResult] = []
    success_rates: dict[str, dict[int, float]] = {}
    pred_valid_rates: dict[str, dict[int, float]] = {}
    plot_paths: PlotRegistry = PlotRegistry()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- convenience adders ---

    def add_descriptive_stats(self, name: str, s: DescriptiveStats) -> None:
        self.descriptive_stats[name] = s

    def add_normality_result(self, name: str, r: NormalityTestResult) -> None:
        self.normality_results[name] = r

    def add_pairwise_result(self, r: PairwiseTestResult) -> None:
        self.pairwise_results.append(r)

    def add_mcnemar_result(self, r: McNemarResult) -> None:
        self.mcnemar_results.append(r)

    def add_mcnemar_pred_valid_result(self, r: McNemarResult) -> None:
        self.mcnemar_pred_valid_results.append(r)

    def add_masking_interaction_result(self, r: MannWhitneyResult) -> None:
        self.masking_interaction_results.append(r)

    def add_friedman_result(self, r: FriedmanResult) -> None:
        self.friedman_results.append(r)

    def add_trend_result(self, r: JonckheereTerpstraResult) -> None:
        self.trend_results.append(r)

    def add_optimal_window_result(self, r: OptimalWindowResult) -> None:
        self.optimal_window_results.append(r)

    def add_spearman_result(self, r: SpearmanResult) -> None:
        self.spearman_results.append(r)

    def add_degradation_result(self, r: DegradationResult) -> None:
        self.degradation_results.append(r)

    def add_anova_result(self, r: AnovaResult) -> None:
        self.anova_results.append(r)

    def add_levene_result(self, r: LeveneResult) -> None:
        self.levene_results.append(r)

    def add_plot_path(self, plot_type: str, path: Path) -> None:
        """Set a plot path by field name (e.g. 'boxplot')."""
        if hasattr(self.plot_paths, plot_type):
            setattr(self.plot_paths, plot_type, path)
