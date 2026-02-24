import itertools as it
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from thund_avoider.schemas.interpretation.config import ReportConfig
from thund_avoider.schemas.interpretation.results import (
    AnovaResult,
    DescriptiveStats,
    LeveneResult,
    McNemarResult,
    NormalityTestResult,
    OptimalWindowResult,
    PairwiseTestResult,
    ReportResults,
)


class StatisticalAnalyzer:
    """
    Performs statistical analysis on pathfinding results.

    This class handles all statistical tests including normality tests,
    pairwise comparisons, ANOVA, and optimal window analysis.
    """

    def __init__(self, config: ReportConfig) -> None:
        self._config = config
        self._results = ReportResults()

    @property
    def results(self) -> ReportResults:
        """Get accumulated results."""
        return self._results

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "ничтожный"
        elif d_abs < 0.5:
            return "малый"
        elif d_abs < 0.8:
            return "средний"
        else:
            return "большой"

    @staticmethod
    def _compute_cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Cohen's d effect size for paired samples."""
        diff = x1 - x2
        return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    @staticmethod
    def _confidence_interval(
        data: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Compute confidence interval for the mean."""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean - h, mean + h

    def compute_descriptive_stats(
        self,
        data: np.ndarray,
        confidence: float | None = None,
    ) -> DescriptiveStats:
        """
        Compute descriptive statistics for a dataset.

        Args:
            data: Input data array.
            confidence: Confidence level for CI.

        Returns:
            DescriptiveStats object.
        """
        confidence = confidence or self._config.confidence_level
        ci_lower, ci_upper = self._confidence_interval(data, confidence)

        return DescriptiveStats(
            mean=float(np.mean(data)),
            median=float(np.median(data)),
            std=float(np.std(data, ddof=1)),
            variance=float(np.var(data, ddof=1)),
            n=len(data),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
        )

    def test_normality(
        self,
        data: np.ndarray,
        alpha: float | None = None,
    ) -> NormalityTestResult:
        """
        Perform Shapiro-Wilk test for normality.

        Args:
            data: Input data array.
            alpha: Significance level.

        Returns:
            NormalityTestResult object.
        """
        alpha = alpha or self._config.alpha

        # Shapiro-Wilk has sample size limit of 5000
        if len(data) > 5000:
            data = np.random.choice(data, 5000, replace=False)

        statistic, p_value = stats.shapiro(data)

        return NormalityTestResult(
            statistic=statistic,
            p_value=p_value,
            is_normal=p_value > alpha,
            alpha=alpha,
        )

    def run_pairwise_test(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        comparison_name: str,
        alpha: float | None = None,
        bonferroni_factor: int = 1,
    ) -> PairwiseTestResult:
        """
        Run pairwise comparison test with automatic test selection.

        Automatically selects t-test (parametric) or Wilcoxon (non-parametric)
        based on normality of relative differences.

        Args:
            x1: First sample.
            x2: Second sample.
            comparison_name: Name for the comparison.
            alpha: Significance level.
            bonferroni_factor: Factor for Bonferroni correction.

        Returns:
            PairwiseTestResult object.
        """
        alpha = alpha or self._config.alpha

        # Compute relative differences
        delta = (x1 - x2) / ((x1 + x2) / 2 + 1e-10)

        # Test normality of differences
        normality = self.test_normality(delta)

        # Select appropriate test
        if normality.is_normal:
            statistic, p_value = stats.ttest_rel(x1, x2)
            test_name = "Парный t-тест"
        else:
            statistic, p_value = stats.wilcoxon(x1, x2)
            test_name = "Тест Вилкоксона"

        # Apply Bonferroni correction
        corrected_alpha = alpha / bonferroni_factor
        p_corrected = min(p_value * bonferroni_factor, 1.0)

        # Compute effect size
        effect_size = self._compute_cohens_d(x1, x2)
        effect_interpretation = self._interpret_cohens_d(effect_size)

        return PairwiseTestResult(
            comparison=comparison_name,
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            p_value_corrected=p_corrected,
            is_significant=p_corrected < corrected_alpha,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            alpha=corrected_alpha,
        )

    def run_mcnemar_test(
        self,
        success1: np.ndarray,
        success2: np.ndarray,
        comparison_name: str,
        alpha: float | None = None,
        bonferroni_factor: int = 1,
    ) -> McNemarResult:
        """
        Run McNemar's test for comparing binary outcomes.

        Args:
            success1: First success array (boolean).
            success2: Second success array (boolean).
            comparison_name: Name for the comparison.
            alpha: Significance level.
            bonferroni_factor: Factor for Bonferroni correction.

        Returns:
            McNemarResult object.
        """
        alpha = alpha or self._config.alpha

        # Build contingency table
        b = np.sum(success1 & ~success2)  # 1 success, 2 failure
        c = np.sum(~success1 & success2)  # 1 failure, 2 success

        # McNemar test with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c + 1e-10)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        # Handle edge case where b + c = 0
        if b + c == 0:
            statistic = 0.0
            p_value = 1.0

        # Apply Bonferroni correction
        corrected_alpha = alpha / bonferroni_factor
        p_corrected = min(p_value * bonferroni_factor, 1.0)

        return McNemarResult(
            comparison=comparison_name,
            statistic=float(statistic),
            p_value=float(p_value),
            p_value_corrected=p_corrected,
            is_significant=p_corrected < corrected_alpha,
            discordant_pairs=int(b + c),
            alpha=corrected_alpha,
        )

    def run_anova_rm(
        self,
        data_dict: dict[str, np.ndarray],
        factor_name: str,
        alpha: float | None = None,
    ) -> AnovaResult:
        """
        Run one-way repeated measures ANOVA.

        Args:
            data_dict: Dictionary mapping condition names to data arrays.
            factor_name: Name of the factor.
            alpha: Significance level.

        Returns:
            AnovaResult object.
        """
        alpha = alpha or self._config.alpha

        # Stack data for ANOVA
        data_matrix = np.column_stack(list(data_dict.values()))
        n_subjects, k_conditions = data_matrix.shape

        # Compute sums of squares
        grand_mean = np.mean(data_matrix)
        subject_means = np.mean(data_matrix, axis=1, keepdims=True)
        condition_means = np.mean(data_matrix, axis=0)

        # SS between conditions
        ss_between = n_subjects * np.sum((condition_means - grand_mean) ** 2)

        # SS within subjects
        ss_within = np.sum((data_matrix - subject_means) ** 2)

        # SS error (interaction)
        ss_error = ss_within - np.sum((subject_means.flatten() - grand_mean) ** 2)

        # Degrees of freedom
        df_between = k_conditions - 1
        df_error = (n_subjects - 1) * (k_conditions - 1)

        # F statistic
        ms_between = ss_between / df_between
        ms_error = ss_error / df_error
        f_stat = ms_between / (ms_error + 1e-10)

        # P value
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_error)

        # Effect size (eta-squared)
        eta_squared = ss_between / (ss_between + ss_error + 1e-10)

        return AnovaResult(
            factor=factor_name,
            f_statistic=float(f_stat),
            p_value=float(p_value),
            degrees_of_freedom=(df_between, df_error),
            effect_size_eta_squared=float(eta_squared),
            is_significant=p_value < alpha,
            alpha=alpha,
        )

    def run_levene_test(
        self,
        samples: list[np.ndarray],
        comparison_name: str,
        alpha: float | None = None,
    ) -> LeveneResult:
        """
        Run Levene's test for equality of variances.

        Args:
            samples: List of sample arrays.
            comparison_name: Name for the comparison.
            alpha: Significance level.

        Returns:
            LeveneResult object.
        """
        alpha = alpha or self._config.alpha

        statistic, p_value = stats.levene(*samples)

        return LeveneResult(
            comparison=comparison_name,
            statistic=float(statistic),
            p_value=float(p_value),
            is_homogeneous=p_value > alpha,
            alpha=alpha,
        )

    def compute_optimal_window(
        self,
        df: pd.DataFrame,
        algorithm_name: str,
    ) -> OptimalWindowResult:
        """
        Find optimal window size by maximizing η = success_rate / mean_length.

        Args:
            df: DataFrame with pathfinding results.
            algorithm_name: Name of the algorithm.

        Returns:
            OptimalWindowResult object.
        """
        # Group by window_size
        grouped = df.groupby("window_size").agg(
            success_rate=("success", "mean"),
            mean_length=("length", "mean"),
            n_success=("success", "sum"),
            n_total=("success", "count"),
        )

        # Compute η metric
        grouped["eta"] = grouped["success_rate"] / (grouped["mean_length"] + 1e-10)

        # Find optimal window
        optimal_idx = grouped["eta"].idxmax()
        optimal_row = grouped.loc[optimal_idx]

        return OptimalWindowResult(
            algorithm=algorithm_name,
            optimal_window=int(optimal_idx),
            eta_score=float(optimal_row["eta"]),
            success_rate=float(optimal_row["success_rate"]),
            mean_length=float(optimal_row["mean_length"]),
            n_success=int(optimal_row["n_success"]),
            n_total=int(optimal_row["n_total"]),
        )

    def compute_success_rates(
        self,
        dfs: dict[str, pd.DataFrame],
    ) -> dict[str, dict[int, float]]:
        """
        Compute success rates by window size for each algorithm.

        Args:
            dfs: Dictionary of DataFrames.

        Returns:
            Dictionary mapping algorithm names to success rate dicts.
        """
        success_rates = {}

        for name, df in dfs.items():
            rates = df.groupby("window_size")["success"].mean().to_dict()
            success_rates[name] = rates

        return success_rates

    def run_full_analysis(
        self,
        preprocessed_data: dict[str, pd.DataFrame],
        pairwise_combinations: list[tuple[str, str]] | None = None,
    ) -> ReportResults:
        """
        Run complete statistical analysis pipeline.

        Args:
            preprocessed_data: Dictionary of preprocessed DataFrames.
            pairwise_combinations: Specific pairs to compare (default: all pairs).

        Returns:
            ReportResults with all analysis results.
        """
        # Compute descriptive stats
        for name, df in preprocessed_data.items():
            stats_result = self.compute_descriptive_stats(df["length"].values)
            self._results.add_descriptive_stats(name, stats_result)

        # Compute success rates
        self._results.success_rates = self.compute_success_rates(preprocessed_data)

        # Compute optimal windows
        for name, df in preprocessed_data.items():
            optimal = self.compute_optimal_window(df, name)
            self._results.add_optimal_window_result(optimal)

        # Run pairwise comparisons
        if pairwise_combinations is None:
            pairwise_combinations = list(it.combinations(preprocessed_data.keys(), 2))

        n_comparisons = len(pairwise_combinations)

        for name1, name2 in pairwise_combinations:
            if name1 not in preprocessed_data or name2 not in preprocessed_data:
                continue

            df1 = preprocessed_data[name1]
            df2 = preprocessed_data[name2]

            # Merge for paired analysis
            merged = pd.merge(
                df1,
                df2,
                on=["subject_id", "window_size"],
                suffixes=("_1", "_2"),
            )

            if len(merged) == 0:
                continue

            # Pairwise test on lengths
            pairwise = self.run_pairwise_test(
                merged["length_1"].values,
                merged["length_2"].values,
                f"{name1} vs {name2}",
                bonferroni_factor=n_comparisons,
            )
            self._results.add_pairwise_result(pairwise)

            # McNemar test on success
            mcnemar = self.run_mcnemar_test(
                merged["success_1"].values,
                merged["success_2"].values,
                f"{name1} vs {name2}",
                bonferroni_factor=n_comparisons,
            )
            self._results.add_mcnemar_result(mcnemar)

        # Run Levene tests
        window_sizes = set()
        for df in preprocessed_data.values():
            window_sizes.update(df["window_size"].unique())

        for ws in sorted(window_sizes):
            samples = []
            names = []
            for name, df in preprocessed_data.items():
                ws_data = df[df["window_size"] == ws]["length"].values
                if len(ws_data) > 0:
                    samples.append(ws_data)
                    names.append(name)

            if len(samples) >= 2:
                levene = self.run_levene_test(samples, f"Window {ws}")
                self._results.add_levene_result(levene)

        return self._results
