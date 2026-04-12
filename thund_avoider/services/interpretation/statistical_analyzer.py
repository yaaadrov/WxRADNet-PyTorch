"""
Statistical analysis pipeline for thunderstorm-avoidance pathfinding experiments.

Phases implemented
------------------
1  Distributional diagnostics   – Shapiro-Wilk on per-config residuals, zero-inflation
                                   check, global use_nonparametric flag.
2  Masking effect                – Paired Wilcoxon/t on δ (per algorithm), Holm
                                   correction within family; Mann-Whitney interaction.
3  Prediction-model comparison   – Friedman omnibus per (algo × window_size),
                                   BH-FDR across the battery; Conover-Iman post-hoc.
4  Window-size trend & optimum   – Jonckheere-Terpstra trend test per config;
                                   bootstrap CI optimal window (primary);
                                   η heuristic (secondary).
5  Aposteriori optimisation      – Paired test per (prediction_model × window_size),
                                   BH-FDR; Spearman gain vs difficulty.
6  Degradation from ground-truth – Descriptive stats on δ(predicted − GT).

Correction strategy
-------------------
  Within a family of ≤ ~3 tests  → Holm FWER  (_apply_holm_correction)
  Battery of many related tests   → BH-FDR     (_apply_fdr_correction)
All test methods return raw (uncorrected) p-values; correction is a separate pass.
"""

import itertools as it
from typing import Any

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from thund_avoider.schemas.interpretation.config import ReportConfig
from thund_avoider.schemas.interpretation.results import (
    AnovaResult,
    DegradationResult,
    DescriptiveStats,
    DiagnosticsResult,
    FriedmanResult,
    JonckheereTerpstraResult,
    LeveneResult,
    MannWhitneyResult,
    McNemarResult,
    NormalityTestResult,
    OptimalWindowResult,
    PairwiseTestResult,
    ReportResults,
    SpearmanResult,
)


class StatisticalAnalyzer:
    """
    Performs statistical analysis on pathfinding results.

    All public run_* and compute_* methods return raw (uncorrected) p-values.
    The caller — or run_full_analysis — groups tests into families and calls
    the appropriate correction helper (_apply_holm_correction / _apply_fdr_correction).
    """

    def __init__(self, config: ReportConfig) -> None:
        self._config = config
        self._results = ReportResults()
        # Conservative default; overwritten by run_phase1_diagnostics
        self._use_nonparametric: bool = True

    @property
    def results(self) -> ReportResults:
        """Get accumulated results."""
        return self._results

    @property
    def use_nonparametric(self) -> bool:
        """Global test-family flag set by run_phase1_diagnostics."""
        return self._use_nonparametric

    def _get_pred_valid_rate_names(self) -> set[str]:
        """Get algorithm names with plot_pred_valid_rate flag enabled."""
        return {
            ds.russian_name
            for ds in self._config.data_sources
            if ds.plot_pred_valid_rate
        }

    # =========================================================================
    # Effect-size helpers
    # =========================================================================

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        d_abs = abs(d)
        if d_abs < 0.2:
            return "ничтожный"
        elif d_abs < 0.5:
            return "малый"
        elif d_abs < 0.8:
            return "средний"
        return "большой"

    @staticmethod
    def _interpret_rank_biserial(r: float) -> str:
        r_abs = abs(r)
        if r_abs < 0.1:
            return "ничтожный"
        elif r_abs < 0.3:
            return "малый"
        elif r_abs < 0.5:
            return "средний"
        return "большой"

    @staticmethod
    def _compute_cohens_d_paired(delta: np.ndarray) -> float:
        """Cohen's d from pre-computed paired differences δ."""
        return float(np.mean(delta) / (np.std(delta, ddof=1) + 1e-10))

    @staticmethod
    def _compute_rank_biserial(wilcoxon_statistic: float, n: int) -> float:
        """Rank-biserial r from scipy Wilcoxon W (minimum rank sum)."""
        return float(1.0 - (4.0 * wilcoxon_statistic) / (n * (n + 1)))

    @staticmethod
    def _confidence_interval(
        data: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """t-based CI for the mean."""
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return float(mean - h), float(mean + h)

    # =========================================================================
    # Descriptive statistics
    # =========================================================================

    def compute_descriptive_stats(
        self,
        data: np.ndarray,
        confidence: float | None = None,
    ) -> DescriptiveStats:
        """Compute mean, median, SD, variance, CI, min, max."""
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

    # =========================================================================
    # Phase 1 — Distributional diagnostics
    # =========================================================================

    def test_normality(
        self,
        data: np.ndarray,
        alpha: float | None = None,
    ) -> NormalityTestResult:
        """
        Shapiro-Wilk test on the full sample (no sub-sampling).

        Valid for 3 ≤ n ≤ 5000. Raises ValueError for larger samples so the
        caller splits by window size first rather than silently degrading.
        """
        alpha = alpha or self._config.alpha
        if len(data) > 5000:
            raise ValueError(
                f"Shapiro-Wilk unreliable for n > 5000 (got n={len(data)}). "
                "Split into per-window-size subsets before testing normality."
            )
        statistic, p_value = stats.shapiro(data)
        return NormalityTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            is_normal=bool(p_value > alpha),
            alpha=alpha,
        )

    def run_phase1_diagnostics(
        self,
        preprocessed_data: dict[str, pd.DataFrame],
        reference_pairs: list[tuple[str, str]],
    ) -> DiagnosticsResult:
        """
        Phase 1: normality of per-configuration residuals + zero-inflation of δ.

        Normality is tested on *within-window-size residuals* (length minus the
        per-window mean) to avoid window-size effects inflating non-normality.
        The test is run per window size; the config is marked non-normal if any
        window fails.

        The global flag use_nonparametric is set True when any configuration is
        non-normal OR any reference pair has zero-inflation > 5 %.

        Args:
            preprocessed_data: Dict of preprocessed DataFrames.
            reference_pairs: Pairs used to compute δ for zero-inflation.

        Returns:
            DiagnosticsResult (also stored in self._results.diagnostics).
        """
        alpha = self._config.alpha
        diag = DiagnosticsResult()
        any_nonnormal = False

        for name, df in preprocessed_data.items():
            config_nonnormal = False
            last_result: NormalityTestResult | None = None
            for ws, group in df.groupby("window_size"):
                ws_resid = (group["length"] - group["length"].mean()).values
                if len(ws_resid) < 3:
                    continue
                result = self.test_normality(ws_resid, alpha=alpha)
                last_result = result
                if not result.is_normal:
                    config_nonnormal = True
            if last_result is not None:
                diag.normality_by_config[name] = last_result
            if config_nonnormal:
                any_nonnormal = True

        for name1, name2 in reference_pairs:
            if name1 not in preprocessed_data or name2 not in preprocessed_data:
                continue
            merged = pd.merge(
                preprocessed_data[name1],
                preprocessed_data[name2],
                on=["subject_id", "window_size"],
                suffixes=("_1", "_2"),
            )
            if len(merged) == 0:
                continue
            x1 = merged["length_1"].values
            x2 = merged["length_2"].values
            delta = (x1 - x2) / ((x1 + x2) / 2 + 1e-10)
            zi = float(np.sum(delta == 0) / len(delta))
            diag.zero_inflation_by_config[f"{name1} vs {name2}"] = zi
            if zi > 0.05:
                any_nonnormal = True

        diag.use_nonparametric = any_nonnormal
        self._use_nonparametric = any_nonnormal
        self._results.diagnostics = diag
        return diag

    # =========================================================================
    # Phase 2 — Paired test (Wilcoxon on δ / one-sample t on δ)
    # =========================================================================

    def run_pairwise_test(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        comparison_name: str,
        alpha: float | None = None,
        force_nonparametric: bool | None = None,
    ) -> PairwiseTestResult:
        """
        Paired comparison on relative differences δ = (x1 − x2) / mid.

        Test selection follows the global use_nonparametric flag (Phase 1)
        unless force_nonparametric overrides it.

        Effect size:
          t-test  → Cohen's d on δ (mean δ / SD δ).
          Wilcoxon → rank-biserial r = 1 − 4W / (n(n+1)).

        Returns an uncorrected result; apply a correction pass afterwards.
        """
        alpha = alpha or self._config.alpha
        nonparam = (
            force_nonparametric
            if force_nonparametric is not None
            else self._use_nonparametric
        )

        delta = (x1 - x2) / ((x1 + x2) / 2 + 1e-10)
        delta_nonzero = delta[delta != 0]

        if not nonparam:
            statistic, p_value = stats.ttest_1samp(delta, popmean=0.0)
            test_name = "Одновыборочный t-тест (на δ)"
            effect_size = self._compute_cohens_d_paired(delta)
            effect_interpretation = self._interpret_cohens_d(effect_size)
        else:
            if len(delta_nonzero) == 0:
                statistic, p_value = 0.0, 1.0
                effect_size = 0.0
            else:
                statistic, p_value = stats.wilcoxon(delta_nonzero)
                effect_size = self._compute_rank_biserial(
                    float(statistic), len(delta_nonzero)
                )
            test_name = "Тест Вилкоксона (на δ)"
            effect_interpretation = self._interpret_rank_biserial(effect_size)

        return PairwiseTestResult(
            comparison=comparison_name,
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            p_value_corrected=float(p_value),
            is_significant=float(p_value) < alpha,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            alpha=alpha,
            correction_method="none",
        )

    def run_mann_whitney_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        comparison_name: str,
        alpha: float | None = None,
    ) -> MannWhitneyResult:
        """
        Two-sided Mann-Whitney U test for independent samples.

        Used in Phase 2 to test whether the masking effect (δ distribution)
        differs between the two algorithm variants — i.e. Algorithm moderates
        the masking effect.

        Effect size: r = |Z| / √N  where Z is the normal approximation of U.
        """
        alpha = alpha or self._config.alpha
        statistic, p_value = stats.mannwhitneyu(
            group_a, group_b, alternative="two-sided"
        )
        n = len(group_a) + len(group_b)
        mean_u = len(group_a) * len(group_b) / 2
        std_u = np.sqrt(len(group_a) * len(group_b) * (n + 1) / 12)
        z = (float(statistic) - mean_u) / (float(std_u) + 1e-10)
        r = float(abs(z) / np.sqrt(n))
        return MannWhitneyResult(
            comparison=comparison_name,
            statistic=float(statistic),
            p_value=float(p_value),
            p_value_corrected=float(p_value),
            is_significant=float(p_value) < alpha,
            effect_size_r=r,
            effect_size_interpretation=self._interpret_rank_biserial(r),
            alpha=alpha,
            correction_method="none",
        )

    # =========================================================================
    # Phase 3 — Friedman omnibus + Conover-Iman post-hoc
    # =========================================================================

    def run_friedman_test(
        self,
        data_dict: dict[str, np.ndarray],
        factor_name: str,
        alpha: float | None = None,
    ) -> FriedmanResult:
        """
        Non-parametric one-way repeated-measures Friedman test.

        All arrays must be the same length (one value per subject/situation).
        Kendall's W = χ² / (n · (k − 1)) is the effect size.

        Returns a result with empty posthoc_conover; populate with
        run_posthoc_conover() after correction.
        """
        alpha = alpha or self._config.alpha
        conditions = list(data_dict.keys())
        arrays = [data_dict[c] for c in conditions]
        statistic, p_value = stats.friedmanchisquare(*arrays)
        n, k = len(arrays[0]), len(arrays)
        kendalls_w = float(statistic / (n * (k - 1)))
        return FriedmanResult(
            factor=factor_name,
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=float(p_value) < alpha,
            alpha=alpha,
            kendalls_w=kendalls_w,
            conditions=conditions,
            condition_medians={c: float(np.median(a)) for c, a in zip(conditions, arrays)},
            posthoc_conover={},
        )

    def run_posthoc_conover(
        self,
        data_dict: dict[str, np.ndarray],
        friedman_result: FriedmanResult,
        p_adjust: str = "holm",
    ) -> FriedmanResult:
        """
        Conover-Iman post-hoc pairwise tests after a significant Friedman.

        Populates friedman_result.posthoc_conover in-place as a flat dict:
        'CondA vs CondB' → adjusted p-value.

        Args:
            data_dict: Same dict passed to run_friedman_test.
            friedman_result: Result to populate.
            p_adjust: Adjustment method for scikit_posthocs ('holm' or 'fdr_bh').
        """
        conditions = list(data_dict.keys())
        df = pd.DataFrame({c: data_dict[c] for c in conditions})
        ph = sp.posthoc_conover_friedman(df, p_adjust_method=p_adjust)
        posthoc: dict[str, float] = {}
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i + 1:]:
                posthoc[f"{c1} vs {c2}"] = float(ph.loc[c1, c2])
        friedman_result.posthoc_conover = posthoc
        return friedman_result

    # =========================================================================
    # Phase 4 — Jonckheere-Terpstra trend test + optimal window
    # =========================================================================

    @staticmethod
    def jonckheere_test(
        samples: list[list[float]],
        alternative: str = "two-sided",
    ) -> tuple[float, float]:
        J = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                xi = samples[i]
                xj = samples[j]
                # count pairwise comparisons
                J += np.sum(xj[:, None] > xi)
                J += 0.5 * np.sum(xj[:, None] == xi)

        # Compute mean and variance under H0
        n = np.array([len(s) for s in samples])
        N = np.sum(n)

        mean_J = (N**2 - np.sum(n**2)) / 4
        var_J = (
            (N**2 * (2 * N + 3) - np.sum(n**2 * (2 * n + 3)))
            / 72
        )

        # Z-score
        z = (J - mean_J) / (var_J ** 0.5)

        # p-value
        if alternative == "greater":
            p_value = 1 - stats.norm.cdf(z)
        elif alternative == "less":
            p_value = stats.norm.cdf(z)
        elif alternative == "two-sided":
            p_value = 2 * min(stats.norm.cdf(z), 1 - stats.norm.cdf(z))
        else:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

        return float(z), float(p_value)

    def run_jonckheere_terpstra(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        factor_name: str,
        alternative: str = "two-sided",
        alpha: float | None = None,
    ) -> JonckheereTerpstraResult:
        """
        Jonckheere-Terpstra test for ordered alternatives (window sizes 0 → 6).

        Tests H₁: median changes monotonically across ordered groups.
        Requires scipy ≥ 1.10 (scipy.stats.jonckheere_terpstra).

        Args:
            df: DataFrame with value_col and group_col.
            value_col: Column to test (e.g. 'length').
            group_col: Ordered grouping column (e.g. 'window_size').
            factor_name: Label for reporting.
            alternative: 'two-sided', 'less' (decreasing), or 'greater'.
            alpha: Significance level.
        """
        alpha = alpha or self._config.alpha
        ordered_groups = sorted(df[group_col].unique())
        samples = [
            df.loc[df[group_col] == g, value_col].values
            for g in ordered_groups
        ]
        statistic, p_value = self.jonckheere_test(samples, alternative)

        # Handle alternative hypothesis direction
        if alternative == "two-sided":
            p_value = min(p_value * 2, 1.0)
        elif alternative == "less":
            # decreasing trend
            pass
        elif alternative == "greater":
            # increasing trend
            pass
        else:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

        if p_value >= alpha:
            trend_direction = "none"
        elif alternative == "less" or (alternative == "two-sided" and statistic < 0):
            trend_direction = "decreasing"
        else:
            trend_direction = "increasing"

        return JonckheereTerpstraResult(
            factor=factor_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < alpha,
            alpha=alpha,
            trend_direction=trend_direction,
        )

    def compute_optimal_window(
        self,
        df: pd.DataFrame,
        algorithm_name: str,
        n_bootstrap: int = 1000,
        random_state: int = 42,
    ) -> OptimalWindowResult:
        """
        Identify the optimal prediction window size.

        Primary criterion (statistically grounded):
            Bootstrap 95 % percentile CIs on the median path length per window.
            Optimal = smallest window whose CI overlaps the best window's CI.

        Secondary criterion (heuristic):
            η = success_rate / mean_length; window maximising η.
        """
        rng = np.random.default_rng(random_state)
        window_sizes_list: list[int] = []
        boot_median_means: list[float] = []
        ci_lowers: list[float] = []
        ci_uppers: list[float] = []
        success_rates: list[float] = []
        mean_lengths: list[float] = []
        n_successes: list[int] = []
        n_totals: list[int] = []

        for ws, group in sorted(df.groupby("window_size")):
            lengths = group["length"].values
            boot_medians = np.array([
                np.median(rng.choice(lengths, size=len(lengths), replace=True))
                for _ in range(n_bootstrap)
            ])
            window_sizes_list.append(int(ws))
            boot_median_means.append(float(np.mean(boot_medians)))
            ci_lowers.append(float(np.percentile(boot_medians, 2.5)))
            ci_uppers.append(float(np.percentile(boot_medians, 97.5)))
            success_rates.append(float(group["is_pred_path_valid"].mean()) if "is_pred_path_valid" in group.columns else 1.0)
            mean_lengths.append(float(lengths.mean()))
            n_successes.append(int(group["is_pred_path_valid"].sum()) if "is_pred_path_valid" in group.columns else int(len(group)))
            n_totals.append(int(len(group)))

        best_idx = int(np.argmin(boot_median_means))
        primary_optimal_idx = best_idx
        for i, (cl, cu) in enumerate(zip(ci_lowers, ci_uppers)):
            if cu >= ci_lowers[best_idx] and cl <= ci_uppers[best_idx]:
                primary_optimal_idx = i
                break

        # η = (relative_delta_rate) / (relative_delta_length)
        # relative_delta_rate = (rate[w] - rate[w-1]) / rate[w-1]
        # relative_delta_length = (length[w] - length[w-1]) / length[w-1]
        etas: list[float] = []
        for i in range(len(mean_lengths)):
            if i == 0:
                # No previous window for window 0
                etas.append(float("-inf"))
            else:
                delta_rate = success_rates[i] - success_rates[i - 1]
                delta_length = mean_lengths[i] - mean_lengths[i - 1]
                rel_delta_rate = delta_rate / (success_rates[i - 1] + 1e-10)
                rel_delta_length = delta_length / (mean_lengths[i - 1] + 1e-10)
                if abs(rel_delta_length) < 1e-10:
                    etas.append(float("-inf"))
                else:
                    etas.append(rel_delta_rate / rel_delta_length)
        eta_optimal_idx = int(np.argmax(etas)) if any(e > float("-inf") for e in etas) else 0

        return OptimalWindowResult(
            algorithm=algorithm_name,
            optimal_window=window_sizes_list[primary_optimal_idx],
            bootstrap_median=boot_median_means[primary_optimal_idx],
            ci_lower=ci_lowers[primary_optimal_idx],
            ci_upper=ci_uppers[primary_optimal_idx],
            eta_optimal_window=window_sizes_list[eta_optimal_idx],
            eta_score=float(etas[eta_optimal_idx]),
            success_rate=success_rates[primary_optimal_idx],
            mean_length=mean_lengths[primary_optimal_idx],
            n_success=n_successes[primary_optimal_idx],
            n_total=n_totals[primary_optimal_idx],
        )

    # =========================================================================
    # Phase 5 — Spearman correlation
    # =========================================================================

    def run_spearman_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str,
        alpha: float | None = None,
    ) -> SpearmanResult:
        """
        Spearman rank correlation.

        Used in Phase 5 to test whether optimisation gain (δ) correlates
        with situation difficulty (ground-truth path length as proxy).
        """
        alpha = alpha or self._config.alpha
        rho, p_value = stats.spearmanr(x, y)
        return SpearmanResult(
            label=label,
            rho=float(rho),
            p_value=float(p_value),
            n=len(x),
            is_significant=float(p_value) < alpha,
            alpha=alpha,
        )

    # =========================================================================
    # Phase 6 — Degradation from ground-truth
    # =========================================================================

    def compute_degradation(
        self,
        df_predicted: pd.DataFrame,
        df_ground_truth: pd.DataFrame,
        predicted_name: str,
        gt_name: str,
    ) -> DegradationResult:
        """
        Descriptive stats of δ(predicted − ground_truth) per situation.

        A positive mean/median δ means the predicted config yields longer
        paths than the ground-truth oracle configuration.
        """
        merged = pd.merge(
            df_ground_truth,
            df_predicted,
            on=["subject_id", "window_size"],
            suffixes=("_gt", "_pred"),
        )
        x_gt = merged["length_gt"].values
        x_pred = merged["length_pred"].values
        delta = (x_pred - x_gt) / ((x_pred + x_gt) / 2 + 1e-10)
        return DegradationResult(
            predicted_config=predicted_name,
            ground_truth_config=gt_name,
            stats=self.compute_descriptive_stats(delta),
        )

    # =========================================================================
    # McNemar (supporting, used in Phase 2 & 5)
    # =========================================================================

    def run_mcnemar_test(
        self,
        success1: np.ndarray,
        success2: np.ndarray,
        comparison_name: str,
        alpha: float | None = None,
    ) -> McNemarResult:
        """McNemar's test with continuity correction. Returns raw p-value."""
        alpha = alpha or self._config.alpha
        b = int(np.sum(success1 & ~success2))
        c = int(np.sum(~success1 & success2))
        if b + c == 0:
            statistic, p_value = 0.0, 1.0
        else:
            statistic = float((abs(b - c) - 1) ** 2 / (b + c))
            p_value = float(1 - stats.chi2.cdf(statistic, df=1))
        return McNemarResult(
            comparison=comparison_name,
            statistic=statistic,
            p_value=p_value,
            p_value_corrected=p_value,
            is_significant=p_value < alpha,
            discordant_pairs=b + c,
            alpha=alpha,
            correction_method="none",
        )

    # =========================================================================
    # Supporting tests
    # =========================================================================

    def run_levene_test(
        self,
        samples: list[np.ndarray],
        comparison_name: str,
        alpha: float | None = None,
    ) -> LeveneResult:
        """Levene's test for equality of variances across groups."""
        alpha = alpha or self._config.alpha
        statistic, p_value = stats.levene(*samples)
        return LeveneResult(
            comparison=comparison_name,
            statistic=float(statistic),
            p_value=float(p_value),
            is_homogeneous=float(p_value) > alpha,
            alpha=alpha,
        )

    def run_anova_rm(
        self,
        data_dict: dict[str, np.ndarray],
        factor_name: str,
        alpha: float | None = None,
    ) -> AnovaResult:
        """One-way RM-ANOVA — parametric fallback; use only after normality confirmed."""
        alpha = alpha or self._config.alpha
        data_matrix = np.column_stack(list(data_dict.values()))
        n_subjects, k_conditions = data_matrix.shape
        grand_mean = np.mean(data_matrix)
        subject_means = np.mean(data_matrix, axis=1, keepdims=True)
        condition_means = np.mean(data_matrix, axis=0)
        ss_between = n_subjects * np.sum((condition_means - grand_mean) ** 2)
        ss_within = np.sum((data_matrix - subject_means) ** 2)
        ss_subjects = k_conditions * np.sum((subject_means.flatten() - grand_mean) ** 2)
        ss_error = ss_within - ss_subjects
        df_between = k_conditions - 1
        df_error = (n_subjects - 1) * (k_conditions - 1)
        f_stat = (ss_between / df_between) / (ss_error / (df_error + 1e-10) + 1e-10)
        p_value = float(1 - stats.f.cdf(f_stat, df_between, df_error))
        eta_squared = float(ss_between / (ss_between + ss_error + 1e-10))
        return AnovaResult(
            factor=factor_name,
            f_statistic=float(f_stat),
            p_value=p_value,
            degrees_of_freedom=(df_between, df_error),
            effect_size_eta_squared=eta_squared,
            is_significant=p_value < alpha,
            alpha=alpha,
        )

    def compute_success_rates(
        self,
        dfs: dict[str, pd.DataFrame],
    ) -> dict[str, dict[int, float]]:
        """Success rate by window size for each configuration."""
        return {
            name: df.groupby("window_size")["success"].mean().to_dict()
            for name, df in dfs.items()
        }

    def compute_pred_valid_rates(
        self,
        dfs: dict[str, pd.DataFrame],
    ) -> dict[str, dict[int, float]]:
        pred_valid_names = self._get_pred_valid_rate_names()
        return {
            name: df.groupby("window_size")["is_pred_path_valid"].mean().to_dict()
            for name, df in dfs.items()
            if "is_pred_path_valid" in df.columns and name in pred_valid_names
        }

    # =========================================================================
    # Correction helpers (in-place, operate on lists of result objects)
    # =========================================================================

    @staticmethod
    def _apply_holm_correction(
        results: list[Any],
        family_size: int | None = None,
        alpha_nominal: float = 0.05,
    ) -> None:
        """
        Holm step-down FWER correction in-place.

        Sorts results by ascending raw p_value, then for rank i (0-based)
        multiplies p by (family_size − i).  family_size defaults to
        len(results) but can be set larger when some tests were skipped.
        """
        if not results:
            return
        m = family_size if family_size is not None else len(results)
        results.sort(key=lambda r: r.p_value)
        for rank, result in enumerate(results):
            p_corr = min(result.p_value * (m - rank), 1.0)
            result.p_value_corrected = p_corr
            result.is_significant = p_corr < alpha_nominal
            result.correction_method = "holm"

    @staticmethod
    def _apply_fdr_correction(
        results: list[Any],
        alpha_nominal: float = 0.05,
    ) -> None:
        """
        Benjamini-Hochberg FDR correction in-place via statsmodels.

        Order of results is preserved; p_value_corrected and is_significant
        are overwritten.
        """
        if not results:
            return
        raw_p = np.array([r.p_value for r in results])
        _, p_corrected, _, _ = multipletests(raw_p, alpha=alpha_nominal, method="fdr_bh")
        for result, p_corr in zip(results, p_corrected):
            result.p_value_corrected = float(p_corr)
            result.is_significant = float(p_corr) < alpha_nominal
            result.correction_method = "fdr_bh"

    # =========================================================================
    # Full pipeline
    # =========================================================================

    def run_full_analysis(
        self,
        preprocessed_data: dict[str, pd.DataFrame],
        masking_pairs: list[tuple[str, str]] | None = None,
        prediction_model_groups: list[dict] | None = None,
        optimisation_pairs: list[tuple[str, str, int]] | None = None,
        degradation_pairs: list[tuple[str, str]] | None = None,
    ) -> ReportResults:
        """
        Orchestrate the full statistical pipeline (Phases 1–6 + supporting).

        Args:
            preprocessed_data: Config name → preprocessed DataFrame.
            masking_pairs: Phase 2 — list of (name_no_mask, name_with_mask) per algorithm.
                Defaults to all pairwise combinations.
            prediction_model_groups: Phase 3 — list of dicts, each with:
                  'factor': str label (e.g. "GT vs LSTM vs GRU (base, ws=3)")
                  'data':   dict[condition_label, np.ndarray of per-situation lengths]
            optimisation_pairs: Phase 5 — list of (base_name, opt_name, window_size).
                Pass window_size = -1 to use all window sizes pooled.
            degradation_pairs: Phase 6 — list of (predicted_name, gt_name).

        Returns:
            Fully populated ReportResults.
        """
        alpha = self._config.alpha
        names = list(preprocessed_data.keys())

        # --- Supporting: descriptive stats + success rates ---
        for name, df in preprocessed_data.items():
            self._results.add_descriptive_stats(
                name, self.compute_descriptive_stats(df["length"].values)
            )
        self._results.success_rates = self.compute_success_rates(preprocessed_data)
        self._results.pred_valid_rates = self.compute_pred_valid_rates(preprocessed_data)

        # -------------------------------------------------------------------
        # Phase 1 — diagnostics
        # -------------------------------------------------------------------
        ref_pairs = masking_pairs or list(it.combinations(names, 2))
        self.run_phase1_diagnostics(preprocessed_data, ref_pairs)

        # -------------------------------------------------------------------
        # Phase 2 — masking effect (Holm within family)
        # -------------------------------------------------------------------
        pairs_p2 = masking_pairs or list(it.combinations(names, 2))
        raw_pw_p2: list[PairwiseTestResult] = []
        raw_mc_p2: list[McNemarResult] = []
        merged_p2: dict[tuple[str, str], pd.DataFrame] = {}

        for name1, name2 in pairs_p2:
            if name1 not in preprocessed_data or name2 not in preprocessed_data:
                continue
            merged = pd.merge(
                preprocessed_data[name1],
                preprocessed_data[name2],
                on=["subject_id", "window_size"],
                suffixes=("_1", "_2"),
            )
            if len(merged) == 0:
                continue
            merged_p2[(name1, name2)] = merged
            label = f"{name1} vs {name2}"
            raw_pw_p2.append(self.run_pairwise_test(
                merged["length_1"].values,
                merged["length_2"].values,
                label,
            ))
            raw_mc_p2.append(self.run_mcnemar_test(
                merged["success_1"].values.astype(bool),
                merged["success_2"].values.astype(bool),
                label,
            ))

        self._apply_holm_correction(raw_pw_p2, family_size=len(pairs_p2), alpha_nominal=alpha)
        self._apply_holm_correction(raw_mc_p2, family_size=len(pairs_p2), alpha_nominal=alpha)
        for r in raw_pw_p2:
            self._results.add_pairwise_result(r)
        for r in raw_mc_p2:
            self._results.add_mcnemar_result(r)

        # Phase 2 — McNemar on is_pred_path_valid (only for pred_valid_names)
        pred_valid_names = self._get_pred_valid_rate_names()
        raw_mc_pred_p2: list[McNemarResult] = []
        for (name1, name2), merged in merged_p2.items():
            if name1 not in pred_valid_names and name2 not in pred_valid_names:
                continue
            if "is_pred_path_valid_1" not in merged.columns:
                continue
            label = f"{name1} vs {name2}"
            raw_mc_pred_p2.append(self.run_mcnemar_test(
                merged["is_pred_path_valid_1"].values.astype(bool),
                merged["is_pred_path_valid_2"].values.astype(bool),
                label,
            ))
        self._apply_holm_correction(raw_mc_pred_p2, alpha_nominal=alpha)
        for r in raw_mc_pred_p2:
            self._results.add_mcnemar_pred_valid_result(r)

        # Phase 2 interaction: Mann-Whitney on δ distributions between algo groups
        # Requires exactly 2 masking pairs (one per algorithm variant)
        if len(pairs_p2) == 2 and all(p in merged_p2 for p in pairs_p2):
            def _delta_from_merged(m: pd.DataFrame) -> np.ndarray:
                x1 = m["length_1"].values
                x2 = m["length_2"].values
                return (x1 - x2) / ((x1 + x2) / 2 + 1e-10)

            delta_a = _delta_from_merged(merged_p2[pairs_p2[0]])
            delta_b = _delta_from_merged(merged_p2[pairs_p2[1]])
            mw = self.run_mann_whitney_test(
                delta_a, delta_b,
                comparison_name=(
                    f"Masking interaction: "
                    f"({pairs_p2[0][0]} vs {pairs_p2[0][1]}) / "
                    f"({pairs_p2[1][0]} vs {pairs_p2[1][1]})"
                ),
            )
            self._results.add_masking_interaction_result(mw)

        # -------------------------------------------------------------------
        # Phase 3 — Friedman omnibus + Conover post-hoc (BH-FDR across battery)
        # -------------------------------------------------------------------
        friedman_raw: list[FriedmanResult] = []
        group_specs_by_factor: dict[str, dict] = {}

        if prediction_model_groups:
            for group_spec in prediction_model_groups:
                fr = self.run_friedman_test(group_spec["data"], group_spec["factor"])
                friedman_raw.append(fr)
                group_specs_by_factor[group_spec["factor"]] = group_spec

        self._apply_fdr_correction(friedman_raw, alpha_nominal=alpha)

        for fr in friedman_raw:
            if fr.is_significant and fr.factor in group_specs_by_factor:
                self.run_posthoc_conover(
                    group_specs_by_factor[fr.factor]["data"], fr, p_adjust="holm"
                )
            self._results.add_friedman_result(fr)

        # -------------------------------------------------------------------
        # Phase 4 — Trend test + optimal window per configuration
        # -------------------------------------------------------------------
        pred_valid_names = self._get_pred_valid_rate_names()
        for name, df in preprocessed_data.items():
            jt = self.run_jonckheere_terpstra(
                df, value_col="length", group_col="window_size",
                factor_name=name, alternative="two-sided",
            )
            self._results.add_trend_result(jt)
            if name in pred_valid_names:
                opt = self.compute_optimal_window(df, name)
                self._results.add_optimal_window_result(opt)

        # -------------------------------------------------------------------
        # Phase 5 — Aposteriori optimisation (BH-FDR) + Spearman
        # -------------------------------------------------------------------
        pred_valid_names = self._get_pred_valid_rate_names()
        if optimisation_pairs:
            raw_pw_p5: list[PairwiseTestResult] = []
            raw_mc_p5: list[McNemarResult] = []
            raw_mc_pred_p5: list[McNemarResult] = []
            raw_sp_p5: list[SpearmanResult] = []
            merged_p5_list: list[tuple[str, str, pd.DataFrame]] = []

            for base_name, opt_name, ws in optimisation_pairs:
                if base_name not in preprocessed_data or opt_name not in preprocessed_data:
                    continue
                df_base = preprocessed_data[base_name]
                df_opt = preprocessed_data[opt_name]
                if ws >= 0:
                    df_base = df_base[df_base["window_size"] == ws]
                    df_opt = df_opt[df_opt["window_size"] == ws]
                merged = pd.merge(
                    df_base, df_opt,
                    on=["subject_id", "window_size"],
                    suffixes=("_base", "_opt"),
                )
                if len(merged) == 0:
                    continue
                merged_p5_list.append((base_name, opt_name, merged))
                label = f"{base_name} vs {opt_name}" + (f" (ws={ws})" if ws >= 0 else "")
                raw_pw_p5.append(self.run_pairwise_test(
                    merged["length_base"].values,
                    merged["length_opt"].values,
                    label,
                ))
                raw_mc_p5.append(self.run_mcnemar_test(
                    merged["success_base"].values.astype(bool),
                    merged["success_opt"].values.astype(bool),
                    label,
                ))
                if len(merged) >= 3:
                    x1 = merged["length_base"].values
                    x2 = merged["length_opt"].values
                    gain = (x1 - x2) / ((x1 + x2) / 2 + 1e-10)
                    raw_sp_p5.append(self.run_spearman_correlation(
                        x1, gain,
                        label=f"Gain vs difficulty: {label}",
                    ))

            # McNemar on is_pred_path_valid (only for pred_valid_names)
            for base_name, opt_name, merged in merged_p5_list:
                if base_name not in pred_valid_names and opt_name not in pred_valid_names:
                    continue
                if "is_pred_path_valid_base" not in merged.columns:
                    continue
                label = f"{base_name} vs {opt_name}"
                raw_mc_pred_p5.append(self.run_mcnemar_test(
                    merged["is_pred_path_valid_base"].values.astype(bool),
                    merged["is_pred_path_valid_opt"].values.astype(bool),
                    label,
                ))

            self._apply_fdr_correction(raw_pw_p5, alpha_nominal=alpha)
            self._apply_fdr_correction(raw_mc_p5, alpha_nominal=alpha)
            self._apply_fdr_correction(raw_mc_pred_p5, alpha_nominal=alpha)
            self._apply_fdr_correction(raw_sp_p5, alpha_nominal=alpha)
            for r in raw_pw_p5:
                self._results.add_pairwise_result(r)
            for r in raw_mc_p5:
                self._results.add_mcnemar_result(r)
            for r in raw_mc_pred_p5:
                self._results.add_mcnemar_pred_valid_result(r)
            for r in raw_sp_p5:
                self._results.add_spearman_result(r)

        # -------------------------------------------------------------------
        # Phase 6 — Degradation from ground-truth
        # -------------------------------------------------------------------
        if degradation_pairs:
            for pred_name, gt_name in degradation_pairs:
                if pred_name not in preprocessed_data or gt_name not in preprocessed_data:
                    continue
                self._results.add_degradation_result(
                    self.compute_degradation(
                        preprocessed_data[pred_name],
                        preprocessed_data[gt_name],
                        pred_name,
                        gt_name,
                    )
                )

        # --- Supporting: Levene per window size ---
        all_ws: set[int] = set()
        for df in preprocessed_data.values():
            all_ws.update(df["window_size"].unique())
        for ws in sorted(all_ws):
            samples = [
                df[df["window_size"] == ws]["length"].values
                for df in preprocessed_data.values()
                if len(df[df["window_size"] == ws]) > 0
            ]
            if len(samples) >= 2:
                self._results.add_levene_result(self.run_levene_test(samples, f"Window {ws}"))

        return self._results
