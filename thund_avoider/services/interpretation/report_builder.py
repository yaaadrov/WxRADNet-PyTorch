from datetime import datetime
from pathlib import Path

import pandas as pd

from thund_avoider.schemas.interpretation.config import ReportConfig
from thund_avoider.schemas.interpretation.results import (
    PlotRegistry,
    ReportResults,
)


class ReportBuilder:
    """
    Builds a Markdown statistical report from ReportResults.

    Each section corresponds to one analysis phase.  Sections are only
    included when their underlying results list is non-empty, so a partial
    analysis still produces a valid report.
    """

    def __init__(self, config: ReportConfig) -> None:
        self._config = config

    # =========================================================================
    # Internal formatting helpers
    # =========================================================================

    @staticmethod
    def _sig(flag: bool) -> str:
        return "✓" if flag else "—"

    @staticmethod
    def _pstr(p: float | None, raw: float | None = None) -> str:
        """Format a corrected p-value; fall back to raw if corrected is None."""
        val = p if p is not None else raw
        if val is None:
            return "—"
        return f"{val:.4f}"

    # =========================================================================
    # Header
    # =========================================================================

    def _build_header(self) -> str:
        return (
            "# Статистический анализ результатов обхода гроз\n\n"
            "Данный отчёт содержит результаты статистического сравнения алгоритмов "
            "обхода грозовых очагов с использованием различных конфигураций.\n\n---\n\n"
        )

    # =========================================================================
    # Data overview
    # =========================================================================

    def _build_data_overview(
        self,
        dfs: dict[str, pd.DataFrame],
        original_counts: dict[str, int],
    ) -> str:
        lines = ["## Обзор данных\n"]
        lines.append("| Алгоритм | Записей (исходно) | Записей (после фильтрации) |")
        lines.append("|----------|-------------------|----------------------------|")
        for name, df in dfs.items():
            lines.append(f"| {name} | {original_counts.get(name, 'N/A')} | {len(df)} |")
        lines += [
            "",
            f"**Доверительный интервал:** {self._config.confidence_level:.0%}",
            f"**Уровень значимости α:** {self._config.alpha}",
            "",
        ]
        return "\n".join(lines)

    # =========================================================================
    # Phase 1 — Diagnostics
    # =========================================================================

    def _build_diagnostics(self, results: ReportResults) -> str:
        diag = results.diagnostics
        lines = ["## Фаза 1 — Диагностика распределений\n"]
        lines.append(
            f"**Выбранная тест-семья:** "
            f"{'непараметрическая (Вилкоксон / Фридман)' if diag.use_nonparametric else 'параметрическая (t-тест / RM-ANOVA)'}  "
            f"({'хотя бы одна конфигурация не прошла тест нормальности или ненулевое нулевое накопление' if diag.use_nonparametric else 'все конфигурации нормальны, нулевое накопление < 5 %'})\n"
        )

        if diag.normality_by_config:
            lines.append("### Тест Шапиро–Уилка (остатки внутри размера окна)\n")
            lines.append("| Конфигурация | W | p | Нормальность |")
            lines.append("|-------------|---|---|--------------|")
            for name, r in diag.normality_by_config.items():
                lines.append(
                    f"| {name} | {r.statistic:.4f} | {r.p_value:.4f} | {self._sig(r.is_normal)} |"
                )
            lines.append("")

        if diag.zero_inflation_by_config:
            lines.append("### Нулевое накопление δ (доля нулевых разностей)\n")
            lines.append("| Пара | Доля нулей |")
            lines.append("|------|-----------|")
            for pair, zi in diag.zero_inflation_by_config.items():
                warn = " ⚠" if zi > 0.05 else ""
                lines.append(f"| {pair} | {zi:.2%}{warn} |")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Descriptive statistics
    # =========================================================================

    def _build_descriptive_stats(self, results: ReportResults) -> str:
        if not results.descriptive_stats:
            return ""
        lines = ["## Описательная статистика\n"]
        lines.append("| Алгоритм | Среднее | Медиана | Ст. откл. | N | 95% ДИ |")
        lines.append("|----------|---------|---------|-----------|---|--------|")
        for name, s in results.descriptive_stats.items():
            lines.append(f"| {name} | {s.to_markdown_table()}")
        lines.append("")
        return "\n".join(lines)

    def _build_success_rates(self, results: ReportResults) -> str:
        if not results.success_rates:
            return ""
        lines = ["## Доли успешных маршрутов\n"]
        lines.append("| Алгоритм | Размер окна | Доля успехов |")
        lines.append("|----------|-------------|--------------|")
        for name, rates in results.success_rates.items():
            for ws, rate in sorted(rates.items()):
                lines.append(f"| {name} | {ws} | {rate:.2%} |")
        lines.append("")
        return "\n".join(lines)

    def _build_pred_path_valid_rates(self, results: ReportResults) -> str:
        if not results.pred_valid_rates:
            return ""
        lines = ["## Доли успешно валидированных маршрутов\n"]
        lines.append("| Алгоритм | Размер окна | Доля успехов валидации |")
        lines.append("|----------|-------------|------------------------|")
        for name, rates in results.pred_valid_rates.items():
            for ws, rate in sorted(rates.items()):
                lines.append(f"| {name} | {ws} | {rate:.2%} |")
        lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # Phase 2 — Masking effect
    # =========================================================================

    def _build_pairwise_tests(self, results: ReportResults) -> str:
        if not results.pairwise_results:
            return ""
        lines = ["## Фаза 2 & 5 — Попарные сравнения длин маршрутов (δ)\n"]
        lines.append(
            "Тест применяется к δ = (L₁ − L₂) / mid.  "
            "Выбор теста определяется флагом Phase 1.  "
            "Метод коррекции указан в столбце «Коррекция».\n"
        )
        lines.append(
            "| Сравнение | Тест | Стат. | p (raw) | p (скорр.) | Знач. | "
            "Эффект | Метрика | Коррекция |"
        )
        lines.append(
            "|-----------|------|-------|---------|------------|-------|"
            "--------|---------|-----------|"
        )
        for r in results.pairwise_results:
            metric = "Cohen's d" if "t-тест" in r.test_name else "r (ранг-бисер.)"
            lines.append(
                f"| {r.comparison} | {r.test_name} | {r.statistic:.4f} "
                f"| {r.p_value:.4f} | {self._pstr(r.p_value_corrected)} "
                f"| {self._sig(r.is_significant)} "
                f"| {r.effect_size:.3f} ({r.effect_size_interpretation}) "
                f"| {metric} | {r.correction_method} |"
            )
        lines += [
            "",
            "**Cohen's d:** ничтожный < 0.2, малый 0.2–0.5, средний 0.5–0.8, большой > 0.8  ",
            "**r:** ничтожный < 0.1, малый 0.1–0.3, средний 0.3–0.5, большой > 0.5",
            "",
        ]
        return "\n".join(lines)

    def _build_mcnemar_tests(self, results: ReportResults) -> str:
        if not results.mcnemar_results:
            return ""
        lines = ["## Фаза 2 & 5 — Сравнение долей успехов (тест МакНемара)\n"]
        lines.append(
            "| Сравнение | χ² | p (raw) | p (скорр.) | Знач. | Дискорд. пары | Коррекция |"
        )
        lines.append(
            "|-----------|-----|---------|------------|-------|---------------|-----------|"
        )
        for r in results.mcnemar_results:
            lines.append(
                f"| {r.comparison} | {r.statistic:.4f} | {r.p_value:.4f} "
                f"| {self._pstr(r.p_value_corrected)} | {self._sig(r.is_significant)} "
                f"| {r.discordant_pairs} | {r.correction_method} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_mcnemar_pred_valid_tests(self, results: ReportResults) -> str:
        if not results.mcnemar_pred_valid_results:
            return ""
        lines = ["## Фаза 2 & 5 — Сравнение долей валидации предсказаний (тест МакНемара)\n"]
        lines.append(
            "Сравнение по колонке is_pred_path_valid (только для конфигураций "
            "с флагом plot_pred_valid_rate=True).\n"
        )
        lines.append(
            "| Сравнение | χ² | p (raw) | p (скорр.) | Знач. | Дискорд. пары | Коррекция |"
        )
        lines.append(
            "|-----------|-----|---------|------------|-------|---------------|-----------|"
        )
        for r in results.mcnemar_pred_valid_results:
            lines.append(
                f"| {r.comparison} | {r.statistic:.4f} | {r.p_value:.4f} "
                f"| {self._pstr(r.p_value_corrected)} | {self._sig(r.is_significant)} "
                f"| {r.discordant_pairs} | {r.correction_method} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_masking_interaction(self, results: ReportResults) -> str:
        if not results.masking_interaction_results:
            return ""
        lines = [
            "## Фаза 2 — Взаимодействие: маскировка × алгоритм (Манн–Уитни)\n",
            "Тест проверяет, различается ли распределение δ(маскировка) "
            "между двумя вариантами алгоритма.\n",
        ]
        lines.append("| Сравнение | U | p | Знач. | r (эффект) |")
        lines.append("|-----------|---|---|-------|------------|")
        for r in results.masking_interaction_results:
            lines.append(
                f"| {r.comparison} | {r.statistic:.2f} | {r.p_value:.4f} "
                f"| {self._sig(r.is_significant)} "
                f"| {r.effect_size_r:.3f} ({r.effect_size_interpretation}) |"
            )
        lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # Phase 3 — Friedman + post-hoc
    # =========================================================================

    def _build_friedman_section(self, results: ReportResults) -> str:
        if not results.friedman_results:
            return ""
        lines = ["## Фаза 3 — Сравнение моделей предсказания (тест Фридмана)\n"]
        lines.append(
            "p-значения скорректированы методом BH-FDR по всей батарее тестов.\n"
        )
        lines.append(
            "| Фактор | χ² | p (raw) | p (BH) | Знач. | W (Кендалл) | "
            "Медианы по условиям |"
        )
        lines.append(
            "|--------|-----|---------|--------|-------|-------------|"
            "--------------------|"
        )
        for r in results.friedman_results:
            medians_str = ", ".join(
                f"{c}: {v:.2f}" for c, v in r.condition_medians.items()
            )
            lines.append(
                f"| {r.factor} | {r.statistic:.4f} | {r.p_value:.4f} "
                f"| {self._pstr(r.p_value_corrected)} | {self._sig(r.is_significant)} "
                f"| {r.kendalls_w:.3f} | {medians_str} |"
            )
        lines.append("")

        # Post-hoc Conover tables (only for significant results)
        sig_with_posthoc = [
            r for r in results.friedman_results
            if r.is_significant and r.posthoc_conover
        ]
        if sig_with_posthoc:
            lines.append("### Post-hoc: Конновер–Иман (только значимые тесты Фридмана)\n")
            for r in sig_with_posthoc:
                lines.append(f"**{r.factor}**\n")
                lines.append("| Пара | p (скорр., Холм) | Знач. |")
                lines.append("|------|-----------------|-------|")
                for pair, p in sorted(r.posthoc_conover.items()):
                    sig = self._sig(p < r.alpha)
                    lines.append(f"| {pair} | {p:.4f} | {sig} |")
                lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Phase 4 — Trend test + optimal window
    # =========================================================================

    def _build_trend_section(self, results: ReportResults) -> str:
        if not results.trend_results:
            return ""
        lines = ["## Фаза 4 — Тренд по размеру окна (Джонкхира–Терпстра)\n"]
        lines.append(
            "Тест проверяет монотонный тренд медианной длины маршрута "
            "по упорядоченным размерам окна (0 → 6).\n"
        )
        lines.append("| Конфигурация | Статистика | p | Знач. | Направление тренда |")
        lines.append("|-------------|------------|---|-------|--------------------|")
        for r in results.trend_results:
            lines.append(
                f"| {r.factor} | {r.statistic:.4f} | {r.p_value:.4f} "
                f"| {self._sig(r.is_significant)} | {r.trend_direction} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_optimal_windows(self, results: ReportResults) -> str:
        if not results.optimal_window_results:
            return ""
        lines = ["## Фаза 4 — Оптимальные размеры окна\n"]
        lines.append(
            "**Основной критерий:** наименьший размер окна, 95 % bootstrap-ДИ которого "
            "перекрывается с ДИ окна с минимальной медианой длины.\n"
        )
        lines.append(
            "**Вспомогательный критерий (η):** окно, максимизирующее "
            "η = доля успехов / средняя длина.\n"
        )
        lines.append(
            "| Конфигурация | Окно (bootstrap) | Медиана | 95% ДИ | "
            "Окно (η) | η | Доля успехов | Ср. длина | N |"
        )
        lines.append(
            "|-------------|------------------|---------|--------|"
            "---------|---|--------------|-----------|---|"
        )
        for r in sorted(results.optimal_window_results, key=lambda x: x.eta_score, reverse=True):
            lines.append(
                f"| {r.algorithm} | {r.optimal_window} | {r.bootstrap_median:.2f} "
                f"| [{r.ci_lower:.2f}, {r.ci_upper:.2f}] "
                f"| {r.eta_optimal_window} | {r.eta_score:.6f} "
                f"| {r.success_rate:.2%} | {r.mean_length:.0f} "
                f"| {r.n_success}/{r.n_total} |"
            )
        lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # Phase 5 — Spearman
    # =========================================================================

    def _build_spearman_section(self, results: ReportResults) -> str:
        if not results.spearman_results:
            return ""
        lines = [
            "## Фаза 5 — Корреляция прироста оптимизации со сложностью ситуации\n",
            "Корреляция Спирмена между δ(base − opt) и сложностью ситуации "
            "(длина маршрута базового алгоритма как прокси).  "
            "p скорректированы BH-FDR.\n",
        ]
        lines.append("| Сравнение | ρ | p (raw) | p (BH) | N | Знач. |")
        lines.append("|-----------|---|---------|--------|---|-------|")
        for r in results.spearman_results:
            lines.append(
                f"| {r.label} | {r.rho:.3f} | {r.p_value:.4f} "
                f"| {self._pstr(r.p_value_corrected)} | {r.n} "
                f"| {self._sig(r.is_significant)} |"
            )
        lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # Phase 6 — Degradation
    # =========================================================================

    def _build_degradation_section(self, results: ReportResults) -> str:
        if not results.degradation_results:
            return ""
        lines = [
            "## Фаза 6 — Деградация относительно эталона (ground-truth)\n",
            "δ = (длина_предсказанная − длина_GT) / mid.  "
            "Положительные значения означают, что предсказанная конфигурация "
            "даёт более длинные маршруты.\n",
        ]
        lines.append(
            "| Предсказанная конф. | GT конф. | Среднее δ | Медиана δ | "
            "Ст. откл. | 95% ДИ | N |"
        )
        lines.append(
            "|---------------------|---------|-----------|-----------|"
            "-----------|--------|---|"
        )
        for r in results.degradation_results:
            s = r.stats
            lines.append(
                f"| {r.predicted_config} | {r.ground_truth_config} "
                f"| {s.mean:.4f} | {s.median:.4f} | {s.std:.4f} "
                f"| [{s.ci_lower:.4f}, {s.ci_upper:.4f}] | {s.n} |"
            )
        lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # Levene
    # =========================================================================

    def _build_levene_section(self, results: ReportResults) -> str:
        if not results.levene_results:
            return ""
        lines = ["## Однородность дисперсий (критерий Левена)\n"]
        lines.append("| Размер окна | Статистика | p | Однородность |")
        lines.append("|-------------|------------|---|--------------|")
        for r in results.levene_results:
            lines.append(
                f"| {r.comparison} | {r.statistic:.4f} | {r.p_value:.4f} "
                f"| {self._sig(r.is_homogeneous)} |"
            )
        lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # Plots
    # =========================================================================

    def _build_plots_section(self, plot_registry: PlotRegistry) -> str:
        plot_descriptions = {
            "boxplot": "Распределение длин маршрутов по размерам окна",
            "lineplot_success": "Зависимость доли успехов от размера окна",
            "optimal_window": "Оптимальные размеры окна по метрике η",
        }
        items = plot_registry.items()
        if not items:
            return ""
        lines = ["## Визуализации\n"]
        for plot_type, path in items:
            description = plot_descriptions.get(plot_type, plot_type)
            relative_path = f"plots/{path.name}"
            lines.append(f"### {description}\n")
            lines.append(f"![{description}]({relative_path})\n")
        return "\n".join(lines)

    # =========================================================================
    # Conclusions
    # =========================================================================

    def _build_conclusions(self, results: ReportResults) -> str:
        lines = ["## Выводы\n"]
        point = 1

        # Phase 1
        diag = results.diagnostics
        nonnormal = [k for k, v in diag.normality_by_config.items() if not v.is_normal]
        if nonnormal:
            lines.append(
                f"{point}. **Нормальность:** конфигурации с ненормальными остатками — "
                f"{', '.join(nonnormal)}.  Применялась непараметрическая тест-семья.\n"
            )
        else:
            lines.append(
                f"{point}. **Нормальность:** все конфигурации прошли тест Шапиро–Уилка. "
                f"Применялась параметрическая тест-семья.\n"
            )
        point += 1

        # Phase 2 — pairwise length
        sig_pw = [r for r in results.pairwise_results if r.is_significant]
        if sig_pw:
            lines.append(f"{point}. **Значимые различия в δ длины маршрутов:**\n")
            for r in sig_pw:
                metric = "Cohen's d" if "t-тест" in r.test_name else "r"
                lines.append(
                    f"   - **{r.comparison}**: {r.test_name}, "
                    f"p ({r.correction_method}) = {self._pstr(r.p_value_corrected)}, "
                    f"{metric} = {r.effect_size:.3f} ({r.effect_size_interpretation}).\n"
                )
        else:
            lines.append(
                f"{point}. **Длины маршрутов:** статистически значимых различий в δ "
                f"не обнаружено ({len(results.pairwise_results)} сравнений).\n"
            )
        point += 1

        # Phase 2 — McNemar
        sig_mc = [r for r in results.mcnemar_results if r.is_significant]
        if sig_mc:
            lines.append(f"{point}. **Значимые различия в долях успехов:**\n")
            for r in sig_mc:
                lines.append(
                    f"   - **{r.comparison}**: χ² = {r.statistic:.4f}, "
                    f"p ({r.correction_method}) = {self._pstr(r.p_value_corrected)}, "
                    f"дискорд. пар = {r.discordant_pairs}.\n"
                )
        else:
            lines.append(
                f"{point}. **Доли успехов:** значимых различий не обнаружено.\n"
            )
        point += 1

        # Phase 2 — interaction
        if results.masking_interaction_results:
            mw = results.masking_interaction_results[0]
            if mw.is_significant:
                lines.append(
                    f"{point}. **Взаимодействие маскировка × алгоритм:** значимо "
                    f"(U = {mw.statistic:.2f}, p = {mw.p_value:.4f}, "
                    f"r = {mw.effect_size_r:.3f}).  "
                    f"Эффект маскировки различается между вариантами алгоритма.\n"
                )
            else:
                lines.append(
                    f"{point}. **Взаимодействие маскировка × алгоритм:** не значимо "
                    f"(p = {mw.p_value:.4f}).  Маскировка влияет одинаково на оба алгоритма.\n"
                )
            point += 1

        # Phase 3 — Friedman
        sig_fr = [r for r in results.friedman_results if r.is_significant]
        if sig_fr:
            lines.append(
                f"{point}. **Тест Фридмана — значимые факторы "
                f"({len(sig_fr)} из {len(results.friedman_results)}):**\n"
            )
            for r in sig_fr:
                posthoc_summary = ""
                if r.posthoc_conover:
                    sig_pairs = [p for p, v in r.posthoc_conover.items() if v < r.alpha]
                    if sig_pairs:
                        posthoc_summary = (
                            f"  Post-hoc значимые пары: {', '.join(sig_pairs)}."
                        )
                lines.append(
                    f"   - **{r.factor}**: χ²={r.statistic:.4f}, "
                    f"p(BH)={self._pstr(r.p_value_corrected)}, W={r.kendalls_w:.3f}."
                    f"{posthoc_summary}\n"
                )
        else:
            lines.append(
                f"{point}. **Тест Фридмана:** значимых различий между моделями "
                f"предсказания не обнаружено.\n"
            )
        point += 1

        # Phase 4 — trend
        sig_jt = [r for r in results.trend_results if r.is_significant]
        if sig_jt:
            lines.append(
                f"{point}. **Монотонный тренд по размеру окна (ДТ):** "
                f"значимо для {len(sig_jt)} конфигураций:\n"
            )
            for r in sig_jt:
                lines.append(
                    f"   - {r.factor}: p = {r.p_value:.4f}, "
                    f"направление — {r.trend_direction}.\n"
                )
        else:
            lines.append(
                f"{point}. **Тренд по размеру окна:** монотонного тренда не обнаружено.\n"
            )
        point += 1

        # Phase 4 — optimal window summary
        if results.optimal_window_results:
            lines.append(f"{point}. **Оптимальные размеры окна (bootstrap):**\n")
            for r in results.optimal_window_results:
                lines.append(
                    f"   - {r.algorithm}: окно = **{r.optimal_window}** "
                    f"(медиана = {r.bootstrap_median:.2f}, "
                    f"ДИ [{r.ci_lower:.2f}, {r.ci_upper:.2f}]); "
                    f"η-оптимум = {r.eta_optimal_window}.\n"
                )
            # Secondary: best by η
            best_eta = max(results.optimal_window_results, key=lambda x: x.eta_score)
            lines.append(
                f"   Лучший алгоритм по эвристике η: **{best_eta.algorithm}** "
                f"(η = {best_eta.eta_score:.6f}, окно = {best_eta.eta_optimal_window}).\n"
            )
            point += 1

        # Phase 6 — degradation
        if results.degradation_results:
            lines.append(
                f"{point}. **Деградация относительно GT:**\n"
            )
            for r in results.degradation_results:
                lines.append(
                    f"   - {r.predicted_config}: медиана δ = {r.stats.median:.4f} "
                    f"(среднее = {r.stats.mean:.4f}, "
                    f"ДИ [{r.stats.ci_lower:.4f}, {r.stats.ci_upper:.4f}]).\n"
                )
            point += 1

        lines += [
            "---\n",
            f"*Отчёт сгенерирован автоматически "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.*\n",
        ]
        return "\n".join(lines)

    # =========================================================================
    # Public entry point
    # =========================================================================

    def build_report(
        self,
        dfs: dict[str, pd.DataFrame],
        results: ReportResults,
        original_counts: dict[str, int],
        output_path: Path,
    ) -> Path:
        """
        Assemble and write the complete Markdown report.

        Args:
            dfs: Preprocessed DataFrames (for data overview counts).
            results: Fully populated ReportResults.
            original_counts: Record counts before outlier filtering.
            output_path: Destination .md file.

        Returns:
            Path to the written file.
        """
        sections = [
            self._build_header(),
            self._build_data_overview(dfs, original_counts),
            self._build_diagnostics(results),
            self._build_descriptive_stats(results),
            self._build_success_rates(results),
            self._build_pred_path_valid_rates(results),
            self._build_pairwise_tests(results),
            self._build_mcnemar_tests(results),
            self._build_mcnemar_pred_valid_tests(results),
            self._build_masking_interaction(results),
            self._build_friedman_section(results),
            self._build_trend_section(results),
            self._build_optimal_windows(results),
            self._build_spearman_section(results),
            self._build_degradation_section(results),
            self._build_levene_section(results),
            self._build_plots_section(results.plot_paths),
            self._build_conclusions(results),
        ]

        report_content = "\n".join(s for s in sections if s)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_content, encoding="utf-8")

        return output_path
