from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from thund_avoider.schemas.interpretation.config import ReportConfig
from thund_avoider.schemas.interpretation.results import ReportResults


class ReportBuilder:
    """
    Builds markdown reports from statistical analysis results.

    This class compiles all analysis results, plots, and automatic
    conclusions into a comprehensive markdown report.
    """

    def __init__(self, config: ReportConfig) -> None:
        self._config = config

    def _build_header(self) -> str:
        """Build report header."""
        return """# Статистический анализ результатов обхода гроз

Данный отчет содержит результаты статистического сравнения алгоритмов
обхода грозовых очагов с использованием различных конфигураций.

---

"""

    def _build_data_overview(
        self,
        dfs: dict[str, pd.DataFrame],
        original_counts: dict[str, int],
    ) -> str:
        """Build data overview section."""
        lines = ["## Обзор данных\n"]
        lines.append("### Загруженные датасеты\n")
        lines.append("| Алгоритм | Записей (исходно) | Записей (после фильтрации) |")
        lines.append("|----------|-------------------|----------------------------|")

        for name, df in dfs.items():
            original = original_counts.get(name, "N/A")
            filtered = len(df)
            lines.append(f"| {name} | {original} | {filtered} |")

        lines.append("")
        lines.append(f"**Доверительный интервал:** {self._config.confidence_level:.0%}")
        lines.append(f"**Уровень значимости α:** {self._config.alpha}")
        lines.append("")

        return "\n".join(lines)

    def _build_descriptive_stats(self, results: ReportResults) -> str:
        """Build descriptive statistics section."""
        lines = ["## Описательная статистика\n"]
        lines.append(
            "| Алгоритм | Среднее | Медиана | Ст. откл. | N | 95% ДИ |"
        )
        lines.append(
            "|----------|---------|---------|-----------|---|--------|"
        )

        for name, stats in results.descriptive_stats.items():
            lines.append(f"| {name} | {stats.to_markdown_table()}")

        lines.append("")
        return "\n".join(lines)

    def _build_success_rates(self, results: ReportResults) -> str:
        """Build success rates section."""
        lines = ["## Доли успешных маршрутов\n"]
        lines.append(
            "| Алгоритм | Размер окна | Доля успехов |"
        )
        lines.append(
            "|----------|-------------|--------------|"
        )

        for name, rates in results.success_rates.items():
            for ws, rate in sorted(rates.items()):
                lines.append(f"| {name} | {ws} | {rate:.2%} |")

        lines.append("")
        return "\n".join(lines)

    def _build_pairwise_tests(self, results: ReportResults) -> str:
        """Build pairwise tests section."""
        if not results.pairwise_results:
            return ""

        lines = ["## Попарные сравнения длин маршрутов\n"]
        lines.append(
            "| Сравнение | Тест | Статистика | p (скорр.) | Значимо | Размер эффекта |"
        )
        lines.append(
            "|-----------|------|------------|------------|---------|----------------|"
        )

        for result in results.pairwise_results:
            lines.append(f"{result.to_markdown_row()}")

        lines.append("")
        lines.append(
            "*p-значения скорректированы методом Бонферрони для множественных сравнений.*"
        )
        lines.append("")
        lines.append(
            "**Интерпретация размера эффекта (Cohen's d):** "
            "ничтожный (< 0.2), малый (0.2-0.5), средний (0.5-0.8), большой (> 0.8)"
        )
        lines.append("")

        return "\n".join(lines)

    def _build_mcnemar_tests(self, results: ReportResults) -> str:
        """Build McNemar tests section."""
        if not results.mcnemar_results:
            return ""

        lines = ["## Сравнение долей успешных маршрутов (тест МакНемара)\n"]
        lines.append(
            "| Сравнение | χ² | p (скорр.) | Значимо | Дискордантные пары |"
        )
        lines.append(
            "|-----------|-----|------------|---------|-------------------|"
        )

        for result in results.mcnemar_results:
            lines.append(f"{result.to_markdown_row()}")

        lines.append("")
        return "\n".join(lines)

    def _build_optimal_windows(self, results: ReportResults) -> str:
        """Build optimal windows section."""
        if not results.optimal_window_results:
            return ""

        lines = ["## Оптимальные размеры окна\n"]
        lines.append(
            "Оптимальный размер окна определяется по метрике η = доля успехов / средняя длина.\n"
        )
        lines.append(
            "| Алгоритм | Оптим. окно | η | Доля успехов | Средняя длина | N успехов |"
        )
        lines.append(
            "|----------|-------------|---|--------------|---------------|-----------|"
        )

        for result in sorted(
            results.optimal_window_results, key=lambda x: x.eta_score, reverse=True
        ):
            lines.append(f"{result.to_markdown_row()}")

        lines.append("")
        return "\n".join(lines)

    def _build_plots_section(self, plot_paths: dict[str, Path]) -> str:
        """Build plots section with embedded images."""
        lines = ["## Визуализации\n"]

        plot_descriptions = {
            "boxplot": "Распределение длин маршрутов обхода по размерам окна",
            "lineplot_success": "Зависимость доли успешных маршрутов от размера окна",
            "optimal_window": "Оптимальные размеры окна по метрике η",
            "dual_axis_marginal": "Сравнение алгоритмов: относительное изменение успехов vs длин",
        }

        for plot_type, path in plot_paths.items():
            if path.exists():
                relative_path = f"plots/{path.name}"
                description = plot_descriptions.get(plot_type, plot_type)
                lines.append(f"### {description}\n")
                lines.append(f"![{description}]({relative_path})\n")

        return "\n".join(lines)

    def _build_conclusions(self, results: ReportResults) -> str:
        """Build automatic conclusions section."""
        lines = ["## Выводы\n"]

        # Best algorithm by eta
        if results.optimal_window_results:
            best = max(results.optimal_window_results, key=lambda x: x.eta_score)
            lines.append(
                f"1. **Лучший алгоритм по метрике η:** {best.algorithm} "
                f"с оптимальным размером окна {best.optimal_window} "
                f"(η = {best.eta_score:.6f}).\n"
            )

        # Significant pairwise differences
        significant_pairwise = [
            r for r in results.pairwise_results if r.is_significant
        ]
        if significant_pairwise:
            lines.append("2. **Значимые различия в длинах маршрутов:**\n")
            for result in significant_pairwise:
                direction = "короче" if result.effect_size < 0 else "длиннее"
                lines.append(
                    f"   - {result.comparison}: {result.test_name} показал "
                    f"значимые различия (p = {result.p_value_corrected:.4f}), "
                    f"размер эффекта {result.effect_size_interpretation} "
                    f"({result.effect_size:.3f}).\n"
                )
        else:
            lines.append(
                "2. **Длины маршрутов:** Статистически значимых различий "
                "между алгоритмами не обнаружено.\n"
            )

        # Success rate differences
        significant_mcnemar = [r for r in results.mcnemar_results if r.is_significant]
        if significant_mcnemar:
            lines.append("3. **Значимые различия в долях успехов:**\n")
            for result in significant_mcnemar:
                lines.append(
                    f"   - {result.comparison}: тест МакНемара показал "
                    f"значимые различия (p = {result.p_value_corrected:.4f}).\n"
                )
        else:
            lines.append(
                "3. **Доли успехов:** Статистически значимых различий "
                "между алгоритмами не обнаружено.\n"
            )

        # Window size recommendations
        if results.optimal_window_results:
            lines.append("4. **Рекомендации по размеру окна:**\n")
            window_counts: dict[int, list[str]] = {}
            for result in results.optimal_window_results:
                ws = result.optimal_window
                if ws not in window_counts:
                    window_counts[ws] = []
                window_counts[ws].append(result.algorithm)

            most_common = max(window_counts.items(), key=lambda x: len(x[1]))
            lines.append(
                f"   - Наиболее часто оптимальный размер окна: {most_common[0]} "
                f"(рекомендован для {len(most_common[1])} алгоритмов).\n"
            )

        lines.append("---\n")
        lines.append(
            f"*Отчет сгенерирован автоматически {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.*\n"
        )

        return "\n".join(lines)

    def build_report(
        self,
        dfs: dict[str, pd.DataFrame],
        results: ReportResults,
        original_counts: dict[str, int],
        output_path: Path,
    ) -> Path:
        """
        Build complete markdown report.

        Args:
            dfs: Dictionary of preprocessed DataFrames.
            results: ReportResults with all analysis results.
            original_counts: Original record counts before filtering.
            output_path: Path to save the report.

        Returns:
            Path to saved report.
        """
        sections = [
            self._build_header(),
            self._build_data_overview(dfs, original_counts),
            self._build_descriptive_stats(results),
            self._build_success_rates(results),
            self._build_pairwise_tests(results),
            self._build_mcnemar_tests(results),
            self._build_optimal_windows(results),
            self._build_plots_section(results.plot_paths),
            self._build_conclusions(results),
        ]

        report_content = "\n".join(sections)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_content, encoding="utf-8")

        return output_path
