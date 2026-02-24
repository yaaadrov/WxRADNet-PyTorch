from datetime import datetime
from pathlib import Path

from thund_avoider.schemas.interpretation.config import (
    DataSourceConfig,
    OutlierConfig,
    ReportConfig,
)
from thund_avoider.services.interpretation import (
    DataPreprocessor,
    PlotGenerator,
    ReportBuilder,
    StatisticalAnalyzer,
)
from thund_avoider.settings import RESULT_PATH, settings


def get_data_sources() -> list[DataSourceConfig]:
    """Define all data sources with Russian names and colors."""
    palette = settings.interpretation_config.default_palette

    return [
        # Masked deterministic
        DataSourceConfig(
            file_path=RESULT_PATH / "dyn_masked_deterministic_base.parquet",
            russian_name="Маскированный (базовый)",
            color=palette[0],
        ),
        DataSourceConfig(
            file_path=RESULT_PATH / "dyn_masked_deterministic_greedy.parquet",
            russian_name="Маскированный (жадная оптимизация)",
            color=palette[1],
        ),
        # Masked predictive
        DataSourceConfig(
            file_path=RESULT_PATH / "dyn_masked_predictive_base.parquet",
            russian_name="Маскированный с предсказаниями (базовый)",
            color=palette[2],
        ),
        DataSourceConfig(
            file_path=RESULT_PATH / "dyn_masked_predictive_greedy.parquet",
            russian_name="Маскированный с предсказаниями (жадная оптимизация)",
            color=palette[3],
        ),
        # Non-masked
        DataSourceConfig(
            file_path=RESULT_PATH / "dynamic_avoider_base.parquet",
            russian_name="Базовый",
            color=palette[4],
        ),
        DataSourceConfig(
            file_path=RESULT_PATH / "dynamic_avoider_greedy.parquet",
            russian_name="Мастер-граф с жадной оптимизацией",
            color=palette[0],
        ),
        # Master graph variants
        DataSourceConfig(
            file_path=RESULT_PATH / "dynamic_avoider_master.parquet",
            russian_name="Мастер-граф",
            color=palette[1],
        ),
        DataSourceConfig(
            file_path=RESULT_PATH / "dynamic_avoider_master_tuned.parquet",
            russian_name="Мастер-граф с тюнингом",
            color=palette[2],
        ),
        DataSourceConfig(
            file_path=RESULT_PATH / "dynamic_avoider_smooth.parquet",
            russian_name="Мастер-граф со сглаживающей оптимизацией",
            color=palette[3],
        ),
    ]


def main() -> Path:
    """
    Generate statistical analysis report for pathfinding results.

    Returns:
        Path to generated report.
    """
    # Generate timestamp for report folder
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Create configuration
    config = ReportConfig(
        reports_path=settings.interpretation_config.reports_path,
        data_sources=get_data_sources(),
        outlier_config=OutlierConfig(
            min_length=300_000.0,
            max_length=1_500_000.0,
            suspicious_timestamps=[],
        ),
        confidence_level=settings.interpretation_config.default_confidence_level,
        alpha=settings.interpretation_config.default_alpha,
        palette=settings.interpretation_config.default_palette,
        figure_dpi=settings.interpretation_config.figure_dpi,
        figure_size=settings.interpretation_config.figure_size,
    )

    # Initialize components
    preprocessor = DataPreprocessor(config)
    analyzer = StatisticalAnalyzer(config)
    plot_generator = PlotGenerator(config)
    report_builder = ReportBuilder(config)

    # Step 1: Load data
    print("Загрузка данных...")
    raw_data = preprocessor.load_data()
    original_counts = {name: len(df) for name, df in raw_data.items()}
    print(f"Загружено {len(raw_data)} датасетов")

    # Step 2: Preprocess data
    print("Предобработка данных...")
    preprocessed_data = preprocessor.preprocess_all()
    print("Предобработка завершена")

    # Step 3: Run statistical analysis
    print("Выполнение статистического анализа...")
    results = analyzer.run_full_analysis(preprocessed_data)
    print(
        f"Анализ завершен: {len(results.pairwise_results)} попарных сравнений, "
        f"{len(results.optimal_window_results)} оптимальных окон"
    )

    # Step 4: Generate plots
    print("Генерация визуализаций...")
    plots_path = config.get_plots_path(timestamp)
    plot_paths = plot_generator.generate_all_plots(
        preprocessed_data, results, plots_path
    )
    results.plot_paths = plot_paths
    print(f"Сгенерировано {len(plot_paths)} графиков")

    # Step 5: Build report
    print("Формирование отчета...")
    report_path = config.get_timestamped_report_path(timestamp) / "report.md"
    report_builder.build_report(
        preprocessed_data, results, original_counts, report_path
    )
    print(f"Отчет сохранен: {report_path}")

    return report_path


if __name__ == "__main__":
    report_path = main()
    print(f"\nГотово! Отчет доступен: {report_path}")
