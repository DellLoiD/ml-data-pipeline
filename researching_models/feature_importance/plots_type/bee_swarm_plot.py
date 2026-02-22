import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

def create_bee_swarm_plot(shap_values, X_sample, feature_names, plot_data, plot_type, sort_order, task_type, explainer_type):
    """
    Создает график 'Пчелиное гнездо' (beeswarm) для SHAP значений.
    
    Параметры:
    - shap_values: значения SHAP
    - X_sample: выборка объектов (данные)
    - feature_names: имена признаков
    - plot_data: кэшированные данные для перестроения
    - plot_type: тип графика
    - sort_order: порядок сортировки
    - task_type: тип задачи ('classification', 'regression')
    - explainer_type: тип используемого объяснителя
    
    Возвращает: фигуру matplotlib
    """
    # Подготовка данных для графика
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    shap_values_data = shap_values.values if hasattr(shap_values, 'values') else shap_values
    
    # Обработка значений в зависимости от размерности
    if np.ndim(shap_values_data) == 1:
        # Уже одномерный массив
        values_flat = shap_values_data
    elif np.ndim(shap_values_data) == 2:
        # Для 2D массива (многоклассовая классификация)
        if shap_values_data.shape[1] == 1:
            # Для бинарной классификации берем первый столбец
            values_flat = shap_values_data[:, 0]
        else:
            # Для многоклассовой классификации берем среднее по абсолютным значениям
            values_flat = np.abs(shap_values_data).mean(axis=1)
    else:
        # В крайнем случае, преобразуем в одномерный массив
        values_flat = np.ravel(shap_values_data)
    
    # Убедимся, что количество признаков соответствует количеству имен признаков
    n_features = len(feature_names)
    X_sample_trimmed = X_sample_df.values[:, :n_features]

    # Убедимся, что количество строк совпадает
    n_samples = min(len(values_flat), len(X_sample_trimmed))
    values_flat = values_flat[:n_samples]
    X_sample_trimmed = X_sample_trimmed[:n_samples]

    # Проверяем соответствие размерности перед созданием Explanation
    if X_sample_trimmed.shape[1] != n_features:
        logger.warning(f"Размерность X_sample_trimmed ({X_sample_trimmed.shape[1]}) не соответствует количеству признаков ({n_features}). Обрезаем признаки.")
        X_sample_trimmed = X_sample_trimmed[:, :n_features]

    # Создаем Explanation объект с одномерными значениями
    # Проверяем количество образцов
    logger.info(f"Количество образцов в X_sample_trimmed: {X_sample_trimmed.shape[0]}")
    logger.info(f"Форма values_flat: {values_flat.shape}")
    logger.info(f"Форма shap_values из plot_data: {plot_data['shap_values'].shape if hasattr(plot_data['shap_values'], 'shape') else type(plot_data['shap_values'])}")
    logger.info(f"Форма X_sample из plot_data: {plot_data['X_sample'].shape if hasattr(plot_data['X_sample'], 'shape') else type(plot_data['X_sample'])}")
    
    if X_sample_trimmed.shape[0] <= 1:
        logger.warning(f"График пчелиного гнезда не поддерживает один образец. Количество образцов: {X_sample_trimmed.shape[0]}.")
        logger.warning("Попытка создать график с одним образцом, будет использован столбчатый график.")
        # Создаем простой график (например, столбчатую диаграмму) для одного образца
        fig, ax = plt.subplots(figsize=(10, 6))
        # Используем feature_names, так как features_display_names не доступен в этой области видимости
        if len(values_flat) > 0:
            ax.bar(feature_names, values_flat)
            ax.set_title("SHAP значения для одного образца")
            ax.set_ylabel("SHAP значение")
            plt.xticks(rotation=45)
            plt.tight_layout()
        else:
            ax.text(0.5, 0.5, 'Нет данных для отображения', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title("SHAP значения (пусто)")
            plt.tight_layout()
        return fig
    
    # Создаем Explanation объект с одномерными значениями
    explanation = shap.Explanation(
        values=values_flat,
        data=X_sample_trimmed,
        feature_names=feature_names
    )
    
    # Строим график
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(explanation, show=False)
    
    # Настройка отображения
    plt.title(f"{plot_type} - {sort_order}")
    plt.tight_layout()
    
    return fig
