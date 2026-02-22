import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def create_summary_plot(shap_values, X_sample, feature_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output=False):
    """
    Создает сводный график (summary plot) для SHAP значений.
    
    Параметры:
    - shap_values: значения SHAP
    - X_sample: выборка объектов (данные)
    - feature_names: имена признаков
    - plot_data: кэшированные данные для перестроения
    - plot_type: тип графика ('Сводный график', 'Столбчатый')
    - sort_order: порядок сортировки
    - task_type: тип задачи ('classification', 'regression')
    - explainer_type: тип используемого объяснителя
    - is_multi_output: флаг, указывающий на multi-output модель
    
    Возвращает: фигуру matplotlib
    """
    # Определяем тип отображения в зависимости от типа графика
    plot_type_shap = "dot" if plot_type == "Сводный график" else "bar"
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Убедимся, что X_sample и shap_values имеют правильную форму
    if X_sample is not None and hasattr(shap_values, 'shape'):
        if X_sample.shape[1] != shap_values.shape[1]:
            print(f"Предупреждение: Размерность X_sample ({X_sample.shape[1]}) не соответствует размерности shap_values ({shap_values.shape[1]}). Приводим к совместимому размеру.")
            # Обрезаем shap_values до размера X_sample
            shap_values = shap_values[:, :X_sample.shape[1]] if shap_values.ndim == 2 else shap_values[:X_sample.shape[1]]

    # Построение сводного графика
    if is_multi_output:
        # Для multi-output используем bar plot
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type=plot_type_shap, show=False)
    else:
        # Для single-output используем dot plot
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type=plot_type_shap, show=False)
    
    # Настройка отображения
    ax.set_title(f"{plot_type} - {sort_order}")
    plt.tight_layout()
    
    return fig
