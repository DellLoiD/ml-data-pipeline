import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def create_bar_plot(shap_values, X_sample, feature_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output=False):
    """
    Создает столбчатый график (bar plot) для SHAP значений.
    
    Параметры:
    - shap_values: значения SHAP
    - X_sample: выборка объектов (данные)
    - feature_names: имена признаков
    - plot_data: кэшированные данные для перестроения
    - plot_type: тип графика ('Столбчатый')
    - sort_order: порядок сортировки
    - task_type: тип задачи ('classification', 'regression')
    - explainer_type: тип используемого объяснителя
    - is_multi_output: флаг, указывающий на multi-output модель
    
    Возвращает: фигуру matplotlib
    """
    # Создание фигуры
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Убедимся, что X_sample и shap_values имеют правильную форму
    if X_sample is not None and hasattr(shap_values, 'shape'):
        if X_sample.shape[1] != shap_values.shape[1]:
            # Обрезаем shap_values до размера X_sample
            shap_values = shap_values[:, :X_sample.shape[1]] if shap_values.ndim == 2 else shap_values[:X_sample.shape[1]]

    # Построение столбчатого графика
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    
    # Настройка отображения
    fig.gca().set_title(f"{plot_type} - {sort_order}")
    plt.tight_layout()
    
    return fig
