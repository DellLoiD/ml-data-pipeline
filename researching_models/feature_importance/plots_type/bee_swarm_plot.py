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
    - shap_values: значения SHAP (уже готовый объект Explanation)
    - X_sample: выборка объектов (данные)
    - feature_names: имена признаков (не используется, берутся из shap_values)
    - plot_data: кэшированные данные
    - plot_type: тип графика
    - sort_order: порядок сортировки
    - task_type: тип задачи ('classification', 'regression')
    - explainer_type: тип используемого объяснителя
    
    Возвращает: фигуру matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Используем shap_values напрямую как explanation, так как он уже содержит правильные feature_names
    explanation = shap_values
    
    # Строим график
    shap.plots.beeswarm(explanation, show=False)
    
    # Настройка отображения
    plt.title(f"{plot_type} - {sort_order}")
    plt.tight_layout()
    
    return fig
