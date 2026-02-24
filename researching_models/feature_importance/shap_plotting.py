import os
import shap
import numpy as np
import pandas as pd
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from .plots_type.summary_plot import create_summary_plot
from .plots_type.bar_plot import create_bar_plot
from .plots_type.bee_swarm_plot import create_bee_swarm_plot
from .prepare_shap_data import prepare_shap_data

def plot_shap(shap_values, X_train, X_sample, task_type, explainer_type="Auto", plot_type="Сводный график", sort_order="По убыванию", df=None):
    """
    Основная функция для построения графиков SHAP и возврата виджета с информацией и кнопками.
    
    Параметры:
    - shap_values: значения SHAP
    - X_train: обучающая выборка (для получения имен признаков)
    - X_sample: выборка, использованная для объяснения
    - task_type: тип задачи ('classification', 'regression')
    - explainer_type: тип объяснителя ("Авто", "TreeExplainer", и т.д.)
    - plot_type: тип графика ("Сводный график", "Столбчатый", "Пчелиное гнездо")
    - sort_order: порядок сортировки ("По убыванию", "По алфавиту", "По исходному порядку")
    - df: исходный DataFrame (для обработки категориальных признаков)
    
    Возвращает: (widget, plot_data, fig) — виджет, данные графика и фигуру matplotlib
    """
    # 1. Подготовка данных (вынесено в отдельную функцию)
    logger = logging.getLogger(__name__)
    logger.info(f"SHAP_PLOTTING: Перед вызовом prepare_shap_data, X_train.columns = {X_train.columns.tolist() if hasattr(X_train, 'columns') else 'no columns'}")
    prepared_data = prepare_shap_data(shap_values, X_train, sort_order)
    
    # 2. Извлечение подготовленных данных
    explanation = prepared_data['explanation']
    feature_names = prepared_data['feature_names']
    features_display_names = prepared_data['features_display_names']
    mean_abs_shap = prepared_data['mean_abs_shap']
    
    # Логирование
    logger.info(f"SHAP_PLOTTING: После prepare_shap_data, explanation.feature_names = {explanation.feature_names}")
    
    # 3. Кэш��рование данных для дальнейшего использования
    plot_data = {
        'shap_values': explanation,
        'X_sample': X_sample,
        'X_train': X_train,
        'plot_type': plot_type,
        'sort_order': sort_order,
        'feature_names': X_train.columns.tolist(),
        'features_display_names': features_display_names,
        'original_feature_names': X_train.columns.tolist(),
        'task_type': task_type,
        'explainer_type': explainer_type,
        'mean_abs_shap': mean_abs_shap.astype(float) if isinstance(mean_abs_shap, np.ndarray) else float(mean_abs_shap)
    }

    # 4. Определение, является ли вывод multi-output
    if isinstance(shap_values, list):
        is_multi_output = True
    else:
        is_multi_output = hasattr(shap_values, 'values') and np.ndim(getattr(shap_values, 'values', [])) > 1 and getattr(getattr(shap_values, 'values', []), 'shape', (1,1))[1] > 1
    
    # 5. Создаем график с помощью соответствующей функции
    # Передаем explanation вместо shap_values для создания графика
    if plot_type == "Сводный график":
        fig = create_summary_plot(explanation, X_sample, feature_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
    elif plot_type == "Столбчатый":
        fig = create_bar_plot(explanation, X_sample, feature_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
    elif plot_type == "Пчелиное гнездо":
        fig = create_bee_swarm_plot(explanation, X_sample, feature_names, plot_data, plot_type, sort_order, task_type, explainer_type)
    else:
        raise ValueError(f"Неподдерживаемый тип графика: {plot_type}")

    # 6. Создание виджета с информацией
    widget = QWidget()
    widget.setFixedWidth(200)
    layout = QVBoxLayout()
    
    # Топ-5 признаков
    top_k = 5
    top_features = [features_display_names[i] for i in range(min(top_k, len(features_display_names)))]
    features_text = f"""
    <b>Метод:</b> {explainer_type}<br>
    <b>Тип графика:</b> {plot_type}<br>
    <b>Сортировка:</b> {sort_order}<br>
    <b>Топ-5 признаков:</b><br>
    """ + "<br>".join(f"{i+1}. {name}" for i, name in enumerate(top_features))
    
    features_label = QLabel(features_text)
    features_label.setWordWrap(True)
    layout.addWidget(features_label)
    
    # Кнопки
    buttons_layout = QHBoxLayout()

    show_btn = QPushButton("👁️📊")
    show_btn.setToolTip("Показать график")
    buttons_layout.addWidget(show_btn)
    
    save_values_btn = QPushButton("💾🔢")
    save_values_btn.setToolTip("Сохранить данные")
    buttons_layout.addWidget(save_values_btn)
    
    save_plot_btn = QPushButton("💾📊")
    save_plot_btn.setToolTip("Сохранить график")
    buttons_layout.addWidget(save_plot_btn)
    
    layout.addLayout(buttons_layout)
    widget.setLayout(layout)    

    return widget, plot_data, fig

def save_shap_plot_for_plot(plot_data):
    """Сохраняет график SHAP на основе кэшированных данных."""
    if plot_data is None or 'shap_values' not in plot_data:
        QMessageBox.warning(None, "Ошибка", "Нет данных графика для сохранения.")
        return

    path, _ = QFileDialog.getSaveFileName(
        None, "Сохранить график", "shap_plot.png", "PNG (*.png);;PDF (*.pdf);;All Files (*)"
    )
    if not path:
        return

    try:
        # Подготовка данных
        shap_values = plot_data['shap_values']
        X_sample = plot_data['X_sample']
        plot_type = plot_data['plot_type']
        features_display_names = plot_data['features_display_names']  
        sort_order = plot_data['sort_order']
        task_type = plot_data['task_type']
        explainer_type = plot_data['explainer_type']
        
        # Определяем, является ли вывод multi-output
        if isinstance(shap_values, list):
            is_multi_output = True
        else:
            is_multi_output = hasattr(shap_values, 'values') and np.ndim(getattr(shap_values, 'values', [])) > 1 and getattr(getattr(shap_values, 'values', []), 'shape', (1,1))[1] > 1
        
        # Создаем график с помощью соответствующей функции
        if plot_type == "Сводный график":
            fig = create_summary_plot(shap_values, X_sample, plot_data['original_feature_names'], plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
        elif plot_type == "Столбчатый":
            fig = create_bar_plot(shap_values, X_sample, plot_data['original_feature_names'], plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
        elif plot_type == "Пчелиное гнездо":
            fig = create_bee_swarm_plot(shap_values, X_sample, plot_data['original_feature_names'], plot_data, plot_type, sort_order, task_type, explainer_type)
        else:
            raise ValueError(f"Неподдерживаемый тип графика: {plot_type}")
        
        # Сохраняем фигуру
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        QMessageBox.information(None, "Сохранено", f"График сохранён:\n{os.path.basename(path)}")
    except Exception as e:
                error_msg = f"Не удалось сохранить график: {e}"
                QMessageBox.critical(None, "Ошибка", error_msg)


def save_shap_values_for_plot(plot_data):
    """Сохраняет SHAP значения из кэшированных данных."""
    if plot_data is None or 'shap_values' not in plot_data:
        QMessageBox.warning(None, "Ошибка", "Нет данных SHAP для сохранения.")
        return

    path, _ = QFileDialog.getSaveFileName(
        None, "Сохранить SHAP значения", "shap_values.npy", "NumPy Files (*.npy);;CSV Files (*.csv);;All Files (*)"
    )
    if not path:
        return

    try:
        shap_values = plot_data['shap_values']
        feature_names = plot_data['feature_names']

        if path.endswith(".npy"):
            np.save(path, shap_values)
        elif path.endswith(".csv"):
            # Преобразуем в DataFrame для CSV
            if isinstance(shap_values, np.ndarray):
                values = shap_values
            elif hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = np.array(shap_values)
            shap_df = pd.DataFrame(values, columns=feature_names)
            shap_df.to_csv(path, index=False)
        else:
            np.save(path, shap_values)  # По умолчанию .npy

        QMessageBox.information(None, "Сохранено", f"SHAP значения сохранены:\n{os.path.basename(path)}")
    except Exception as e:
                error_msg = f"Не удалось сохранить SHAP значения: {e}"
                QMessageBox.critical(None, "Ошибка", error_msg)