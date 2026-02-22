import os
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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    feature_names = X_train.columns.tolist()

    # Определение порядка сортировки
    if sort_order == "По убыванию":
        # Проверяем, является ли shap_values списком (multi-output)
        if isinstance(shap_values, list):
            # Если это список, используем первый элемент или агрегируем
            values_list = []
            for v in shap_values:
                if hasattr(v, 'values'):
                    values_list.append(v.values)
                else:
                    values_list.append(v)
            values = np.array(values_list)
            # Усредняем по выходам, если нужно
            if values.ndim > 2:
                values = values.mean(axis=0)
        else:
            # Для одиночного вывода
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values
            
        # Преобразуем в массив и усредняем по выборке
        values = np.array(values)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        # Превращаем в flat array перед argsort
        mean_abs_shap = np.abs(values).mean(axis=0)
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        feature_order = np.argsort(-mean_abs_shap)
    elif sort_order == "По алфавиту":
        feature_order = np.argsort(feature_names)
    else:  # По исходному порядку
        feature_order = np.arange(len(feature_names))


    
    # Создание отображаемых имен (с учетом категориальных признаков)
    # Проверка корректности индексов перед доступом
    max_index = len(feature_names) - 1
    feature_order = [i for i in feature_order if 0 <= i <= max_index]
    if not feature_order:
        raise ValueError("Нет допустимых индексов в feature_order после фильтрации")
    features_display_names = [feature_names[i] for i in feature_order]
    if df is not None:
        try:
            cat_columns = df.select_dtypes(include=['object']).columns
            if len(cat_columns) > 0:
                name_mapping = {}
                for col in cat_columns:
                    unique_vals = df[col].astype(str).unique()
                    for val in unique_vals:
                        encoded_name = f"{col}_{val}"
                        display_name = f"{col}={val}"
                        if encoded_name in feature_names:
                            name_mapping[encoded_name] = display_name
                features_display_names = [name_mapping.get(name, name) for name in features_display_names]
        except Exception as e:
            logger.error(f"Ошибка при создании имен: {e}")

    # Агрегация значений для категориальных признаков
    if df is not None:
        try:
            cat_columns = df.select_dtypes(include=['object']).columns
            if len(cat_columns) > 0:
                original_feature_names = {}
                for col in cat_columns:
                    unique_vals = df[col].astype(str).unique()
                    for val in unique_vals:
                        encoded_name = f"{col}_{val}"
                        if encoded_name in feature_names:
                            if col not in original_feature_names:
                                original_feature_names[col] = []
                            original_feature_names[col].append(encoded_name)
                
                # Используем shap_values напрямую, если нет .values
                shap_values_agg = getattr(shap_values, 'values', shap_values)
                # Проверяем, нужно ли копировать
                if hasattr(shap_values_agg, 'copy'):
                    shap_values_agg = shap_values_agg.copy()
                else:
                    shap_values_agg = np.array(shap_values_agg)
                
                feature_names_agg = feature_names[:]
                
                for orig_col, encoded_cols in original_feature_names.items():
                    if len(encoded_cols) <= 1:
                        continue
                    idxs = [feature_names.index(col) for col in encoded_cols if col in feature_names]
                    if len(idxs) < 2:
                        continue
                    
                    if shap_values_agg.ndim == 2:
                        aggregated_values = np.sum(shap_values_agg[:, idxs], axis=1)
                        new_shap_values = np.delete(shap_values_agg, idxs[1:], axis=1)
                        new_shap_values[:, idxs[0]] = aggregated_values
                    else:
                        aggregated_values = np.sum(shap_values_agg[idxs])
                        new_shap_values = np.delete(shap_values_agg, idxs[1:])
                        new_shap_values[idxs[0]] = aggregated_values
                    shap_values_agg = new_shap_values
                    
                    new_feature_names = [name for i, name in enumerate(feature_names_agg) if i not in idxs[1:]]
                    new_feature_names[idxs[0]] = orig_col
                    feature_names_agg = new_feature_names

                # После всех изменений проверяем соответствие размерностей
                if shap_values_agg.ndim == 2 and shap_values_agg.shape[1] != len(feature_names_agg):
                    raise ValueError(f"Размерность shap_values_agg ({shap_values_agg.shape[1]}) не соответствует количеству признаков ({len(feature_names_agg)}) после агрегации")

                values = shap_values_agg
                feature_names = feature_names_agg
                # Пересоздаем feature_order после агрегации
                if sort_order == "По убыванию":
                    mean_abs_shap = np.abs(values).mean(axis=0)
                    if mean_abs_shap.ndim > 1:
                        mean_abs_shap = mean_abs_shap.flatten()
                    feature_order = np.argsort(-mean_abs_shap).tolist()
                
                # Проверка корректности индексов перед доступом
                max_index = len(feature_names) - 1
                feature_order = [i for i in feature_order if 0 <= i <= max_index]
                if not feature_order:
                    raise ValueError("Нет допустимых индексов в feature_order после фильтрации")
                features_display_names = [feature_names[i] for i in feature_order]
        except Exception as e:
            logger.error(f"Ошибка при агрегации SHAP значений: {e}")

    # Кэшируем только необходимые данные, исключая X_train
    plot_data = {
        'shap_values': shap_values,
        'X_sample': X_sample,
        'X_train': X_train,
        'plot_type': plot_type,
        'sort_order': sort_order,
        'feature_names': feature_names,
        'features_display_names': features_display_names,
        'task_type': task_type,
        'explainer_type': explainer_type
    }

    # Логирование успешного получения данных
    logger.info(f"Данные для графика SHAP успешно сформированы.")
    logger.info(f"Тип графика: {plot_type}")
    logger.info(f"Метод объяснения: {explainer_type}")
    logger.info(f"Количество признаков: {len(feature_names)}")
    if isinstance(shap_values, np.ndarray):
        logger.info(f"Форма значений SHAP: {shap_values.shape}")
    elif hasattr(shap_values, 'values'):
        logger.info(f"Форма значений SHAP: {shap_values.values.shape}")
    else:
        logger.info(f"Тип значений SHAP: {type(shap_values)}")

    fig = None

    # Создание виджета с информацией и кнопками
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
            fig = create_summary_plot(shap_values, X_sample, features_display_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
        elif plot_type == "Столбчатый":
            fig = create_bar_plot(shap_values, X_sample, features_display_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
        elif plot_type == "Пчелиное гнездо":
            fig = create_bee_swarm_plot(shap_values, X_sample, features_display_names, plot_data, plot_type, sort_order, task_type, explainer_type)
        else:
            raise ValueError(f"Неподдерживаемый тип графика: {plot_type}")
        
        # Сохраняем фигуру
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        QMessageBox.information(None, "Сохранено", f"График сохранён:\n{os.path.basename(path)}")
    except Exception as e:
        error_msg = f"Не удалось сохранить график: {e}"
        QMessageBox.critical(None, "Ошибка", error_msg)
        logger.error(error_msg)


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
        logger.error(error_msg)
