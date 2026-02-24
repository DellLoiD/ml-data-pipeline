# Эти методы — это "контроллеры", которые связывают UI и логику. Вынос их в отдельный модуль четко разделяет поток управления.
from PySide6.QtWidgets import (QMessageBox)
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def analyze_shap(trained_models, X_train, shap_explainer, shap_values, X_sample, explainer_combo, sample_size_combo, plot_shap, update_button_states, task_type=None):
    logger.info("Начало выполнения analyze_shap в shap_interaction.py")
    
    if not trained_models:
        logger.error("Нет обученных моделей для анализа.")
        QMessageBox.warning(None, "Ошибка", "Сначала обучите модель.")
        return None  # Возвращаем None при ошибке
    
    model_name, model = list(trained_models.items())[0]
    logger.info(f"Используется модель: {model_name}")
    
    from .feature_importance_shap_logic import analyze_shap as logic_analyze_shap
    
    explainer_type = explainer_combo.currentText()
    sample_size_text = sample_size_combo.currentText()
    
    logger.info(f"Параметры анализа: explainer={explainer_type}, sample_size={sample_size_text}, task_type={task_type}")
    
    result = logic_analyze_shap(
        explainer_type=explainer_type,
        model=model,
        X_train=X_train,
        sample_size=sample_size_text,
        model_task=task_type
    )
    
    logger.info(f"Результат из logic_analyze_shap получен: {list(result.keys()) if result else 'None'}")
    
    # Принудительная проверка X_train
    if X_train is None:
        logger.error("X_train не должен быть None. Данные не были загружены или обработаны.")
        QMessageBox.critical(None, "Ошибка", "X_train не должен быть None. Данные не были загружены или обработаны.")
        return None
    
    if not hasattr(X_train, 'shape') or X_train.shape[0] == 0 or X_train.shape[1] == 0:
        logger.error(f"X_train пуст или имеет некорректную форму: {getattr(X_train, 'shape', 'unknown')}")
        QMessageBox.critical(None, "Ошибка", "X_train пуст или имеет некорректную форму.")
        return None
    
    if result['success']:
        logger.info("Анализ SHAP успешен, обновляем данные и строим график.")
        logger.info(f"X_train.columns в shap_interaction перед возвратом: {X_train.columns.tolist() if X_train is not None and hasattr(X_train, 'columns') else 'No columns'}")
        logger.info(f"Типы X_train.columns в shap_interaction: {[type(name).__name__ for name in X_train.columns] if X_train is not None and hasattr(X_train, 'columns') else 'No columns'}")
        
        # Возвращаем результат вместо обновления глобальных переменных
        return result
    else:
        error_msg = f"Ошибка при анализе SHAP: {result['error']}"
        logger.error(error_msg)
        QMessageBox.critical(None, "Ошибка", error_msg)
        print(error_msg)
        return None  # Возвращаем None при ошибке

def train_model(self):
    from .feature_importance_shap_logic import kill_child_processes
    kill_child_processes()
    self.update_memory_usage()
    if self.X_train is None or self.y_train is None:
        QMessageBox.warning(self, "Ошибка", "Нет данных для обучения.")
        return
    if not self.target_col:
        QMessageBox.warning(self, "Ошибка", "Целевая переменная не выбрана.")
        return
    selected = [cb.text() for cb in self.checkboxes if cb.isChecked()]
    if not selected:
        QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну модель.")
        return
    
    from .feature_importance_shap_logic import train_model as logic_train_model
    
    feature_names = self.X_train.columns.tolist()
    self.trained_models = {}
    
    for model_name in selected:
        try:
            params = self.labels_and_lines.get(model_name, {})
            n_jobs = self.safe_int(params, 'n_jobs', 1)
            
            result = logic_train_model(model_name, params, self.X_train, self.y_train, n_jobs)
            
            if result['success']:
                self.trained_models[model_name] = result['model']
                self.feature_importances[model_name] = result.get('importances')
                QMessageBox.information(self, "Обучение", f"Модель {model_name} обучена.")
            else:
                QMessageBox.critical(self, "Ошибка", f"Ошибка обучения {model_name}: {result['error']}")
                
        except Exception as e:
            error_msg = f"Ошибка обучения {model_name}: {e}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            print(error_msg)
    
    self.update_button_states()
    self.update_memory_usage()
    

def show_single_plot(self, fig, plot_data):
    """Показывает отдельный график в новом окне"""
    # Убедимся, что фигура всё ещё существует
    if fig and plt.fignum_exists(fig.number):
        plt.figure(fig.number)
        plt.show()
    else:
        # Перестраиваем график заново из кэшированных данных
        try:
            # Импортируем функции из модулей plots_type
            from .plots_type.summary_plot import create_summary_plot
            from .plots_type.bar_plot import create_bar_plot
            from .plots_type.bee_swarm_plot import create_bee_swarm_plot
            
            # Подготовка данных
            shap_values = plot_data['shap_values']
            X_sample = plot_data['X_sample']
            X_train = plot_data['X_train']
            plot_type = plot_data['plot_type']
            features_display_names = plot_data['features_display_names']
            sort_order = plot_data['sort_order']
            task_type = plot_data['task_type']
            explainer_type = plot_data['explainer_type']
            
            # Проверяем соответствие размерности shap_values и X_train
            if X_train is not None:
                expected_features = X_train.shape[1]
                shap_vals = plot_data['shap_values']
                if hasattr(shap_vals, 'values'):
                    shap_vals = shap_vals.values
                if isinstance(shap_vals, np.ndarray):
                    if shap_vals.ndim == 1:
                        if shap_vals.shape[0] != expected_features:
                            logger.warning(f"Размерность X_train ({expected_features}) не соответствует размерности shap_values ({shap_vals.shape[0]}). Приводим к совместимому размеру.")
                            # Обрезаем или дополняем shap_vals
                            if shap_vals.shape[0] > expected_features:
                                shap_vals = shap_vals[:expected_features]
                            else:
                                shap_vals = np.pad(shap_vals, (0, expected_features - shap_vals.shape[0]))
                    elif shap_vals.ndim == 2:
                        if shap_vals.shape[1] != expected_features:
                            logger.warning(f"Размерность X_train ({expected_features}) не соответствует размерности shap_values ({shap_vals.shape[1]}). Приводим к совместимому размеру.")
                            if shap_vals.shape[1] > expected_features:
                                shap_vals = shap_vals[:, :expected_features]
                            else:
                                padding = ((0, 0), (0, expected_features - shap_vals.shape[1]))
                                shap_vals = np.pad(shap_vals, padding)
                    elif shap_vals.ndim == 3:
                        if shap_vals.shape[2] != expected_features:
                            logger.warning(f"Размерность X_train ({expected_features}) не соответствует размерности shap_values ({shap_vals.shape[2]}). Приводим к совместимому размеру.")
                            if shap_vals.shape[2] > expected_features:
                                shap_vals = shap_vals[:, :, :expected_features]
                            else:
                                padding = ((0, 0), (0, 0), (0, expected_features - shap_vals.shape[2]))
                                shap_vals = np.pad(shap_vals, padding)
                # Обновляем shap_values в plot_data
                plot_data['shap_values'] = shap_vals

            # Определяем, является ли вывод multi-output
            is_multi_output = isinstance(plot_data['shap_values'], list) or (hasattr(plot_data['shap_values'], 'values') and np.ndim(plot_data['shap_values'].values) > 1 and plot_data['shap_values'].values.shape[1] > 1)
            
            # Создаем график с помощью соответствующей функции
            if plot_type == "Сводный график":
                fig = create_summary_plot(plot_data['shap_values'], X_sample, features_display_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
            elif plot_type == "Столбчатый":
                fig = create_bar_plot(plot_data['shap_values'], X_sample, features_display_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
            elif plot_type == "Пчелиное гнездо":
                fig = create_bee_swarm_plot(plot_data['shap_values'], X_sample, plot_data['features_display_names'], plot_data, plot_type, sort_order, task_type, explainer_type)
                if fig is None:
                    logger.error("create_bee_swarm_plot вернул None, используем альтернативный график")
                    fig = create_bar_plot(plot_data['shap_values'], X_sample, plot_data['features_display_names'], plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
            else:
                raise ValueError(f"Неподдерживаемый тип графика: {plot_type}")
            
            # Проверяем, что фигура создана
            if fig is None:
                logger.error("График не был создан - fig is None")
                return
                
            # Показываем фигуру
            plt.figure(fig.number)
            plt.show()
        except Exception as e:
            error_msg = f"Не удалось перестроить график: {e}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            logger.error(error_msg)


def show_plot_help(self):
    from .feature_importance_help_dialog import HelpDialog, PLOT_HELP_TEXT
    text = PLOT_HELP_TEXT
    HelpDialog("Справка по графикам", text, self).exec_()