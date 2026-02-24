import os, shap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QLineEdit, QDialog,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton, QInputDialog, QScrollArea, QFrame, QComboBox, QSpinBox, QGridLayout
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from utils.meta_tracker import MetaTracker
import psutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .feature_importance_help_dialog import HelpDialog, MODEL_PARAM_HELP, N_JOBS_HELP, PLOT_HELP_TEXT
from .shap_ui_management_logic import ShapUiLogic


class FeatureImportanceSHAPUI(QWidget):
    def __init__(self):
        super().__init__()
        self.meta_tracker = MetaTracker()
        self.plot_settings = {} 
        self.shap_explainer = None
        self.plot_figures = []
        self.plot_data_cache = [] 
        self.logic = ShapUiLogic()
        # Добавляем атрибуты, которые используются в analyze_shap
        self.X_train = None
        self.X_sample = None
        self.df = None
        self.task_type = "classification" 
        self.init_ui()

    def set_trained_model(self, model, model_name):
        """Устанавливает предварительно обученную модель извне."""
        return self.logic.set_trained_model(model, model_name)
    
    def set_data(self, df, target_col):
        """Устанавливает данные для анализа извне. Вызывает подготовку данных."""
        # Сохраняем ссылку на исходный DataFrame
        self.df = df
        # Передаем дальше в логику
        return self.logic.set_data(df, target_col)

    def init_ui(self):
        self.setWindowTitle("Анализ SHAP")
        self.main_layout = QVBoxLayout()

        title_label = QLabel("Анализ SHAP")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        # Создаем горизонтальный контейнер для заголовка и "Тип анализа"
        header_analysis_layout = QHBoxLayout()
        header_analysis_layout.addWidget(title_label)
        
        # Тип анализа
        analysis_type_group = QGroupBox("Тип анализа")
        analysis_type_layout = QHBoxLayout()

        self.global_analysis_radio = QRadioButton("Глобальный (важность признаков)")
        self.local_analysis_radio = QRadioButton("Локальный (вклад признаков для объекта № ___ )")
        self.global_analysis_radio.setChecked(True)
        self.instance_num_le = QLineEdit("0")
        self.instance_num_le.setFixedWidth(20)
        self.instance_num_le.setEnabled(False)
        self.local_analysis_radio.toggled.connect(lambda checked: self.instance_num_le.setEnabled(checked))

        analysis_type_layout.addWidget(self.global_analysis_radio)
        analysis_type_layout.addWidget(self.local_analysis_radio)
        analysis_type_layout.addWidget(QLabel("№ объекта"))
        analysis_type_layout.addWidget(self.instance_num_le)
        analysis_type_layout.addStretch()
        
        analysis_type_group.setLayout(analysis_type_layout)
        header_analysis_layout.addWidget(analysis_type_group)
        header_analysis_layout.addStretch()

        self.main_layout.addLayout(header_analysis_layout)

        # Настройки SHAP
        shap_settings_group = QGroupBox("Настройки")
        shap_settings_layout = QHBoxLayout()

        # Метод объяснения
        self.explainer_combo = QComboBox()
        self.explainer_combo.addItems(["Авто", "TreeExplainer", "KernelExplainer", "LinearExplainer"])
        self.explainer_combo.setCurrentText("Авто")
        
        # Размер выборки
        self.sample_size_combo = QComboBox()
        self.sample_size_combo.addItems(["100", "500", "1000", "все"])
        self.sample_size_combo.setCurrentText("100")
        
        # Группировка: Метод + Размер
        method_size_layout = QHBoxLayout()
        method_size_layout.addWidget(QLabel("Метод:"))
        method_size_layout.addWidget(self.explainer_combo)
        method_size_layout.addWidget(QLabel("Размер:"))
        method_size_layout.addWidget(self.sample_size_combo)

        # Добавляем сгруппированные макеты в общий layout
        shap_settings_layout.addLayout(method_size_layout)

        shap_settings_group.setLayout(shap_settings_layout)

        # График
        plot_group = QGroupBox("График")
        plot_layout = QHBoxLayout()

        # Создание комбобоксов
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Сводный график", "Столбчатый"])
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["По убыванию", "По алфавиту", "По исходному порядку"])

        # Кнопка помощи
        self.help_plot_btn = QPushButton("?")
        self.help_plot_btn.setFixedSize(20, 20)
        self.help_plot_btn.clicked.connect(self.logic.show_plot_help)

        # Комбинированный виджет для Тип и Сортировка
        combined_layout = QVBoxLayout()
        type_sort_layout = QHBoxLayout()
        type_sort_layout.addWidget(QLabel("Тип:"))
        type_sort_layout.addWidget(self.plot_type_combo)
        type_sort_layout.addWidget(QLabel("Сортировка:"))
        type_sort_layout.addWidget(self.sort_combo)
        type_sort_layout.addWidget(self.help_plot_btn)
        type_sort_layout.addStretch()
        
        combined_layout.addLayout(type_sort_layout)
        plot_layout.addLayout(combined_layout)
        plot_group.setLayout(plot_layout)

        # Создаем горизонтальный контейнер для Настроек и Графика
        settings_plot_layout = QHBoxLayout()
        settings_plot_layout.addWidget(shap_settings_group)
        settings_plot_layout.addWidget(plot_group)
        
        # Добавляем контейнер в основной макет
        self.main_layout.addLayout(settings_plot_layout)

        # Результаты
        results_group = QGroupBox("📊 Результаты важности признаков")

        # Кнопка анализа перемещена ниже настроек и над результатами
        self.analyze_btn = QPushButton("Анализировать")
        self.analyze_btn.clicked.connect(self.analyze_shap)
        self.analyze_btn.setEnabled(False)
        self.main_layout.addWidget(self.analyze_btn)        

        # Результаты (продолжение)
        results_layout = QVBoxLayout()
        # Ограничим количество отображаемых графиков
        self.max_displayed_plots = 5
        self.plots_history = []

        self.results_layout = QHBoxLayout()
        self.results_layout.setSpacing(15)

        scroll_content = QWidget()
        scroll_content.setLayout(self.results_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        scroll.setFixedHeight(250)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        results_layout.addWidget(scroll)

        results_group.setLayout(results_layout)
        self.main_layout.addWidget(results_group)        
        
        self.setLayout(self.main_layout)
        
        self.update()
        pass

    def update_button_states(self):
        """Обновляет состояние всех кнопок на основе текущего состояния."""
        logic_state = self.logic.update_button_states()
        self.analyze_btn.setEnabled(logic_state['analyze_btn_enabled'])
        # Обновляем состояние кнопки "Удалить колонки" в основном UI
        if hasattr(self.parent(), 'delete_columns_btn') and 'delete_columns_btn_enabled' in logic_state:
            self.parent().delete_columns_btn.setEnabled(logic_state['delete_columns_btn_enabled'])

    def _add_model_to_layout(self, model_name, params, defaults, layout):
        hbox = QHBoxLayout()
        cb = QCheckBox(model_name)
        self.checkboxes.append(cb)
        hbox.addWidget(cb)
        lines = {}

        for param in params:
            lbl = QLabel(param)
            if param in ['Fit Intercept', 'Normalize']:
                le = QComboBox()
                le.addItems(['Истина', 'Ложь'])
                le.setCurrentText(defaults.get(param, "Истина"))
            else:
                le = QLineEdit()
                le.setFixedWidth(80)
                le.setText(defaults.get(param, "0"))

            help_text = MODEL_PARAM_HELP.get(param, param)

            btn = QPushButton("?")
            btn.setFixedSize(20, 20)
            btn.clicked.connect(lambda ch, t=param, h=help_text: HelpDialog(t, h, self.parent()).exec_())

            hbox.addWidget(lbl)
            hbox.addWidget(le)
            hbox.addWidget(btn)
            lines[param] = le

        # n_jobs
        n_jobs_lbl = QLabel("n_jobs")
        n_jobs_le = QLineEdit("1")
        n_jobs_le.setFixedWidth(50)
        n_jobs_help = QPushButton("?")
        n_jobs_help.setFixedSize(20, 20)
        n_jobs_help.clicked.connect(lambda: HelpDialog("n_jobs", N_JOBS_HELP, self.parent() or self).exec_())
        hbox.addWidget(n_jobs_lbl)
        hbox.addWidget(n_jobs_le)
        hbox.addWidget(n_jobs_help)
        lines['n_jobs'] = n_jobs_le

        self.labels_and_lines[model_name] = lines
        hbox.addStretch()
        layout.addLayout(hbox)

    def create_model(self, model_name, params):
        # This function has been moved to logic
        from .feature_importance_shap_logic import create_model
        return create_model(model_name, params)

    def analyze_shap(self):
        
            from .shap_interaction import analyze_shap
            
            # Проверка данных перед анализом
            if self.logic.X_train is None:
                return None
                
            if not self.logic.trained_models:
                return None            
            # Передаем атрибуты напрямую, избегая передачи всего объекта
            result = analyze_shap(
                trained_models=self.logic.trained_models,
                X_train=self.logic.X_train,
                shap_explainer=self.shap_explainer,
                shap_values=self.logic.shap_values,
                X_sample=self.X_sample,
                explainer_combo=self.explainer_combo,
                sample_size_combo=self.sample_size_combo,
                plot_shap=self.plot_shap,
                update_button_states=self.update_button_states,
                task_type=self.logic.task_type
            )
            
            # ОБНОВЛЕНИЕ: Сохраняем результаты в атрибутах UI
            if result and result['success']:
                self.shap_explainer = result.get('explainer')
                self.shap_values = result.get('shap_values')
                self.X_sample = result.get('X_sample')
                # Синхронизируем shap_values с логикой
                self.logic.shap_values = self.shap_values
                # Убедимся, что X_train и df тоже обновлены
                self.X_train = self.logic.X_train
                
                # Логирование значений важности после анализа SHAP и построения графика
                if self.shap_values is not None:
                    # Получаем имена признаков
                    feature_names = self.X_train.columns.tolist()
                    # Обработка различных типов shap_values
                    if hasattr(self.shap_values, 'values'):
                        # Для объектов Explanation, используемых в новых версиях SHAP
                        shap_vals = self.shap_values.values
                        # Для multi-output может быть массивом, берем первый выход
                        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim > 1:
                            shap_vals = shap_vals[0] if shap_vals.shape[0] == len(feature_names) else shap_vals.mean(axis=0)
                        mean_abs_shap = np.abs(shap_vals).mean(axis=0) if shap_vals.ndim > 1 else np.abs(shap_vals)
                    else:
                        # Для старого формата
                        mean_abs_shap = np.abs(self.shap_values).mean(axis=0) if self.shap_values.ndim > 1 else np.abs(self.shap_values)
                    # Создаем словарь с именами и значениями
                    feature_importances = dict(zip(feature_names, mean_abs_shap))
                    # Выводим в консоль
                    print("[LOG] Значения важности признаков после анализа SHAP и построения графика:", feature_importances)
                
                # Теперь вызываем plot_shap с актуальными данными
                explainer_type = self.explainer_combo.currentText()
                self.plot_shap(explainer_type=explainer_type)
                
                # После построения графика, активируем кнопку "Удалить колонки" через update_button_states
                if self.update_button_states:
                    self.update_button_states()

    def train_model(self):
        from .shap_interaction import train_model
        # Передаем атрибуты напрямую
        return train_model(
            X_train=self.X_train,
            y_train=self.y_train,
            target_col=self.target_col,
            checkboxes=self.checkboxes,
            labels_and_lines=self.labels_and_lines,
            logic=self.logic,
            update_button_states=self.update_button_states,
            update_memory_usage=getattr(self, 'update_memory_usage', lambda: None)
        )

    def plot_shap(self, explainer_type="Auto"):
        if self.shap_values is None:
            print("plot_shap: shap_values is None, пропуск построения графика.")
            return
            
        # Проверка X_train
        if self.X_train is None:
            print("plot_shap: self.X_train is None, пропуск построения графика.")
            return

        plot_type = self.plot_type_combo.currentText()
        sort_order = self.sort_combo.currentText()

        # Используем внешний модуль для визуализации
        from .shap_plotting import plot_shap
        
        print(f"Начало построения графика SHAP: тип={plot_type}, сортировка={sort_order}")
        
        widget, plot_data, fig = plot_shap(
            shap_values=self.shap_values,
            X_train=self.X_train,
            X_sample=self.X_sample,
            task_type=self.task_type,
            explainer_type=explainer_type,
            plot_type=plot_type,
            sort_order=sort_order
        )

        print(f"График SHAP построен и добавлен в интерфейс.")

        # Подключаем сигналы кнопок
        show_btn = widget.layout().itemAt(1).layout().itemAt(0).widget()
        show_btn.clicked.connect(lambda checked, d=plot_data: self.show_single_plot(None, d))

        save_values_btn = widget.layout().itemAt(1).layout().itemAt(1).widget()
        save_values_btn.clicked.connect(lambda: self.save_shap_values_for_plot(plot_data))

        save_plot_btn = widget.layout().itemAt(1).layout().itemAt(2).widget()
        save_plot_btn.clicked.connect(lambda: self.save_shap_plot_for_plot(plot_data))

        # Сохраняем фигуру в plot_data для избежания повторного построения
        plot_data['fig'] = fig

        # Добавляем виджет и данные в историю (фигура теперь в plot_data)
        self.plots_history.append((widget, None))
        self.plot_data_cache.append(plot_data)
        
        # Если графиков больше 5, удаляем самый левый (старый)
        if len(self.plots_history) > self.max_displayed_plots:
            old_widget, old_fig = self.plots_history.pop(0)
            old_widget.setParent(None)
            if old_fig in self.plot_figures:
                self.plot_figures.remove(old_fig)
            plt.close(old_fig)
            if len(self.plot_data_cache) > 0:
                self.plot_data_cache.pop(0)

        # Удаляем все виджеты из макета
        for i in reversed(range(self.results_layout.count())):
            item = self.results_layout.itemAt(i)
            if item:
                widget_item = item.widget()
                if widget_item:
                    widget_item.setParent(None)
        
        # Добавляем все виджеты из истории
        for w, f in self.plots_history:
            self.results_layout.addWidget(w)
 
   
    def show_single_plot(self, fig, plot_data):
        """Показывает отдельный график в новом окне"""
        # Если фигура существует и активна, показываем её и выходим
        if fig and plt.fignum_exists(fig.number):
            plt.figure(fig.number)
            plt.show()
            return  # Возвращаемся, чтобы избежать повторного построения
        # Перестраиваем график заново из кэшированных данных
        try:
                # Импортируем функции из модулей plots_type
                from .plots_type.summary_plot import create_summary_plot
                from .plots_type.bar_plot import create_bar_plot
                from .plots_type.bee_swarm_plot import create_bee_swarm_plot
                
                # Подготовка данных
                shap_values = plot_data['shap_values']
                X_sample = plot_data['X_sample']
                plot_type = plot_data['plot_type']
                features_display_names = plot_data['features_display_names']
                sort_order = plot_data['sort_order']
                task_type = plot_data['task_type']
                explainer_type = plot_data['explainer_type']
                
                # Определяем, является ли вывод multi-output
                is_multi_output = isinstance(shap_values, list) or (hasattr(shap_values, 'values') and np.ndim(shap_values.values) > 1 and shap_values.values.shape[1] > 1)
                
                # Создаем график с помощью соответствующей функции
                if plot_type == "Сводный график":
                    fig = create_summary_plot(shap_values, X_sample, features_display_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
                elif plot_type == "Столбчатый":
                    fig = create_bar_plot(shap_values, X_sample, features_display_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
                elif plot_type == "Пчелиное гнездо":
                    fig = create_bee_swarm_plot(shap_values, X_sample, plot_data['features_display_names'], plot_data, plot_type, sort_order, task_type, explainer_type)
                    if fig is None:
                        print("create_bee_swarm_plot вернул None, используем альтернативный график")
                        fig = create_bar_plot(shap_values, X_sample, features_display_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output)
                    else:
                        print("График Пчелиное гнездо успешно построен")
                else:
                    raise ValueError(f"Неподдерживаемый тип графика: {plot_type}") 
                
                # Сохраняем фигуру в кэш
                plot_data['fig'] = fig                
                # Показываем фигуру
                plt.figure(fig.number)
                plt.show()
        except Exception as e:
            error_msg = f"Не удалось перестроить график: {e}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            print(error_msg)