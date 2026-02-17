import os
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

class FeatureImportanceSHAPUI(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.X_train = None
        self.y_train = None
        self.target_col = None
        self.categorical_columns = []
        self.label_encoders = {}
        self.checkboxes = []
        self.labels_and_lines = {}
        self.task_type = "classification"
        self.results_layout = None
        self.original_path = None
        self.meta_tracker = MetaTracker()
        self.feature_importances = {}
        self.process = psutil.Process(os.getpid())
        self.plot_settings = {} 
        self.trained_models = {}  
        self.shap_explainer = None
        self.shap_values = None
        self.plot_figures = []  # Храним все построенные фигуры
        self.plot_data_cache = []  # Кэшируем данные для перестроения графиков
        self.init_ui()

    def set_trained_model(self, model, model_name):
        """Устанавливает предварительно обученную модель извне."""
        if model is not None and model_name:
            self.trained_models = {model_name: model}
            self.update_button_states()
            return True
        return False

    def set_data(self, df, target_col):
        """Устанавливает данные для анализа извне. Вызывает подготовку данных."""
        if df is None or target_col is None or target_col not in df.columns:
            return False
        
        self.df = df.copy()
        self.target_col = target_col
        
        # Подготовка данных
        self._prepare_data()
        
        # После подготовки данных обновляем состояние кнопок
        self.update_button_states()
        return True

    def _prepare_data(self):
        """Подготавливает X_train и y_train из self.df и self.target_col."""
        if self.df is None or self.target_col is None:
            self.X_train = None
            self.y_train = None
            return
        
        # Разделяем на признаки и целевую переменную
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Обработка категориальных признаков
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X_encoded = X.copy()
        
        self.label_encoders = {}
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Преобразуем в строку, чтобы избежать проблем с типами
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        # Сохраняем обработанные данные
        self.X_train = X_encoded
        self.y_train = y
        
        # Определяем тип задачи
        if y.dtype.kind in ['i', 'u'] and len(y.unique()) < 20:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

    def init_ui(self):
        self.setWindowTitle("Анализ SHAP")
        self.main_layout = QVBoxLayout()

        title_label = QLabel("Анализ SHAP")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.main_layout.addWidget(title_label)

        # Тип анализа
        analysis_type_group = QGroupBox("Тип анализа")
        analysis_type_layout = QHBoxLayout()

        self.global_analysis_radio = QRadioButton("Глобальный (важность признаков)")
        self.local_analysis_radio = QRadioButton("Локальный (вклад признаков для объекта № ___ )")
        self.global_analysis_radio.setChecked(True)
        self.instance_num_le = QLineEdit("0")
        self.instance_num_le.setFixedWidth(60)
        self.instance_num_le.setEnabled(False)
        self.local_analysis_radio.toggled.connect(lambda checked: self.instance_num_le.setEnabled(checked))

        analysis_type_layout.addWidget(self.global_analysis_radio)
        analysis_type_layout.addWidget(self.local_analysis_radio)
        analysis_type_layout.addWidget(QLabel("номер объекта"))
        analysis_type_layout.addWidget(self.instance_num_le)
        analysis_type_layout.addStretch()
        
        analysis_type_group.setLayout(analysis_type_layout)
        self.main_layout.addWidget(analysis_type_group)

        # Настройки SHAP
        shap_settings_group = QGroupBox("Настройки")
        shap_settings_layout = QHBoxLayout()

        # Метод объяснения
        self.explainer_combo = QComboBox()
        self.explainer_combo.addItems(["Auto", "TreeExplainer", "KernelExplainer", "LinearExplainer"])
        self.explainer_combo.setCurrentText("Auto")
        shap_settings_layout.addWidget(QLabel("Метод:"))
        shap_settings_layout.addWidget(self.explainer_combo)

        # Размер выборки
        self.sample_size_combo = QComboBox()
        self.sample_size_combo.addItems(["100", "500", "1000", "all"])
        self.sample_size_combo.setCurrentText("1000")
        shap_settings_layout.addWidget(QLabel("Размер:"))
        shap_settings_layout.addWidget(self.sample_size_combo)

        # Топ-N признаков
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 100)
        self.top_n_spin.setValue(15)
        shap_settings_layout.addWidget(QLabel("Топ-N:"))
        shap_settings_layout.addWidget(self.top_n_spin)

        shap_settings_group.setLayout(shap_settings_layout)
        self.main_layout.addWidget(shap_settings_group)

        # График
        plot_group = QGroupBox("График")
        plot_layout = QHBoxLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Summary Plot", "Bar", "Beeswarm"])
        plot_layout.addWidget(QLabel("Тип:"))
        plot_layout.addWidget(self.plot_type_combo)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Descending", "Alphabetical", "Original Order"])
        plot_layout.addWidget(QLabel("Сортировка:"))
        plot_layout.addWidget(self.sort_combo)

        self.help_plot_btn = QPushButton("?")
        self.help_plot_btn.clicked.connect(self.show_plot_help)
        plot_layout.addWidget(self.help_plot_btn)

        plot_group.setLayout(plot_layout)
        self.main_layout.addWidget(plot_group)

        # Кнопки управления
        action_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Анализировать")
        self.analyze_btn.clicked.connect(self.analyze_shap)
        self.analyze_btn.setEnabled(False)

        self.save_shap_btn = QPushButton("Сохранить SHAP-значения")
        self.save_shap_btn.clicked.connect(self.save_shap_values)
        self.save_shap_btn.setEnabled(False)

        self.save_plot_btn = QPushButton("Сохранить график")
        self.save_plot_btn.clicked.connect(self.save_shap_plot)
        self.save_plot_btn.setEnabled(False)

        action_layout.addStretch()
        action_layout.addWidget(self.analyze_btn)
        action_layout.addWidget(self.save_shap_btn)
        action_layout.addWidget(self.save_plot_btn)
        self.main_layout.addLayout(action_layout)

        # Результаты
        results_group = QGroupBox("📊 Результаты важности признаков")
        results_layout = QVBoxLayout()

        help_label = QLabel(
            "Топ-5 признаков и кнопка графика.\n"
            "Прокручивайте вправо, чтобы увидеть все модели."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("font-size: 11px; color: #555;")
        results_layout.addWidget(help_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        results_layout.addWidget(line)

        # Ограничим количество отображаемых графиков
        self.max_displayed_plots = 5
        self.plots_history = []  # Список для хранения предыдущих графиков

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
        
        self.update()  # Принудительное обновление виджета

        # Импорт shap после инициализации UI
        global shap
        import shap

    def update_button_states(self):
        """Обновляет состояние всех кнопок на основе текущего состояния."""
        model_trained = len(self.trained_models) > 0

        self.analyze_btn.setEnabled(model_trained)
        self.save_shap_btn.setEnabled(self.shap_values is not None)
        self.save_plot_btn.setEnabled(self.shap_values is not None)

    def save_shap_plot(self):
        """Сохраняет текущий график SHAP как изображение."""
        if self.shap_values is None:
            QMessageBox.warning(self, "Ошибка", "Нет графика для сохранения.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "shap_plot.png", "PNG (*.png);;PDF (*.pdf);;All Files (*)"
        )
        if not path:
            return

        try:
            # Например, сохранение последнего построенного графика через matplotlib
            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            QMessageBox.information(self, "Сохранено", f"График сохранён:\n{os.path.basename(path)}")
        except Exception as e:
            error_msg = f"Не удалось сохранить график: {e}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            print(error_msg)

    def save_shap_values(self):
        """Сохраняет SHAP значения в файл .npy или .csv"""
        if self.shap_values is None:
            QMessageBox.warning(self, "Ошибка", "Нет SHAP значений для сохранения.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить SHAP значения", "shap_values.npy", "NumPy Files (*.npy);;CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            if path.endswith(".npy"):
                np.save(path, self.shap_values)
            elif path.endswith(".csv"):
                # Преобразуем в DataFrame для CSV
                shap_df = pd.DataFrame(self.shap_values.values, columns=self.X_train.columns)
                shap_df.to_csv(path, index=False)
            else:
                np.save(path, self.shap_values)  # По умолчанию .npy

            QMessageBox.information(self, "Сохранено", f"SHAP значения сохранены:\n{os.path.basename(path)}")
        except Exception as e:
            error_msg = f"Не удалось сохранить SHAP значения: {e}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            print(error_msg)

    def show_plot_help(self):
        text = """
        <b>Типы графиков:</b><br>
        • <b>Summary Plot</b> — суммирует важность признаков и направление влияния<br>
        • <b>Bar</b> — столбчатая диаграмма важности<br>
        • <b>Beeswarm</b> — распределение вкладов признаков по объектам<br><br>
        <b>Сортировка:</b><br>
        • По убыванию — по среднему |SHAP значению|<br>
        • По алфавиту — по имени признака<br>
        • Исходный порядок — как в датасете
        """
        HelpDialog("Plot Help", text, self).exec_()

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
                le.addItems(['True', 'False'])
                le.setCurrentText(defaults.get(param, "True"))
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

    def train_model(self):
        from .feature_importance_shap_logic import kill_child_processes
        kill_child_processes()
        self.update_memory_usage()
        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "Error", "No data for training.")
            return
        if not self.target_col:
            QMessageBox.warning(self, "Error", "Target variable is not selected.")
            return
        selected = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Error", "Select at least one model.")
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
                    QMessageBox.information(self, "Training", f"Model {model_name} trained.")
                else:
                    QMessageBox.critical(self, "Error", f"Training error {model_name}: {result['error']}")
                    
            except Exception as e:
                error_msg = f"Training error {model_name}: {e}"
                QMessageBox.critical(self, "Error", error_msg)
                print(error_msg)
        
        self.update_button_states()
        self.update_memory_usage()

    def analyze_shap(self):
        if not self.trained_models:
            QMessageBox.warning(self, "Error", "First, train a model.")
            return
        
        model_name, model = list(self.trained_models.items())[0]
        
        from .feature_importance_shap_logic import analyze_shap as logic_analyze_shap
        
        result = logic_analyze_shap(
            explainer_type=self.explainer_combo.currentText(),
            model=model,
            X_train=self.X_train,
            sample_size=self.sample_size_combo.currentText(),
            model_task=self.task_type
        )
        
        if result['success']:
            self.shap_explainer = result['explainer']
            self.shap_values = result['shap_values']
            self.X_sample = result['X_sample']
            self.plot_shap()
            self.update_button_states()
        else:
            error_msg = f"Error analyzing SHAP: {result['error']}"
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)

    def plot_shap(self):
        if self.shap_values is None:
            return

        plot_type = self.plot_type_combo.currentText()
        sort_order = self.sort_combo.currentText()
        feature_names = self.X_train.columns.tolist()

        # Determine sorting
        if sort_order == "Descending":
            # Sort by mean |value|
            values = np.array(self.shap_values.values)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            feature_order = np.argsort(-np.abs(values).mean(axis=0))
        elif sort_order == "Alphabetical":
            feature_order = np.argsort(feature_names)
        else:  # Original Order
            feature_order = np.arange(len(feature_names))

        # Limit by Top-N
        top_n = self.top_n_spin.value()
        feature_order = feature_order[:top_n]

        # Создание отображаемых имён
        # Преобразуем feature_order в одномерный массив индексов
        feature_order = np.array(feature_order).flatten()
        features_display_names = [feature_names[i] for i in feature_order]  
        if hasattr(self, 'df') and self.df is not None:
            try:
                cat_columns = self.df.select_dtypes(include=['object']).columns
                if len(cat_columns) > 0:
                    # Создаём словарь для замены
                    name_mapping = {}
                    for col in cat_columns:
                        unique_vals = self.df[col].astype(str).unique()
                        for val in unique_vals:
                            # Предполагаем, что закодированное имя содержит имя столбца и значение
                            encoded_name = f"{col}_{val}"
                            display_name = f"{col}={val}"
                            if encoded_name in feature_names:
                                name_mapping[encoded_name] = display_name
                    # Создаём новый список имён для отображения
                    features_display_names = [name_mapping.get(name, name) for name in features_display_names]
            except Exception as e:
                print(f"Error creating display names: {e}")

        # Создание и отображение Top-5 признаков
        top_k = 5
        top_indices = feature_order[:top_k]
        top_features = [features_display_names[i] for i in range(min(top_k, len(features_display_names)))]
        
        features_text = "<b>Top-5 признаков:</b><br>" + "<br>".join(
            f"{i+1}. {name}" for i, name in enumerate(top_features)
        )
        
        features_label = QLabel(features_text)
        features_label.setWordWrap(True)

        # Создание фигуры
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Генерация графика в зависимости от типа
        if plot_type == "Summary Plot":
            # Проверяем, является ли shap_values списком (multi-output)
            if isinstance(self.shap_values, list):
                # Для multi-output используем bar plot
                shap.summary_plot(self.shap_values, self.X_sample, feature_names=features_display_names, plot_type="bar", show=False)
            else:
                # Для single-output можно использовать dot
                shap.summary_plot(self.shap_values, self.X_sample, feature_names=features_display_names, plot_type="dot", show=False)
        elif plot_type == "Bar":
            shap.summary_plot(self.shap_values, features=self.X_sample, feature_names=features_display_names, plot_type="bar", show=False)
        elif plot_type == "Beeswarm":
            shap.plots.beeswarm(self.shap_values, features=self.X_sample, feature_names=features_display_names, show=False)
            
        # Настройка отображения
        ax.set_title(f"{plot_type} - {sort_order}")
        plt.tight_layout()

        # Кэшируем данные для перестроения
        plot_data = {
            'shap_values': self.shap_values,
            'X_sample': self.X_sample,
            'plot_type': plot_type,
            'sort_order': sort_order,
            'top_n': top_n,
            'feature_names': self.X_train.columns.tolist(),
            'features_display_names': features_display_names,
            'task_type': self.task_type
        }

        # Создание виджета с кнопкой "Показать график"
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(features_label)
        
        # Кнопка для показа только этого графика
        show_btn = QPushButton("Показать график")
        show_btn.clicked.connect(lambda: self.show_single_plot(fig, plot_data))
        layout.addWidget(show_btn)
        
        widget.setLayout(layout)
        
        # Добавляем виджет, фигуру и данные в историю
        self.plots_history.append((widget, fig))
        self.plot_figures.append(fig)
        self.plot_data_cache.append(plot_data)  # Сохраняем данные для перестроения
        
        # Если графиков больше 5, удаляем самый левый (старый)
        if len(self.plots_history) > self.max_displayed_plots:
            old_widget, old_fig = self.plots_history.pop(0)
            old_widget.setParent(None)
            if old_fig in self.plot_figures:
                self.plot_figures.remove(old_fig)
            plt.close(old_fig)
            if len(self.plot_data_cache) > 0:
                self.plot_data_cache.pop(0)

        # Очистка текущего макета перед добавлением всех графиков
        for i in reversed(range(self.results_layout.count())): 
            self.results_layout.itemAt(i).widget().setParent(None)

        # Добавляем все виджеты из истории слева направо
        for widget, fig in self.plots_history:
            self.results_layout.addWidget(widget)

    def show_single_plot(self, fig, plot_data):
        """Показывает отдельный график в новом окне"""
        # Убедимся, что фигура всё ещё существует
        if fig and plt.fignum_exists(fig.number):
            plt.figure(fig.number)
            plt.show()
        else:
            # Перестраиваем график заново из кэшированных данных
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            
            # Используем данные из кэша для перестроения
            shap_values = plot_data['shap_values']
            X_sample = plot_data['X_sample']
            plot_type = plot_data['plot_type']
            features_display_names = plot_data['features_display_names']

            if plot_type == "Summary Plot":
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values, X_sample, feature_names=features_display_names, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values, X_sample, feature_names=features_display_names, plot_type="dot", show=False)
            elif plot_type == "Bar":
                shap.summary_plot(shap_values, features=X_sample, feature_names=features_display_names, plot_type="bar", show=False)
            elif plot_type == "Beeswarm":
                shap.plots.beeswarm(shap_values, features=X_sample, feature_names=features_display_names, show=False)
            
            plt.title(f"{plot_type} - {plot_data['sort_order']}")
            plt.tight_layout()
            plt.show()

    def show_full_shap_plot(self):
        """Отображает полный график в отдельном окне matplotlib"""
        if self.current_fig is None:
            return
        
        # Показываем график
        plt.show()