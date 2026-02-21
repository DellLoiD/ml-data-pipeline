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
        self.instance_num_le.setFixedWidth(60)
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
        self.plot_type_combo.addItems(["Сводный график", "Столбчатый", "Пчелиное гнездо"])
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["По убыванию", "По алфавиту", "По исходному порядку"])

        # Кнопка помощи
        self.help_plot_btn = QPushButton("?")
        self.help_plot_btn.setFixedSize(20, 20)
        self.help_plot_btn.clicked.connect(self.show_plot_help)

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

        #help_label = QLabel("Топ-5 признаков и кнопка графика.\n""Прокручивайте вправо, чтобы увидеть все модели.")
        #help_label.setWordWrap(True)
        #help_label.setStyleSheet("font-size: 11px; color: #555;")
        #results_layout.addWidget(help_label)

        #line = QFrame()
        #line.setFrameShape(QFrame.HLine)
        #line.setFrameShadow(QFrame.Sunken)
        #results_layout.addWidget(line)

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

        # Импорт shap после инициализации UI
        global shap
        import shap

    def update_button_states(self):
        """Обновляет состояние всех кнопок на основе текущего состояния."""
        model_trained = len(self.trained_models) > 0

        self.analyze_btn.setEnabled(model_trained)

    def save_shap_plot_for_plot(self, plot_data):
        """Сохраняет график SHAP на основе кэшированных данных."""
        if plot_data is None or 'shap_values' not in plot_data:
            QMessageBox.warning(self, "Ошибка", "Нет данных графика для сохранения.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "shap_plot.png", "PNG (*.png);;PDF (*.pdf);;All Files (*)"
        )
        if not path:
            return

        try:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            
            shap_values = plot_data['shap_values']
            X_sample = plot_data['X_sample']
            plot_type = plot_data['plot_type']
            features_display_names = plot_data['features_display_names']

            if plot_type == "Сводный график":
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values, X_sample, feature_names=features_display_names, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values, X_sample, feature_names=features_display_names, plot_type="dot", show=False)
            elif plot_type == "Столбчатый":
                shap.summary_plot(shap_values, features=X_sample, feature_names=features_display_names, plot_type="bar", show=False)
            elif plot_type == "Пчелиное гнездо":
                X_sample_df = pd.DataFrame(X_sample, columns=features_display_names)
                shap_values_data = shap_values.values if hasattr(shap_values, 'values') else shap_values
                if isinstance(shap_values_data, list) and len(shap_values_data) > 0:
                    shap_values_data = shap_values_data[0]
                # Создаем Explanation объект
            explanation = shap.Explanation(
                values=shap_values_data,
                data=X_sample_df.values,
                feature_names=features_display_names
            )
            # Для beeswarm нужен одномерный массив значений
            if explanation.values.ndim > 1:
                # Берем первый класс или усредняем
                if explanation.values.shape[1] > 1:
                    values_flat = np.abs(explanation.values).mean(axis=1)  # Усредняем |values| по классам
                else:
                    values_flat = explanation.values[:, 0]
                # Обрезаем data до нужного количества признаков (количество столбцов = количеству имен признаков)
                data_trimmed = X_sample_df.values[:, :len(features_display_names)]
                # Убедимся, что значения и данные имеют одинаковое количество строк
                if len(values_flat) != len(data_trimmed):
                    min_len = min(len(values_flat), len(data_trimmed))
                    values_flat = values_flat[:min_len]
                    data_trimmed = data_trimmed[:min_len]
                # Создаем новое объяснение с одномерными значениями
                explanation = shap.Explanation(
                    values=values_flat,
                    data=data_trimmed,
                    feature_names=features_display_names
                )
            shap.plots.beeswarm(explanation, show=False)
            plt.title(f"{plot_type} - {plot_data['sort_order']}")
            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            QMessageBox.information(self, "Сохранено", f"График сохранён:\n{os.path.basename(path)}")
        except Exception as e:
            error_msg = f"Не удалось сохранить график: {e}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            print(error_msg)

    def save_shap_values_for_plot(self, plot_data):
        """Сохраняет SHAP значения из кэшированных данных."""
        if plot_data is None or 'shap_values' not in plot_data:
            QMessageBox.warning(self, "Ошибка", "Нет данных SHAP для сохранения.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить SHAP значения", "shap_values.npy", "NumPy Files (*.npy);;CSV Files (*.csv);;All Files (*)"
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
                else:
                    values = shap_values.values
                shap_df = pd.DataFrame(values, columns=feature_names)
                shap_df.to_csv(path, index=False)
            else:
                np.save(path, shap_values)  # По умолчанию .npy

            QMessageBox.information(self, "Сохранено", f"SHAP значения сохранены:\n{os.path.basename(path)}")
        except Exception as e:
            error_msg = f"Не удалось сохранить SHAP значения: {e}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            print(error_msg)

    def show_plot_help(self):
        text = """
        <b>Типы графиков:</b><br>
        • <b>Сводный график</b> — суммирует важность признаков и направление влияния<br>
        • <b>Столбчатый</b> — диаграмма важности признаков<br>
        • <b>Пчелиное гнездо</b> — распределение вкладов признаков по объектам<br><br>
        <b>Сортировка:</b><br>
        • По убыванию — по среднему |SHAP значению|<br>
        • По алфавиту — по имени признака<br>
        • По исходному порядку — как в датасете
        """
        HelpDialog("Справка по графикам", text, self).exec_()

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
        
    def analyze_shap(self):
        if not self.trained_models:
            QMessageBox.warning(self, "Ошибка", "Сначала обучите модель.")
            return
        
        model_name, model = list(self.trained_models.items())[0]
        
        from .feature_importance_shap_logic import analyze_shap as logic_analyze_shap
        
        explainer_type = self.explainer_combo.currentText()
        
        result = logic_analyze_shap(
            explainer_type=explainer_type,
            model=model,
            X_train=self.X_train,
            sample_size=self.sample_size_combo.currentText(),
            model_task=self.task_type
        )
        
        if result['success']:
            self.shap_explainer = result['explainer']
            self.shap_values = result['shap_values']
            self.X_sample = result['X_sample']
            
            # Добавим реальное имя объяснителя
            actual_explainer_name = result['explainer'].__class__.__name__.replace("Explainer", "") if result['explainer'] else "Unknown"
            
            # Передаём explainer_type в plot_shap
            self.plot_shap(explainer_type=explainer_type)
            self.update_button_states()
        else:
            error_msg = f"Ошибка при анализе SHAP: {result['error']}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            print(error_msg)

    def plot_shap(self, explainer_type="Auto"):
        if self.shap_values is None:
            return

        plot_type = self.plot_type_combo.currentText()
        sort_order = self.sort_combo.currentText()
        feature_names = self.X_train.columns.tolist()

        # Determine sorting
        if sort_order == "По убыванию":
            # Sort by mean |value|
            values = np.array(self.shap_values.values)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            feature_order = np.argsort(-np.abs(values).mean(axis=0))
        elif sort_order == "По алфавиту":
            feature_order = np.argsort(feature_names)
        else:  # Original Order
            feature_order = np.arange(len(feature_names))



        # Создание отображаемых имён
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
                print(f"Ошибка при создании имён: {e}")

        # Создание и отображение Top-5 признаков
        top_k = 5
        top_indices = feature_order[:top_k]
        top_features = [features_display_names[i] for i in range(min(top_k, len(features_display_names)))]
        
        features_text = "<b>Топ-5 признаков:</b><br>" + "<br>".join(
            f"{i+1}. {name}" for i, name in enumerate(top_features)
        )        

        # Получаем информацию для отображения
        method_display_name = explainer_type
        if explainer_type == "Авто" and hasattr(self, 'shap_explainer'):
            # Получаем реальный тип из построенного экземпляра объяснителя
            explainer_class = self.shap_explainer.__class__.__name__
            if "Tree" in explainer_class:
                method_display_name = "TreeExplainer"
            elif "Linear" in explainer_class:
                method_display_name = "LinearExplainer"
            elif "Kernel" in explainer_class:
                method_display_name = "KernelExplainer"
            else:
                method_display_name = explainer_class
        
        features_text = f"""
        <b>Метод:</b> {method_display_name}<br>
        <b>Тип графика:</b> {plot_type}<br>
        <b>Сортировка:</b> {sort_order}<br>
        <b>Топ-5 признаков:</b><br>
        """ + "<br>".join(f"{i+1}. {name}" for i, name in enumerate(top_features))
        
        features_label = QLabel(features_text)
        features_label.setWordWrap(True)

        # Создание фигуры
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # По умолчанию используем оригинальные значения
        values = self.shap_values.values
        feature_names = self.X_train.columns.tolist()
        X_sample_current = self.X_sample
        features_display_names = feature_names[:]  # По умолчанию без изменений
        
        # Обновляем порядок сортировки
        if sort_order == "По убыванию":
            # Sort by mean |value|
            values_array = np.array(values)
            if values_array.ndim == 1:
                values_array = values_array.reshape(1, -1)
            feature_order = np.argsort(-np.abs(values_array).mean(axis=0))
        elif sort_order == "По алфавиту":
            feature_order = np.argsort(feature_names)
        else:  # Original Order
            feature_order = np.arange(len(feature_names))

        # Агрегация значений для категориальных признаков
        if hasattr(self, 'df') and self.df is not None:
            try:
                cat_columns = self.df.select_dtypes(include=['object']).columns
                if len(cat_columns) > 0:
                    # Создаём словарь для отображения закодированных имён в исходные
                    original_feature_names = {}
                    for col in cat_columns:
                        unique_vals = self.df[col].astype(str).unique()
                        for val in unique_vals:
                            encoded_name = f"{col}_{val}"
                            if encoded_name in feature_names:
                                if col not in original_feature_names:
                                    original_feature_names[col] = []
                                original_feature_names[col].append(encoded_name)
                    
                    print(f"[DEBUG] Найдены категориальные признаки: {dict(original_feature_names)}")
                    
                    # Агрегируем SHAP значения по исходным признакам
                    shap_values_agg = self.shap_values.values.copy()
                    feature_names_agg = feature_names[:]
                    
                    # Для каждого категориального признака объединяем значения
                    for orig_col, encoded_cols in original_feature_names.items():
                        if len(encoded_cols) <= 1:
                            print(f"[DEBUG] Пропуск {orig_col}: {len(encoded_cols)} закодированных признаков (нужно >1)")
                            continue
                        # Находим индексы закодированных признаков
                        idxs = [feature_names.index(col) for col in encoded_cols if col in feature_names]
                        if len(idxs) < 2:
                            print(f"[DEBUG] Недостаточно индексов для {orig_col}: {idxs}")
                            continue
                        print(f"[DEBUG] Агрегация {orig_col}: {encoded_cols} -> индексы {idxs}")
                        
                        # Суммируем SHAP значения по всем объектам
                        if shap_values_agg.ndim == 2:
                            aggregated_values = np.sum(shap_values_agg[:, idxs], axis=1)
                            print(f"[DEBUG] Агрегированные значения (mean |SHAP|): {np.abs(aggregated_values).mean():.4f}")
                        else:
                            aggregated_values = np.sum(shap_values_agg[idxs])
                            print(f"[DEBUG] Агрегированное значение: {aggregated_values:.4f}")
                        
                        # Заменяем первый признак суммой, удаляем остальные
                        # Создаём новые массивы
                        if shap_values_agg.ndim == 2:
                            new_shap_values = np.delete(shap_values_agg, idxs[1:], axis=1)
                            new_shap_values[:, idxs[0]] = aggregated_values
                        else:
                            new_shap_values = np.delete(shap_values_agg, idxs[1:])
                            new_shap_values[idxs[0]] = aggregated_values
                        shap_values_agg = new_shap_values
                        
                        # Обновляем имена признаков
                        new_feature_names = [name for i, name in enumerate(feature_names_agg) if i not in idxs[1:]]
                        new_feature_names[idxs[0]] = orig_col
                        feature_names_agg = new_feature_names
                        
                        print(f"[DEBUG] После агрегации: {orig_col} -> {new_feature_names[idxs[0]]}, осталось признаков: {len(new_feature_names)}")
                    
                    # Обновляем значения после агрегации
                    values = shap_values_agg
                    feature_names = feature_names_agg
                    
                    # Убедимся, что X_sample_current соответствует размеру
                    if self.X_sample.shape[1] == len(feature_names):
                        X_sample_current = self.X_sample
                    else:
                        # Если размеры не совпадают, используем только нужные столбцы
                        X_sample_current = self.X_sample[:, :len(feature_names)]
                    
                    print(f"[DEBUG] Финальное количество признаков после агрегации: {len(feature_names)}")
                    
                    # Обновляем порядок сортировки
                    if sort_order == "По убыванию":
                        values_array = np.array(values)
                        if values_array.ndim == 1:
                            values_array = values_array.reshape(1, -1)
                        feature_order = np.argsort(-np.abs(values_array).mean(axis=0))
                    elif sort_order == "По алфавиту":
                        feature_order = np.argsort(feature_names)
                    else:
                        feature_order = np.arange(len(feature_names))
                    
                    # Обновляем отображаемые имена
                    features_display_names = [feature_names[i] for i in feature_order]
                    
            except Exception as e:
                print(f"Ошибка при агрегации SHAP значений: {e}")

        # Генерация графика в зависимости от типа
        if plot_type == "Сводный график":
            # Проверяем, является ли shap_values списком (multi-output)
            if isinstance(self.shap_values, list):
                # Для multi-output используем bar plot
                shap.summary_plot([values], X_sample_current, feature_names=features_display_names, plot_type="bar", show=False)
            else:
                # Для single-output можно использовать dot
                shap.summary_plot(values, X_sample_current, feature_names=features_display_names, plot_type="dot", show=False)
        elif plot_type == "Столбчатый":
            shap.summary_plot(values, features=X_sample_current, feature_names=features_display_names, plot_type="bar", show=False)
        elif plot_type == "Пчелиное гнездо":
            X_sample_df = pd.DataFrame(X_sample_current, columns=features_display_names)
            # Используем shap_values.values для beeswarm, так как сам объект может быть списком
            shap_values_data = self.shap_values.values if hasattr(self.shap_values, 'values') else self.shap_values
            if isinstance(shap_values_data, list):
                shap_values_data = shap_values_data[0]  # Берём первый выход для multi-output
            
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
            n_features = len(features_display_names)
            X_sample_trimmed = X_sample_df.values[:, :n_features]
            
            # Убедимся, что количество строк совпадает
            n_samples = min(len(values_flat), len(X_sample_trimmed))
            values_flat = values_flat[:n_samples]
            X_sample_trimmed = X_sample_trimmed[:n_samples]
            
            # Создаем Explanation объект с одномерными значениями
            explanation = shap.Explanation(
                values=values_flat,
                data=X_sample_trimmed,
                feature_names=features_display_names
            )
            
            # Логируем размер explanation перед построением
            print(f"[DEBUG] Размер explanation: {explanation.shape}")
            print(f"[DEBUG] Количество экземпляров: {explanation.shape[0]}")
            # В SHAP значение values[0] может быть значением для одного признака, но мы должны смотреть на общую форму
            # Для пчелиного гнезда важно, что бы было много экземпляров (строк), а признаков может быть один
            print(f"[DEBUG] Форма values: {explanation.values.shape}")
            if explanation.values.ndim == 1:
                print(f"[DEBUG] Интерпретация: {explanation.values.shape[0]} экземпляров, 1 признак")
            else:
                print(f"[DEBUG] Интерпретация: {explanation.values.shape[0]} экземпляров, {explanation.values.shape[1]} признаков")
            
            # Теперь формируем Explanation с гарантированно согласованными размерностями
            # Убедимся, что data имеет только один признак, так как values имеет одну колонку
            X_sample_corrected = X_sample_trimmed[:, :1] if X_sample_trimmed.ndim > 1 else X_sample_trimmed.reshape(-1, 1)
            
            # Обрезаем оба массива до минимального количества строк
            min_rows = min(values_flat.shape[0], X_sample_corrected.shape[0])
            values_final = values_flat[:min_rows].reshape(-1, 1)  # Двумерный массив (n, 1)
            data_final = X_sample_corrected[:min_rows]            # Двумерный массив (n, 1)
            
            explanation = shap.Explanation(
                values=values_final,
                data=data_final,
                feature_names=features_display_names[:1] if len(features_display_names) >= 1 else ["feature_0"]
            )
            
            # Логируем окончательные размеры для отладки
            print(f"[DEBUG] Финальный размер explanation.values: {explanation.values.shape}")
            print(f"[DEBUG] Финальный размер explanation.data: {explanation.data.shape}")
            
            # Теперь строим график
            shap.plots.beeswarm(explanation, show=False)
        # Настройка отображения
        ax.set_title(f"{plot_type} - {sort_order}")
        plt.tight_layout()

        # Кэшируем данные для перестроения
        plot_data = {
            'shap_values': self.shap_values,
            'X_sample': X_sample_current,
            'plot_type': plot_type,
            'sort_order': sort_order,

            'feature_names': feature_names,
            'features_display_names': features_display_names,
            'task_type': self.task_type,
            'explainer_type': explainer_type
        }

        # Создание виджета с кнопкой "Показать график"
        widget = QWidget()
        widget.setFixedWidth(200)
        layout = QVBoxLayout()
        layout.addWidget(features_label)
        
        # Горизонтальный макет для кнопок
        buttons_layout = QHBoxLayout()
        
        # Кнопка для показа только этого графика
        show_btn = QPushButton("👁️📊")
        show_btn.setToolTip("Показать график")
        show_btn.clicked.connect(lambda: self.show_single_plot(fig, plot_data))
        buttons_layout.addWidget(show_btn)

        # Кнопка 'Сохранить значения'
        save_values_btn = QPushButton("💾🔢")
        save_values_btn.setToolTip("Сохранить данные")
        save_values_btn.clicked.connect(lambda: self.save_shap_values_for_plot(plot_data))
        buttons_layout.addWidget(save_values_btn)

        # Кнопка 'Сохранить график'
        save_plot_btn = QPushButton("💾📊")
        save_plot_btn.setToolTip("Сохранить график")
        save_plot_btn.clicked.connect(lambda: self.save_shap_plot_for_plot(plot_data))
        buttons_layout.addWidget(save_plot_btn)
        
        layout.addLayout(buttons_layout)
        widget.setLayout(layout)
        
        # Добавляем виджет, фигуру и данные в историю
        self.plots_history.append((widget, fig))
        self.plot_figures.append(fig)
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

            if plot_type == "Сводный график":
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values, X_sample, feature_names=features_display_names, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values, X_sample, feature_names=features_display_names, plot_type="dot", show=False)
            elif plot_type == "Столбчатый":
                shap.summary_plot(shap_values, features=X_sample, feature_names=features_display_names, plot_type="bar", show=False)
            elif plot_type == "Пчелиное гнездо":
                X_sample_df = pd.DataFrame(X_sample, columns=features_display_names)
                shap_values_data = shap_values.values if hasattr(shap_values, 'values') else shap_values
                if isinstance(shap_values_data, list) and len(shap_values_data) > 0:
                    shap_values_data = shap_values_data[0]
                # Создаем Explanation объект
            explanation = shap.Explanation(
                values=shap_values_data,
                data=X_sample_df.values,
                feature_names=features_display_names
            )
            # Для beeswarm нужен одномерный массив значений
            if explanation.values.ndim > 1:
                # Берем первый класс или усредняем
                if explanation.values.shape[1] > 1:
                    values_flat = np.abs(explanation.values).mean(axis=1)  # Усредняем |values| по классам
                else:
                    values_flat = explanation.values[:, 0]
                # Обрезаем data до нужного количества признаков (количество столбцов = количеству имен признаков)
                data_trimmed = X_sample_df.values[:, :len(features_display_names)]
                # Убедимся, что значения и данные имеют одинаковое количество строк
                if len(values_flat) != len(data_trimmed):
                    min_len = min(len(values_flat), len(data_trimmed))
                    values_flat = values_flat[:min_len]
                    data_trimmed = data_trimmed[:min_len]
                # Создаем новое объяснение с одномерными значениями
                explanation = shap.Explanation(
                    values=values_flat,
                    data=data_trimmed,
                    feature_names=features_display_names
                )
            shap.plots.beeswarm(explanation, show=False)
            plt.title(f"{plot_type} - {plot_data['sort_order']}")
            plt.tight_layout()
            plt.show()

    def show_full_shap_plot(self):
        """Отображает полный график в отдельном окне matplotlib"""
        if self.current_fig is None:
            return        
        # Показываем график
        plt.show()