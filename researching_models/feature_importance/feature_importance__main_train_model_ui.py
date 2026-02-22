# feature_importance__main_train_model_ui.py
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QLineEdit, QDialog,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton, QInputDialog, QScrollArea, QTextEdit, QFrame,
    QGridLayout, QSpacerItem, QSizePolicy, QComboBox, QSpinBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from utils.meta_tracker import MetaTracker
import gc
import psutil
from joblib import parallel_backend

class HelpDialog(QDialog):
    """Справка по метрикам и параметрам"""
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.setModal(True)
        self.resize(400, 300)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
        self.setLayout(layout)

class DeleteColumnsDialog(QDialog):
    """Диалог для выбора колонок для удаления — сортирует по важности (от низкой к высокой)"""
    def __init__(self, columns, importances_dict=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Удалить колонки")
        self.resize(150, 200)

        layout = QVBoxLayout()

        info_label = QLabel("Выберите колонки для удаления:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        scroll = QScrollArea()
        scroll_content = QWidget()
        grid = QGridLayout(scroll_content)
        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(250)

        self.checkboxes = []
        sorted_columns = columns

        if importances_dict:
            # Вычисляем среднюю важность для каждого признака
            col_importance = {}
            for col in columns:
                imp_list = importances_dict.get(col, [0])
                avg_imp = sum(imp_list) / len(imp_list) if len(imp_list) > 0 else 0
                col_importance[col] = avg_imp
            # Сортируем по возрастанию важности (сначала наименее важные — удобнее удалять)
            sorted_columns = sorted(columns, key=lambda col: col_importance.get(col, 0))
            self.col_importance = col_importance  
            
            # Отладочный вывод
            #print("[DEBUG] Веса признаков (до сортировки):", col_importance)
            print("[DEBUG] Отсортированные колонки по возрастанию важности:", sorted_columns)
        else:
            sorted_columns = sorted(columns)
            self.col_importance = {col: 0 for col in columns}

        for idx, col in enumerate(sorted_columns):
            # Отображение значения важности рядом с чекбоксом
            cb = QCheckBox(f"{col} (важность: {self.col_importance.get(col, 0):.4f})")
            cb.setChecked(False)
            cb.setProperty("column_name", col)
            grid.addWidget(cb, idx, 0)
            self.checkboxes.append(cb)

        grid.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), len(sorted_columns), 0)

        layout.addWidget(scroll)

        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        delete_btn = QPushButton("Удалить")
        delete_btn.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(delete_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_selected_columns(self):
        selected = []
        for cb in self.checkboxes:
            if cb.isChecked():
                col_name = cb.text().split(' (важность:')[0]
                selected.append(col_name)
        return selected

class FeatureImportanceUI(QWidget):
    def safe_int(self, params, key, default):
        try:
            val = params[key].text().strip() if key in params else self.sender().parent().findChild(QLineEdit, key).text().strip()
            return int(val) if val else default
        except:
            return default

    def safe_float(self, params, key, default):
        try:
            val = params[key].text().strip()
            return float(val) if val else default
        except:
            return default

    def safe_int_or_none(self, params, key, default):
        try:
            val = params[key].text().strip()
            if not val or val.lower() in ('none', 'null'):
                return None
            return int(val)
        except:
            return default
    def __init__(self):
        super().__init__()
        self.df = None
        self.X_train = None
        self.y_train = None
        self.target_col = None
        self.checkboxes = []
        self.labels_and_lines = {}
        self.task_type = "classification"
        self.results_layout = None
        self.original_path = None
        self.meta_tracker = MetaTracker()
        self.feature_importances = {}
        self.process = psutil.Process(os.getpid())
        self.plot_settings = {}
        
        # SHAP-related attributes
        self.trained_models = {}
        self.shap_explainer = None
        self.shap_values = None
        self.X_sample = None
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Анализ важности признаков")
        main_layout = QVBoxLayout()

        # Горизонтальный макет для строки с заголовком, целевой переменной, памятью, R.S. и n_jobs
        info_layout = QHBoxLayout()

        # Заголовок (остаётся слева)
        title_label = QLabel("Анализ важности признаков")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        info_layout.addWidget(title_label)
        
        # Целевая переменная — в центре
        self.target_label = QLabel("Целевая переменная: не выбрана")
        self.target_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.target_label)

        # Метка памяти — справа
        self.memory_label = QLabel("📊 Память: ? МБ")
        self.memory_label.setStyleSheet("color: #555; font-size: 11px;")
        info_layout.addWidget(self.memory_label)

        # Добавим R.S. и n_jobs в ту же строку
        info_layout.addWidget(QLabel("R.S.:"))
        self.global_random_state = QLineEdit("42")
        self.global_random_state.setFixedWidth(20)
        info_layout.addWidget(self.global_random_state)

        help_random = QPushButton("?")
        help_random.setFixedSize(20, 20)
        help_random.clicked.connect(lambda: HelpDialog(
            "Random State",
            "Фиксация случайности. Для воспроизводимости результатов",
            self
        ).exec_())
        info_layout.addWidget(help_random)

        info_layout.addWidget(QLabel("n_jobs:"))
        self.global_n_jobs = QLineEdit("1")
        self.global_n_jobs.setFixedWidth(20)
        info_layout.addWidget(self.global_n_jobs)

        help_njobs = QPushButton("?")
        help_njobs.setFixedSize(20, 20)
        help_njobs.clicked.connect(lambda: HelpDialog(
            "n_jobs",
            "Количество ядер CPU для параллельных вычислений.\n"
            "1 — последовательно (по умолчанию)\n"
            "-1 — использовать все ядра",
            self
        ).exec_())       
        info_layout.addWidget(help_njobs)

        # Добавляем горизонтальный макет в основной вертикальный
        main_layout.addLayout(info_layout)
        
         # Основной горизонтальный макет для кнопок
        main_horizontal_layout = QHBoxLayout()        

        # === Строка с пометкой "Задача" и переключателями в одной строке ===
        main_horizontal_layout.addWidget(QLabel("Задача:"))
        self.classification_radio = QRadioButton("Классификация")
        self.regression_radio = QRadioButton("Регрессия")
        self.classification_radio.setChecked(True)
        self.regression_radio.setChecked(False)
        self.classification_radio.toggled.connect(self.on_task_selected)
        self.regression_radio.toggled.connect(self.on_task_selected)
        main_horizontal_layout.addWidget(self.classification_radio)
        main_horizontal_layout.addWidget(self.regression_radio)

        self.load_btn = QPushButton("Загрузить датасет")
        self.load_btn.clicked.connect(self.load_dataset)
        main_horizontal_layout.addWidget(self.load_btn)

        self.delete_columns_btn = QPushButton("🗑️ Удалить колонки")
        self.delete_columns_btn.clicked.connect(self.delete_selected_columns)
        self.delete_columns_btn.setEnabled(False)
        main_horizontal_layout.addWidget(self.delete_columns_btn)

        self.save_btn = QPushButton("💾 Сохранить датасет")
        self.save_btn.clicked.connect(self.save_dataset)
        self.save_btn.setEnabled(False)
        main_horizontal_layout.addWidget(self.save_btn)

        main_horizontal_layout.addStretch()  # Растяжка справа

        # Добавляем макет с кнопками в основной вертикальный макет
        main_layout.addLayout(main_horizontal_layout)

        # === Модели (без внешней группировки) ===
        self.classification_box = QGroupBox("Классификация")
        self.classification_layout = QGridLayout()
        self.classification_box.setLayout(self.classification_layout)
        main_layout.addWidget(self.classification_box)

        self.regression_box = QGroupBox("Регрессия")
        self.regression_layout = QGridLayout()
        self.regression_box.setLayout(self.regression_layout)
        main_layout.addWidget(self.regression_box)
        
        # === SHAP Analysis Section ===
        from .feature_importance_shap_ui import FeatureImportanceSHAPUI
        
        # Инициализация и добавление SHAP UI
        self.shap_ui = FeatureImportanceSHAPUI()
        main_layout.addWidget(self.shap_ui)
        
        # Кнопка для обучения модели
        self.train_model_btn = QPushButton("Обучить модель")
        self.train_model_btn.clicked.connect(self.train_selected_model)
        self.train_model_btn.setEnabled(False)
        
        # Вставка кнопки перед блоком SHAP UI
        main_layout.insertWidget(main_layout.indexOf(self.shap_ui), self.train_model_btn)

        self.setLayout(main_layout)
        self.shap_ui.update()
        self.create_models()
        self.classification_box.setVisible(self.task_type == "classification")
        self.regression_box.setVisible(self.task_type == "regression")
        self.adjustSize()
        self.show()
        # Кнопка Удалить колонки активна только если есть важности признаков
        self.delete_columns_btn.setEnabled(bool(self.feature_importances))
        self.update_memory_usage()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        
    def delete_selected_columns(self):
        """Открывает диалог для удаления выбранных колонок"""
        if self.X_train is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет.")
            return

        columns = self.X_train.columns.tolist()
        dialog = DeleteColumnsDialog(columns, importances_dict=self.feature_importances, parent=self)
        if dialog.exec() == QDialog.Accepted:
            to_delete = dialog.get_selected_columns()
            if not to_delete:
                return

            to_delete_existing = [col for col in to_delete if col in self.X_train.columns]
            if not to_delete_existing:
                return

            self.X_train = self.X_train.drop(columns=to_delete_existing)
            self.meta_tracker.add_change(f"удалены колонки: {', '.join(to_delete_existing)}")
            self.save_btn.setEnabled(True)

            QMessageBox.information(
                self, "Готово",
                f"Удалены колонки:\n" + "\n".join(to_delete_existing)
            )

            # Кнопка Удалить колонки активна только если есть важности признаков
        self.delete_columns_btn.setEnabled(bool(self.feature_importances))
        self.update_memory_usage()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
            
    def save_dataset(self):
        """Сохраняет текущий X_train + y_train в CSV с метаданными"""
        if self.X_train is None or len(self.X_train) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения.")
            return

        df_to_save = self.X_train.copy()
        df_to_save[self.target_col] = self.y_train

        base_name = "importance_dataset"
        if self.original_path:
            base_name = os.path.splitext(os.path.basename(self.original_path))[0].split("_v")[0]

        save_path = os.path.join("dataset", f"{base_name}_v{self.meta_tracker.version}.csv")

        try:
            success = self.meta_tracker.save_to_file(save_path, df_to_save)
            if success:
                self.meta_tracker.version += 1
                QMessageBox.information(
                    self, "Сохранено",
                    f"✅ Датасет сохранён:\n\n{os.path.basename(save_path)}\n\nВерсия: v{self.meta_tracker.version - 1}"
                )
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

        # Кнопка Удалить колонки активна только если есть важности признаков
        self.delete_columns_btn.setEnabled(bool(self.feature_importances))
        self.update_memory_usage()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()

    def kill_child_processes(self):
        """Принудительно завершает все дочерние процессы (например, от joblib)"""
        try:
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            if not children:
                return
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            gone, alive = psutil.wait_procs(children, timeout=3)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"Ошибка при завершении процессов: {e}")

    def update_memory_usage(self):
        try:
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            self.memory_label.setText(f"📊 Память: {mem_mb:.1f} МБ")
        except:
            self.memory_label.setText("📊 Память: ошибка")

    def on_task_selected(self):
        self.task_type = "classification" if self.classification_radio.isChecked() else "regression"
        self.classification_box.setVisible(self.task_type == "classification")
        self.regression_box.setVisible(self.task_type == "regression")

    def create_models(self):
        clf_models = {
            'Random Forest': ['Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Random State'],
            'Gradient Boosting': ['Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Random State'],
            'Logistic Regression': ['C', 'Max Iterations', 'Penalty', 'Random State']
        }
        reg_models = {
            'Random Forest': ['Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Random State'],
            'Gradient Boosting': ['Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Random State']
        }
        defaults = {
            'Кол-во деревьев': '100',
            'Max Depth': 'None',
            'Min Samples Split': '2',
            'Random State': '42',
            'Learning Rate': '0.1',
            'C': '1.0',
            'Max Iterations': '100',
            'Penalty': 'l2'
        }

        for model_name, params in clf_models.items():
            self._add_model_to_layout(model_name, params, defaults, self.classification_layout)
        for model_name, params in reg_models.items():
            self._add_model_to_layout(model_name, params, defaults, self.regression_layout)
            
    def _add_model_to_layout(self, model_name, params, defaults, layout):
        # Основной layout для модели — горизонтальный
        group_box = QGroupBox("")
        group_layout = QHBoxLayout()
        group_box.setLayout(group_layout)
        group_layout.setContentsMargins(10, 4, 10, 4)

        # Чекбокс для выбора модели
        model_checkbox = QCheckBox("")
        model_checkbox.setChecked(False)
        model_checkbox.setFixedWidth(25)
        # Добавляем имя модели как свойство
        model_checkbox.setProperty("model_name", model_name)
        group_layout.addWidget(model_checkbox)

        # Название модели как QLabel
        model_label = QLabel(model_name)
        model_label.setFixedWidth(110)
        group_layout.addWidget(model_label)

        # Контейнер для параметров модели
        lines = {}
        for param in params:
            if param not in ['Random State', 'n_jobs']:
                # Виджет для одного параметра
                param_widget = QWidget()
                param_hbox = QHBoxLayout(param_widget)
                param_hbox.setContentsMargins(3, 1, 3, 1)

                lbl = QLabel(param)
                lbl.setFixedWidth(100)
                le = QLineEdit()
                le.setFixedWidth(60)
                le.setText(defaults.get(param, "0"))

                help_text = {
                    'Кол-во деревьев': "Число деревьев в ансамбле. Больше → точнее, но дольше",
                    'Max Depth': "Максимальная глубина дерева. None — без ограничений. Большое → переобучение",
                    'Min Samples Split': "Минимальное число объектов для разбиения узла. Больше → проще модель",
                    'Learning Rate': "Скорость обучения в GB. Меньше → стабильнее, но медленнее",
                    'C': "Сила регуляризации в Logistic Regression. Больше → слабее регуляризация",
                    'Max Iterations': "Максимальное число итераций обучения. Увеличьте, если модель не сходится",
                    'Penalty': "Тип регуляризации: l1, l2, none",
                    'Random State': "Фиксация случайности. Для воспроизводимости"
                }.get(param, param)

                btn = QPushButton("?")
                btn.setFixedSize(20, 20)
                btn.clicked.connect(lambda ch, t=param, h=help_text: HelpDialog(t, h, self).exec_())

                param_hbox.addWidget(lbl)
                param_hbox.addWidget(le)
                param_hbox.addWidget(btn)

                group_layout.addWidget(param_widget)
                lines[param] = le

        # Сохраняем ссылку
        self.labels_and_lines[model_name] = lines
        self.checkboxes.append(model_checkbox)
        layout.addWidget(group_box)

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", "./dataset/", "CSV (*.csv)")
        if not path:
            return
        try:
            self.meta_tracker.load_from_file(path)
            # Игнорирование строк, начинающихся с # META:, при загрузке датасета
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line for line in f if not line.strip().startswith('# META:')]
            from io import StringIO
            df = pd.read_csv(StringIO(''.join(lines)), comment='#', skipinitialspace=True)
            self.df = df.copy()
            self.original_path = path
            # self.X_train = self.y_train = None
            self.select_target_variable()
            filename = os.path.basename(path)
            self.load_btn.setText(f"📁 {filename}")
            self.delete_columns_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            # Кнопка Удалить колонки активна только если есть важности признаков
            self.delete_columns_btn.setEnabled(bool(self.feature_importances))
            self.update_memory_usage()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def select_target_variable(self):
        if self.df is None:
            return
        possible_targets = [col for col in self.df.columns]
        if not possible_targets:
            QMessageBox.critical(self, "Ошибка", "Датасет пуст.")
            return
        target, ok = QInputDialog.getItem(self, "Целевая", "Выберите целевую переменную:", sorted(possible_targets), 0, False)
        if not ok or not target:
            return
        df_local = self.df.copy()
        original_dtype = df_local[target].dtype
        if self.task_type == "classification" and df_local[target].dtype == 'object':
            df_local[target] = LabelEncoder().fit_transform(df_local[target])
        X = df_local.drop(columns=[target]).select_dtypes(include=['number'])
        y = df_local[target]
        if X.empty:
            QMessageBox.critical(self, "Ошибка", "Нет числовых признаков.")
            return
        self.X_train, self.y_train = X, y
        self.target_col = target
        self.y_display = self.df[target].copy() if original_dtype == 'object' else self.y_train.copy()
        self.target_label.setText(f"Целевая переменная: {target}")
        #self.analyze_btn.setEnabled(True)
        self.delete_columns_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.train_model_btn.setEnabled(True)
        # Кнопка Удалить колонки активна только если есть важности признаков
        self.delete_columns_btn.setEnabled(bool(self.feature_importances))
        self.update_memory_usage()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()

    def on_analyze(self):
        self.kill_child_processes()
        # Кнопка Удалить колонки активна только если есть важности признаков
        self.delete_columns_btn.setEnabled(bool(self.feature_importances))
        self.update_memory_usage()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
            return
        if not self.target_col:
            QMessageBox.warning(self, "Ошибка", "Целевая переменная не выбрана.")
            return
        selected = {cb.text(): True for cb in self.checkboxes if cb.isChecked()}
        if not selected:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну модель.")
            return
        X_scaled = StandardScaler().fit_transform(self.X_train)
        feature_names = self.X_train.columns.tolist()
        self.feature_importances = {col: [] for col in feature_names}
        for model_name in selected:
            try:
                params = self.labels_and_lines.get(model_name, {})
                clf = self._create_model(model_name, params)
                with parallel_backend('loky', n_jobs=self.safe_int({'n_jobs': self.global_n_jobs}, 'n_jobs', 1)):
                    clf.fit(X_scaled, self.y_train)
                importances = self._get_importances(clf)
                for idx, col in enumerate(feature_names):
                    if col in self.feature_importances:
                        self.feature_importances[col].append(importances[idx])
                idx_sorted = np.argsort(importances)[::-1]
                top_5 = [feature_names[i] for i in idx_sorted[:5]]
                model_group = QGroupBox(f" {model_name} ")
                model_group.setStyleSheet("""
                    QGroupBox {
                        font-weight: bold;
                        border: 1px solid #aaa;
                        border-radius: 6px;
                        margin: 0;
                        padding: 10px;
                        min-width: 240px;
                    }
                """)
                model_layout = QVBoxLayout()
                model_layout.setSpacing(8)
                top_text = QTextEdit()
                top_text.setPlainText(f"ТОП-5:\n" + "\n".join([f"• {f}" for f in top_5]))
                top_text.setFixedHeight(100)
                top_text.setReadOnly(True)
                model_layout.addWidget(top_text)
                param_text = "<br>".join([f"{k}: {v.text().strip()}" for k, v in params.items()])
                params_label = QLabel(f"<small><b>Параметры:</b><br>{param_text}</small>")
                params_label.setWordWrap(True)
                params_label.setStyleSheet("font-size: 14px; color: #777;")
                model_layout.addWidget(params_label)
                plot_btn = QPushButton("📊 График")
                plot_btn.clicked.connect(
                    lambda ch, imp=importances.copy(), names=feature_names.copy(), mn=model_name:
                    self.plot_importance(imp, names, mn)
                )
                model_layout.addWidget(plot_btn)
                model_group.setLayout(model_layout)
                self.results_layout.addWidget(model_group)
                while self.results_layout.count() > 3:
                    item = self.results_layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка в {model_name}:\n{e}")
        # Кнопка Удалить колонки активна только если есть важности признаков
        self.delete_columns_btn.setEnabled(bool(self.feature_importances))
        self.update_memory_usage()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()

    def _create_model(self, name, params):
        random_state = self.safe_int(params, 'Random State', 42)
        n_estimators = self.safe_int(params, 'Кол-во деревьев', 100)
        # Используем глобальные параметры
        random_state = self.safe_int({'Random State': self.global_random_state}, 'Random State', 42)
        n_jobs = self.safe_int({'n_jobs': self.global_n_jobs}, 'n_jobs', 1)
        
        if name == 'Random Forest':
            max_depth = self.safe_int_or_none(params, 'Max Depth', None)
            min_samples_split = self.safe_int(params, 'Min Samples Split', 2)
            if self.task_type == "classification":
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state)
            else:
                return RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state)
        
        elif name == 'Gradient Boosting':
            max_depth = self.safe_int_or_none(params, 'Max Depth', 3)
            learning_rate = self.safe_float(params, 'Learning Rate', 0.1)
            if self.task_type == "classification":
                return GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state)
            else:
                return GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state)
        
        elif name == 'Logistic Regression':
            C = self.safe_float(params, 'C', 1.0)
            max_iter = self.safe_int(params, 'Max Iterations', 100)
            penalty = params.get('Penalty', None)
            penalty = penalty.text().strip() if penalty else 'l2'
            penalty = penalty if penalty in ['l1', 'l2', 'none'] else 'l2'
            solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
            return LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, random_state=random_state)

    def train_selected_model(self):
        """Обучает выбранную модель и передает её в SHAP UI"""
        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет.")
            return

        selected = [cb.property("model_name") for cb in self.checkboxes if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну модель.")
            return
        
        if len(selected) > 1:
            QMessageBox.warning(self, "Ошибка", "Выберите только одну модель для обучения.")
            return
        
        model_name = selected[0]
        params = self.labels_and_lines.get(model_name, {})
        
        try:
            # Создание и обучение модели
            model = self._create_model(model_name, params)
            if model is None:
                QMessageBox.critical(self, "Ошибка", f"Не удалось создать модель: неизвестное имя '{model_name}'")
                return

            X_scaled = StandardScaler().fit_transform(self.X_train)
            
            # Обучение модели
            with parallel_backend('loky', n_jobs=self.safe_int({'n_jobs': self.global_n_jobs}, 'n_jobs', 1)):
                model.fit(X_scaled, self.y_train)
            
            # Передача данных и модели в SHAP UI
            self.shap_ui.set_data(self.df, self.target_col)
            success = self.shap_ui.set_trained_model(model, model_name)
            
            # === ДОБАВЛЕНО: Обновление self.feature_importances ===
            try:
                # Инициализируем словарь, если он ещё не создан
                if not self.feature_importances:
                    self.feature_importances = {col: [] for col in self.X_train.columns}
                
                # Получаем важности признаков
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Для линейных моделей
                    importances = np.abs(model.coef_)
                    if importances.ndim > 1:
                        importances = importances.mean(axis=0)
                    importances = importances.ravel()
                else:
                    raise AttributeError("Модель не поддерживает важность признаков")
                
                # Записываем важности в общий словарь
                for idx, col in enumerate(self.X_train.columns):
                    if col in self.feature_importances:
                        # Заменяем список на одно значение (или можно append, если нужно хранить историю)
                        self.feature_importances[col] = [importances[idx]]
                
                print(f"[DEBUG] Важности признаков обновлены для модели {model_name}")
            except Exception as e_imp:
                print(f"[WARNING] Не удалось получить важность признаков: {e_imp}")
                # Инициализируем нулевыми значениями, если возникла ошибка
                self.feature_importances = {col: [0] for col in self.X_train.columns}
            
            # === КОНЕЦ ДОБАВЛЕНИЯ ===
            
            if success:
                QMessageBox.information(self, "Успех", f"Модель {model_name} обучена и передана в SHAP.")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось передать модель в SHAP.")
                
        except Exception as e:
            error_msg = f"Ошибка при обучении модели {model_name}: {e}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            print(error_msg)
        
        # Кнопка Удалить колонки активна только если есть важности признаков
        self.delete_columns_btn.setEnabled(bool(self.feature_importances))
        self.update_memory_usage()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()
        
        # Обновляем состояние кнопки 'Анализировать' в SHAP UI после успешного обучения
        self.shap_ui.update_button_states()