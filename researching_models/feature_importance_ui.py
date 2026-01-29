# feature_importance_ui.py
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QLineEdit, QDialog,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton, QInputDialog, QScrollArea, QTextEdit, QFrame,
    QGridLayout, QSpacerItem, QSizePolicy, QComboBox
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
        self.resize(350, 400)

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
            col_importance = {col: sum(importances_dict.get(col, [0])) / len(importances_dict.get(col, [0])) for col in columns}
            sorted_columns = sorted(columns, key=lambda col: col_importance.get(col, 0))
        else:
            sorted_columns = sorted(columns)

        for idx, col in enumerate(sorted_columns):
            cb = QCheckBox(str(col))
            cb.setChecked(False)
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
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]


class FeatureImportanceUI(QWidget):
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
        self.plot_settings = {}  # Для хранения настроек графика
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Анализ важности признаков")
        main_layout = QVBoxLayout()

        title_label = QLabel("Анализ важности признаков")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)

        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Задача:"))
        self.classification_radio = QRadioButton("Классификация")
        self.regression_radio = QRadioButton("Регрессия")
        self.classification_radio.setChecked(True)
        self.regression_radio.setChecked(False)
        self.task_group = QButtonGroup()
        self.task_group.addButton(self.classification_radio, 1)
        self.task_group.addButton(self.regression_radio, 2)
        self.task_group.buttonClicked.connect(self.on_task_selected)
        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.regression_radio)
        task_layout.addStretch()
        main_layout.addLayout(task_layout)

        self.load_btn = QPushButton("Загрузить датасет")
        self.load_btn.clicked.connect(self.load_dataset)
        main_layout.addWidget(self.load_btn)

        self.target_label = QLabel("Целевая переменная: не выбрана")
        self.target_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.target_label)

        # 🔺 МЕТКА ДЛЯ ОТОБРАЖЕНИЯ ПАМЯТИ
        self.memory_label = QLabel("📊 Память: ? МБ")
        self.memory_label.setStyleSheet("color: #555; font-size: 11px;")
        main_layout.addWidget(self.memory_label)

        btn_layout = QHBoxLayout()

        self.delete_columns_btn = QPushButton("🗑️ Удалить колонки")
        self.delete_columns_btn.clicked.connect(self.delete_selected_columns)
        self.delete_columns_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_columns_btn)

        self.save_btn = QPushButton("💾 Сохранить датасет")
        self.save_btn.clicked.connect(self.save_dataset)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        main_layout.addLayout(btn_layout)

        models_group = QGroupBox("Модели для анализа")
        models_layout = QVBoxLayout()

        self.classification_box = QGroupBox("Классификация")
        self.classification_layout = QVBoxLayout()
        self.classification_box.setLayout(self.classification_layout)
        models_layout.addWidget(self.classification_box)

        self.regression_box = QGroupBox("Регрессия")
        self.regression_layout = QVBoxLayout()
        self.regression_box.setLayout(self.regression_layout)
        models_layout.addWidget(self.regression_box)

        models_group.setLayout(models_layout)
        main_layout.addWidget(models_group)

        # 🔧 Настройки графика
        plot_settings_group = QGroupBox("🎨 Настройки графика")
        plot_layout = QHBoxLayout()

        # Top N Features
        top_n_group = QGroupBox("Кол-во признаков")
        top_n_layout = QHBoxLayout()
        self.top_n_le = QLineEdit("15")
        self.top_n_le.setFixedWidth(50)
        help_top_n = QPushButton("?")
        help_top_n.setFixedSize(20, 20)
        help_top_n.clicked.connect(lambda: HelpDialog(
            "Кол-во признаков",
            "Сколько топ-признаков показывать на графике (по убыванию важности).",
            self
        ).exec_())
        top_n_layout.addWidget(self.top_n_le)
        top_n_layout.addWidget(help_top_n)
        top_n_group.setLayout(top_n_layout)
        plot_layout.addWidget(top_n_group)

        # Sort Order
        sort_group = QGroupBox("Сортировка")
        sort_layout = QHBoxLayout()
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["По убыванию", "По алфавиту", "По исходному порядку"])
        help_sort = QPushButton("?")
        help_sort.setFixedSize(20, 20)
        help_sort.clicked.connect(lambda: HelpDialog(
            "Сортировка",
            "Как отсортировать признаки на графике:\n"
            "• По убыванию — важность (по умолчанию)\n"
            "• По алфавиту — A-Z\n"
            "• По исходному порядку — как в датасете",
            self
        ).exec_())
        sort_layout.addWidget(self.sort_combo)
        sort_layout.addWidget(help_sort)
        sort_group.setLayout(sort_layout)
        plot_layout.addWidget(sort_group)

        plot_settings_group.setLayout(plot_layout)
        main_layout.addWidget(plot_settings_group)

        self.analyze_btn = QPushButton("Анализировать важность признаков")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analyze_btn.setEnabled(False)
        main_layout.addWidget(self.analyze_btn)

        results_group = QGroupBox("📊 Результаты важности признаков")
        results_layout = QVBoxLayout()

        help_label = QLabel(
            "ТОП-5 признаков и кнопка графика.\n"
            "Прокрутите вправо, чтобы увидеть все модели."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("font-size: 11px; color: #555;")
        results_layout.addWidget(help_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        results_layout.addWidget(line)

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
        main_layout.addWidget(results_group)

        self.setLayout(main_layout)
        self.resize(1000, 850)
        self.create_models()
        self.classification_box.setVisible(self.task_type == "classification")
        self.regression_box.setVisible(self.task_type == "regression")
        self.show()

        self.update_memory_usage()
        
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

            self.update_memory_usage()
            
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

        self.update_memory_usage()

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
            'Random Forest Classification': ['Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Random State'],
            'Gradient Boosting Classification': ['Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Random State'],
            'Logistic Regression Classification': ['C', 'Max Iterations', 'Penalty', 'Random State']
        }
        reg_models = {
            'Random Forest Regression': ['Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Random State'],
            'Gradient Boosting Regression': ['Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Random State']
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
        hbox = QHBoxLayout()
        cb = QCheckBox(model_name)
        self.checkboxes.append(cb)
        hbox.addWidget(cb)
        lines = {}

        for param in params:
            lbl = QLabel(param)
            le = QLineEdit()
            le.setFixedWidth(80)
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

            hbox.addWidget(lbl)
            hbox.addWidget(le)
            hbox.addWidget(btn)
            lines[param] = le

        # 🔧 Добавляем n_jobs (по умолчанию = 1)
        n_jobs_lbl = QLabel("n_jobs")
        n_jobs_le = QLineEdit("1")
        n_jobs_le.setFixedWidth(50)
        n_jobs_help = QPushButton("?")
        n_jobs_help.setFixedSize(20, 20)
        n_jobs_help.clicked.connect(lambda: HelpDialog(
            "n_jobs",
            "Количество ядер CPU для параллельных вычислений.\n"
            "1 — последовательно (по умолчанию)\n"
            "-1 — использовать все ядра",
            self
        ).exec_())
        hbox.addWidget(n_jobs_lbl)
        hbox.addWidget(n_jobs_le)
        hbox.addWidget(n_jobs_help)
        lines['n_jobs'] = n_jobs_le

        self.labels_and_lines[model_name] = lines
        hbox.addStretch()
        layout.addLayout(hbox)

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", "./dataset/", "CSV (*.csv)")
        if not path:
            return
        try:
            self.meta_tracker.load_from_file(path)
            df = pd.read_csv(path, comment='#')
            self.df = df.copy()
            self.original_path = path
            self.X_train = self.y_train = None
            self.select_target_variable()
            filename = os.path.basename(path)
            self.load_btn.setText(f"📁 {filename}")
            self.delete_columns_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
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
        self.analyze_btn.setEnabled(True)
        self.delete_columns_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.update_memory_usage()

    def on_analyze(self):
        self.kill_child_processes()
        self.update_memory_usage()
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
                with parallel_backend('loky', n_jobs=self.safe_int(params, 'n_jobs', 1)):
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
        self.update_memory_usage()

    def _create_model(self, name, params):
        random_state = self.safe_int(params, 'Random State', 42)
        n_estimators = self.safe_int(params, 'Кол-во деревьев', 100)
        if 'Random Forest Classification' in name:
            max_depth = self.safe_int_or_none(params, 'Max Depth', None)
            min_samples_split = self.safe_int(params, 'Min Samples Split', 2)
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        elif 'Gradient Boosting Classification' in name:
            max_depth = self.safe_int_or_none(params, 'Max Depth', 3)
            learning_rate = self.safe_float(params, 'Learning Rate', 0.1)
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        elif 'Logistic Regression Classification' in name:
            C = self.safe_float(params, 'C', 1.0)
            max_iter = self.safe_int(params, 'Max Iterations', 100)
            penalty = params.get('Penalty', None)
            penalty = penalty.text().strip() if penalty else 'l2'
            penalty = penalty if penalty in ['l1', 'l2', 'none'] else 'l2'
            solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
            return LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, random_state=random_state)
        elif 'Random Forest Regression' in name:
            max_depth = self.safe_int_or_none(params, 'Max Depth', None)
            min_samples_split = self.safe_int(params, 'Min Samples Split', 2)
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        elif 'Gradient Boosting Regression' in name:
            max_depth = self.safe_int_or_none(params, 'Max Depth', 3)
            learning_rate = self.safe_float(params, 'Learning Rate', 0.1)
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            raise ValueError(f"Неизвестная модель: {name}")

    def _get_importances(self, clf):
        if hasattr(clf, 'feature_importances_'):
            return clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            coef = np.abs(clf.coef_)
            return coef.mean(axis=0) if coef.ndim > 1 else coef.ravel()
        else:
            raise AttributeError("Модель не поддерживает важность признаков")

    def plot_importance(self, importances, feature_names, model_name):
        self.kill_child_processes()
        top_n = self.safe_int({}, 'top_n', int(self.top_n_le.text()) if self.top_n_le.text().strip().isdigit() else 15)
        sort_mode = self.sort_combo.currentText()
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        if sort_mode == "По убыванию":
            df_imp = df_imp.sort_values('Importance', ascending=False)
        elif sort_mode == "По алфавиту":
            df_imp = df_imp.sort_values('Feature', ascending=True)
        elif sort_mode == "По исходному порядку":
            df_imp['Original Order'] = range(len(df_imp))
            df_imp = df_imp.sort_values('Original Order', ascending=True)
        df_imp = df_imp.head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_imp, x='Importance', y='Feature')
        plt.title(f"Важность признаков — {model_name}")
        plt.tight_layout()
        plt.show()
        self.update_memory_usage()

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

    def closeEvent(self, event):
        self.kill_child_processes()
        plt.close('all')
        self.df = None
        self.X_train = None
        self.y_train = None
        self.feature_importances = {}
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        gc.collect()
        self.update_memory_usage()
        super().closeEvent(event)
