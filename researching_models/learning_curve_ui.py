# learning_curve_ui.py — кривые обучения + финальная оценка на test

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QInputDialog,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton, QLineEdit, QScrollArea, QDialog, QFrame
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
import psutil  
from joblib import parallel_backend  

class HelpDialog(QDialog):
    """Модальное окно с пояснением метрик или параметров"""
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.setModal(True)
        self.resize(400, 300)

        layout = QVBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)

        text_label = QLabel(text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
        self.setLayout(layout)
class LearningCurveUI(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.target_col = None
        self.checkboxes = []
        self.labels_and_lines = {}
        self.task_type = "classification"
        self.results_layout = None
        self.curve_params = {}
        self.process = psutil.Process(os.getpid())
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Кривые обучения")
        main_layout = QVBoxLayout()

        title_label = QLabel("Построение кривых обучения")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)

        # Тип задачи
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

        # Кнопка загрузки
        self.load_btn = QPushButton("Загрузить датасет")
        self.load_btn.clicked.connect(self.on_load_dataset)
        main_layout.addWidget(self.load_btn)

        # Целевая переменная
        self.target_label = QLabel("Целевая переменная: не выбрана")
        self.target_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.target_label)

        # 🔺 МЕТКА ДЛЯ ОТОБРАЖЕНИЯ ПАМЯТИ
        self.memory_label = QLabel("📊 Память: ? МБ")
        self.memory_label.setStyleSheet("color: #555; font-size: 11px;")
        main_layout.addWidget(self.memory_label)

        # Модели
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

        # === ПАРАМЕТРЫ КРИВОЙ ОБУЧЕНИЯ — ГОРИЗОНТАЛЬНО ===
        curve_group = QGroupBox("⚙️ Параметры кривой обучения")
        curve_layout = QHBoxLayout()
        curve_layout.setSpacing(15)
        curve_layout.setContentsMargins(10, 10, 10, 10)

        params = [
            ("CV", "5", "Количество фолдов кросс-валидации.\n\n• 5 — баланс\n• Меньше → быстрее, менее стабильно"),
            ("n_jobs", "1", "Число процессов.\n\n• -1 = все ядра\n• 1 = один поток (стабильнее)"),
            ("Число точек", "10", "Сколько точек на графике.\n\n• 10 — оптимально\n• Больше → точнее, но дольше"),
            ("Random State", "42", "Контроль случайности.\n\n• Фиксированное значение → воспроизводимость")
        ]

        for label_text, default_value, help_text in params:
            group_box = QGroupBox(label_text)
            group_box.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    padding: 8px;
                    margin-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
            """)
            group_layout = QHBoxLayout()
            group_layout.setContentsMargins(5, 25, 5, 5)

            le = QLineEdit(default_value)
            le.setFixedWidth(60)

            btn = QPushButton("?")
            btn.setFixedSize(20, 20)
            btn.clicked.connect(lambda ch, t=label_text, h=help_text: HelpDialog(t, h, self).exec_())

            group_layout.addWidget(le)
            group_layout.addWidget(btn)
            group_box.setLayout(group_layout)

            if label_text == "CV":
                self.curve_params['cv'] = le
            elif label_text == "n_jobs":
                self.curve_params['n_jobs'] = le
            elif label_text == "Число точек":
                self.curve_params['train_points'] = le
            elif label_text == "Random State":
                self.curve_params['random_state'] = le

            curve_layout.addWidget(group_box)

        curve_group.setLayout(curve_layout)
        main_layout.addWidget(curve_group)

        # Кнопка анализа
        self.analyze_btn = QPushButton("Построить кривые обучения")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analyze_btn.setEnabled(False)
        main_layout.addWidget(self.analyze_btn)

        # === БЛОК РЕЗУЛЬТАТОВ ===
        results_group = QGroupBox("📊 Результаты кривой обучения (история)")
        results_layout = QVBoxLayout()

        help_label = QLabel(
            "Результаты моделей и финальный тест.\n"
            "Прокрутите вправо, чтобы увидеть все."
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

        # === Инициализация ===
        self.setLayout(main_layout)
        self.resize(1000, 850)
        self.show()

        self.create_models()
        self.classification_box.setVisible(self.task_type == "classification")
        self.regression_box.setVisible(self.task_type == "regression")

        # 🔺 Обновляем память при старте
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
        """Обновляет метку с текущим использованием ОЗУ"""
        try:
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024  # в МБ
            self.memory_label.setText(f"📊 Память: {mem_mb:.1f} МБ")
        except Exception as e:
            self.memory_label.setText("📊 Память: ошибка")

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
                'Test Size': "Доля данных, выделяемая на тест.\nНапример, 0.2 = 20%",
                'Кол-во деревьев': "Число деревьев в ансамбле.\nБольше → точнее, но дольше",
                'Max Depth': "Максимальная глубина одного дерева.\nNone = не ограничена\nБольше → риск переобучения",
                'Min Samples Split': "Минимальное число образцов, чтобы разделить узел.\nБольше → проще модель",
                'Learning Rate': "Скорость обучения в Gradient Boosting.\nМеньше → стабильнее, но медленнее",
                'C': "Сила регуляризации в Logistic Regression.\nБольше → слабее регуляризация",
                'Max Iterations': "Максимальное число итераций обучения.\nУвеличьте, если модель не сходится",
                'Penalty': "Тип регуляризации: l1, l2, elasticnet, none",
                'Random State': "Фиксация случайных процессов для воспроизводимости результатов"
            }.get(param, param)

            btn = QPushButton("?")
            btn.setFixedSize(20, 20)
            btn.clicked.connect(lambda ch, t=param, h=help_text: HelpDialog(t, h, self).exec_())

            hbox.addWidget(lbl)
            hbox.addWidget(le)
            hbox.addWidget(btn)
            lines[param] = le
        self.labels_and_lines[model_name] = lines
        hbox.addStretch()
        layout.addLayout(hbox)

    def create_models(self):
        clf_models = {
            'Random Forest Classification': ['Test Size', 'Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Random State'],
            'Gradient Boosting Classification': ['Test Size', 'Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Random State'],
            'Logistic Regression Classification': ['Test Size', 'C', 'Max Iterations', 'Penalty', 'Random State']
        }
        reg_models = {
            'Random Forest Regression': ['Test Size', 'Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Random State'],
            'Gradient Boosting Regression': ['Test Size', 'Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Random State']
        }
        defaults = {
            'Test Size': '0.2', 'Кол-во деревьев': '100', 'Max Depth': 'None', 'Min Samples Split': '2',
            'Random State': '42', 'Learning Rate': '0.1', 'C': '1.0', 'Max Iterations': '100', 'Penalty': 'l2'
        }

        for model_name, params in clf_models.items():
            self._add_model_to_layout(model_name, params, defaults, self.classification_layout)
        for model_name, params in reg_models.items():
            self._add_model_to_layout(model_name, params, defaults, self.regression_layout)

    def on_task_selected(self):
        self.task_type = "classification" if self.classification_radio.isChecked() else "regression"
        self.classification_box.setVisible(self.task_type == "classification")
        self.regression_box.setVisible(self.task_type == "regression")

    def on_load_dataset(self):
        reply = QMessageBox.question(
            self, "Режим загрузки",
            "Разделить датасет на train/test?\n"
            "• Да — загрузить два файла\n"
            "• Нет — один файл, разделю автоматически",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.load_separate_datasets()
        else:
            self.load_single_dataset()

    def load_single_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", "./dataset/", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path, comment='#')
            self.df = df
            self.X_train = self.y_train = self.X_test = self.y_test = None
            self.select_target_variable()

            for lines in self.labels_and_lines.values():
                if 'Test Size' in lines:
                    lines['Test Size'].setEnabled(True)
                if 'Random State' in lines:
                    lines['Random State'].setEnabled(True)

            self.update_memory_usage()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не загружен:\n{e}")

    def load_separate_datasets(self):
        train_path, _ = QFileDialog.getOpenFileName(self, "Train", "./dataset/", "CSV (*.csv)")
        if not train_path:
            return
        test_path, _ = QFileDialog.getOpenFileName(self, "Test", "./dataset/", "CSV (*.csv)")
        if not test_path:
            return
        try:
            df_train = pd.read_csv(train_path, comment='#')
            df_test = pd.read_csv(test_path, comment='#')

            common_cols = set(df_train.columns) & set(df_test.columns)
            if not common_cols:
                QMessageBox.critical(self, "Ошибка", "Нет общих колонок!")
                return

            possible_targets = [col for col in common_cols if df_train[col].nunique() < 0.9 * len(df_train)]
            if not possible_targets:
                possible_targets = list(common_cols)

            target, ok = QInputDialog.getItem(self, "Целевая", "Выберите:", sorted(possible_targets), 0, False)
            if not ok or not target:
                return

            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]

            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test
            self.df = None
            self.target_col = target
            self.target_label.setText(f"Целевая переменная: {target}")
            self.analyze_btn.setEnabled(True)

            train_name = os.path.basename(train_path)
            test_name = os.path.basename(test_path)
            self.load_btn.setText(f"📁 train: {train_name}\n   test: {test_name}")

            for lines in self.labels_and_lines.values():
                if 'Test Size' in lines:
                    lines['Test Size'].setEnabled(False)
                if 'Random State' in lines:
                    lines['Random State'].setEnabled(False)

            self.update_memory_usage()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки:\n{e}")

    def select_target_variable(self):
        if self.df is None:
            return
        possible_targets = [col for col in self.df.columns]
        target, ok = QInputDialog.getItem(self, "Целевая", "Выберите:", sorted(possible_targets), 0, False)
        if not ok or not target:
            return

        df_local = self.df.copy()
        if self.task_type == "classification" and df_local[target].dtype == 'object':
            df_local[target] = LabelEncoder().fit_transform(df_local[target])

        X = df_local.drop(columns=[target]).select_dtypes(include=['number'])
        y = df_local[target]

        if X.empty:
            QMessageBox.critical(self, "Ошибка", "Нет числовых признаков.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.target_col = target
        self.target_label.setText(f"Целевая переменная: {target}")
        self.analyze_btn.setEnabled(True)

        for lines in self.labels_and_lines.values():
            if 'Test Size' in lines:
                lines['Test Size'].setEnabled(True)
            if 'Random State' in lines:
                lines['Random State'].setEnabled(True)

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

        cv = self.safe_int(self.curve_params, 'cv', 5)
        n_jobs = self.safe_int(self.curve_params, 'n_jobs', 1)
        n_points = self.safe_int(self.curve_params, 'train_points', 10)
        curve_random_state = self.safe_int(self.curve_params, 'random_state', 42)

        for model_name in selected:
            try:
                params = self.labels_and_lines.get(model_name, {})
                clf = self._create_model(model_name, params)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(self.X_train)
                X_test_scaled = scaler.transform(self.X_test)

                scoring = 'accuracy' if 'Classification' in model_name else 'r2'

                with parallel_backend('loky', n_jobs=n_jobs):
                    train_sizes, train_scores, val_scores = learning_curve(
                        clf, X_train_scaled, self.y_train,
                        train_sizes=np.linspace(0.1, 1.0, n_points),
                        cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=curve_random_state
                    )

                train_mean = np.mean(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                final_val = val_mean[-1]
                gap = train_mean[-1] - final_val

                clf.fit(X_train_scaled, self.y_train)
                final_test_score = clf.score(X_test_scaled, self.y_test)

                # === UI: Отображение результатов ===
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

                # Val Final
                row1 = QHBoxLayout()
                lbl1 = QLabel(f"Val Final: {final_val:.4f}")
                btn1 = QPushButton("?")
                btn1.setFixedSize(20, 20)
                btn1.clicked.connect(lambda: HelpDialog(
                    "Финал валидации",
                    "Значение на кривой обучения при 100% данных.\n"
                    "Показывает, как модель обобщает.", self).exec_())
                row1.addWidget(lbl1)
                row1.addWidget(btn1)
                model_layout.addLayout(row1)

                # Gap
                row2 = QHBoxLayout()
                lbl2 = QLabel(f"Gap: {gap:.4f}")
                btn2 = QPushButton("?")
                btn2.setFixedSize(20, 20)
                btn2.clicked.connect(lambda: HelpDialog(
                    "Переобучение",
                    "Разница между обучением и валидацией.\n"
                    "❌ > 0.1 — признак переобучения.", self).exec_())
                row2.addWidget(lbl2)
                row2.addWidget(btn2)
                model_layout.addLayout(row2)

                # Test
                row3 = QHBoxLayout()
                lbl3 = QLabel(f"Test: {final_test_score:.4f}")
                btn3 = QPushButton("?")
                btn3.setFixedSize(20, 20)
                btn3.clicked.connect(lambda: HelpDialog(
                    "Финальный тест",
                    "Оценка на независимом тестовом наборе.\n"
                    "Настоящая проверка обобщающей способности.", self).exec_())
                row3.addWidget(lbl3)
                row3.addWidget(btn3)
                model_layout.addLayout(row3)

                # 🔺 Параметры модел��
                param_text = "<br>".join([f"{k}: {v.text().strip()}" for k, v in params.items()])
                params_label = QLabel(f"<small><b>Параметры:</b><br>{param_text}</small>")
                params_label.setWordWrap(True)
                params_label.setStyleSheet("font-size: 14px; color: #777;")
                model_layout.addWidget(params_label)

                # Кнопка графика
                plot_btn = QPushButton("📈 График")
                plot_btn.clicked.connect(
                    lambda ch, sizes=train_sizes.copy(),
                                    t_mean=train_mean.copy(),
                                    v_mean=val_mean.copy(),
                                    mn=model_name,
                                    s=scoring:
                        self.plot_curve(sizes, t_mean, v_mean, mn, s)
                )
                model_layout.addWidget(plot_btn)

                model_group.setLayout(model_layout)
                self.results_layout.addWidget(model_group)

                # 🔺 Логика: максимум 3 блока, удаляем самый левый
                while self.results_layout.count() > 3:
                    item = self.results_layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при анализе {model_name}:\n{e}")

        self.update_memory_usage()

    def plot_curve(self, train_sizes, train_mean, val_mean, model_name, scoring):
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Обучение')
        plt.plot(train_sizes, val_mean, 'o-', color='green', label='Валидация')
        plt.xlabel('Размер обучающей выборки')
        plt.ylabel(scoring.capitalize())
        plt.title(f"Кривая обучения — {model_name}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.kill_child_processes()
        plt.show()

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

    def safe_int(self, container, key, default):
        try:
            val = container[key].text().strip()
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
        self.X_test = None
        self.y_test = None
        self.clear_results()
        gc.collect()
        super().closeEvent(event)