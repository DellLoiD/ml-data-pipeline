# cross_validation_ui.py — анализ кросс-валидации моделей (с финальной проверкой на test)

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
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


class CrossValidationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None  # ✅ Добавлено
        self.y_test = None  # ✅ Добавлено
        self.target_col = None
        self.checkboxes = []
        self.labels_and_lines = {}
        self.task_type = "classification"
        self.results_layout = None
        self.cv_params = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Кросс-валидация моделей")
        main_layout = QVBoxLayout()

        title_label = QLabel("Анализ кросс-валидации")
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

        # === ПАРАМЕТРЫ КРОСС-ВАЛИДАЦИИ ===
        cv_group = QGroupBox("⚙️ Параметры кросс-валидации")
        cv_layout = QHBoxLayout()
        cv_layout.setSpacing(15)
        cv_layout.setContentsMargins(10, 10, 10, 10)

        params = [
            ("CV", "5", "Количество фолдов.\n\n"
                      "• 5 — стандарт\n• 3 → быстрее, менее надёжно\n• 10 → точнее, но дольше"),
            ("n_jobs", "-1", "Параллельные процессы.\n\n"
                            "• -1 = все ядра\n• 1 = один поток (стабильнее при ошибках)"),
            ("Random State", "42", "Контроль воспроизводимости.\n\n"
                                  "• Фиксировано → одинаковые фолды")
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
                self.cv_params['cv'] = le
            elif label_text == "n_jobs":
                self.cv_params['n_jobs'] = le
            elif label_text == "Random State":
                self.cv_params['random_state'] = le

            cv_layout.addWidget(group_box)

        cv_group.setLayout(cv_layout)
        main_layout.addWidget(cv_group)

        # Кнопка анализа
        self.analyze_btn = QPushButton("Построить кросс-валидацию")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analyze_btn.setEnabled(False)
        main_layout.addWidget(self.analyze_btn)

        # === БЛОК РЕЗУЛЬТАТОВ ===
        results_group = QGroupBox("📊 Результаты кросс-валидации")
        results_layout = QVBoxLayout()

        help_label = QLabel(
            "Результаты CV и финальный тест.\n"
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

        # === Создаём модели и устанавливаем видимость ===
        self.create_models()
        self.classification_box.setVisible(self.task_type == "classification")
        self.regression_box.setVisible(self.task_type == "regression")

        self.setLayout(main_layout)
        self.resize(1000, 850)
        self.show()

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

            if param == 'Test Size':
                help_text = "Доля данных для теста. Игнорируется при CV, если загружены train/test"
            elif param == 'Кол-во деревьев':
                help_text = "Число деревьев в ансамбле. Больше → точнее, но дольше"
            elif param == 'Max Depth':
                help_text = "Макс. глубина дерева. None — без ограничений. Большое → переобучение"
            elif param == 'Min Samples Split':
                help_text = "Мин. число объектов для разбиения узла. Больше → проще модель"
            elif param == 'Learning Rate':
                help_text = "Скорость обучения в GB. Меньше → стабильнее, но медленнее"
            elif param == 'C':
                help_text = "Сила регуляризации в Logistic Regression. Больше → слабее регуляризация"
            elif param == 'Max Iterations':
                help_text = "Макс. итераций обучения. Увеличьте, если модель не сходится"
            elif param == 'Penalty':
                help_text = "Тип регуляризации: l1, l2, elasticnet, none"
            elif param == 'Random State':
                help_text = "Фиксация случайности. Для воспроизводимости"
            else:
                help_text = param

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
            'Test Size': '0.2', 'Кол-во деревьев': '100', 'Max Depth': 'None', 'Min Samples Split': '2', 'Random State': '42',
            'Learning Rate': '0.1', 'C': '1.0', 'Max Iterations': '100', 'Penalty': 'l2'
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
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Датасет не загружен:\n{e}")

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

            # Сохраняем и обучаемые, и тестовые данные
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
        self.X_test, self.y_test = X_test, y_test  # ✅ Сохраняем test
        self.target_col = target
        self.target_label.setText(f"Целевая переменная: {target}")
        self.analyze_btn.setEnabled(True)

        for lines in self.labels_and_lines.values():
            if 'Test Size' in lines:
                lines['Test Size'].setEnabled(True)
            if 'Random State' in lines:
                lines['Random State'].setEnabled(True)

    def on_analyze(self):
        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
            return
        if not self.target_col:
            QMessageBox.warning(self, "Ошибка", "Целевая переменная не выбрана.")
            return

        selected = {}
        for cb in self.checkboxes:
            if cb.isChecked():
                selected[cb.text()] = True

        if not selected:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну модель.")
            return

        cv = self.safe_int(self.cv_params, 'cv', 5)
        n_jobs = self.safe_int(self.cv_params, 'n_jobs', -1)
        random_state = self.safe_int(self.cv_params, 'random_state', 42)

        # Удаляем старые результаты (максимум 6)
        while self.results_layout.count() >= 6:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for model_name in selected:
            try:
                params = self.labels_and_lines.get(model_name, {})
                model = self._create_model(model_name, params)

                # Подготовка данных
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(self.X_train)
                X_test_scaled = scaler.transform(self.X_test)  # ✅ Трансформируем test

                scoring = 'accuracy' if 'Classification' in model_name else 'r2'

                # Кросс-валидация на train
                scores = cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
                cv_mean = np.mean(scores)
                cv_std = np.std(scores)

                # Обучение на всём train и оценка на test
                model.fit(X_train_scaled, self.y_train)
                final_score = model.score(X_test_scaled, self.y_test)

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

                # CV среднее
                row1 = QHBoxLayout()
                lbl1 = QLabel(f"CV среднее: {cv_mean:.4f}")
                btn1 = QPushButton("?")
                btn1.setFixedSize(20, 20)
                btn1.clicked.connect(lambda: HelpDialog(
                    "Кросс-валидация",
                    f"Средняя метрика по {cv} фолдам.\n"
                    "Оценка стабильности и обобщающей способности.", self).exec_())
                row1.addWidget(lbl1)
                row1.addWidget(btn1)
                model_layout.addLayout(row1)

                # CV std
                row2 = QHBoxLayout()
                lbl2 = QLabel(f"CV std: ±{cv_std:.4f}")
                btn2 = QPushButton("?")
                btn2.setFixedSize(20, 20)
                btn2.clicked.connect(lambda: HelpDialog(
                    "Стандартное отклонение",
                    "Разброс по фолдам. Малый std — стабильная модель.", self).exec_())
                row2.addWidget(lbl2)
                row2.addWidget(btn2)
                model_layout.addLayout(row2)

                # Final test
                row3 = QHBoxLayout()
                lbl3 = QLabel(f"Final Test: {final_score:.4f}")
                btn3 = QPushButton("?")
                btn3.setFixedSize(20, 20)
                btn3.clicked.connect(lambda: HelpDialog(
                    "Финальный тест",
                    "Оценка на независимом test датасете.\n"
                    "Настоящая проверка обобщения.", self).exec_())
                row3.addWidget(lbl3)
                row3.addWidget(btn3)
                model_layout.addLayout(row3)

                # Кнопка графика
                plot_btn = QPushButton("📊 График CV")
                plot_btn.clicked.connect(
                    lambda ch, s=scores, mn=model_name, sc=scoring, rs=random_state:
                    self.plot_cv_scores(s, mn, sc, rs)
                )
                model_layout.addWidget(plot_btn)

                model_group.setLayout(model_layout)
                self.results_layout.addWidget(model_group)

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при анализе {model_name}:\n{e}")

    def plot_cv_scores(self, scores, model_name, scoring, random_state):
        folds = np.arange(1, len(scores) + 1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        plt.figure(figsize=(8, 5))
        plt.bar(folds, scores, color='skyblue', edgecolor='black', alpha=0.7, label='Оценка фолда')
        plt.axhline(mean_score, color='red', linestyle='--', label=f'Среднее: {mean_score:.4f}')
        plt.fill_between(folds, mean_score - std_score, mean_score + std_score,
                         color='orange', alpha=0.2, label=f'±std ({std_score:.4f})')

        for i, score in enumerate(scores):
            plt.text(i + 1, score + 0.005 * (max(scores) - min(scores)), f"{score:.3f}",
                     ha='center', fontsize=9, color='darkblue')

        plt.xlabel('Фолд')
        plt.ylabel(scoring.capitalize())
        plt.title(f"Кросс-валидация — {model_name}")
        plt.xticks(folds)
        plt.legend(loc='best')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
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
            if isinstance(container, dict) and key in container:
                val = container[key].text().strip()
            else:
                val = container.text().strip()
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
