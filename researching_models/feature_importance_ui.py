# feature_importance_ui.py
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QLineEdit, QDialog,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton, QInputDialog, QScrollArea, QTextEdit, QFrame,
    QGridLayout, QSpacerItem, QSizePolicy
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

        # Если есть данные о важности — сортируем по возрастанию (сначала наименее важные)
        if importances_dict:
            col_importance = {col: sum(importances_dict.get(col, [0])) / len(importances_dict.get(col, [0])) for col in columns}
            sorted_columns = sorted(columns, key=lambda col: col_importance.get(col, 0))  # от низкой к высокой
        else:
            sorted_columns = sorted(columns)

        for idx, col in enumerate(sorted_columns):
            cb = QCheckBox(str(col))
            cb.setChecked(False)  # ✅ Теперь не отмечены по умолчанию
            grid.addWidget(cb, idx, 0)
            self.checkboxes.append(cb)

        # Пустое пространство внизу
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
        self.results_layout = None  # Горизонтальная прокрутка результатов
        self.original_path = None  # Путь к загруженному файлу
        self.meta_tracker = MetaTracker()
        self.feature_importances = {}  # Словарь: {'feature': [imp1, imp2, ...]}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Анализ важности признаков")
        main_layout = QVBoxLayout()

        title_label = QLabel("Анализ важности признаков")
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

        # Кнопка загрузки — только один файл
        self.load_btn = QPushButton("Загрузить датасет")
        self.load_btn.clicked.connect(self.load_dataset)
        main_layout.addWidget(self.load_btn)

        # Целевая переменная
        self.target_label = QLabel("Целевая переменная: не выбрана")
        self.target_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.target_label)

        # === Кнопки удаления и сохранения ===
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

        # Кнопка анализа
        self.analyze_btn = QPushButton("Анализировать важность признаков")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analyze_btn.setEnabled(False)
        main_layout.addWidget(self.analyze_btn)

        # === БЛОК РЕЗУЛЬТАТОВ — горизонтальная прокрутка ===
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

        # === Инициализация ===
        self.setLayout(main_layout)
        self.resize(1000, 850)
        self.create_models()
        self.classification_box.setVisible(self.task_type == "classification")
        self.regression_box.setVisible(self.task_type == "regression")
        self.show()

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
            'Кол-во деревьев': '100', 'Max Depth': 'None', 'Min Samples Split': '2', 'Random State': '42',
            'Learning Rate': '0.1', 'C': '1.0', 'Max Iterations': '100', 'Penalty': 'l2'
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
            hbox.addWidget(lbl)
            hbox.addWidget(le)
            lines[param] = le
        self.labels_and_lines[model_name] = lines
        hbox.addStretch()
        layout.addLayout(hbox)

    def load_dataset(self):
        """Загружаем только один CSV файл"""
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

            # Обновляем текст кнопки
            filename = os.path.basename(path)
            self.load_btn.setText(f"📁 {filename}")

            # Активируем кнопку удаления
            self.delete_columns_btn.setEnabled(True)
            self.save_btn.setEnabled(False)

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

        # Оригинальные значения для display (не используется в обучении)
        if original_dtype == 'object':
            self.y_display = self.df[target].copy()
        else:
            self.y_display = self.y_train.copy()

        self.target_label.setText(f"Целевая переменная: {target}")
        self.analyze_btn.setEnabled(True)
        self.delete_columns_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

    def delete_selected_columns(self):
        """Открывает диалог для удаления колонок — сортировка по важности (от слабых к сильным)"""
        if self.X_train is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет.")
            return

        columns = self.X_train.columns.tolist()

        # Передаём в диалог словарь с важностями (или None, если ещё не было анализа)
        dialog = DeleteColumnsDialog(columns, importances_dict=self.feature_importances, parent=self)
        if dialog.exec() == QDialog.Accepted:
            to_delete = dialog.get_selected_columns()
            if not to_delete:
                return

            to_delete_existing = [col for col in to_delete if col in self.X_train.columns]
            if not to_delete_existing:
                return

            # Удаляем
            self.X_train = self.X_train.drop(columns=to_delete_existing)

            # Обновляем интерфейс
            self.meta_tracker.add_change(f"удалены колонки: {', '.join(to_delete_existing)}")
            self.save_btn.setEnabled(True)

            QMessageBox.information(
                self, "Готово",
                f"Удалены колонки:\n" + "\n".join(to_delete_existing)
            )

    def save_dataset(self):
        if self.X_train is None or len(self.X_train) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения.")
            return

        # Собираем датасет
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

        # Очищаем старые результаты
        while self.results_layout.count() >= 6:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Общие преобразования
        X_scaled = StandardScaler().fit_transform(self.X_train)
        feature_names = self.X_train.columns.tolist()

        # Сбросим словарь важности
        self.feature_importances = {col: [] for col in feature_names}

        for model_name in selected:
            try:
                params = self.labels_and_lines.get(model_name, {})
                clf = self._create_model(model_name, params)
                clf.fit(X_scaled, self.y_train)
                importances = self._get_importances(clf)

                # Записываем важность для каждой колонки
                for idx, col in enumerate(feature_names):
                    if col in self.feature_importances:
                        self.feature_importances[col].append(importances[idx])

                # ТОП-5 признаков
                idx_sorted = np.argsort(importances)[::-1]
                top_5 = [feature_names[i] for i in idx_sorted[:5]]

                # === UI: Блок для модели ===
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

                # ТОП-5
                top_text = QTextEdit()
                top_text.setPlainText(f"ТОП-5:\n" + "\n".join([f"• {f}" for f in top_5]))
                top_text.setFixedHeight(100)
                top_text.setReadOnly(True)
                model_layout.addWidget(top_text)

                # Кнопка графика
                plot_btn = QPushButton("📊 График")
                plot_btn.clicked.connect(
                    lambda ch, imp=importances, names=feature_names, mn=model_name:
                    self.plot_importance(imp, names, mn)
                )
                model_layout.addWidget(plot_btn)

                model_group.setLayout(model_layout)
                self.results_layout.addWidget(model_group)

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка в {model_name}:\n{e}")

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
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values('Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_imp, x='Importance', y='Feature')
        plt.title(f"Важность признаков — {model_name}")
        plt.tight_layout()
        plt.show()

    def safe_int(self, params, key, default):
        try:
            val = params[key].text().strip()
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
