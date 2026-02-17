# model_evaluation_ui.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton, QInputDialog, QScrollArea
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import os
import pandas as pd
import gc  # Для принудительной очистки памяти
from .model_evaluation_logic import ModelEvaluator


class ModelEvaluationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_file_name = ""
        self.checkboxes = []
        self.labels_and_lines = {}
        self.selected_task = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results_layout = None 
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        title_label = QLabel('Оценка моделей — Классификация и Регрессия')
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)

        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Тип задачи:"))
        self.classification_radio = QRadioButton("Классификация")
        self.regression_radio = QRadioButton("Регрессия")
        self.classification_radio.setChecked(False)
        self.regression_radio.setChecked(False)

        self.task_group = QButtonGroup()
        self.task_group.addButton(self.classification_radio, 1)
        self.task_group.addButton(self.regression_radio, 2)
        self.task_group.buttonClicked.connect(self.on_task_selected)

        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.regression_radio)
        task_layout.addStretch()
        main_layout.addLayout(task_layout)

        self.select_dataset_btn = QPushButton("Выбрать датасет")
        self.select_dataset_btn.clicked.connect(self.on_select_dataset_clicked)
        self.select_dataset_btn.setEnabled(False)
        main_layout.addWidget(self.select_dataset_btn)

        # Переместили метку целевой переменной и памяти сюда, сразу после строки выбора датасета
        target_memory_layout = QHBoxLayout()
        self.target_label = QLabel("Целевая переменная: не выбрана")
        self.target_label.setStyleSheet("font-weight: bold;")
        self.memory_label = QLabel("📊 Память: ? МБ")
        self.memory_label.setStyleSheet("color: #555; font-size: 11px;")
        target_memory_layout.addWidget(self.target_label)
        target_memory_layout.addWidget(self.memory_label)
        target_memory_layout.addStretch() 
        main_layout.addLayout(target_memory_layout)

        models_group = QGroupBox("Модели для оценки")
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

        self.evaluate_models_btn = QPushButton('Оценить выбранные модели')
        self.evaluate_models_btn.clicked.connect(self.on_evaluate_models_clicked)
        self.evaluate_models_btn.setEnabled(False)
        main_layout.addWidget(self.evaluate_models_btn)

        results_group = QGroupBox("Результаты оценки моделей")
        results_layout = QVBoxLayout()

        self.results_layout = QHBoxLayout()
        scroll_content = QWidget()
        scroll_content.setLayout(self.results_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        scroll.setFixedHeight(250)
        results_layout.addWidget(scroll)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        self.evaluator = ModelEvaluator(
            parent=self,
            checkboxes=self.checkboxes,
            labels_and_lines=self.labels_and_lines,
            results_layout=self.results_layout,
            task_type="classification"
        )

        self.create_classification_models()
        self.create_regression_models()

        self.classification_box.setVisible(False)
        self.regression_box.setVisible(False)

        self.setLayout(main_layout)
        self.resize(1000, 850)
        self.setWindowTitle("Оценка моделей")
        self.show()

    def on_task_selected(self):
        if self.classification_radio.isChecked():
            self.selected_task = "classification"
        elif self.regression_radio.isChecked():
            self.selected_task = "regression"
        else:
            return
        self.evaluator.task_type = self.selected_task
        self.classification_box.setVisible(self.selected_task == "classification")
        self.regression_box.setVisible(self.selected_task == "regression")
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
        self.select_dataset_btn.setEnabled(True)
        self.evaluate_models_btn.setEnabled(True)

    def create_classification_models(self):
        models = {
            'Random Forest Classification': ['Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Test Size', 'Random State'],
            'Gradient Boosting Classification': ['Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Test Size', 'Random State'],
            'Logistic Regression Classification': ['C', 'Max Iterations', 'Penalty', 'Test Size', 'Random State']
        }
        defaults = {
            'Кол-во деревьев': '100', 'Max Depth': 'None', 'Min Samples Split': '2', 'Test Size': '0.2', 'Random State': '42',
            'Learning Rate': '0.1', 'C': '1.0', 'Max Iterations': '100', 'Penalty': 'l2'
        }
        for model_name, params in models.items():
            hbox = QHBoxLayout()
            cb = QCheckBox(model_name)
            self.checkboxes.append(cb)
            hbox.addWidget(cb)
            lines = {}
            for param_name in params:
                lbl = QLabel(param_name)
                le = QLineEdit()
                le.setFixedWidth(80)
                le.setText(defaults[param_name])
                hbox.addWidget(lbl)
                hbox.addWidget(le)
                lines[param_name] = le
            self.labels_and_lines[model_name] = lines
            self.classification_layout.addLayout(hbox)
        self.classification_box.setLayout(self.classification_layout)

    def create_regression_models(self):
        models = {
            'Random Forest Regression': ['Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Test Size', 'Random State'],
            'Gradient Boosting Regression': ['Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Test Size', 'Random State']
        }
        defaults = {
            'Кол-во деревьев': '100', 'Max Depth': 'None', 'Min Samples Split': '2', 'Test Size': '0.2', 'Random State': '42',
            'Learning Rate': '0.1'
        }
        for model_name, params in models.items():
            hbox = QHBoxLayout()
            cb = QCheckBox(model_name)
            self.checkboxes.append(cb)
            hbox.addWidget(cb)
            lines = {}
            for param_name in params:
                lbl = QLabel(param_name)
                le = QLineEdit()
                le.setFixedWidth(80)
                le.setText(defaults[param_name])
                hbox.addWidget(lbl)
                hbox.addWidget(le)
                lines[param_name] = le
            self.labels_and_lines[model_name] = lines
            self.regression_layout.addLayout(hbox)
        self.regression_box.setLayout(self.regression_layout)

    def on_select_dataset_clicked(self):
        reply = QMessageBox.question(
            self, "Режим загрузки",
            "Разделить датасет на train и test?\n\n"
            "• Да → загрузить train и test отдельно\n"
            "• Нет → загрузить один датасет, разделю при обучении",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.load_separate_datasets()
        else:
            self.load_single_dataset()

    def load_single_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл CSV", "./dataset/", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path, comment='#')
            self.df = df
            self.dataset_file_name = os.path.basename(file_path)
            self.select_dataset_btn.setText(f"📁 {self.dataset_file_name}")
            self.X_train = self.X_test = self.y_train = self.y_test = None
            self.disable_test_size_fields(disable=False)
            self.select_target_variable()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def load_separate_datasets(self):
        train_path, _ = QFileDialog.getOpenFileName(self, "Выберите train-файл", "./dataset/", "CSV Files (*.csv)")
        if not train_path: return
        test_path, _ = QFileDialog.getOpenFileName(self, "Выберите test-файл", "./dataset/", "CSV Files (*.csv)")
        if not test_path: return

        try:
            df_train = pd.read_csv(train_path, comment='#')
            df_test = pd.read_csv(test_path, comment='#')

            common_cols = set(df_train.columns) & set(df_test.columns)
            if not common_cols:
                QMessageBox.critical(self, "Ошибка", "Нет общих колонок между train и test!")
                return

            possible_targets = [col for col in common_cols
                if col not in ['index', 'id', 'Id', 'ID', 'Index'] and
                df_train[col].nunique() < len(df_train) * 0.9]

            if not possible_targets:
                possible_targets = list(common_cols)

            target, ok = QInputDialog.getItem(
                self, "Целевая переменная", "Выберите целевую переменную:", sorted(possible_targets), 0, False)
            if not ok or not target:
                QMessageBox.warning(self, "Отмена", "Целевая переменная не выбрана.")
                return

            if target not in df_train.columns or target not in df_test.columns:
                QMessageBox.critical(self, "Ошибка", f"Колонка '{target}' отсутствует в одном из файлов.")
                return

            X_train = df_train.drop(columns=[target])
            X_test = df_test.drop(columns=[target])
            y_train = df_train[target]
            y_test = df_test[target]

            self.evaluator.set_split_data(X_train, X_test, y_train, y_test, target)
            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
            self.df = None

            self.target_label.setText(f"Целевая переменная: {target}")
            self.select_dataset_btn.setText(f"📁 train: {os.path.basename(train_path)}\n   test: {os.path.basename(test_path)}")
            self.disable_test_size_fields(disable=True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файлы:\n{e}")

    def disable_test_size_fields(self, disable=True):
        """Блокировка Test Size и Random State при загрузке двух датасетов"""
        for model_name, lines in self.labels_and_lines.items():
            if 'Test Size' in lines:
                lines['Test Size'].setEnabled(not disable)
            if 'Random State' in lines:
                lines['Random State'].setEnabled(not disable)

    def select_target_variable(self):
        if self.df is None:
            return
        possible_targets = [col for col in self.df.columns if self.df[col].dtype != 'object']
        if len(possible_targets) == 0:
            possible_targets = self.df.columns.tolist()

        target, ok = QInputDialog.getItem(self, "Целевая переменная", "Выберите целевую переменную:", sorted(possible_targets), 0, False)
        if not ok or not target:
            QMessageBox.warning(self, "Отмена", "Целевая переменная не выбрана.")
            return

        self.target_label.setText(f"Целевая переменная: {target}")
        self.evaluator.update_dataframe(self.df, target)

    def on_evaluate_models_clicked(self):
        if self.df is None and (self.X_train is None or self.y_train is None):
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите датасет!")
            return

        if self.evaluator.target_col is None:
            QMessageBox.warning(self, "Предупреждение", "Выберите целевую переменную!")
            return

        self.evaluator.task_type = self.selected_task
        self.evaluator.evaluate_models()

    # ✅ НОВЫЙ МЕТОД: Очистка при закрытии окна
    def closeEvent(self, event):
        """Очищает данные и ресурсы при закрытии окна"""
        # Очищаем данные
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Очищаем результаты
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Удаляем evaluator, если он содержит ссылки
        if hasattr(self, 'evaluator'):
            self.evaluator = None

        # Принудительная сборка мусора
        gc.collect()

        super().closeEvent(event)
