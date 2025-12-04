# check_models_ui.py
import sys
from .check_models_logic import DataModelHandler
import pandas as pd
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QCheckBox, QFileDialog, QMessageBox, QGroupBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

class ClassificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_file_name = ""
        self.init_ui()

        # Инициализация обработчика данных
        self.data_handler = DataModelHandler(
            parent=self,
            df=None,
            combobox=self.target_var_combobox,
            checkboxes=self.checkboxes,
            labels_and_lines=self.labels_and_lines,
            accuracy_label=self.metrics_container,
            time_label=self.time_label
        )

    def init_ui(self):
        main_layout = QVBoxLayout()
        title_label = QLabel('Выбор модели и обучение')
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)
        # 1. Выбор датасета
        self.select_dataset_btn = QPushButton("Выбрать датасет")
        self.select_dataset_btn.clicked.connect(self.on_select_dataset_clicked)
        main_layout.addWidget(self.select_dataset_btn)
        # 2. Выбор целевой переменной
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Целевая переменная:"))
        self.target_var_combobox = QComboBox()
        self.target_var_combobox.setEnabled(False)
        target_layout.addWidget(self.target_var_combobox)
        main_layout.addLayout(target_layout)
        # 3. Модель классификации
        model_group_box = QGroupBox("Модель классификации")
        model_vlayout = QVBoxLayout()
        self.checkboxes = []
        self.labels_and_lines = {}

        models_params = {
            'Random Forest': ['Количество деревьев', 'Test Size', 'Random State'],
            'Gradient Boosting': ['Количество деревьев', 'Test Size', 'Random State'],
            'Logistic Regression': ['C', 'Max Iterations', 'Penalty']
        }

        for model_name, params_list in models_params.items():
            hbox = QHBoxLayout()
            cb = QCheckBox(model_name)
            cb.setChecked(True if model_name == "Random Forest" else False)
            self.checkboxes.append(cb)
            hbox.addWidget(cb)

            lines = {}
            for param_name in params_list:
                lbl = QLabel(param_name)
                le = QLineEdit()
                defaults = {
                    'Количество деревьев': '100',
                    'C': '0.01',
                    'Max Iterations': '100',
                    'Penalty': 'l2',
                    'Test Size': '0.2',
                    'Random State': '42'
                }
                le.setText(defaults.get(param_name, ''))
                hbox.addWidget(lbl)
                hbox.addWidget(le)
                lines[param_name] = le

            self.labels_and_lines[model_name] = lines
            model_vlayout.addLayout(hbox)

        model_group_box.setLayout(model_vlayout)
        main_layout.addWidget(model_group_box)

        # 4. Оценка моделей
        evaluate_models_btn = QPushButton('Оценить модели')
        evaluate_models_btn.clicked.connect(self.on_evaluate_models_clicked)
        main_layout.addWidget(evaluate_models_btn)

        # 5. Результаты оценки — с кнопками-подсказками
        results_group = QGroupBox("Результаты оценки моделей")
        results_layout = QVBoxLayout()

        self.metrics_container = QVBoxLayout()  # Контейнер для метрик (вставится сюда)
        results_layout.addLayout(self.metrics_container)

        self.time_label = QLabel('')
        results_layout.addWidget(self.time_label)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        # 6. Показать важность признаков
        models_group_box = QGroupBox("Выберите модель для анализа важности признаков")
        models_layout = QVBoxLayout()

        self.rf_checkbox = QCheckBox('Random Forest')
        self.gb_checkbox = QCheckBox('Gradient Boosting')
        self.lr_checkbox = QCheckBox('Logistic Regression')

        checkboxes_layout = QHBoxLayout()
        checkboxes_layout.addWidget(self.rf_checkbox)
        checkboxes_layout.addWidget(self.gb_checkbox)
        checkboxes_layout.addWidget(self.lr_checkbox)
        models_layout.addLayout(checkboxes_layout)

        show_importance_btn = QPushButton("Показать важность признаков")
        show_importance_btn.clicked.connect(self.on_show_feature_importance)
        models_layout.addWidget(show_importance_btn)

        models_group_box.setLayout(models_layout)
        main_layout.addWidget(models_group_box)

        self.setLayout(main_layout)
        self.resize(800, 700)
        self.setWindowTitle("Оценка моделей классификации")
        self.show()

    # === Подсказки по метрикам ===
    def show_metric_help(self, metric_name):
        descriptions = {
            "Precision": (
                "<b>Precision (Точность)</b><br>"
                "Доля правильно предсказанных положительных объектов среди всех предсказанных как положительные.<br><br>"
                "Формула: TP / (TP + FP)<br>"
                "Высокая точность — мало ложных срабатываний."
            ),
            "Recall": (
                "<b>Recall (Полнота)</b><br>"
                "Доля правильно предсказанных положительных объектов среди всех реальных положительных.<br><br>"
                "Формула: TP / (TP + FN)<br>"
                "Высокая полнота — мало пропущенных случаев (важно в медицине)."
            ),
            "F1-Score": (
                "<b>F1-Score</b><br>"
                "Гармоническое среднее между Precision и Recall.<br><br>"
                "Формула: 2 * (Precision * Recall) / (Precision + Recall)<br>"
                "Хорошо работает при несбалансированных данных."
            ),
            "ROC-AUC": (
                "<b>ROC-AUC</b><br>"
                "Площадь под ROC-кривой. Показывает, насколько хорошо модель различает классы.<br><br>"
                "Чем ближе к 1.0 — тем лучше.<br>"
                "Работает на вероятностях, а не на предсказаниях."
            )
        }
        text = descriptions.get(metric_name, "Нет описания.")
        QMessageBox.information(self, f"Что такое {metric_name}?", text)

    def on_select_dataset_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл CSV", "./dataset/", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.dataset_file_name = os.path.basename(file_path)
                self.select_dataset_btn.setText(f"📁 {self.dataset_file_name}")
                self.data_handler.update_dataframe(df)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")
        else:
            print("Файл не выбран.")

    def on_evaluate_models_clicked(self):
        if self.data_handler.df is None or self.data_handler.df.empty:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите датасет!")
            return
        if self.target_var_combobox.currentText() == "":
            QMessageBox.warning(self, "Предупреждение", "Выберите целевую переменную!")
            return
        self.data_handler.evaluate_models()

    def on_show_feature_importance(self):
        selected_models = {}
        if self.rf_checkbox.isChecked():
            selected_models['Random Forest'] = True
        if self.gb_checkbox.isChecked():
            selected_models['Gradient Boosting'] = True
        if self.lr_checkbox.isChecked():
            selected_models['Logistic Regression'] = True

        if selected_models:
            self.data_handler.calculate_feature_importances(selected_models)
        else:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну модель!")

    # === Метод для обновления UI с метриками и кнопками ===
    def update_metrics_display(self, report_lines):
        # Очищаем старые метрики
        while self.metrics_container.count():
            child = self.metrics_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Добавляем каждую строку
        for line in report_lines:
            if not line.strip():
                continue

            model_label = QLabel(line)
            model_label.setTextFormat(Qt.TextFormat.RichText)
            self.metrics_container.addWidget(model_label)

            # Извлекаем метрики и добавляем кнопки
            self.add_metric_row("Precision", line)
            self.add_metric_row("Recall", line)
            self.add_metric_row("F1-Score", line)
            self.add_metric_row("ROC-AUC", line)

            # Разделитель
            self.metrics_container.addWidget(self.create_separator())

    def add_metric_row(self, metric_name, line):
        if metric_name in line:
            start = line.find(metric_name + "=") + len(metric_name) + 1
            end = line.find(",", start)
            if end == -1:
                end = len(line)
            value = line[start:end].strip()

            row = QHBoxLayout()
            label = QLabel(f"<b>{metric_name}:</b> {value}")
            label.setTextFormat(Qt.TextFormat.RichText)
            btn = QPushButton("❓")
            btn.setFixedSize(24, 24)
            btn.clicked.connect(lambda: self.show_metric_help(metric_name))
            row.addWidget(label)
            row.addWidget(btn)
            row.addStretch()
            self.metrics_container.addLayout(row)

    def create_separator(self):
        line = QLabel()
        line.setFrameShape(QLabel.HLine)
        line.setFrameShadow(QLabel.Sunken)
        return line
