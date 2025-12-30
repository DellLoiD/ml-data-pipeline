# check_models_ui.py
import sys
import pandas as pd
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QApplication,
    QComboBox, QCheckBox, QFileDialog, QMessageBox, QGroupBox
)
from PySide6.QtGui import QFont, QClipboard
from PySide6.QtCore import Qt

from .check_models_logic import DataModelHandler


class ClassificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_file_name = ""
        self.checkboxes = []
        self.labels_and_lines = {}
        self.report_text = ""  # Для хранения текста отчёта
        self.init_ui()

    def init_ui(self):
        # === Основной layout ===
        main_layout = QVBoxLayout()

        # === Заголовок ===
        title_label = QLabel('Оценка моделей — Классификация и Регрессия')
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)

        # === Кнопка выбора датасета ===
        self.select_dataset_btn = QPushButton("Выбрать датасет")
        self.select_dataset_btn.clicked.connect(self.on_select_dataset_clicked)
        main_layout.addWidget(self.select_dataset_btn)

        # === Выбор целевой переменной ===
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Целевая переменная:"))
        self.target_var_combobox = QComboBox()
        self.target_var_combobox.setEnabled(False)
        target_layout.addWidget(self.target_var_combobox)
        main_layout.addLayout(target_layout)

        # === Группа: Модели ===
        models_group = QGroupBox("Модели для оценки")
        models_layout = QVBoxLayout()

        # --- Классификация ---
        classification_box = QGroupBox("Классификация")
        self.classification_layout = QVBoxLayout()
        classification_box.setLayout(self.classification_layout)
        models_layout.addWidget(classification_box)

        # --- Регрессия ---
        regression_box = QGroupBox("Регрессия")
        self.regression_layout = QVBoxLayout()
        regression_box.setLayout(self.regression_layout)
        models_layout.addWidget(regression_box)

        models_group.setLayout(models_layout)
        main_layout.addWidget(models_group)

        # === Кнопка оценки ===
        evaluate_models_btn = QPushButton('Оценить выбранные модели')
        evaluate_models_btn.clicked.connect(self.on_evaluate_models_clicked)
        main_layout.addWidget(evaluate_models_btn)

        # === Результаты ===
        results_group = QGroupBox("Результаты оценки моделей")
        results_layout = QVBoxLayout()

        self.metrics_container = QVBoxLayout()
        results_layout.addLayout(self.metrics_container)

        self.time_label = QLabel('')
        results_layout.addWidget(self.time_label)

        # === Кнопка копирования ===
        copy_btn = QPushButton('Копировать результаты в буфер')
        copy_btn.clicked.connect(self.on_copy_results)
        results_layout.addWidget(copy_btn)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        # === Анализ важности признаков ===
        importance_group = QGroupBox("Анализ важности признаков")
        self.importance_layout = QVBoxLayout()
        importance_group.setLayout(self.importance_layout)
        main_layout.addWidget(importance_group)

        # === Инициализация обработчика ===
        self.data_handler = DataModelHandler(
            parent=self,
            df=None,
            combobox=self.target_var_combobox,
            checkboxes=self.checkboxes,
            labels_and_lines=self.labels_and_lines,
            accuracy_label=self.metrics_container,
            time_label=self.time_label,
            task_type="both"
        )

        # === Заполняем UI ===
        self.create_classification_models()
        self.create_regression_models()
        self.create_importance_checkboxes()

        # === Финальные настройки ===
        self.setLayout(main_layout)
        self.resize(900, 800)
        self.setWindowTitle("Оценка моделей — Все задачи")
        self.show()

    def create_classification_models(self):
        """Создаёт UI для моделей классификации"""
        models = {
            'Random Forest Classification': ['Количество деревьев', 'Test Size', 'Random State'],
            'Gradient Boosting Classification': ['Количество деревьев', 'Test Size', 'Random State'],
            'Logistic Regression Classification': ['C', 'Max Iterations', 'Penalty']
        }

        defaults = {
            'Количество деревьев': '100',
            'Test Size': '0.2',
            'Random State': '42',
            'C': '1.0',
            'Max Iterations': '100',
            'Penalty': 'l2'
        }

        for model_name, params in models.items():
            hbox = QHBoxLayout()
            cb = QCheckBox(model_name)
            cb.setChecked("Random Forest" in model_name)
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

    def create_regression_models(self):
        """Создаёт UI для моделей регрессии"""
        models = {
            'Random Forest Regression': ['Количество деревьев', 'Test Size', 'Random State'],
            'Gradient Boosting Regression': ['Количество деревьев', 'Test Size', 'Random State']
        }

        defaults = {
            'Количество деревьев': '100',
            'Test Size': '0.2',
            'Random State': '42'
        }

        for model_name, params in models.items():
            hbox = QHBoxLayout()
            cb = QCheckBox(model_name)
            cb.setChecked("Random Forest" in model_name)
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

    def create_importance_checkboxes(self):
        """Создаёт чекбоксы для анализа важности"""
        layout = QHBoxLayout()

        self.rfc_cb = QCheckBox("Random Forest Classification")
        self.rfc_cb.setChecked(True)
        layout.addWidget(self.rfc_cb)

        self.gbc_cb = QCheckBox("Gradient Boosting Classification")
        self.gbc_cb.setChecked(True)
        layout.addWidget(self.gbc_cb)

        self.lrc_cb = QCheckBox("Logistic Regression Classification")
        self.lrc_cb.setChecked(True)
        layout.addWidget(self.lrc_cb)

        self.rfr_cb = QCheckBox("Random Forest Regression")
        self.rfr_cb.setChecked(True)
        layout.addWidget(self.rfr_cb)

        self.gbr_cb = QCheckBox("Gradient Boosting Regression")
        self.gbr_cb.setChecked(True)
        layout.addWidget(self.gbr_cb)

        layout.addStretch()
        self.importance_layout.addLayout(layout)

        # Кнопка
        btn = QPushButton("Показать важность признаков")
        btn.clicked.connect(self.on_show_feature_importance)
        self.importance_layout.addWidget(btn)

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
        if self.rfc_cb.isChecked():
            selected_models['Random Forest Classification'] = True
        if self.gbc_cb.isChecked():
            selected_models['Gradient Boosting Classification'] = True
        if self.lrc_cb.isChecked():
            selected_models['Logistic Regression Classification'] = True
        if self.rfr_cb.isChecked():
            selected_models['Random Forest Regression'] = True
        if self.gbr_cb.isChecked():
            selected_models['Gradient Boosting Regression'] = True

        if selected_models:
            self.data_handler.calculate_feature_importances(selected_models)
        else:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну модель для анализа!")

    def update_metrics_display(self, report_lines):
        """Обновляет отображение метрик — удаляет старые, добавляет новые"""
        # === Очищаем старые метрики ===
        while self.metrics_container.count():
            child = self.metrics_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                # Очистка вложенных лэйаутов
                while child.layout().count():
                    subchild = child.layout().takeAt(0)
                    if subchild.widget():
                        subchild.widget().deleteLater()

        # === Формируем текст отчёта для копирования ===
        self.report_text = ""

        for line in report_lines:
            if not line.strip():
                continue
            # Добавляем в отображение
            model_label = QLabel(line)
            model_label.setTextFormat(Qt.RichText)
            self.metrics_container.addWidget(model_label)

            # Добавляем в текст для копирования
            clean_line = line.replace("<b>", "").replace("</b>", "").replace("<br>", "\n  ")
            self.report_text += clean_line + "\n\n"

            # Определяем, какие метрики показывать
            metrics_to_show = ["R²", "MSE", "MAE"] if "R²" in line else ["Precision", "Recall", "F1-Score", "ROC-AUC"]

            for metric in metrics_to_show:
                self.add_metric_row(metric, line)

            self.metrics_container.addWidget(self.create_separator())

        # Добавляем время выполнения
        self.report_text += self.time_label.text()

    def add_metric_row(self, metric_name, line):
        if metric_name not in line:
            return
        start = line.find(metric_name + "=") + len(metric_name) + 1
        end = line.find(",", start)
        if end == -1:
            end = len(line)
        value = line[start:end].strip()

        row = QHBoxLayout()
        label = QLabel(f"<b>{metric_name}:</b> {value}")
        label.setTextFormat(Qt.RichText)
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
            ),
            "R²": (
                "<b>R² Score</b><br>"
                "Показывает, насколько модель объясняет дисперсию данных.<br>"
                "1.0 — идеально, 0.0 — не лучше среднего, может быть отрицательным."
            ),
            "MSE": (
                "<b>MSE (Mean Squared Error)</b><br>"
                "Средний квадрат ошибки.<br>Чем ближе к 0 — тем лучше.<br>Чувствителен к выбросам."
            ),
            "MAE": (
                "<b>MAE (Mean Absolute Error)</b><br>"
                "Среднее абсолютное отклонение.<br>Более устойчива к выбросам, чем MSE."
            )
        }
        text = descriptions.get(metric_name, "Нет описания.")
        QMessageBox.information(self, f"Что такое {metric_name}?", text)

    def on_copy_results(self):
        """Копирует текст отчёта в буфер обмена"""
        if not self.report_text.strip():
            QMessageBox.information(self, "Копирование", "Нет данных для копирования.")
            return
        
        clipboard = QApplication.clipboard()  # Импортируем ниже
        clipboard.setText(self.report_text)
        QMessageBox.information(self, "Копирование", "Результаты скопированы в буфер обмена!")

    # Если не импортирован QApplication — добавь в начало файла:
    # from PySide6.QtWidgets import QApplication
