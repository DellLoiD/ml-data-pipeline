import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
import os
import joblib
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox, QComboBox
)
from PySide6.QtGui import QFont
import pandas as pd


class SurveyForm(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data = {}
        self.questions = {}
        self.question_order = []
        self.current_question_idx = 0
        self.file_chosen = False
        self.dataset_name = None
        self.target_variable = None  # Целевая переменная

    def initUI(self):
        layout = QVBoxLayout()
        font = QFont("Arial", 12)

        # Кнопка для выбора датасета
        self.choose_dataset_button = QPushButton("Выберите датасет (.csv)", self)
        self.choose_dataset_button.clicked.connect(self.choose_dataset)
        layout.addWidget(self.choose_dataset_button)

        # Кнопка для выбора файла с вопросами
        self.choose_file_button = QPushButton("Выберите файл с вопросами (.txt)", self)
        self.choose_file_button.clicked.connect(self.choose_questions_file)
        layout.addWidget(self.choose_file_button)

        # Показываем текущие вопросы
        self.label = QLabel("", self)
        self.label.setFont(font)
        layout.addWidget(self.label)

        # Поле для ввода ответа
        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)

        # Кнопка для продолжения анкетирования
        self.button = QPushButton("Записать ответ", self)
        self.button.clicked.connect(self.save_answer_and_continue)
        self.button.setEnabled(False)  # Изначально кнопка выключена
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.setWindowTitle("Анкетирование здоровья")

    def choose_dataset(self):
        """Выбор датасета"""
        dataset_dialog = QFileDialog.getOpenFileName(
            self, "Выберите датасет", "", "CSV-файлы (*.csv)"
        )
        if dataset_dialog[0]:
            self.dataset_name = dataset_dialog[0]
            self.choose_dataset_button.setText(os.path.basename(dataset_dialog[0]))
            self.df = pd.read_csv(self.dataset_name)
            self.choose_target_variable()  # Просим пользователя выбрать целевую переменную

    def choose_target_variable(self):
        """Окно для выбора целевой переменной"""
        dialog = QDialog(self)
        layout = QVBoxLayout()
        label = QLabel("Выберите целевую переменную:", dialog)
        combobox = QComboBox(dialog)
        combobox.addItems(self.df.columns.tolist())  # Перечисляем все столбцы
        button = QPushButton("Подтвердить", dialog)
        button.clicked.connect(lambda: self.confirm_target(combobox.currentText()))
        layout.addWidget(label)
        layout.addWidget(combobox)
        layout.addWidget(button)
        dialog.setLayout(layout)
        dialog.exec_()

    def confirm_target(self, target_var):
        """Фиксируем выбранную целевую переменную"""
        self.target_variable = target_var
        QMessageBox.information(self, "Подтверждено", f"Целевая переменная установлена: {target_var}.")

    def choose_questions_file(self):
        """Выбор файла с вопросами"""
        file_dialog = QFileDialog.getOpenFileName(
            self, "Выберите файл с вопросами", "",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        if file_dialog[0]:
            filename = file_dialog[0]
            self.choose_file_button.setText(os.path.basename(filename))
            self.read_questions_from_file(filename)

            # Продолжаем только после выбора датасета
            if hasattr(self, 'df'):
                columns = [col for col in self.df.columns if col != self.target_variable]
                self.update_question_order(columns)
            else:
                QMessageBox.information(self, "Внимание", "Сначала выберите датасет.")

    def update_question_order(self, column_names):
        """Обновляем порядок вопросов, пропуская целевую переменную"""
        missing_columns = []
        self.question_order = []

        for col in column_names:
            if col in self.questions:
                question = self.questions[col]
                self.question_order.append((col, question))
            else:
                missing_columns.append(col)

        if missing_columns:
            message = ", ".join(missing_columns)
            QMessageBox.critical(self, "Ошибка", f"Следующие метрики не имеют вопросов:\n{message}\n\nПопробуйте выбрать другой файл с вопросами.")
        else:
            self.current_question_idx = 0
            self.ask_next_question()

    def read_questions_from_file(self, filename):
        """Читаем вопросы из файла"""
        with open(filename, encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(": ")
                if len(parts) == 2:
                    key, text = parts
                    self.questions[key] = text

    def ask_next_question(self):
        """Показ следующего вопроса пользователю"""
        if self.current_question_idx < len(self.question_order):
            col_name, question = self.question_order[self.current_question_idx]
            self.label.setText(f"{col_name}: {question}")
            self.input_field.clear()
            self.input_field.setFocus()
            self.button.setEnabled(True)  # Активируем кнопку
        else:
            self.show_results()  # Завершаем опрос и показываем результаты

    def save_answer_and_continue(self):
        """Сохраняем ответ и идём дальше"""
        answer_text = self.input_field.text().strip()
        current_col_name, _ = self.question_order[self.current_question_idx]
        try:
            value = float(answer_text)
            self.data[current_col_name] = value
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите числовое значение.")
            return

        self.current_question_idx += 1
        self.ask_next_question()

    def show_results(self):
        """Покажем результаты анкетирования и предлагаем выбрать модель для анализа"""
        # Форматируем введённые данные для отображения
        results_string = ", ".join([f"{key}={val:.2f}" for key, val in self.data.items()])

        # Каталог с готовыми моделями
        models_dir = "trained_models"
        # Получаем список моделей (.pkl файлов)
        model_files = [model for model in os.listdir(models_dir) if model.endswith('.pkl')]

        # Создаем новое окно для выбора модели
        dialog = QDialog(self)
        dialog.setWindowTitle("Анализ риска диабета")
        dialog_layout = QVBoxLayout(dialog)

        # Элемент с результатом ввода данных
        info_label = QLabel(f"Вы ввели следующие данные:\n\n{results_string}\n\nВыберите модель для анализа:")
        dialog_layout.addWidget(info_label)

        # Выбор модели из выпадающего меню
        combo_box = QComboBox()
        combo_box.addItems(model_files)
        dialog_layout.addWidget(combo_box)

        # Кнопка для запуска анализа
        button_analyze = QPushButton("Запустить анализ")
        button_analyze.clicked.connect(lambda: self.run_analysis(combo_box.currentText(), dialog))
        dialog_layout.addWidget(button_analyze)

        # Закрываем старое окно опроса
        self.close()

        # Открываем диалоговое окно выбора модели
        dialog.exec_()

    def run_analysis(self, model_filename, dialog):
        """
        Запускает анализ на основе выбранной модели.
        """
        # Полный путь к модели
        model_path = os.path.join("trained_models", model_filename)
        # Загружаем модель
        with open(model_path, 'rb') as f:
            model = joblib.load(f)

        # Входные данные преобразовываем в DataFrame
        df_input = pd.DataFrame([self.data])

        # Прогнозируем риск диабета
        prediction = model.predict(df_input)

        # Обрабатываем результат
        if prediction[0] == 1:
            result_msg = "По результатам диагностики, вероятно наличие диабета. Рекомендуем обратиться к специалисту."
        else:
            result_msg = "Риск диабета низкий. Продолжайте вести здоровый образ жизни!"

        # Показываем результат пользователю
        QMessageBox.information(None, "Результат анализа", result_msg)

        # Закрываем диалоговое окно
        dialog.accept()