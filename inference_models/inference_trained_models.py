import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QDialog,
    QFileDialog, QMessageBox, QComboBox, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import os
import joblib
import pandas as pd
import math
from pathlib import Path



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
        self.target_variable = None
        self.df = None

    def initUI(self):
        layout = QVBoxLayout()
        font = QFont("Arial", 12)

        # Кнопка для выбора датасета
        self.choose_dataset_button = QPushButton("Выберите датасет (.csv)", self)
        self.choose_dataset_button.clicked.connect(self.choose_dataset)
        layout.addWidget(self.choose_dataset_button)

        # Кнопка для создания шаблона вопросов
        self.generate_questions_button = QPushButton("Создать шаблон вопросов", self)
        self.generate_questions_button.clicked.connect(self.generate_questions_template)
        self.generate_questions_button.setEnabled(False)
        layout.addWidget(self.generate_questions_button)

        # Кнопка для выбора файла с вопросами
        self.choose_file_button = QPushButton("Выберите файл с вопросами (.txt)", self)
        self.choose_file_button.clicked.connect(self.choose_questions_file)
        layout.addWidget(self.choose_file_button)

        # Показываем текущий вопрос
        self.label = QLabel("", self)
        self.label.setFont(font)
        layout.addWidget(self.label)

        # Поле для ввода ответа
        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)

        # Кнопка для продолжения анкетирования
        self.button = QPushButton("Записать ответ", self)
        self.button.clicked.connect(self.save_answer_and_continue)
        self.button.setEnabled(False)
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.setWindowTitle("Анкетирование для инференса")

    def choose_dataset(self):
        """Выбор датасета"""
        base_dir = str(Path(__file__).resolve().parent.parent)
        dataset_folder = os.path.join(base_dir, 'dataset')

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выбор датасета", dataset_folder, "CSV Files (*.csv)"
        )

        if file_name:
            self.dataset_name = file_name
            self.df = pd.read_csv(self.dataset_name)
            self.choose_dataset_button.setText(os.path.basename(file_name))
            self.choose_target_variable()

    def choose_target_variable(self):
        """Открывает диалог выбора целевой переменной и закрывается автоматически после выбора"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор целевой переменной")
        layout = QVBoxLayout()

        label = QLabel("Выберите целевую переменную:")
        combobox = QComboBox()
        combobox.addItems(self.df.columns.tolist())
        layout.addWidget(label)
        layout.addWidget(combobox)

        button = QPushButton("Подтвердить")
        button.clicked.connect(lambda: self.confirm_target_and_close(combobox.currentText(), dialog))
        layout.addWidget(button)

        dialog.setLayout(layout)
        dialog.exec_()  # блокирующий вызов — закрывается при accept()

    def confirm_target_and_close(self, target_var, dialog):
        """Фиксирует целевую переменную и закрывает диалог"""
        self.target_variable = target_var
        QMessageBox.information(self, "Подтверждено", f"Целевая переменная установлена: {target_var}")
        self.generate_questions_button.setEnabled(True)
        dialog.accept()  # закрывает диалог

    def generate_questions_template(self):
        """Создаёт шаблон .txt файла с вопросами и подсказками по категориям"""
        if not self.df is not None or not self.target_variable:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите датасет и целевую переменную.")
            return

        feature_columns = [col for col in self.df.columns if col != self.target_variable]

        lines = []
        for col in feature_columns:
            col_data = self.df[col].dropna()
            unique_vals = sorted(col_data.unique()) if col_data.dtype.kind in 'biufc' else col_data.unique()
            unique_vals = [str(x) for x in unique_vals]

            if len(unique_vals) <= 10:  # Мало уникальных значений → вероятно категория
                if all(v.isdigit() for v in unique_vals):
                    # Числовые категории: показываем как числа
                    values_str = ", ".join(unique_vals)
                    prompt = f"Введите значение ({values_str})"
                else:
                    # Текстовые категории
                    values_str = ", ".join(f'"{v}"' for v in unique_vals)
                    prompt = f"Введите значение ({values_str})"
            else:
                # Много уникальных значений — вероятно числовая переменная (возраст, глюкоза и т.д.)
                prompt = 'Введите числовое значение'

            line = f'{col}: {prompt}'
            lines.append(line)

        # Формируем имя файла
        dataset_base = Path(self.dataset_name).stem
        target_clean = self.target_variable.replace(" ", "_")
        filename = f"{dataset_base}_{target_clean}_inference.txt"
        save_dir = Path("inference_models")
        save_dir.mkdir(exist_ok=True)
        file_path = save_dir / filename

        # Сохраняем
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        QMessageBox.information(self, "Успех", f"Шаблон вопросов сохранён как:\n{file_path}")

        # Автоматически загружаем файл
        self.load_questions_file(str(file_path))

    def load_questions_file(self, file_path):
        """Загружает файл с вопросами и обновляет интерфейс"""
        self.choose_file_button.setText(os.path.basename(file_path))
        self.read_questions_from_file(file_path)

        # Обновляем порядок вопросов
        if self.df is not None and self.target_variable:
            columns = [col for col in self.df.columns if col != self.target_variable]
            self.update_question_order(columns)

    def choose_questions_file(self):
        """Выбор файла с вопросами вручную"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с вопросами", "",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        if file_path:
            self.load_questions_file(file_path)

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
        self.questions.clear()
        try:
            with open(filename, encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    parts = line.split(": ", 1)  # Разделяем только по первому ": "
                    if len(parts) == 2:
                        key, text = parts
                        self.questions[key.strip()] = text.strip()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл вопросов:\n{str(e)}")

    def ask_next_question(self):
        """Показ следующего вопроса пользователю"""
        if self.current_question_idx < len(self.question_order):
            col_name, question = self.question_order[self.current_question_idx]
            self.label.setText(f"{col_name}: {question}")
            self.input_field.clear()
            self.input_field.setFocus()
            self.button.setEnabled(True)
        else:
            self.show_results()

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

    def format_two_columns(self, data):
        keys = list(data.keys())
        values = list(data.values())
        num_rows = len(keys)
        half_num_rows = math.ceil(num_rows / 2)

        left_keys = keys[:half_num_rows]
        right_keys = keys[half_num_rows:]
        left_values = values[:half_num_rows]
        right_values = values[half_num_rows:]

        LINE_LENGTH = 30
        result_lines = []
        for i in range(half_num_rows):
            left_key = left_keys[i]
            left_value = str(left_values[i])
            padding_left = ' ' * (LINE_LENGTH - len(left_key) - len(left_value))
            left_formatted = f'{left_key}{padding_left}{left_value}'

            right_key = right_keys[i] if i < len(right_keys) else ''
            right_value = str(right_values[i]) if i < len(right_values) else ''
            padding_right = ' ' * (LINE_LENGTH - len(right_key) - len(right_value)) if right_key else ''
            right_formatted = f'{right_key}{padding_right}{right_value}'

            formatted_line = f'{left_formatted}      {right_formatted}'
            result_lines.append(formatted_line)

        return '\n'.join(result_lines)

    def show_results(self):
        """Покажем результаты анкетирования и предлагаем выбрать модель для анализа"""
        results_string = self.format_two_columns(self.data)

        models_dir = "trained_models"
        if not os.path.exists(models_dir):
            QMessageBox.critical(self, "Ошибка", "Папка trained_models не найдена!")
            return

        model_files = [model for model in os.listdir(models_dir) if model.endswith('.pkl')]
        if not model_files:
            QMessageBox.critical(self, "Ошибка", "Нет обученных моделей (.pkl) в папке trained_models!")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Анализ риска диабета")
        dialog_layout = QVBoxLayout(dialog)

        info_label = QLabel(f"Введены следующие данные:\n{results_string}\n\nВыберите модель для анализа:")
        info_label.setFont(QFont('Consolas', 10))
        dialog_layout.addWidget(info_label)

        combo_box = QComboBox()
        combo_box.addItems(model_files)
        dialog_layout.addWidget(combo_box)

        button_analyze = QPushButton("Запустить анализ")
        button_analyze.clicked.connect(lambda: self.run_analysis(combo_box.currentText(), dialog))
        dialog_layout.addWidget(button_analyze)

        self.close()
        dialog.exec_()

    def run_analysis(self, model_filename, dialog):
        """Запускает анализ на основе выбранной модели."""
        model_path = os.path.join("trained_models", model_filename)
        try:
            with open(model_path, 'rb') as f:
                model = joblib.load(f)
            df_input = pd.DataFrame([self.data])
            prediction = model.predict(df_input)

            # Формируем динамическое сообщение
            target_name = self.target_variable
            predicted_class = int(prediction[0])  # преобразуем в int (на случай np.int64)

            result_msg = f'"{target_name}" предсказано как: {predicted_class}'

            # Дополнительный текст (по желанию)
            if target_name.lower() in ['diabetes', 'outcome', 'diabetic'] and predicted_class == 1:
                result_msg += "\n\nПо результатам диагностики, вероятно наличие диабета.\nРекомендуем обратиться к специалисту."
            elif target_name.lower() in ['diabetes', 'outcome', 'diabetic'] and predicted_class == 0:
                result_msg += "\n\nРиск диабета низкий.\nПродолжайте вести здоровый образ жизни!"

            QMessageBox.information(None, "Результат анализа", result_msg)
            dialog.accept()
        except Exception as e:
            QMessageBox.critical(None, "Ошибка", f"Не удалось выполнить анализ:\n{str(e)}")
            logger.error(f"Ошибка при запуске анализа: {e}")


# === Запуск приложения ===
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SurveyForm()
    window.show()
    sys.exit(app.exec())
