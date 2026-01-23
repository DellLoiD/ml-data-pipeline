# inference_trained_models.py
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QDialog,
    QFileDialog, QMessageBox, QComboBox, QApplication, QHBoxLayout,
    QGroupBox, QFrame, QLineEdit
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
        self.dataset_name = None
        self.target_variable = None
        self.df = None
        self.input_widgets = {}

    def initUI(self):
        layout = QVBoxLayout()

        # ============= ШАГ 1: ПОДГОТОВКА ДАННЫХ =============
        prep_group = QGroupBox("Шаг 1: Подготовка")
        prep_layout = QVBoxLayout()

        # Выбор датасета
        self.choose_dataset_button = QPushButton("Выберите датасет (.csv)", self)
        self.choose_dataset_button.clicked.connect(self.choose_dataset)
        prep_layout.addWidget(self.choose_dataset_button)

        # Выбор целевой переменной и генерация шаблона
        self.generate_questions_button = QPushButton("Создать шаблон вопросов", self)
        self.generate_questions_button.clicked.connect(self.generate_questions_template)
        self.generate_questions_button.setEnabled(False)
        prep_layout.addWidget(self.generate_questions_button)

        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        prep_layout.addWidget(line)

        # Загрузка шаблона вопросов
        self.choose_file_button = QPushButton("Загрузить шаблон вопросов (.txt)", self)
        self.choose_file_button.clicked.connect(self.choose_questions_file)
        prep_layout.addWidget(self.choose_file_button)

        prep_group.setLayout(prep_layout)
        layout.addWidget(prep_group)

        # ============= ШАГ 2: ПРОХОЖДЕНИЕ ОПРОСА =============
        survey_group = QGroupBox("Шаг 2: Прохождение опроса")
        survey_layout = QVBoxLayout()

        # Кнопка запуска опроса
        self.start_survey_button = QPushButton("▶️ Запустить опрос")
        self.start_survey_button.clicked.connect(self.start_survey)
        self.start_survey_button.setEnabled(False)
        survey_layout.addWidget(self.start_survey_button)

        # Прогресс
        self.progress_label = QLabel("", self)
        self.progress_label.setStyleSheet("color: gray;")
        survey_layout.addWidget(self.progress_label)

        # Вопрос
        self.label = QLabel("Выберите шаблон вопросов для начала.", self)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Arial", 12))
        survey_layout.addWidget(self.label)

        # Поле ввода (или комбобокс)
        self.input_widget_layout = QHBoxLayout()
        self.input_widgets_container = QWidget()
        self.input_widgets_container.setLayout(self.input_widget_layout)
        survey_layout.addWidget(self.input_widgets_container)

        # Кнопка "Записать ответ"
        self.answer_button = QPushButton("Записать ответ и продолжить")
        self.answer_button.clicked.connect(self.save_answer_and_continue)
        self.answer_button.setEnabled(False)
        survey_layout.addWidget(self.answer_button)

        survey_group.setLayout(survey_layout)
        layout.addWidget(survey_group)

        # ============= ЗАВЕРШЕНИЕ =============
        layout.addStretch()
        self.setLayout(layout)
        self.setWindowTitle("Анкетирование для инференса")
        self.resize(700, 500)

    def choose_dataset(self):
        """Выбор датасета с игнорированием строк #META"""
        base_dir = str(Path(__file__).resolve().parent.parent)
        dataset_folder = os.path.join(base_dir, 'dataset')

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выбор датасета", dataset_folder, "CSV Files (*.csv)"
        )
        if not file_name:
            return

        try:
            self.dataset_name = file_name
            self.df = pd.read_csv(self.dataset_name, comment='#')
            self.choose_dataset_button.setText(f"✅ {os.path.basename(file_name)}")
            self.choose_target_variable()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить датасет:\n{e}")

    def choose_target_variable(self):
        """Выбор целевой переменной"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор целевой переменной")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Выберите целевую переменную:"))
        combo = QComboBox()
        combo.addItems(self.df.columns.tolist())
        layout.addWidget(combo)

        button = QPushButton("Подтвердить")
        button.clicked.connect(lambda: self.confirm_target_and_close(combo.currentText(), dialog))
        layout.addWidget(button)

        dialog.setLayout(layout)
        dialog.exec()

    def confirm_target_and_close(self, target_var, dialog):
        self.target_variable = target_var
        self.generate_questions_button.setEnabled(True)
        self.label.setText(f"Целевая переменная: {target_var}. Загрузите или создайте шаблон вопросов.")
        dialog.accept()

    def generate_questions_template(self):
        """Генерация шаблона на основе датасета"""
        if self.df is None or not self.target_variable:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите датасет и целевую переменную.")
            return

        feature_columns = [col for col in self.df.columns if col != self.target_variable]
        lines = []

        for col in feature_columns:
            col_data = self.df[col].dropna()
            unique_vals = col_data.unique()
            unique_vals = [str(x) for x in unique_vals if pd.notna(x)]

            if len(unique_vals) <= 10:
                values_str = ", ".join(f'"{v}"' for v in unique_vals)
                prompt = f"Выберите значение ({values_str})"
            else:
                prompt = 'Введите числовое значение'

            lines.append(f"{col}: {prompt}")

        # Сохранение
        dataset_base = Path(self.dataset_name).stem
        target_clean = self.target_variable.replace(" ", "_")
        filename = f"{dataset_base}_{target_clean}_inference.txt"
        save_dir = Path("inference_models")
        save_dir.mkdir(exist_ok=True)
        file_path = save_dir / filename

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            QMessageBox.information(self, "Успех", f"Шаблон сохранён:\n{file_path}")
            self.load_questions_file(str(file_path))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")

    def choose_questions_file(self):
        """Загрузка внешнего шаблона вопросов"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить шаблон вопросов", "",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        if file_path:
            self.load_questions_file(file_path)

    def load_questions_file(self, file_path):
        """Загрузка вопросов из файла"""
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Ошибка", "Файл не найден.")
            return

        self.choose_file_button.setText(f"✅ {os.path.basename(file_path)}")
        self.read_questions_from_file(file_path)

        # Активируем кнопку запуска опроса
        self.start_survey_button.setEnabled(True)

        # Обновляем порядок вопросов, если есть датасет
        if self.df is not None and self.target_variable:
            columns = [col for col in self.df.columns if col != self.target_variable]
            self.update_question_order(columns)
        else:
            # Можно пройти опрос и без датасета, если вопросы совпадают
            keys_in_questions = [line.split(":")[0].strip() for line in open(file_path, encoding="utf-8") if ":" in line]
            self.question_order = [(key, self.questions[key]) for key in keys_in_questions if key in self.questions]
            self.label.setText("Датасет не загружен. Опрос начнётся по загруженным вопросам.")

    def read_questions_from_file(self, filename):
        self.questions.clear()
        try:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    key, *text = line.split(": ", 1)
                    if text:
                        self.questions[key.strip()] = text[0].strip()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка чтения файла:\n{e}")

    def update_question_order(self, column_names):
        missing = [col for col in column_names if col not in self.questions]
        if missing:
            QMessageBox.warning(self, "Внимание", f"Нет вопросов для:\n{', '.join(missing)}")
            # Продолжаем без них
            column_names = [col for col in column_names if col in self.questions]

        self.question_order = [(col, self.questions[col]) for col in column_names]
        self.label.setText(f"Готово: {len(self.question_order)} вопросов. Нажмите 'Запустить опрос'.")

    def start_survey(self):
        """Запуск анкетирования"""
        if not self.question_order:
            QMessageBox.warning(self, "Ошибка", "Нет вопросов для отображения.")
            return

        self.current_question_idx = 0
        self.data.clear()
        self.ask_next_question()

    def ask_next_question(self):
        if self.current_question_idx >= len(self.question_order):
            self.show_results()
            return

        col_name, prompt = self.question_order[self.current_question_idx]
        self.label.setText(f"<b>{col_name}</b>: {prompt}")

        # Очистка предыдущего виджета
        for i in reversed(range(self.input_widget_layout.count())):
            widget = self.input_widget_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Создание нового виджета
        if 'Выберите значение' in prompt:
            values = [s.strip('"\n\r\t ') for s in prompt.split('(')[1].split(')')[0].split(',')]
            combo = QComboBox()
            combo.addItems(values)
            self.input_widget_layout.addWidget(combo)
            self.input_widgets[col_name] = combo
        else:
            line_edit = QLineEdit()
            self.input_widget_layout.addWidget(line_edit)
            self.input_widgets[col_name] = line_edit

        self.progress_label.setText(f"Вопрос {self.current_question_idx + 1} из {len(self.question_order)}")
        self.answer_button.setEnabled(True)
        self.answer_button.setFocus()

    def save_answer_and_continue(self):
        col_name, _ = self.question_order[self.current_question_idx]
        widget = self.input_widgets.get(col_name)

        if isinstance(widget, QComboBox):
            value = widget.currentText()
        else:
            text = widget.text().strip()
            if not text:
                QMessageBox.warning(self, "Ошибка", "Введите значение.")
                return
            try:
                value = float(text)
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Введите числовое значение.")
                return

        self.data[col_name] = value
        self.current_question_idx += 1
        self.ask_next_question()

    def format_two_columns(self, data):
        keys = list(data.keys())
        values = list(data.values())
        n = len(keys)
        half = math.ceil(n / 2)
        left_k, right_k = keys[:half], keys[half:]
        left_v, right_v = values[:half], values[half:]
        LINE_LEN = 30
        lines = []
        for i in range(half):
            left = f"{left_k[i]}{' '*(LINE_LEN - len(str(left_k[i])) - len(str(left_v[i])))}{left_v[i]}"
            right = ""
            if i < len(right_k):
                right = f"{right_k[i]}{' '*(LINE_LEN - len(str(right_k[i])) - len(str(right_v[i])))}{right_v[i]}"
            lines.append(f"{left}      {right}")
        return "\n".join(lines)

    def show_results(self):
        results_str = self.format_two_columns(self.data)
        models_dir = "trained_models"

        if not os.path.exists(models_dir):
            QMessageBox.critical(self, "Ошибка", "Папка trained_models не найдена!")
            return

        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if not model_files:
            QMessageBox.critical(self, "Ошибка", "Нет обученных моделей!")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Анализ данных")
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Введены данные:\n{results_str}\n\nВыберите модель:"))

        combo = QComboBox()
        combo.addItems(model_files)
        layout.addWidget(combo)

        btn = QPushButton("Запустить анализ")
        btn.clicked.connect(lambda: self.run_analysis(combo.currentText(), dialog))
        layout.addWidget(btn)

        dialog.setLayout(layout)
        dialog.exec()

    def run_analysis(self, model_filename, dialog):
        model_path = os.path.join("trained_models", model_filename)
        try:
            model = joblib.load(model_path)

            # Формируем введённый датасет
            df_input = pd.DataFrame([self.data])

            # Извлекаем ожидаемый порядок колонок
            expected_columns = None
            if hasattr(model, 'feature_names_in_'):
                expected_columns = list(model.feature_names_in_)
            else:
                # Пытаемся загрузить из JSON
                json_path = model_path.replace('.pkl', '_features.json')
                if os.path.exists(json_path):
                    import json
                    with open(json_path, 'r', encoding='utf-8') as f:
                        expected_columns = json.load(f)
                else:
                    # Нельзя продолжить без порядка
                    QMessageBox.critical(
                        self, "Ошибка",
                        "Не удалось определить порядок признаков.\n"
                        "Модель не имеет feature_names_in_, а JSON-файл не найден.\n"
                        "Убедитесь, что при обучении был сохранён порядок признаков."
                    )
                    dialog.reject()
                    return

            # Проверяем совпадение, но НЕ порядок
            data_columns_set = set(df_input.columns)
            expected_columns_set = set(expected_columns)

            if data_columns_set != expected_columns_set:
                QMessageBox.critical(
                    self, "Ошибка",
                    f"Разные признаки:\n"
                    f"Ожидалось: {sorted(expected_columns_set)}\n"
                    f"Получено: {sorted(data_columns_set)}"
                )
                dialog.reject()
                return

            # Проверяем порядок
            if list(df_input.columns) != expected_columns:
                reply = QMessageBox.question(
                    self, "Порядок признаков",
                    "Порядок признаков не совпадает с тем, на котором обучалась модель.\n"
                    "Хотите выровнять порядок по датасету?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    # Открываем диалог выбора датасета для определения порядка
                    dataset_file, _ = QFileDialog.getOpenFileName(
                        self, "Выберите датасет для выравнивания порядка", "",
                        "CSV Files (*.csv)"
                    )
                    if not dataset_file:
                        QMessageBox.warning(self, "Отменено", "Выравнивание отменено.")
                        dialog.reject()
                        return

                    try:
                        df_ref = pd.read_csv(dataset_file, comment='#')
                        feature_cols = [col for col in df_ref.columns if col != self.target_variable]
                        # Удаляем отсутствующие признаки
                        feature_cols = [col for col in feature_cols if col in expected_columns]
                        if set(feature_cols) != set(expected_columns):
                            QMessageBox.critical(
                                self, "Ошибка",
                                "Выбранный датасет не содержит все необходимые признаки."
                            )
                            dialog.reject()
                            return
                        # Выравниваем порядок
                        df_input = df_input[feature_cols]
                    except Exception as e:
                        QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать датасет:\n{e}")
                        dialog.reject()
                        return
                else:
                    # Попробуем использовать ожидаемый порядок
                    try:
                        df_input = df_input[expected_columns]
                    except KeyError as e:
                        QMessageBox.critical(self, "Ошибка", f"Не удаётся выровнять порядок:\n{e}")
                        dialog.reject()
                        return
                    
            # === ПРОГНОЗ ===
            prediction = model.predict(df_input)[0]

            # Определяем имя целевой переменной
            target_name = self.target_variable or "Целевая_переменная"

            # Форматируем значение (целое число, если возможно)
            if isinstance(prediction, (int, float)):
                pred_value = int(prediction) if float(prediction).is_integer() else float(prediction)
            else:
                pred_value = str(prediction)

            # Универсальное сообщение
            msg = f"📊 Результат инференса:\n\n"
            msg += f"🔹 Целевая переменная: {target_name}\n"
            msg += f"🎯 Предсказанное значение: {pred_value}"

            # Показываем результат
            QMessageBox.information(self, "Результат инференса", msg)
            dialog.accept()
            self.close()


        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка анализа:\n{str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SurveyForm()
    window.show()
    sys.exit(app.exec())
