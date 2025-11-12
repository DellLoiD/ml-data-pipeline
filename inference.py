import os
import joblib
import pandas as pd
from PySide6.QtWidgets import QVBoxLayout, QComboBox, QPushButton, QLabel, QMessageBox
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLineEdit, QDialog
)

# Вопросы для анкеты
questions = {
    "HighBP": "Артериальное давление (0 = низкое; 1 = высокое)",
    "HighChol": "Уровень холестерина (0 = низкий; 1 = высокий)",
    "CholCheck": "Проверялись ли вы на уровень холестерина в течение последних 5 лет? (0 = нет; 1 = да)",
    "BMI": "Ваш индекс массы тела:",
    "Smoker": "Выкуривали ли вы за всю свою жизнь хотя бы 100 сигарет? (0 = нет; 1 = да)",
    "Stroke": "Был ли у вас когда-нибудь зафиксирован инсульт? (0 = нет; 1 = да)",
    "HeartDiseaseorAttack": "Были ли у вас когда-нибудь зафиксированы болезни сердца или инфаркт? (0 = нет; 1 = да)",
    "PhysActivity": "Занимаетесь ли вы в последние 30 дней активными физическими упражнениями, не считая работу? (0 = нет; 1 = да)",
    "Fruits": "Употребляете ли вы регулярно фрукты один или более раз в день? (0 = нет; 1 = да)",
    "Veggies": "Употребляете ли вы регулярно овощи один или более раз в день? (0 = нет; 1 = да)",
    "HvyAlcoholConsump": "Употребляете ли вы крепкие спиртные напитки часто (мужчины ≥14 порций/неделя, женщины ≥7 порций)? (0 = нет; 1 = да)",
    "AnyHealthcare": "Есть ли у вас медицинская страховка? (0 = нет; 1 = да)",
    "NoDocbcCost": "За последний год были ли ситуации, когда вам хотелось попасть к врачу, но не смогли оплатить визит? (0 = нет; 1 = да)",
    "GenHlth": "Как бы вы оцениваете своё общее состояние здоровья? (1 = прекрасное; 2 = очень хорошее; 3 = хорошее; 4 = удовлетворительное; 5 = плохое)",
    "MentHlth": "Количество дней за последние 30 дней, когда ваше психическое здоровье было неудовлетворительно (0–30 дней)",
    "PhysHlth": "Количество дней за последние 30 дней, когда ваше физическое здоровье было неудовлетворительно (0–30 дней)",
    "DiffWalk": "Испытываете ли серьёзные трудности при пеших прогулках или подъёме по лестнице? (0 = нет; 1 = да)",
    "Sex": "Ваш пол (0 = женский; 1 = мужской)",
    "Age": "Возрастная группа (1 = 18-24; 2 = 25-29; … ; 13 = старше 80)",
    "Education": "Уровень образования (1 = дошкольник; 2 = начальная школа; 3 = средняя школа; 4 = старшая школа; 5 = среднее специальное или бакалавриат; 6 = магистратура)",
    "Income": "Годовой доход ( $ ) (1 = <10k; 2 = <15k; 3 = <20k; 4 = <25k; 5 = <35k; 6 = <50k; 7 = <75k; 8 = >75k)"
}

class SurveyForm(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data = {}
        self.question_order = list(reversed(list(questions.keys())))
        self.current_question_idx = len(self.question_order) - 1
        self.ask_next_question()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.label = QLabel()
        self.input_field = QLineEdit()
        self.button = QPushButton("Записать ответ")
        self.button.clicked.connect(self.save_answer_and_continue)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.input_field)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        self.setWindowTitle("Анкетирование здоровья")

    def ask_next_question(self):
        if self.current_question_idx >= 0:
            question_key = self.question_order[self.current_question_idx]
            self.label.setText(f"{question_key}: {questions[question_key]}")
            self.input_field.clear()
            self.input_field.setFocus()
        else:
            self.show_results()

    def save_answer_and_continue(self):
        answer_text = self.input_field.text().strip()
        current_question = self.question_order[self.current_question_idx]

        if not answer_text:
            QMessageBox.warning(self, "Предупреждение", "Необходимо ввести ответ.")
            return

        try:
            answer_value = float(answer_text)
            self.data[current_question] = answer_value
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Неверный формат ответа! Введите число.")
            return

        # Переходим к следующему вопросу
        self.current_question_idx -= 1
        self.ask_next_question()

    def show_results(self):
        # Форматируем введённые данные для отображения
        results_string = ", ".join([f"{key}={val}" for key, val in self.data.items()])
        
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

        # Здесь закрываем старое окно опроса
        self.close()

        # Открываем диалоговое окно выбора модели
        dialog.exec()

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

def main():
    app = QApplication(sys.argv)
    window = SurveyForm()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

#Описание работы программы:

#Вопросы и порядок: Программа начинается с последнего вопроса и постепенно движется к первому, показывая их пользователю один за другим.


#Интерфейс: Каждый новый вопрос появляется вместе с полем для ввода ответа и кнопкой "Записать ответ". Пользователь должен заполнить поле и подтвердить ответ.


#Проверка ввода: Все ответы обязательно должны быть числами (например, целые числа 0 или 1 для бинарных вопросов, и вещественные числа для индекса массы тела). Если введен неправильный формат ответа, появится предупреждение.


#Завершение анкеты: Когда пользователь заполнит все вопросы, появится информационное окно с результатами и предложением продолжить обработку данных.


#Формат сохранения: Итоговая строка выглядит следующим образом:

#HighBP=1.0, HighChol=1.0, ..., Income=3.0

#Эти данные легко сопоставляются с форматом входных данных вашей обученной модели, которая должна уметь интерпретировать такую строку и выдавать прогноз.

#Таким образом, этот скрипт обеспечивает удобный интерактивный способ сбора данных для последующего анализа состояния здоровья пациента.