import os
import sys
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QComboBox,
    QLabel,
    QLineEdit,
    QMessageBox, 
    QApplication
)
from PySide6.QtGui import QStandardItemModel
from PySide6.QtCore import Slot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class DatasetSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Выбор и оценка моделей')
        layout = QVBoxLayout()
        # 1. Кнопка "Выбор датасета"
        select_dataset_button = QPushButton("Выбор датасета")
        select_dataset_button.clicked.connect(self.select_dataset)
        layout.addWidget(select_dataset_button)
        # Добавляем QLabel для отображения названия датасета
        self.dataset_label = QLabel('Нет выбранного датасета.')
        layout.addWidget(self.dataset_label)
        # 2. Комбобокс для выбора целевой переменной
        self.target_variable_combo = QComboBox()
        layout.addWidget(QLabel("Выберите целевую переменную:"))
        layout.addWidget(self.target_variable_combo)
        # 3. Метка для отображения количества экземпляров по классам
        self.class_count_label = QLabel("")
        layout.addWidget(self.class_count_label)
        # 4. Текстовое поле для ввода количества записей для тестирования
        self.number_input_field = QLineEdit()
        self.number_input_field.setPlaceholderText("Количество записей для тестирования")
        layout.addWidget(self.number_input_field)
        # 5. Кнопка "Отобрать записи для теста"
        test_records_button = QPushButton("Отобрать записи для теста")
        test_records_button.clicked.connect(self.test_data_selection)  # Сигнал подключаем ПОСЛЕ определения метода!
        layout.addWidget(test_records_button)
        # 6. Метка для отображения отобранных записей
        self.random_samples_label = QLabel("")
        layout.addWidget(self.random_samples_label)
        # 7. Кнопка выбора модели
        self.model_selection_button = QPushButton("Выбор модели")  # Сделайте ссылку на кнопку доступной в классе
        self.model_selection_button.clicked.connect(self.select_model)
        layout.addWidget(self.model_selection_button)
        # 8. Кнопка "Запустить тестирование"
        run_test_button = QPushButton("Запустить тестирование")
        run_test_button.clicked.connect(self.run_test)
        layout.addWidget(run_test_button)
        # 9. Метка для отображения результатов тестирования
        self.result_label = QLabel("")
        layout.addWidget(self.result_label)
        # 10. Кнопка "Сохранить отчет точности"
        save_report_button = QPushButton("Сохранить отчёт точности")
        save_report_button.clicked.connect(self.save_report)
        layout.addWidget(save_report_button)
        self.setLayout(layout)
        
    def select_dataset(self):
        file_dialog = QFileDialog()
        filename, _ = file_dialog.getOpenFileName(self, 'Выбрать датасет', './dataset/', 'CSV Files (*.csv)')
        if not filename:
            return
        try:
            df = pd.read_csv(filename)
            self.df = df  
            self.dataset_path = filename
            # Очищаем список колонок и добавляем новые
            self.target_variable_combo.clear()
            feature_cols = list(df.columns)
            for col in feature_cols:
                self.target_variable_combo.addItem(col)
            # Устанавливаем первую колонку по умолчанию
            first_column = feature_cols[0]
            self.update_class_distribution(first_column)
            # Обновляем метку с названием датасета
            self.dataset_label.setText(f"Датасет: {filename}")
            # Подключаем обработчик события для изменения целевой переменной
            self.target_variable_combo.currentTextChanged.connect(self.update_class_distribution)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при чтении датасета:\\\\n{e}")
            
    def update_class_distribution(self, selected_class):
        try:# Подсчитываем количество записей по каждому классу
            classes_info = self.df[selected_class].value_counts().sort_index()            
            # Преобразуем классы в пары индекс-количество
            values_list = list(classes_info.items())
            third_length = len(values_list) // 3 
            # Разбиваем список на три части
            first_column = values_list[:third_length]
            second_column = values_list[third_length:2*third_length]
            third_column = values_list[2*third_length:]
            # Создаем строковую переменную для отображения
            distribution_message = ""
            # Проходим по максимуму длин колонок
            max_len = max(len(first_column), len(second_column), len(third_column))
            for i in range(max_len):
                # Элементы первой колонки
                if i < len(first_column):
                    first_value = first_column[i]
                else:
                    first_value = ("", "")                
                # Элементы второй колонки
                if i < len(second_column):
                    second_value = second_column[i]
                else:
                    second_value = ("", "")                
                # Элементы третьей колонки
                if i < len(third_column):
                    third_value = third_column[i]
                else:
                    third_value = ("", "")                
                # Формирование строки с тремя колонками
                line = (
                    f"{first_value[0]:<15}{first_value[1]:>5}\t\t"
                    f"{second_value[0]:<15}{second_value[1]:>5}\t\t"
                    f"{third_value[0]:<15}{third_value[1]:>5}"
                )
                distribution_message += line + "\n"
            # Обновляем метку с информацией о классах
            self.class_count_label.setText(f"Распределение классов:\n{distribution_message}")        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при расчете распределения классов:\n{e}")
            
    def test_data_selection(self):
        number_to_select = int(self.number_input_field.text())
        selected_class = self.target_variable_combo.currentText()        
        try:
            random_samples_per_class = {}
            display_samples_per_class = {}
            unique_classes = self.df[selected_class].unique()
            unique_classes_for_display = unique_classes[:3]
            
            # Для всех классов делаем две вещи:
            # 1. Отбираем нужное количество записей для моделирования (number_to_select)
            # 2. Дополнительно отбираем по три записи для отображения в интерфейсе
            for cls in unique_classes:
                subset = self.df[self.df[selected_class] == cls]                
                # Выбираем нужное количество записей для моделирования
                model_sample = subset.sample(min(number_to_select, len(subset)))
                random_samples_per_class[cls] = model_sample                
                # Только для первых трёх классов дополнительно берем по три записи для отображения
                if cls in unique_classes_for_display:
                    display_subset = subset.sample(min(3, len(subset)))  # Берём максимум 3 записи
                    display_samples_per_class[cls] = display_subset            
            # Готовим данные для моделирования (объединяем все отобранные записи)
            random_sample_model = pd.concat(list(random_samples_per_class.values()), ignore_index=True)
            self.random_sample = random_sample_model
            # Подготавливаем данные для отображения (только первые три класса и по три записи на класс)
            random_sample_display = pd.concat(list(display_samples_per_class.values()), ignore_index=True)            
            # Форматируем текст для отображения в интерфейсе
            samples_text = ''
            for idx, row in random_sample_display.iterrows():
                record_values = ', '.join([f'{col}: {val}' for col, val in zip(row.index, row)])
                samples_text += f'\nЗапись №{idx+1}: {record_values}'            
            # Выводим результат в интерфейс
            self.random_samples_label.setText(samples_text)        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при отборе записей:\\n{e}")
            
    def select_model(self):
        # Открываем диалог выбора модели из папки trained_models
        model_file, _ = QFileDialog.getOpenFileName(self, 'Выбрать модель', './trained_models/', 'Model files (*.pkl *.joblib)')
        if model_file:
            self.model_file = model_file  
            # Получаем базовое имя файла модели (без расширения и полного пути)
            model_basename = os.path.basename(model_file).split('.')[0]
            # Обновляем текст кнопки
            self.model_selection_button.setText(f"{model_basename}")
            print(f"Модель успешно выбрана: {model_file}")
        else:
            QMessageBox.warning(self, "Предупреждение", "Файл модели не выбран.")

    def run_test(self):
        try:# Загружаем выбранную модель
            model = joblib.load(self.model_file)
            print("\\n--- Содержимое random_sample ---")
            #print(self.random_sample.to_string(index=False))
            print("-------------------------------")
            # Проверяем наличие данных
            if hasattr(self, 'random_sample'):
                print("Условие выполнено: данные для тестирования найдены.")
                X_test = self.random_sample.drop(columns=[self.target_variable_combo.currentText()])
                y_true = self.random_sample[self.target_variable_combo.currentText()]
                # Прогнозируем
                y_pred = model.predict(X_test)
                # Оцениваем точность модели
                accuracy = accuracy_score(y_true, y_pred)
                result_text = f"Точность модели: {accuracy:.2%}"
                # Создаем отчет по каждому классу
                results_by_class = {}
                for cls in y_true.unique():
                    correct = sum((y_pred == y_true)[y_true == cls])
                    total = len(y_true[y_true == cls])
                    results_by_class[cls] = f"({total}) из них верно предсказаны: {correct}"
                detailed_results = '\n'.join(results_by_class.values())
                final_result = f"{result_text}\n\nДетали по классам:\n{detailed_results}"
                self.result_label.setText(final_result)
            else:
                QMessageBox.warning(self, "Предупреждение", "Сначала нужно отобрать записи для теста.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при выполнении теста:\n{e}")
        
    # Обновляем метод save_report
    def save_report(self):        
        # Формируем имя файла на основе модели и датасета
        model_name = os.path.splitext(os.path.basename(self.model_file))[0]
        dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        report_filename = f'{model_name}_{dataset_name}_report.txt'        
        # Полный путь к файлу отчета
        report_dir = 'accuracy_reports_tested_models'
        full_report_path = os.path.join(report_dir, report_filename)        
        # Создаем директорию, если её ещё нет
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)            
        # Сохраняем отчёт
        with open(full_report_path, 'w') as f:
            f.write(f'Модель: {os.path.basename(self.model_file)}\\n'
                f'Датасет: {self.dataset_path}\\n'
                f'Результат: {self.result_label.text()}')        
        QMessageBox.information(self, "Успех", f"Отчёт сохранён в {full_report_path}")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetSelectionWindow()
    window.show()
    sys.exit(app.exec())
