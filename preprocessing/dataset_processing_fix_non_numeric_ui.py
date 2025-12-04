from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from PySide6.QtCore import Qt
# Метод Binary Encoding
from category_encoders.binary import BinaryEncoder
#from preprocessing.methods_handling_non_numeric_values_ui import DataEncodingWindow
import pandas as pd
import os
import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

class OneHotEncodingWindow(QWidget):
    def __init__(self, dataset=None):
        super().__init__()
                # Настройка размеров окна
        self.setMinimumSize(400, 300)     
        self.resize(400, 300)          
        self.setWindowTitle("Обработка датасета")
        # Главный вертикальный макет
        main_layout = QVBoxLayout()  

        # Загрузка датасета
        self.dataset_df = dataset
        self.btn_select_dataset = QPushButton('Выбрать датасет')
        self.btn_select_dataset.clicked.connect(self.select_raw_dataset)
        main_layout.addWidget(self.btn_select_dataset)

        # Кнопка показа нечисловых признаков
        btn_show_non_numeric = QPushButton('Показать нечисловые значения')
        btn_show_non_numeric.clicked.connect(self.display_unique_values)
        main_layout.addWidget(btn_show_non_numeric)
        # Верхняя панель с выбором колонки и методами
        top_panel = QHBoxLayout()
        
        # Список нечисловых колонок
        self.column_selector = QComboBox()
        top_panel.addWidget(self.column_selector)
        
        # Таблица для вывода уникальных значений
        self.table_widget = QTableWidget()
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.verticalHeader().hide()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(['Колонка', 'Значения'])
        main_layout.addWidget(self.table_widget)
        
        # Верхняя панель с выбором колонки и методами
        top_panel = QHBoxLayout()
        
        # Список нечисловых колонок
        self.column_selector = QComboBox()
        top_panel.addWidget(self.column_selector)

        # Раздел методов обработки
        methods_layout = QVBoxLayout()
        method_buttons = [
            ("One-Hot Encoding", self.process_one_hot_encoding),
            ("Label Encoding", self.process_label_encoding),
            ("Target Encoding", self.process_target_encoding),
            ("Frequency Encoding", self.process_frequency_encoding),
            ("Binary Encoding", self.process_binary_encoding),
            ("Обработать как дату", self.process_date_column)  
        ]

        for name, func in method_buttons:
            hbox = QHBoxLayout()
            button_method = QPushButton(name)
            button_help = QPushButton("Справка")
            button_method.clicked.connect(lambda checked=False, f=func: self.apply_method(f))
            button_help.clicked.connect(lambda checked=False, n=name: self.show_help(n))
            hbox.addWidget(button_method)
            hbox.addWidget(button_help)
            methods_layout.addLayout(hbox)

        top_panel.addLayout(methods_layout)
        main_layout.addLayout(top_panel)

        # === КНОПКА: Удалить выбранную категорию ===
        remove_button = QPushButton("🗑️ Удалить выбранную категорию")
        remove_button.setStyleSheet("color: red; font-weight: bold;")
        remove_button.clicked.connect(self.remove_selected_column)
        main_layout.addWidget(remove_button)

        # === КНОПКА: Сохранить датасет ===
        save_button = QPushButton('💾 Сохранить датасет')
        save_button.clicked.connect(self.save_processed_dataset)
        main_layout.addWidget(save_button)
        
        self.setLayout(main_layout)
        
    def remove_selected_column(self):
        """Удаляет выбранный столбец из датасета"""
        column_name = self.column_selector.currentText()
        if not column_name:
            QMessageBox.warning(self, "Предупреждение", "Выберите столбец для удаления!")
            return

        if self.dataset_df is None:
            QMessageBox.critical(self, "Ошибка", "Датасет не загружен!")
            return

        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Вы действительно хотите удалить столбец '{column_name}'?"
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self.dataset_df.drop(columns=[column_name], inplace=True)
            QMessageBox.information(self, "Готово", f"Столбец '{column_name}' удалён.")

            # Обновляем отображение
            self.display_unique_values()  # автоматически очистит комбобокс и таблицу
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось удалить столбец:\n{e}")
    # Окно помощи
    def show_help(self, method_name):
        help_text = {
            "One-Hot Encoding": "Метод преобразования категориальных переменных в бинарные признаки.",
            "Label Encoding": "Кодирование категориальных переменных числовыми индексами.",
            "Target Encoding": "Замещение категорий средними значениями целевой переменной.",
            "Frequency Encoding": "Замена категорий частотностью встречаемости.",
            "Binary Encoding": "Использование бинарного представления индекса категорий.",
            "Обработать как дату":("Преобразует столбец с датами (например, 4/02/2016) в числовые признаки:\n"
                "• Год\n• Месяц\n• День\n• День недели\n• Номер недели\n• Квартал\n\n"
                "Позволяет использовать временные паттерны в обучении модели."
            )
        }
        QMessageBox.information(self, f"Справка: {method_name}", help_text.get(method_name, ""))

    # Выбор датасета
    def select_raw_dataset(self):
        file_dialog = QFileDialog()
        filename, _ = file_dialog.getOpenFileName(self, 'Выбрать датасет', './dataset', 'CSV Files (*.csv)')
        if not filename:
            return
        try:
            self.dataset_df = pd.read_csv(filename)
            basename = os.path.basename(filename)
            self.btn_select_dataset.setText(f'Файл загружен: {basename}')
        except Exception as e:
            QMessageBox.critical(None, "Ошибка", f"Произошла ошибка при чтении датасета:\n{e}")
            
    # Сохранение обработанного датасета
    def save_processed_dataset(self):
        if self.dataset_df is None:
            QMessageBox.warning(self, "Предупреждение", "Датасет ещё не загружен или не обработан!")
            return
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self,"Сохранить обработанный датасет","","CSV Files (*.csv);;All Files (*) ",options=options)
        if filename:
            try:
                self.dataset_df.to_csv(filename, index=False)
                QMessageBox.information(self, "Готово", "Датасет сохранён успешно!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Возникла проблема при сохранении файла:\n{e}")

    # Метод для отображения уникальных значений нечисловых столбцов
    def display_unique_values(self):
        if self.dataset_df is None:
            QMessageBox.critical(self, "Ошибка", "Датасет не выбран!")
            return
        
        # Исключаем числовые и булевые столбцы
        excluded_types = ["number", "bool"]
        non_numeric_and_non_bool_columns = self.dataset_df.select_dtypes(exclude=excluded_types).columns.tolist()
        
        if len(non_numeric_and_non_bool_columns) > 0:
            rows_to_display = min(len(non_numeric_and_non_bool_columns), 20)  # Ограничиваем число отображаемых колонок до 20
            self.table_widget.clearContents()
            self.table_widget.setRowCount(rows_to_display)
            
            row_idx = 0
            for column in non_numeric_and_non_bool_columns[:rows_to_display]:
                unique_values = self.dataset_df[column].unique()
                value_string = ', '.join(map(str, unique_values))
                
                item_column_name = QTableWidgetItem(column)
                item_value_list = QTableWidgetItem(value_string)
                
                self.table_widget.setItem(row_idx, 0, item_column_name)
                self.table_widget.setItem(row_idx, 1, item_value_list)
                row_idx += 1
                
            # Заполняем выпадающий список колонок
            self.column_selector.clear()
            self.column_selector.addItems(non_numeric_and_non_bool_columns)
            
            # Приспосабливаем размер окна под новую высоту таблицы
            self.adjustSize()
        else:
            QMessageBox.information(self, "Информация", "Нет столбцов, кроме числовых и булевых.")
    # Применение выбранного метода
    def apply_method(self, method_func):
        selected_column = self.column_selector.currentText()
        if not selected_column:
            QMessageBox.warning(self, "Предупреждение", "Необходимо выбрать колонку!")
            return        
        # Здесь применяем указанный метод к выбранной колонке
        method_func(selected_column) 
        
    # Методы выборки нечисловых колонок
    def select_non_numeric_columns(self):
        """
        Возвращает список всех нечисловых колонок из датасета.
        """
        if self.dataset_df is None:
            raise ValueError("Датасет не выбран!")
        return self.dataset_df.select_dtypes(exclude=['number']).columns.tolist()
           
    def process_one_hot_encoding(self, column_name=None):
        non_numeric_columns = self.select_non_numeric_columns()
        if column_name:
            # Если указана конкретная колонка, работаем только с ней
            one_hot_cols = [column_name]
        elif len(non_numeric_columns) > 0:
            # Иначе используем все категориальные колонки
            one_hot_cols = non_numeric_columns
        else:
            QMessageBox.information(self, "Информация", "Нечисловых столбцов нет.")
            return
        
        # Выполняем One-Hot-кодирование
        encoded_dataframe = pd.get_dummies(self.dataset_df, columns=one_hot_cols)
        
        self.dataset_df = encoded_dataframe
        QMessageBox.information(self, "Преобразование", "One-Hot кодирование применено успешно!")
    # Метод Label Encoding
    def process_label_encoding(self, column_name=None):
            if self.dataset_df is None:
                QMessageBox.critical(self, "Ошибка", "Датасет не выбран!")
                return
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            non_numeric_columns = self.select_non_numeric_columns()
            if column_name:
                # Работаем только с выбранной колонкой
                cols_to_encode = [column_name]
            else:
                cols_to_encode = non_numeric_columns
            
            for col in cols_to_encode:
                self.dataset_df[col] = le.fit_transform(self.dataset_df[col]).astype(int)
            
            QMessageBox.information(self, "Преобразование", f"Label Encoding применён успешно для {len(cols_to_encode)} столбцов!")

    # Метод Target Encoding
    def process_target_encoding(self, column_name=None):
        if self.dataset_df is None or 'target' not in self.dataset_df.columns:
            QMessageBox.critical(self, "Ошибка", "Датасет не выбран или отсутствует целевая переменная!")
            return
        target_col = 'target'
        non_numeric_columns = self.select_non_numeric_columns()
        if column_name:
            # Работаем только с выбранной колонкой
            cols_to_encode = [column_name]
        else:
            cols_to_encode = non_numeric_columns
        
        for col in cols_to_encode:
            mean_map = self.dataset_df.groupby(col)[target_col].mean().to_dict()
            new_col_name = f"{col}_encoded"
            self.dataset_df[new_col_name] = self.dataset_df[col].map(mean_map).astype(float)
        
        QMessageBox.information(self, "Преобразование", f"Target Encoding применён успешно для {len(cols_to_encode)} столбцов!")

    # Метод Frequency Encoding
    def process_frequency_encoding(self, column_name=None):
        if self.dataset_df is None:
            QMessageBox.critical(self, "Ошибка", "Датасет не выбран!")
            return
        non_numeric_columns = self.select_non_numeric_columns()
        if column_name:
            # Работаем только с выбранной колонкой
            cols_to_encode = [column_name]
        else:
            cols_to_encode = non_numeric_columns
        
        for col in cols_to_encode:
            freq_map = self.dataset_df[col].value_counts(normalize=True).to_dict()
            new_col_name = f"{col}_freq_encoded"
            self.dataset_df[new_col_name] = self.dataset_df[col].map(freq_map).astype(float)
        
        QMessageBox.information(self, "Преобразование", f"Frequency Encoding применён успешно для {len(cols_to_encode)} столбцов!")

    def process_binary_encoding(self, column_name=None):
        if self.dataset_df is None:
            QMessageBox.critical(self, "Ошибка", "Датасет не выбран!")
            return
        from category_encoders import BinaryEncoder
        non_numeric_columns = self.select_non_numeric_columns()
        if column_name:
            # Работаем только с выбранной колонкой
            cols_to_encode = [column_name]
        else:
            cols_to_encode = non_numeric_columns
        
        binary_encoder = BinaryEncoder(cols=cols_to_encode)
        transformed_data = binary_encoder.fit_transform(self.dataset_df)
        self.dataset_df = transformed_data
        QMessageBox.information(self, "Преобразование", "Binary Encoding применён успешно!")
        
    def process_date_column(self, column_name=None):
        if self.dataset_df is None:
            QMessageBox.critical(self, "Ошибка", "Датасет не загружен!")
            return

        if not column_name:
            column_name = self.column_selector.currentText()
            if not column_name:
                QMessageBox.warning(self, "Предупреждение", "Выберите столбец с датой!")
                return

        try:
            series = self.dataset_df[column_name]

            # 🟩 Используем несколько форматов, как в detect_date_columns
            date_formats = [
                '%m/%d/%Y',    # 4/02/2016
                '%m/%d/%y',    # 4/02/16
                '%d/%m/%Y',    # 02/04/2016
                '%Y-%m-%d',    # 2016-04-02
                '%d.%m.%Y',    # 02.04.2016
                '%Y/%m/%d',
            ]

            # Пытаемся распарсить с помощью каждого формата
            parsed_series = pd.Series([pd.NaT] * len(series), dtype='datetime64[ns]')
            remaining = series.copy()

            for fmt in date_formats:
                # Убираем уже распознанные
                not_parsed = parsed_series.isna()
                if not not_parsed.any():
                    break

                temp = remaining[not_parsed]
                if len(temp) == 0:
                    continue

                # Пытаемся распарсить оставшиеся с этим форматом
                converted = pd.to_datetime(temp, format=fmt, errors='coerce')
                # Обновляем только где получилось
                parsed_series[not_parsed] = converted.where(
                    converted.notna(),
                    parsed_series[not_parsed]
                )

            # Если ни один формат не помог — попробуем auto-parse (fallback)
            still_na = parsed_series.isna()
            if still_na.any():
                fallback = pd.to_datetime(remaining[still_na], errors='coerce')
                parsed_series[still_na] = fallback

            # Теперь parsed_series — результат
            valid_count = parsed_series.notna().sum()
            total_count = len(series)

            if valid_count == 0:
                QMessageBox.critical(self, "Ошибка", f"Не удалось распознать ни одну дату в столбце '{column_name}'")
                return

            # 🟩 Извлекаем компоненты
            self.dataset_df[f"{column_name}_year"] = parsed_series.dt.year.astype('Int64')
            self.dataset_df[f"{column_name}_month"] = parsed_series.dt.month.astype('Int64')
            self.dataset_df[f"{column_name}_day"] = parsed_series.dt.day.astype('Int64')
            self.dataset_df[f"{column_name}_dayofweek"] = parsed_series.dt.dayofweek.astype('Int64')
            self.dataset_df[f"{column_name}_week"] = parsed_series.dt.isocalendar().week.astype('Int64')
            self.dataset_df[f"{column_name}_quarter"] = parsed_series.dt.quarter.astype('Int64')

            # Отчёт
            reply = QMessageBox.question(
                self, "Удалить исходный столбец?",
                f"Распознано: {valid_count}/{total_count} валидных дат.\nУдалить столбец '{column_name}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.dataset_df.drop(columns=[column_name], inplace=True)

            QMessageBox.information(
                self, "Успех",
                f"Дата успешно обработана!\n"
                f"Созданы признаки: год, месяц, день, день недели, неделя, квартал."
            )
            self.display_unique_values()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обработке даты:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OneHotEncodingWindow()
    window.show()
    sys.exit(app.exec())