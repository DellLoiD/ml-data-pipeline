# preprocessing/one_hot_encoding_ui.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QMessageBox, QTableWidget, QTableWidgetItem, QComboBox, QLabel
)
from PySide6.QtCore import Qt
import pandas as pd
import os
import sys


class OneHotEncodingWindow(QWidget):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset_df = dataset
        self._meta_line = "# META:"  # Хранение строки метаданных
        self._has_changes = False  # Контроль кнопки сохранения
        self._last_loaded_path = None

        # Настройка окна
        self.setMinimumSize(400, 300)
        self.resize(500, 500)
        self.setWindowTitle("Обработка категориальных признаков")

        # Главный макет
        main_layout = QVBoxLayout()

        # === Кнопка: Выбрать датасет ===
        self.btn_select_dataset = QPushButton('📂 Выбрать датасет')
        self.btn_select_dataset.clicked.connect(self.select_raw_dataset)
        main_layout.addWidget(self.btn_select_dataset)

        # === Кнопка: Показать нечисловые значения ===
        btn_show_non_numeric = QPushButton('🔍 Показать нечисловые значения')
        btn_show_non_numeric.clicked.connect(self.display_unique_values)
        main_layout.addWidget(btn_show_non_numeric)

        # === Таблица: Уникальные значения ===
        self.table_widget = QTableWidget()
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.verticalHeader().hide()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(['Колонка', 'Значения'])
        main_layout.addWidget(self.table_widget)

        # === Выбор колонки ===
        top_panel = QHBoxLayout()
        self.column_selector = QComboBox()
        self.column_selector.setPlaceholderText("Выберите колонку")
        top_panel.addWidget(self.column_selector)

        # === Методы кодирования ===
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
            button_help = QPushButton("?")
            button_method.clicked.connect(lambda checked=False, f=func: self.apply_method(f))
            button_help.clicked.connect(lambda checked=False, n=name: self.show_help(n))
            button_method.setMinimumHeight(30)
            button_help.setFixedSize(30, 30)
            hbox.addWidget(button_method)
            hbox.addWidget(button_help)
            methods_layout.addLayout(hbox)

        top_panel.addLayout(methods_layout)
        main_layout.addLayout(top_panel)

        # === Кнопка: Удалить выбранную категорию ===
        remove_button = QPushButton("🗑️ Удалить выбранную колонку")
        remove_button.setStyleSheet("color: red; font-weight: bold;")
        remove_button.clicked.connect(self.remove_selected_column)
        main_layout.addWidget(remove_button)

        # === Кнопка: Сохранить датасет ===
        self.save_button = QPushButton('💾 Сохранить датасет')
        self.save_button.clicked.connect(self.save_processed_dataset)
        self.save_button.setEnabled(False)  # Активна только после изменений
        main_layout.addWidget(self.save_button)

        self.setLayout(main_layout)
        self.reset_ui()

    def reset_ui(self):
        """Сброс всех полей"""
        self.dataset_df = None
        self._meta_line = "# META:"
        self._has_changes = False
        self._last_loaded_path = None
        self.btn_select_dataset.setText('📂 Выбрать датасет')
        self.column_selector.clear()
        self.table_widget.setRowCount(0)
        self.save_button.setEnabled(False)

    def select_raw_dataset(self):
        """Загрузка датасета с учётом #META"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Выбрать датасет', './dataset', 'CSV Files (*.csv)'
        )
        if not filename:
            return

        try:
            # Читаем #META строку
            with open(filename, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            if first_line.startswith("# META:"):
                self._meta_line = first_line
            else:
                self._meta_line = "# META:"

            # Загружаем CSV, игнорируя строки с комментариями
            self.dataset_df = pd.read_csv(filename, comment='#')
            self._last_loaded_path = filename

            basename = os.path.basename(filename)
            self.btn_select_dataset.setText(f'✅ Файл загружен: {basename}')

            # Обновляем отображение
            self.display_unique_values()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")
            self.reset_ui()

    def display_unique_values(self):
        """Отображение нечисловых колонок и их уникальных значений"""
        if self.dataset_df is None:
            QMessageBox.critical(self, "Ошибка", "Датасет не выбран!")
            return

        excluded_types = ["number", "bool"]
        non_numeric_columns = self.dataset_df.select_dtypes(exclude=excluded_types).columns.tolist()

        # Ограничим до 20 колонок
        rows_to_display = min(len(non_numeric_columns), 20)
        self.table_widget.clearContents()
        self.table_widget.setRowCount(rows_to_display)

        row_idx = 0
        for col in non_numeric_columns[:rows_to_display]:
            unique_vals = self.dataset_df[col].dropna().unique()
            value_string = ', '.join(map(str, unique_vals))
            self.table_widget.setItem(row_idx, 0, QTableWidgetItem(col))
            self.table_widget.setItem(row_idx, 1, QTableWidgetItem(value_string))
            row_idx += 1

        # Обновляем комбобокс
        self.column_selector.clear()
        if non_numeric_columns:
            self.column_selector.addItems(non_numeric_columns)
        else:
            self.column_selector.addItem("Нет нечисловых колонок")

        self._has_changes = False
        self.save_button.setEnabled(False)

    def remove_selected_column(self):
        """Удаление выбранной колонки"""
        column_name = self.column_selector.currentText()
        if not column_name or column_name == "Нет нечисловых колонок":
            QMessageBox.warning(self, "Предупреждение", "Выберите колонку для удаления!")
            return

        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Удалить столбец '{column_name}'?"
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self.dataset_df.drop(columns=[column_name], inplace=True)
            self._meta_line += f", удалён столбец '{column_name}'"
            self._has_changes = True
            self.save_button.setEnabled(True)
            QMessageBox.information(self, "Готово", f"Столбец '{column_name}' удалён.")
            self.display_unique_values()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось удалить столбец:\n{e}")

    def apply_method(self, method_func):
        """Применение выбранного метода"""
        column_name = self.column_selector.currentText()
        if not column_name or column_name == "Нет нечисловых колонок":
            QMessageBox.warning(self, "Предупреждение", "Выберите колонку для обработки!")
            return

        # Применяем метод
        try:
            method_func(column_name)
            self._has_changes = True
            self.save_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при применении метода:\n{e}")

    def show_help(self, method_name):
        help_text = {
            "One-Hot Encoding": "Преобразует категориальные переменные в бинарные признаки (0/1). Создаёт новую колонку для каждого уникального значения.",
            "Label Encoding": "Кодирует категории целыми числами (0, 1, 2, ...). Подходит для ординальных переменных.",
            "Target Encoding": "Заменяет категорию средним значением целевой переменной для этой категории.",
            "Frequency Encoding": "Заменяет категорию долей её встречаемости в датасете.",
            "Binary Encoding": "Преобразует категорию в бинарное представление её индекса. Требует меньше колонок, чем One-Hot.",
            "Обработать как дату": (
                "Извлекает числовые признаки из даты:\n"
                "• Год\n• Месяц\n• День\n• День недели\n• Неделя года\n• Квартал\n\n"
                "Поддерживает форматы: 4/02/2016, 2016-04-02, 02.04.2016 и др."
            )
        }
        QMessageBox.information(self, f"Справка: {method_name}", help_text.get(method_name, ""))

    def select_non_numeric_columns(self):
        """Возвращает нечисловые колонки"""
        if self.dataset_df is None:
            raise ValueError("Датасет не выбран!")
        return self.dataset_df.select_dtypes(exclude=['number']).columns.tolist()

    def process_one_hot_encoding(self, column_name):
        if column_name not in self.dataset_df.columns:
            QMessageBox.warning(self, "Ошибка", f"Колонка '{column_name}' не найдена!")
            return

        try:
            encoded_df = pd.get_dummies(self.dataset_df, columns=[column_name])
            self.dataset_df = encoded_df
            self._meta_line += f", One-Hot Encoding для '{column_name}'"
            QMessageBox.information(self, "Готово", f"One-Hot Encoding применён к '{column_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить One-Hot Encoding:\n{e}")

    def process_label_encoding(self, column_name):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        try:
            self.dataset_df[column_name] = le.fit_transform(self.dataset_df[column_name].astype(str))
            self._meta_line += f", Label Encoding для '{column_name}'"
            QMessageBox.information(self, "Готово", f"Label Encoding применён к '{column_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при Label Encoding:\n{e}")

    def process_target_encoding(self, column_name):
        if 'target' not in self.dataset_df.columns:
            QMessageBox.critical(self, "Ошибка", "Отсутствует колонка 'target' для Target Encoding!")
            return
        try:
            mean_map = self.dataset_df.groupby(column_name)['target'].mean().to_dict()
            new_col = f"{column_name}_encoded"
            self.dataset_df[new_col] = self.dataset_df[column_name].map(mean_map)
            self._meta_line += f", Target Encoding для '{column_name}'"
            QMessageBox.information(self, "Готово", f"Target Encoding применён к '{column_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при Target Encoding:\n{e}")

    def process_frequency_encoding(self, column_name):
        try:
            freq_map = self.dataset_df[column_name].value_counts(normalize=True).to_dict()
            new_col = f"{column_name}_freq_encoded"
            self.dataset_df[new_col] = self.dataset_df[column_name].map(freq_map)
            self._meta_line += f", Frequency Encoding для '{column_name}'"
            QMessageBox.information(self, "Готово", f"Frequency Encoding применён к '{column_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при Frequency Encoding:\n{e}")

    def process_binary_encoding(self, column_name):
        try:
            from category_encoders import BinaryEncoder
            encoder = BinaryEncoder(cols=[column_name])
            encoded_df = encoder.fit_transform(self.dataset_df)
            self.dataset_df = encoded_df
            self._meta_line += f", Binary Encoding для '{column_name}'"
            QMessageBox.information(self, "Готово", f"Binary Encoding применён к '{column_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при Binary Encoding:\n{e}")

    def process_date_column(self, column_name):
        if column_name not in self.dataset_df.columns:
            QMessageBox.warning(self, "Ошибка", f"Колонка '{column_name}' не найдена!")
            return

        series = self.dataset_df[column_name]
        date_formats = [
            '%m/%d/%Y', '%m/%d/%y',
            '%d/%m/%Y', '%Y-%m-%d',
            '%d.%m.%Y', '%Y/%m/%d'
        ]

        parsed_series = pd.to_datetime(series, format='mixed', errors='coerce')

        if parsed_series.isna().all():
            QMessageBox.critical(self, "Ошибка", f"Не удалось распознать ни одну дату в '{column_name}'.")
            return

        # Извлекаем признаки
        self.dataset_df[f"{column_name}_year"] = parsed_series.dt.year.astype('Int64')
        self.dataset_df[f"{column_name}_month"] = parsed_series.dt.month.astype('Int64')
        self.dataset_df[f"{column_name}_day"] = parsed_series.dt.day.astype('Int64')
        self.dataset_df[f"{column_name}_dayofweek"] = parsed_series.dt.dayofweek.astype('Int64')
        self.dataset_df[f"{column_name}_week"] = parsed_series.dt.isocalendar().week.astype('Int64')
        self.dataset_df[f"{column_name}_quarter"] = parsed_series.dt.quarter.astype('Int64')

        # Предлагаем удалить оригинальный столбец
        reply = QMessageBox.question(
            self, "Удалить оригинал?",
            f"Распознано: {parsed_series.notna().sum()}/{len(series)}\nУдалить '{column_name}'?"
        )
        if reply == QMessageBox.Yes:
            self.dataset_df.drop(columns=[column_name], inplace=True)
            self._meta_line += f", обработана как дата, удалена колонка '{column_name}'"
        else:
            self._meta_line += f", обработана как дата, колонка '{column_name}' сохранена"

        QMessageBox.information(self, "Успех", "Дата успешно разбита на признаки.")
        self._has_changes = True
        self.save_button.setEnabled(True)
        self.display_unique_values()

    def save_processed_dataset(self):
        """Сохранение с обновлением #META и версионированием"""
        if self.dataset_df is None or not self._has_changes:
            QMessageBox.warning(self, "Предупреждение", "Нет изменений для сохранения.")
            return

        # Определяем имя и версию
        base_name = "dataset"
        if self._last_loaded_path:
            path = os.path.basename(self._last_loaded_path)
            name, ext = os.path.splitext(path)
            if "_v" in name:
                try:
                    base, ver = name.rsplit("_v", 1)
                    version = int(ver) + 1
                    base_name = base
                except:
                    base_name = name
                    version = 1
            else:
                base_name = name
                version = 1
        else:
            version = 1

        save_path = os.path.join("dataset", f"{base_name}_v{version}.csv")

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(self._meta_line + "\n")
                self.dataset_df.to_csv(f, index=False)

            QMessageBox.information(
                self, "Сохранено",
                f"✅ Датасет сохранён:\n{save_path}\n\nВерсия: v{version}"
            )
            self.save_button.setEnabled(False)
            self._has_changes = False
            self._last_loaded_path = save_path

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = OneHotEncodingWindow()
    window.show()
    sys.exit(app.exec())
