# checking_data_formats_ui.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QComboBox, QScrollArea, QTableWidget,
    QTableWidgetItem, QFrame, QGroupBox
)
from PySide6.QtCore import Qt
import os
import pandas as pd
import numpy as np
from datetime import datetime


class CheckingDataFormatsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Проверка форматов данных")
        self.resize(800, 700)

        layout = QVBoxLayout()

        # === Заголовок ===
        title = QLabel("Проверка форматов данных")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # === Кнопка загрузки ===
        self.load_btn = QPushButton("📂 Загрузить датасет из папки 'dataset'")
        self.load_btn.clicked.connect(self.load_dataset)
        self.load_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.load_btn)

        # === Общая информация ===
        self.info_group = QGroupBox("Общая информация о датасете")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("Датасет не загружен.")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        self.info_group.setLayout(info_layout)
        layout.addWidget(self.info_group)

        # === Разделитель ===
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line1)

        # === Выбор колонки ===
        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel("Выберите колонку:"))
        self.column_combo = QComboBox()
        self.column_combo.currentTextChanged.connect(self.on_column_selected)
        self.column_combo.setEnabled(False)
        col_layout.addWidget(self.column_combo)
        layout.addLayout(col_layout)

        # === Результат анализа ===
        self.result_group = QGroupBox("Анализ выбранной колонки")
        result_layout = QVBoxLayout()

        # Пропуски
        self.missing_label = QLabel("Пропуски не анализировались.")
        result_layout.addWidget(self.missing_label)

        # Форматы
        self.format_label = QLabel("Форматы не определены.")
        result_layout.addWidget(self.format_label)

        # Примеры
        self.examples_label = QLabel("Примеры значений по форматам:")
        result_layout.addWidget(self.examples_label)

        # Таблица примеров
        self.examples_table = QTableWidget()
        self.examples_table.setColumnCount(2)
        self.examples_table.setHorizontalHeaderLabels(["Формат", "Примеры (до 3)"])
        self.examples_table.horizontalHeader().setStretchLastSection(True)
        result_layout.addWidget(self.examples_table)

        self.result_group.setLayout(result_layout)
        layout.addWidget(self.result_group)

        # === Кнопка анализа вручную (резерв) ===
        self.analyze_btn = QPushButton("🔍 Повторно проанализировать колонку")
        self.analyze_btn.clicked.connect(self.analyze_current_column)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)

        self.setLayout(layout)
        self.reset_state()

    def reset_state(self):
        """Сброс состояния интерфейса"""
        self.df = None
        self.column_combo.clear()
        self.column_combo.setEnabled(False)
        self.info_label.setText("Датасет не загружен.")
        self.missing_label.setText("Пропуски не анализировались.")
        self.format_label.setText("Форматы не определены.")
        self.examples_table.setRowCount(0)
        self.analyze_btn.setEnabled(False)

    def load_dataset(self):
        """Загрузка CSV из папки dataset"""
        dataset_dir = "dataset"
        if not os.path.exists(dataset_dir):
            QMessageBox.critical(self, "Ошибка", f"Папка '{dataset_dir}' не найдена!")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите датасет", dataset_dir, "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            # Читаем с автоматическим распознаванием пропусков
            self.df = pd.read_csv(
                file_path,
                na_values=['', 'NA', 'N/A', 'NULL', '?', 'none', 'null', '.', ' '],
                skipinitialspace=True
            )
            filename = os.path.basename(file_path)
            rows, cols = self.df.shape

            # Анализ типов
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            date_cols = self.detect_date_columns()
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()

            # Исключаем даты и булевы из категориальных
            categorical_without_dates_and_bools = [col for col in categorical_cols if col not in date_cols + bool_cols]

            type_info = []
            if numeric_cols:
                cols_str = ', '.join(numeric_cols)
                type_info.append(f"🔢 Числовые: {len(numeric_cols)} ({cols_str})")
            if date_cols:
                cols_str = ', '.join(date_cols)
                type_info.append(f"📅 Даты: {len(date_cols)} ({cols_str})")
            if categorical_without_dates_and_bools:
                cols_str = ', '.join(categorical_without_dates_and_bools)
                type_info.append(f"🔤 Категориальные: {len(categorical_without_dates_and_bools)} ({cols_str})")
            if bool_cols:
                cols_str = ', '.join(bool_cols)
                type_info.append(f"✅ Булевы: {len(bool_cols)} ({cols_str})")

            info_text = f"""
            <b>Загружен датасет:</b> {filename}<br><br>
            <b>Размер:</b> {rows} строк × {cols} столбцов<br>
            <b>Пропусков в датасете:</b> {self.df.isnull().sum().sum()}<br><br>
            <b>Типы данных:</b><br>
            {'<br>'.join(type_info)}
            """
            self.info_label.setText(info_text)

            # Заполнение комбобокса
            self.column_combo.clear()
            self.column_combo.addItems(self.df.columns)
            self.column_combo.setEnabled(True)
            self.analyze_btn.setEnabled(True)

            # Автоанализ первой колонки
            self.on_column_selected(self.column_combo.currentText())

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить датасет:\n{str(e)}")
            self.reset_state()


    def on_column_selected(self, column):
        """Автоматический анализ при выборе колонки"""
        if self.df is not None and column:
            self.analyze_column(column)

    def detect_date_columns(self):
        """Поиск колонок с датами с поддержкой частых форматов"""
        candidates = []
        date_formats = [
            '%m/%d/%Y',    # 4/02/2016
            '%m/%d/%y',    # 4/02/16
            '%d/%m/%Y',    # 02/04/2016
            '%Y-%m-%d',    # 2016-04-02
            '%d.%m.%Y',    # 02.04.2016
            '%Y/%m/%d',
        ]

        for col in self.df.select_dtypes(include=['object']).columns:
            sample = self.df[col].dropna().astype(str).head(10)
            if len(sample) == 0:
                continue

            valid_count = 0
            for fmt in date_formats:
                try:
                    parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                    valid_ratio = parsed.notna().mean()
                    if valid_ratio > 0.8:
                        valid_count += 1
                except:
                    continue

            # Если хотя бы один формат подошёл хорошо
            if valid_count > 0:
                candidates.append(col)

        return candidates

    def analyze_column(self, column):
        """Анализ выбранной колонки"""
        if self.df is None or column not in self.df.columns:
            return

        series = self.df[column]

        # === Пропуски ===
        missing_count = series.isnull().sum()
        total_count = len(series)
        missing_ratio = missing_count / total_count
        if missing_count > 0:
            self.missing_label.setText(
                f"<span style='color: red;'>⚠️ Пропуски: {missing_count} ({missing_ratio:.1%})</span>"
            )
        else:
            self.missing_label.setText("✅ Нет пропусков")

        # === Определяем форматы ===
        non_null = series.dropna()
        if len(non_null) == 0:
            self.format_label.setText("⚠️ Все значения — пропуски")
            self.examples_table.setRowCount(0)
            return

        # Попробуем определить тип данных
        unique_sample = non_null.astype(str).str.strip().unique()
        if len(unique_sample) == 0:
            fmt = "пусто"
        elif self.is_numeric_series(non_null):
            fmt = "число (int/float)"
        elif self.is_datetime_series(non_null):
            fmt = "дата/время"
        elif self.is_boolean_like(non_null):
            fmt = "логическое (да/нет, true/false)"
        elif len(unique_sample) <= 10:
            fmt = "категория (мало уникальных)"
        else:
            fmt = "текст (строка)"

        self.format_label.setText(f"Определённый формат: <b>{fmt}</b>")

        # === Сбор примеров по форматам ===
        formats = {}
        if self.is_numeric_series(non_null):
            nums = pd.to_numeric(non_null, errors='coerce').dropna()
            # Берём 3 уникальных числа
            unique_nums = pd.Series(nums).drop_duplicates().head(5).tolist()
            formats["Число"] = unique_nums

        if self.is_datetime_series(non_null):
            dates = pd.to_datetime(non_null, errors='coerce').dropna()
            # Берём 3 уникальные даты
            unique_dates = pd.Series(dates).drop_duplicates().head(5)
            date_strings = [d.strftime("%Y-%m-%d") for d in unique_dates if not pd.isna(d)]
            formats["Дата"] = date_strings

        if self.is_boolean_like(non_null):
            # Уникальные логические значения
            bools = non_null.drop_duplicates().head(5).tolist()
            formats["Логическое"] = bools

        if fmt == "категория (мало уникальных)" or fmt == "текст (строка)":
            # 🔹 Вот здесь — главное изменение: уникальные значения
            unique_values = non_null.drop_duplicates().head(5).tolist()
            key = "Категория" if len(unique_sample) <= 10 else "Текст"
            formats[key] = unique_values

        # === Заполняем таблицу ===
        self.examples_table.setRowCount(len(formats))
        for i, (fmt_name, examples) in enumerate(formats.items()):
            self.examples_table.setItem(i, 0, QTableWidgetItem(fmt_name))
            self.examples_table.setItem(i, 1, QTableWidgetItem(", ".join(map(str, examples))))

    def is_numeric_series(self, series):
        numeric_ratio = pd.to_numeric(series, errors='coerce').notna().mean()
        return numeric_ratio > 0.9

    def is_datetime_series(self, series):
        if series.empty:
            return False

        date_formats = [
            '%m/%d/%Y',
            '%m/%d/%y',
            '%d/%m/%Y',
            '%Y-%m-%d',
            '%d.%m.%Y',
            '%Y/%m/%d'
        ]

        sample = series.astype(str).head(20)

        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                valid_ratio = parsed.notna().mean()
                if valid_ratio > 0.8:
                    return True
            except:
                continue
        return False

    def is_boolean_like(self, series):
        bool_values = ['да', 'нет', 'yes', 'no', 'true', 'false', '1', '0', 'True', 'False']
        lower_values = series.astype(str).str.lower()
        match_ratio = lower_values.isin(bool_values).mean()
        return match_ratio > 0.9

    def analyze_current_column(self):
        """Кнопка для ручного перезапуска анализа"""
        col = self.column_combo.currentText()
        if col:
            self.analyze_column(col)