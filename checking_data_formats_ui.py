# checking_data_formats_ui.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidgetItem,
    QFileDialog, QMessageBox, QComboBox, QFrame, QGroupBox
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
        self._last_loaded_path = None  # Сохраняем путь для генерации имени
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Проверка форматов данных")
        self.resize(800, 900)

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

        # === Статистика по категориальным признакам ===
        self.categories_group = QGroupBox("Категориальные признаки и количество классов")
        categories_layout = QVBoxLayout()
        self.categories_label = QLabel("Категориальные колонки не загружены.")
        self.categories_label.setWordWrap(True)
        self.categories_label.setStyleSheet("font-family: monospace;")
        categories_layout.addWidget(self.categories_label)
        self.categories_group.setLayout(categories_layout)
        layout.addWidget(self.categories_group)

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

        # === Кнопки анализа и удаления ===
        btn_layout = QHBoxLayout()

        self.freq_btn = QPushButton("📊 Показать частоту классов")
        self.freq_btn.clicked.connect(self.show_category_frequency)
        self.freq_btn.setEnabled(False)
        self.freq_btn.setStyleSheet("font-size: 13px;")
        btn_layout.addWidget(self.freq_btn)

        self.delete_btn = QPushButton("🗑️ Удалить колонку")
        self.delete_btn.clicked.connect(self.delete_selected_column)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet("font-size: 13px; color: red;")
        btn_layout.addWidget(self.delete_btn)

        layout.addLayout(btn_layout)

        # === Результат анализа ===
        self.result_group = QGroupBox("Анализ выбранной колонки")
        result_layout = QVBoxLayout()

        self.missing_label = QLabel("Пропуски не анализировались.")
        result_layout.addWidget(self.missing_label)

        self.format_label = QLabel("Форматы не определены.")
        result_layout.addWidget(self.format_label)

        self.examples_label = QLabel("Примеры значений по форматам:")
        result_layout.addWidget(self.examples_label)

        from PySide6.QtWidgets import QTableWidget
        self.examples_table = QTableWidget()
        self.examples_table.setColumnCount(2)
        self.examples_table.setHorizontalHeaderLabels(["Формат", "Примеры (до 5)"])
        self.examples_table.horizontalHeader().setStretchLastSection(True)
        result_layout.addWidget(self.examples_table)

        self.result_group.setLayout(result_layout)
        layout.addWidget(self.result_group)

        # === Кнопка анализа вручную (резерв) ===
        self.analyze_btn = QPushButton("🔍 Повторно проанализировать колонку")
        self.analyze_btn.clicked.connect(self.analyze_current_column)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)

        # === Кнопка сохранения (внизу) ===
        self.save_btn = QPushButton("💾 Сохранить датасет")
        self.save_btn.clicked.connect(self.save_dataset)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.save_btn)

        self.setLayout(layout)
        self.reset_state()

    def reset_state(self):
        """Сброс состояния интерфейса"""
        self.df = None
        self._last_loaded_path = None
        self.column_combo.clear()
        self.column_combo.setEnabled(False)
        self.freq_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.info_label.setText("Датасет не загружен.")
        self.missing_label.setText("Пропуски не анализировались.")
        self.format_label.setText("Форматы не определены.")
        self.examples_table.setRowCount(0)
        self.analyze_btn.setEnabled(False)
        self.categories_label.setText("Категориальные колонки не загружены.")

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
            self.df = pd.read_csv(
                file_path,
                na_values=['', 'NA', 'N/A', 'NULL', '?', 'none', 'null', '.', ' '],
                skipinitialspace=True
            )
            self._last_loaded_path = file_path
            filename = os.path.basename(file_path)
            rows, cols = self.df.shape

            # Анализ типов
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            date_cols = self.detect_date_columns()
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
            categorical_without_dates_and_bools = [col for col in categorical_cols if col not in date_cols + bool_cols]

            # Только истинно строковые категориальные
            true_categorical = []
            for col in categorical_without_dates_and_bools:
                sample = self.df[col].dropna().astype(str).head(100)
                if not pd.to_numeric(sample, errors='coerce').notna().all():
                    true_categorical.append(col)

            # === Общая информация ===
            type_info = []
            if numeric_cols:
                cols_str = ', '.join(numeric_cols)
                type_info.append(f"🔢 Числовые: {len(numeric_cols)} ({cols_str})")
            if date_cols:
                cols_str = ', '.join(date_cols)
                type_info.append(f"📅 Даты: {len(date_cols)} ({cols_str})")
            if true_categorical:
                cols_str = ', '.join(true_categorical)
                type_info.append(f"🔤 Категориальные: {len(true_categorical)} ({cols_str})")
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

            # === Статистика категорий ===
            self.update_categories_display()

            # === Заполнение комбобокса ===
            self.column_combo.clear()
            self.column_combo.addItems(self.df.columns)
            self.column_combo.setEnabled(True)
            self.freq_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)

            self.on_column_selected(self.column_combo.currentText())

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить датасет:\n{str(e)}")
            self.reset_state()

    def update_categories_display(self):
        """Обновляет отображение категориальных признаков и число классов"""
        if self.df is None:
            self.categories_label.setText("Категориальные колонки не загружены.")
            return

        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        date_cols = self.detect_date_columns()
        bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        true_categorical = [col for col in categorical_cols
                            if col not in date_cols + bool_cols
                            and not self.is_numeric_series(self.df[col])]

        if true_categorical:
            cat_lines = []
            for col in true_categorical:
                unique_count = self.df[col].dropna().astype(str).nunique()
                cat_lines.append(f"<b>{col:20}</b> — {unique_count} классов")
            cat_text = "<br>".join(cat_lines)
        else:
            cat_text = "❌ Нет подходящих строковых категориальных колонок."

        self.categories_label.setText(cat_text)

    def on_column_selected(self, column):
        """Автоматический анализ при выборе колонки"""
        if self.df is not None and column:
            self.analyze_column(column)

    def detect_date_columns(self):
        """Поиск колонок с датами с поддержкой частых форматов"""
        candidates = []
        date_formats = [
            '%m/%d/%Y', '%m/%d/%y', '%d/%m/%Y', '%Y-%m-%d', '%d.%m.%Y', '%Y/%m/%d'
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

            if valid_count > 0:
                candidates.append(col)

        return candidates

    def is_numeric_series(self, series):
        numeric_ratio = pd.to_numeric(series, errors='coerce').notna().mean()
        return numeric_ratio > 0.9

    def is_datetime_series(self, series):
        if series.empty:
            return False
        date_formats = ['%m/%d/%Y', '%m/%d/%y', '%d/%m/%Y', '%Y-%m-%d', '%d.%m.%Y', '%Y/%m/%d']
        sample = series.astype(str).head(20)
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                if parsed.notna().mean() > 0.8:
                    return True
            except:
                continue
        return False

    def is_boolean_like(self, series):
        bool_values = ['да', 'нет', 'yes', 'no', 'true', 'false', '1', '0', 'True', 'False']
        lower_values = series.astype(str).str.lower()
        match_ratio = lower_values.isin(bool_values).mean()
        return match_ratio > 0.9

    def analyze_column(self, column):
        if self.df is None or column not in self.df.columns:
            return

        series = self.df[column]
        non_null = series.dropna()

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

        # === Определяем формат ===
        if len(non_null) == 0:
            self.format_label.setText("⚠️ Все значения — пропуски")
            self.examples_table.setRowCount(0)
            return

        unique_sample = non_null.astype(str).str.strip().unique()
        if self.is_numeric_series(non_null):
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

        # === Сбор примеров ===
        formats = {}
        if self.is_numeric_series(non_null):
            nums = pd.to_numeric(non_null, errors='coerce').dropna()
            unique_nums = pd.Series(nums).drop_duplicates().head(5).tolist()
            formats["Число"] = unique_nums
        if self.is_datetime_series(non_null):
            dates = pd.to_datetime(non_null, errors='coerce').dropna()
            unique_dates = pd.Series(dates).drop_duplicates().head(5)
            date_strings = [d.strftime("%Y-%m-%d") for d in unique_dates if not pd.isna(d)]
            formats["Дата"] = date_strings
        if self.is_boolean_like(non_null):
            bools = non_null.drop_duplicates().head(5).tolist()
            formats["Логическое"] = bools
        if fmt in ["категория (мало уникальных)", "текст (строка)"]:
            unique_vals = non_null.drop_duplicates().head(5).tolist()
            key = "Категория" if len(unique_sample) <= 10 else "Текст"
            formats[key] = unique_vals

        # === Заполняем таблицу ===
        self.examples_table.setRowCount(len(formats))
        for i, (fmt_name, examples) in enumerate(formats.items()):
            self.examples_table.setItem(i, 0, QTableWidgetItem(fmt_name))
            self.examples_table.setItem(i, 1, QTableWidgetItem(", ".join(map(str, examples))))

    def analyze_current_column(self):
        """Кнопка для ручного перезапуска анализа"""
        col = self.column_combo.currentText()
        if col:
            self.analyze_column(col)

    def show_category_frequency(self):
        """Показывает статистику по частоте значений в категориальной колонке"""
        column = self.column_combo.currentText()
        if not column:
            QMessageBox.warning(self, "Внимание", "Сначала выберите колонку!")
            return
        if self.df is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите датасет!")
            return
        if column not in self.df.columns:
            QMessageBox.critical(self, "Ошибка", f"Колонка '{column}' не найдена.")
            return

        series = self.df[column]
        non_null = series.dropna()

        if self.is_numeric_series(non_null) or self.is_datetime_series(non_null):
            QMessageBox.information(self, "Информация", "Эта колонка числовая или дата — не подходит для анализа категорий.")
            return

        value_counts = non_null.value_counts()
        unique_count = len(value_counts)
        top3 = value_counts.head(3)
        min_freq = value_counts.min()
        rare_classes_count = (value_counts == min_freq).sum()

        msg = f"<b>📊 Статистика колонки '{column}'</b><br><br>"
        msg += f"🔢 Всего уникальных классов: <b>{unique_count}</b><br><br>"
        msg += "<b>🏆 Самые частые значения:</b><br>"
        for val, count in top3.items():
            msg += f"• {val} — <b>{count}</b><br>"
        msg += f"<br><b>🔻 Классы с частотой {min_freq}:</b><br>"
        msg += f"• Всего таких: <b>{rare_classes_count}</b><br>"

        if rare_classes_count <= 10:
            rare_values = value_counts[value_counts == min_freq].index.tolist()
            msg += f"• Примеры: {', '.join(map(str, rare_values[:5]))}" + ("..." if len(rare_values) > 5 else "")

        msg += "<br><br><b>💡 Рекомендации:</b><br>"
        if unique_count <= 5:
            msg += "✅ Подходит для <b>One-Hot Encoding</b>."
        elif unique_count <= 50:
            msg += "🟡 Лучше <b>Label Encoding</b> или <b>Target Encoding</b>."
        else:
            msg += "🔴 Рассмотрите <b>Label Encoding</b> или <b>хэширование</b>."

        QMessageBox.information(self, "Частота классов", msg)

    def delete_selected_column(self):
        """Удаляет выбранную колонку и обновляет интерфейс"""
        column = self.column_combo.currentText()
        if not column:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите колонку для удаления.")
            return
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Датасет не загружен.")
            return

        reply = QMessageBox.question(
            self,
            "Подтверждение удаления",
            f"Удалить колонку '{column}'?\n\n"
            "Это действие нельзя отменить.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        try:
            self.df = self.df.drop(columns=[column]).copy()

            # Обновляем интерфейс
            self.column_combo.removeItem(self.column_combo.currentIndex())
            self.update_categories_display()
            self.reset_analysis_display()

            if len(self.df.columns) > 0:
                new_col = self.df.columns[0]
                self.column_combo.setCurrentText(new_col)
                self.on_column_selected(new_col)
            else:
                self.reset_state()

            QMessageBox.information(self, "Успех", f"✅ Колонка '{column}' удалена.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось удалить колонку:\n{e}")

    def save_dataset(self):
        """Сохраняет текущий датасет в CSV-файл"""
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Ошибка", "Нечего сохранять — датасет пуст или не загружен.")
            return

        # Генерация имени по умолчанию
        if self._last_loaded_path:
            original_name = os.path.basename(self._last_loaded_path)
            default_name = f"cleaned_{original_name}"
        else:
            default_name = "cleaned_dataset.csv"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить датасет",
            f"./dataset/{default_name}",
            "CSV Files (*.csv)"
        )

        if not save_path:
            return  # Пользователь отменил

        try:
            self.df.to_csv(save_path, index=False)
            QMessageBox.information(
                self,
                "Сохранено",
                f"✅ Датасет успешно сохранён:\n{save_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")

    def reset_analysis_display(self):
        """Сброс отображения анализа колонки"""
        self.missing_label.setText("Пропуски не анализировались.")
        self.format_label.setText("Форматы не определены.")
        self.examples_table.setRowCount(0)
