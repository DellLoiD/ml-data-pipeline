# preprocessing/checking_data_formats_ui.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QComboBox, QFrame, QGroupBox, QTextEdit, QLineEdit, QInputDialog
)
from PySide6.QtCore import Qt
import os
import shutil
import pandas as pd
import numpy as np

# Импорт нового класса
from utils.meta_tracker import MetaTracker


class CheckingDataFormatsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self._last_loaded_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)
        self.param_descriptions = {}
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Проверка форматов данных")
        self.resize(800, 600)

        layout = QVBoxLayout()

        title = QLabel("Проверка форматов данных")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # === Кнопки загрузки ===
        buttons_layout = QHBoxLayout()

        self.import_btn = QPushButton("📥 Загрузка датасета в проект")
        self.import_btn.clicked.connect(self.import_dataset_to_project)
        self.import_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        self.import_btn.setMinimumWidth(250)
        buttons_layout.addWidget(self.import_btn)

        self.load_btn = QPushButton("📂 Загрузить датасет из папки dataset")
        self.load_btn.clicked.connect(self.load_dataset)
        self.load_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        self.load_btn.setMinimumWidth(250)
        buttons_layout.addWidget(self.load_btn)

        self.load_desc_btn = QPushButton("📄 Загрузить описание (txt)")
        self.load_desc_btn.clicked.connect(self.load_parameter_descriptions)
        self.load_desc_btn.setStyleSheet("font-size: 13px; padding: 8px;")
        self.load_desc_btn.setMinimumWidth(250)
        buttons_layout.addWidget(self.load_desc_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        # === Три группы в одной строке ===
        top_row_layout = QHBoxLayout()

        self.info_group = QGroupBox("Общая информация")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("Датасет не загружен.")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        self.info_group.setLayout(info_layout)
        top_row_layout.addWidget(self.info_group, 1)

        self.categories_group = QGroupBox("Категориальные признаки")
        categories_layout = QVBoxLayout()
        self.categories_label = QLabel("Категориальные колонки не загружены.")
        self.categories_label.setWordWrap(True)
        self.categories_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        categories_layout.addWidget(self.categories_label)
        self.categories_group.setLayout(categories_layout)
        top_row_layout.addWidget(self.categories_group, 1)

        self.missing_group = QGroupBox("Пропуски")
        missing_layout = QVBoxLayout()
        self.missing_label_summary = QLabel("Пропуски не рассчитаны.")
        self.missing_label_summary.setWordWrap(True)
        missing_layout.addWidget(self.missing_label_summary)
        self.missing_group.setLayout(missing_layout)
        top_row_layout.addWidget(self.missing_group, 1)

        layout.addLayout(top_row_layout)

        # === Разделитель ===
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line1)

        # === Контрольная строка: выбор и действия ===
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Выберите колонку:"))

        self.column_combo = QComboBox()
        self.column_combo.addItem("Выберите колонку")
        self.column_combo.setEnabled(False)
        self.column_combo.setFixedWidth(180)
        control_layout.addWidget(self.column_combo)

        self.analyze_btn = QPushButton("Найти классы по параметру")
        self.analyze_btn.setToolTip("Анализ редких значений (≤ N)")
        self.analyze_btn.clicked.connect(self.analyze_rare_classes)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setFixedWidth(170)
        control_layout.addWidget(self.analyze_btn)

        self.merge_btn = QPushButton("🔗 Объединить классы.")
        self.merge_btn.setToolTip("Объединить значения в диапазоне")
        self.merge_btn.clicked.connect(self.merge_interval_values)
        self.merge_btn.setEnabled(False)
        self.merge_btn.setFixedWidth(150)
        control_layout.addWidget(self.merge_btn)

        self.delete_btn = QPushButton("🗑️ Удалить колонку")
        self.delete_btn.setToolTip("Удалить выбранную колонку")
        self.delete_btn.setStyleSheet("color: red; font-weight: bold;")
        self.delete_btn.clicked.connect(self.delete_selected_column)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setFixedWidth(130)
        control_layout.addWidget(self.delete_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # === Описание выбранной колонки ===
        self.description_label = QLabel("Описание: не загружено или отсутствует.")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("font-style: italic; color: #555; padding: 4px;")
        layout.addWidget(self.description_label)

        # === Анализ редких классов ===
        outlier_group = QGroupBox("Редкие значения (≤ N)")
        outlier_layout = QVBoxLayout()

        # Фильтр по интервалу
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Фильтр (от):"))
        self.min_val_input = QLineEdit()
        self.min_val_input.setPlaceholderText("мин, напр. 1800")
        self.min_val_input.setFixedWidth(90)
        self.min_val_input.setEnabled(False)
        range_layout.addWidget(self.min_val_input)

        range_layout.addWidget(QLabel("до:"))
        self.max_val_input = QLineEdit()
        self.max_val_input.setPlaceholderText("макс, напр. 1950")
        self.max_val_input.setFixedWidth(90)
        self.max_val_input.setEnabled(False)
        range_layout.addWidget(self.max_val_input)
        outlier_layout.addLayout(range_layout)

        # Поле N
        n_layout = QHBoxLayout()
        n_layout.addWidget(QLabel("Макс. записей (N):"))
        self.n_input = QLineEdit("5")
        self.n_input.setPlaceholderText("Напр.: 5")
        self.n_input.setFixedWidth(90)
        n_layout.addWidget(self.n_input)
        outlier_layout.addLayout(n_layout)

        # Результаты
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Результаты анализа...")
        self.results_text.setFixedHeight(220)
        outlier_layout.addWidget(self.results_text)

        outlier_group.setLayout(outlier_layout)
        layout.addWidget(outlier_group)

        # === Кнопка сохранения ===
        self.save_btn = QPushButton("💾 Сохранить датасет")
        self.save_btn.clicked.connect(self.save_dataset)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.save_btn)

        self.setLayout(layout)
        self.reset_state()

    def reset_state(self):
        self.df = None
        self._last_loaded_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)
        self.param_descriptions = {}

        self.column_combo.clear()
        self.column_combo.addItem("Выберите колонку")
        self.column_combo.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.merge_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self.min_val_input.clear()
        self.max_val_input.clear()
        self.n_input.setText("5")
        self.results_text.clear()
        self.description_label.setText("Описание: не загружено или отсутствует.")

        self.info_label.setText("Датасет не загружен.")
        self.categories_label.setText("Категориальные колонки не загружены.")
        self.missing_label_summary.setText("Пропуски не рассчитаны.")
        self.load_btn.setText("📂 Загрузить датасет из папки dataset")

    def import_dataset_to_project(self):
        """Загружает датасет из любого места на ПК в папку dataset с именем _v0"""
        dataset_dir = "dataset"
        os.makedirs(dataset_dir, exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите датасет для импорта", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            original_name = os.path.splitext(os.path.basename(file_path))[0]
            safe_name = "".join(c for c in original_name if c.isalnum() or c in " _-")
            new_filename = f"{safe_name}_v0.csv"
            save_path = os.path.join(dataset_dir, new_filename)

            df.to_csv(save_path, index=False, encoding="utf-8")

            QMessageBox.information(
                self, "Успех",
                f"✅ Датасет импортирован в проект:\n{new_filename}\n\nТеперь его можно загрузить из папки dataset."
            )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось импортировать датасет:\n{e}")


    def load_parameter_descriptions(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл описаний", "", "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return

        try:
            self.param_descriptions = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    key, *desc_parts = line.split(":", 1)
                    description = desc_parts[0].strip() if desc_parts else ""
                    key = key.strip()
                    self.param_descriptions[key] = description

            QMessageBox.information(self, "Успех", f"✅ Описания загружены:\n{os.path.basename(file_path)}\n"
                                                  f"Найдено параметров: {len(self.param_descriptions)}")

            current_col = self.column_combo.currentText()
            if current_col != "Выберите колонку" and current_col in self.param_descriptions:
                self.description_label.setText(f"<b>{current_col}:</b> {self.param_descriptions[current_col]}")
            else:
                self.description_label.setText("Описание: не найдено.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл описаний:\n{e}")

    def load_dataset(self):
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
            self.meta_tracker.load_from_file(file_path)
            self.df = pd.read_csv(
                file_path,
                na_values=['', 'NA', 'N/A', 'NULL', '?', 'none', 'null', '.', ' '],
                skipinitialspace=True,
                comment='#'
            )
            self._last_loaded_path = file_path

            rows, cols = self.df.shape
            total_missing = self.df.isnull().sum().sum()

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            date_cols = self.detect_date_columns()
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
            categorical_without_dates_and_bools = [
                col for col in categorical_cols if col not in date_cols + bool_cols
            ]

            true_categorical = []
            for col in categorical_without_dates_and_bools:
                sample = self.df[col].dropna().astype(str).head(100)
                if not pd.to_numeric(sample, errors='coerce').notna().all():
                    true_categorical.append(col)

            cat_counts = []
            for col in true_categorical:
                unique_count = self.df[col].dropna().astype(str).nunique()
                cat_counts.append((col, unique_count))
            cat_counts.sort(key=lambda x: x[1])

            type_info = []
            if numeric_cols:
                type_info.append(f"🔢 Числовые: {len(numeric_cols)}")
            if date_cols:
                type_info.append(f"📅 Даты: {len(date_cols)}")
            if true_categorical:
                type_info.append(f"🔤 Категориальные: {len(true_categorical)}")
            if bool_cols:
                type_info.append(f"✅ Булевы: {len(bool_cols)}")

            info_text = f"""
            <b>Размер:</b> {rows}×{cols}<br>
            <b>Пропусков:</b> {total_missing}<br><br>
            <b>Типы данных:</b><br>
            {'<br>'.join(type_info)}
            """
            self.info_label.setText(info_text)

            self.update_categories_display(cat_counts)
            self.update_missing_summary()

            self.column_combo.clear()
            self.column_combo.addItem("Выберите колонку")
            self.column_combo.addItems(self.df.columns)
            self.column_combo.setCurrentText("Выберите колонку")
            self.column_combo.setEnabled(True)
            self.delete_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.merge_btn.setEnabled(True)

            self.column_combo.currentTextChanged.connect(self.on_column_changed)

            self.load_btn.setText(f"✅ Загружен: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить датасет:\n{str(e)}")
            self.reset_state()

    def on_column_changed(self, column_name):
        """Показывает описание и автоматически запускает анализ при выборе колонки"""
        if not column_name or column_name == "Выберите колонку":
            self.description_label.setText("Описание: не загружено или отсутствует.")
            return

        if column_name in self.param_descriptions:
            self.description_label.setText(f"<b>{column_name}:</b> {self.param_descriptions[column_name]}")
        else:
            self.description_label.setText("Описание: не найдено.")

        if self.df is not None and column_name in self.df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(self.df[column_name])
            self.min_val_input.setEnabled(is_numeric)
            self.max_val_input.setEnabled(is_numeric)
            if not is_numeric:
                self.min_val_input.clear()
                self.max_val_input.clear()

        self.analyze_rare_classes()

    def update_categories_display(self, cat_counts):
        if not cat_counts:
            self.categories_label.setText("❌ Нет строковых категориальных колонок.")
            return
        cat_lines = [f"<b>{col:20}</b> — {count}" for col, count in cat_counts]
        self.categories_label.setText("<br>".join(cat_lines))

    def update_missing_summary(self):
        if self.df is None:
            self.missing_label_summary.setText("Пропуски не рассчитаны.")
            return

        missing_data = self.df.isnull().sum()
        missing_cols = missing_data[missing_data > 0].sort_values(ascending=False)

        if missing_cols.empty:
            self.missing_label_summary.setText("✅ Нет пропусков.")
            return

        lines = []
        for col, count in missing_cols.items():
            pct = count / len(self.df)
            marker = " 🔴" if pct > 0.5 else ""
            lines.append(f"<b>{col:12}</b> — {count:3} ({pct:.1%}){marker}")

        text = "<br>".join(lines)
        self.missing_label_summary.setText(text)

    def detect_date_columns(self):
        candidates = []
        date_formats = ['%m/%d/%Y', '%m/%d/%y', '%d/%m/%Y', '%Y-%m-%d', '%d.%m.%Y', '%Y/%m/%d']
        for col in self.df.select_dtypes(include=['object']).columns:
            sample = self.df[col].dropna().astype(str).head(10)
            if len(sample) == 0:
                continue
            valid_count = sum(
                pd.to_datetime(sample, format=fmt, errors='coerce').notna().mean() > 0.8
                for fmt in date_formats
            )
            if valid_count > 0:
                candidates.append(col)
        return candidates

    def delete_selected_column(self):
        column = self.column_combo.currentText()
        if not column or self.df is None or column == "Выберите колонку":
            QMessageBox.warning(self, "Ошибка", "Выберите колонку для удаления.")
            return

        reply = QMessageBox.question(self, "Подтверждение", f"Удалить колонку '{column}'?")
        if reply != QMessageBox.Yes:
            return

        try:
            self.df = self.df.drop(columns=[column]).copy()
            self.meta_tracker.add_change(f"удалена колонка '{column}'")
            self.column_combo.removeItem(self.column_combo.currentIndex())

            if len(self.df.columns) == 0:
                self.reset_state()
            else:
                QMessageBox.information(self, "Успех", f"✅ Колонка '{column}' удалена.")
                self.save_btn.setEnabled(True)
                self.update_categories_display(self.get_categorical_counts())
                self.update_missing_summary()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось удалить колонку:\n{e}")

    def get_categorical_counts(self):
        if self.df is None:
            return []

        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        date_cols = self.detect_date_columns()
        bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        categorical_without_dates_and_bools = [
            col for col in categorical_cols if col not in date_cols + bool_cols
        ]

        true_categorical = []
        for col in categorical_without_dates_and_bools:
            sample = self.df[col].dropna().astype(str).head(100)
            if not pd.to_numeric(sample, errors='coerce').notna().all():
                true_categorical.append(col)

        cat_counts = []
        for col in true_categorical:
            unique_count = self.df[col].dropna().astype(str).nunique()
            cat_counts.append((col, unique_count))
        cat_counts.sort(key=lambda x: x[1])
        return cat_counts

    def analyze_rare_classes(self):
        """Поиск редких значений с фильтрацией по интервалу"""
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет!")
            return

        column_name = self.column_combo.currentText()
        if not column_name or column_name not in self.df.columns:
            return

        is_numeric = pd.api.types.is_numeric_dtype(self.df[column_name])
        min_val, max_val = None, None
        use_range = False

        if is_numeric:
            min_text = self.min_val_input.text().strip()
            max_text = self.max_val_input.text().strip()
            if min_text or max_text:
                try:
                    min_val = float(min_text) if min_text else None
                    max_val = float(max_text) if max_text else None
                    use_range = True
                except ValueError:
                    return

        if use_range and is_numeric:
            mask = True
            if min_val is not None:
                mask &= (self.df[column_name] >= min_val)
            if max_val is not None:
                mask &= (self.df[column_name] <= max_val)
            filtered_series = self.df[column_name][mask]
        else:
            filtered_series = self.df[column_name]

        try:
            n = int(self.n_input.text().strip())
            if n < 0:
                return
        except ValueError:
            return

        value_counts = filtered_series.value_counts(dropna=False).sort_index()
        rare_values = value_counts[value_counts <= n]

        total_filtered = len(filtered_series)
        total_unique = len(value_counts)
        summary_line = (f"📊 Сводка: • Записей: <b>{total_filtered}</b> • Уникальных: <b>{total_unique}</b> • "
                        f"Мин/макс: <b>{value_counts.min() if len(value_counts) else 0}</b> / "
                        f"<b>{value_counts.max() if len(value_counts) else 0}</b>")

        if rare_values.empty:
            result_text = f"✅ Нет значений ≤ {n}.<br><br><i>{summary_line}</i>"
            if use_range:
                result_text += f"<br><i>(в диапазоне от {min_val} до {max_val})</i>"
        else:
            count_rare = len(rare_values)
            result_text = (f"🔍 <b>{count_rare}</b> редких значений (≤ {n}): {summary_line}"
                           f"<pre>Значение → К-во</pre>\n"
                           f"<pre>" + "-" * 30 + "</pre>\n")
            for value, count in rare_values.items():
                val_str = "(пусто)" if pd.isna(value) else str(value)
                val_str = val_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                result_text += f"<pre>{val_str:<15} → {count:>6}</pre>\n"

        self.results_text.setHtml(result_text)

    def merge_interval_values(self):
        """Объединяет значения в указанном интервале"""
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет!")
            return

        column_name = self.column_combo.currentText()
        if not column_name or column_name not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", "Выберите корректный столбец!")
            return

        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            QMessageBox.warning(self, "Ошибка", f"Столбец '{column_name}' должен быть числовым.")
            return

        min_text = self.min_val_input.text().strip()
        max_text = self.max_val_input.text().strip()

        if not min_text or not max_text:
            QMessageBox.warning(self, "Ошибка", "Введите оба значения: 'от' и 'до'.")
            return

        try:
            min_val = float(min_text)
            max_val = float(max_text)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректные числа.")
            return

        if min_val > max_val:
            QMessageBox.warning(self, "Ошибка", "'от' не может быть больше 'до'.")
            return

        target_val, ok = QInputDialog.getDouble(
            self,
            "Объединить значения",
            f"В какое значение объединить все записи\nв диапазоне [{min_val}, {max_val}]?",
            decimals=0 if self.df[column_name].dtype == 'int64' else 2,
            value=min_val
        )
        if not ok:
            return

        if target_val < -1e10 or target_val > 1e10:
            QMessageBox.warning(self, "Ошибка", "Значение вне допустимого диапазона.")
            return

        mask = (self.df[column_name] >= min_val) & (self.df[column_name] <= max_val)
        count = mask.sum()
        if count == 0:
            QMessageBox.information(self, "Нет данных", "Нет записей в диапазоне.")
            return

        self.df.loc[mask, column_name] = target_val
        self.meta_tracker.add_change(
            f"объединены значения в '{column_name}' от {min_val} до {max_val} в {target_val}"
        )
        self.save_btn.setEnabled(True)

        QMessageBox.information(
            self, "Успешно", f"✅ {count} записей\nобъединены в значение: <b>{target_val}</b>"
        )
        self.analyze_rare_classes()

    def save_dataset(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Ошибка", "Нечего сохранять.")
            return

        if not self._last_loaded_path:
            QMessageBox.critical(self, "Ошибка", "Неизвестен путь загрузки.")
            return

        base_name = os.path.splitext(os.path.basename(self._last_loaded_path))[0]
        base_name = base_name.split("_v")[0] if "_v" in base_name else base_name
        save_path = os.path.join("dataset", f"{base_name}_v{self.meta_tracker.version}.csv")

        try:
            success = self.meta_tracker.save_to_file(save_path, self.df)
            if success:
                self._last_loaded_path = save_path
                self.meta_tracker.version += 1  # Увеличиваем для следующего сохранения
                self.save_btn.setEnabled(False)
                self.update_missing_summary()

                QMessageBox.information(
                    self, "Сохранено",
                    f"✅ Датасет сохранён:\n{os.path.basename(save_path)}\n\nВерсия: v{self.meta_tracker.version - 1}"
                )
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")
