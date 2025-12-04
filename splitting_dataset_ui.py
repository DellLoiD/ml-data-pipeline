# splitting_dataset.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
    QScrollArea, QComboBox, QHBoxLayout, QFrame, QLineEdit
)
from PySide6.QtCore import Qt
import os
import pandas as pd


class SplittingDatasetWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.target_column = None
        self.complete_df = None  # где target НЕ NaN
        self.missing_df = None   # где target NaN
        self.df_path = None      # путь к исходному файлу
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # === Заголовок ===
        title = QLabel("Разделение датасета")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel("Выберите датасет. Доступны два режима разделения:")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # === Кнопка загрузки — СРАЗУ после описания ===
        self.load_btn = QPushButton("📂 Выбрать датасет из папки 'dataset'")
        self.load_btn.clicked.connect(self.load_dataset)
        self.load_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.load_btn)

        # === Информация о датасете ===
        self.info_label = QLabel("Датасет не загружен.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # === Разделитель: Пропуски ===
        self.add_section_separator(layout, "1. Разделение по пропускам в целевой переменной")

        list_widget = QLabel("• Полный набор — строки, где целевая переменная заполнена<br>"
                            "• Набор с пропусками — строки, где целевая переменная отсутствует")
        list_widget.setTextFormat(Qt.RichText)
        list_widget.setWordWrap(True)
        layout.addWidget(list_widget)

        # === Выбор целевой переменной ===
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Целевая переменная:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.currentTextChanged.connect(self.on_target_changed)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)

        # === Кнопка разделения по пропускам ===
        self.split_btn = QPushButton("✂️ Выполнить разделение по пропускам")
        self.split_btn.clicked.connect(self.split_by_target)
        self.split_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        self.split_btn.setEnabled(False)
        layout.addWidget(self.split_btn)

        # === Результат разделения по пропускам ===
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: 'Courier'; font-size: 12px;")
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.result_label)
        scroll.setMaximumHeight(200)
        layout.addWidget(QLabel("<b>Результат разделения по пропускам:</b>"))
        layout.addWidget(scroll)

        # === Кнопки сохранения по пропускам ===
        btn_layout = QHBoxLayout()
        self.save_complete_btn = QPushButton("💾 Сохранить полный набор (target не NaN)")
        self.save_complete_btn.clicked.connect(self.save_complete)
        self.save_complete_btn.setEnabled(False)
        btn_layout.addWidget(self.save_complete_btn)

        self.save_missing_btn = QPushButton("💾 Сохранить с пропусками (target = NaN)")
        self.save_missing_btn.clicked.connect(self.save_with_missing)
        self.save_missing_btn.setEnabled(False)
        btn_layout.addWidget(self.save_missing_btn)
        layout.addLayout(btn_layout)

        # === Разделитель: По классу ===
        self.add_section_separator(layout, "2. Разделение датасета по классу")

        # === Выбор колонки для разделения ===
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Колонка для разделения:"))
        self.class_combo = QComboBox()
        self.class_combo.addItem("Выберите категорию")
        self.class_combo.setEnabled(False)
        self.class_combo.currentTextChanged.connect(self.on_class_column_changed)
        class_layout.addWidget(self.class_combo)
        layout.addLayout(class_layout)

        # === Подсказка о типе данных ===
        self.type_label = QLabel("")
        self.type_label.setWordWrap(True)
        self.type_label.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(self.type_label)

        # === Контейнер для ввода значений ===
        self.input_container = QVBoxLayout()
        layout.addLayout(self.input_container)

        # === Кнопка: Разделить по классу ===
        self.split_class_btn = QPushButton("✂️ Выполнить разделение по классу")
        self.split_class_btn.clicked.connect(self.split_by_class)
        self.split_class_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        self.split_class_btn.setEnabled(False)
        layout.addWidget(self.split_class_btn)

        # === Результат разделения по классу ===
        self.class_result_label = QLabel("")
        self.class_result_label.setWordWrap(True)
        self.class_result_label.setStyleSheet("font-family: 'Courier'; font-size: 12px;")
        self.class_result_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setWidget(self.class_result_label)
        scroll2.setMaximumHeight(150)
        layout.addWidget(QLabel("<b>Результат разделения по классу:</b>"))
        layout.addWidget(scroll2)

        # === Кнопка сохранения нового набора ===
        self.save_class_btn = QPushButton("💾 Сохранить отфильтрованный датасет")
        self.save_class_btn.clicked.connect(self.save_class_dataset)
        self.save_class_btn.setEnabled(False)
        layout.addWidget(self.save_class_btn)

        # === Финал ===
        self.setLayout(layout)
        self.reset_state()

    def add_section_separator(self, layout, text):
        """Добавляет визуальный заголовок-разделитель в указанный layout"""
        label = QLabel(f"<b>{text}</b>")
        label.setStyleSheet("font-size: 14px; margin-top: 15px; margin-bottom: 5px;")
        layout.addWidget(label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

    def reset_state(self):
        """Сброс всех полей и состояний"""
        self.df = None
        self.target_column = None
        self.complete_df = None
        self.missing_df = None
        self.class_filtered_df = None
        self.df_path = None
        self.info_label.setText("Датасет не загружен.")
        self.target_combo.clear()
        self.target_combo.setEnabled(False)
        self.split_btn.setEnabled(False)
        self.save_complete_btn.setEnabled(False)
        self.save_missing_btn.setEnabled(False)
        self.result_label.setText("")

        self.class_combo.clear()
        self.class_combo.addItem("Выберите категорию")
        self.class_combo.setEnabled(False)
        self.type_label.setText("")
        self.clear_input_fields()
        self.split_class_btn.setEnabled(False)
        self.class_result_label.setText("")
        self.save_class_btn.setEnabled(False)

        # Удаляем ссылки на виджеты
        if hasattr(self, 'from_edit'):
            delattr(self, 'from_edit')
        if hasattr(self, 'to_edit'):
            delattr(self, 'to_edit')
        if hasattr(self, 'str_edit'):
            delattr(self, 'str_edit')

    def clear_input_fields(self):
        """Очищает поля ввода"""
        while self.input_container.count():
            child = self.input_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

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
            self.df = pd.read_csv(file_path)
            self.df_path = file_path
            filename = os.path.basename(file_path)
            rows, cols = self.df.shape

            self.info_label.setText(f"✅ Загружен: <b>{filename}</b><br>"
                                    f"Размер: <b>{rows} строк × {cols} столбцов</b><br>"
                                    f"Общее количество пропусков: <b>{self.df.isnull().sum().sum()}</b>")

            # Заполняем комбобоксы
            columns = list(self.df.columns)
            self.target_combo.clear()
            self.target_combo.addItems(columns)
            self.target_combo.setEnabled(True)
            self.target_combo.setCurrentIndex(0)
            self.on_target_changed(self.target_combo.currentText())

            self.class_combo.clear()
            self.class_combo.addItem("Выберите категорию")
            self.class_combo.addItems(columns)
            self.class_combo.setEnabled(True)
            self.class_combo.setCurrentIndex(0)

            self.split_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить датасет:\n{str(e)}")
            self.reset_state()

    def on_target_changed(self, column):
        """Обновление информации при смене целевой переменной"""
        if self.df is None or column not in self.df.columns:
            return

        missing_count = self.df[column].isnull().sum()
        not_missing_count = len(self.df) - missing_count

        self.result_label.setText(f"<b>Статистика по столбцу '{column}':</b><br><br>"
                                  f"• Заполнено: <b>{not_missing_count}</b> строк<br>"
                                  f"• Пропущено: <b>{missing_count}</b> строк<br><br>"
                                  f"Выберите 'Выполнить разделение', чтобы создать два набора.")

        self.target_column = column

    def on_class_column_changed(self, column):
        self.clear_input_fields()
        self.type_label.setText("")
        self.class_result_label.setText("")
        self.save_class_btn.setEnabled(False)

        # ✅ Исправляем передачу строки вместо bool
        self.split_class_btn.setEnabled(column not in ["", "Выберите категорию"])

        if self.df is None or column == "Выберите категорию" or column not in self.df.columns:
            return

        series = self.df[column].dropna()
        if series.empty:
            self.type_label.setText("⚠️ Нет данных для анализа")
            return

        # === Числовая колонка ===
        if pd.api.types.is_numeric_dtype(series):
            min_val, max_val = series.min(), series.max()
            self.type_label.setText(f"Тип: числовая (int/float). Диапазон: от {min_val} до {max_val}")

            row1 = QHBoxLayout()
            row1.addWidget(QLabel("Значение от:"))
            self.from_edit = QLineEdit()
            self.from_edit.setPlaceholderText(str(min_val))
            row1.addWidget(self.from_edit)
            self.input_container.addLayout(row1)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel("Значение до:"))
            self.to_edit = QLineEdit()
            self.to_edit.setPlaceholderText(str(max_val))
            row2.addWidget(self.to_edit)
            self.input_container.addLayout(row2)

        # === Строковая колонка ===
        else:
            self.type_label.setText("Тип: строка (str). Введите значения через запятую.")
            row = QHBoxLayout()
            row.addWidget(QLabel("Значения (через запятую):"))
            self.str_edit = QLineEdit()
            self.str_edit.setPlaceholderText("напр. Northern, Western")
            row.addWidget(self.str_edit)
            self.input_container.addLayout(row)

    def split_by_class(self):
        """Фильтрация по выбранной колонке"""
        if self.df is None:  # ✅ так правильно
            return
        column = self.class_combo.currentText()
        if not column or column == "Выберите категорию":
            QMessageBox.warning(self, "Предупреждение", "Выберите колонку для фильтрации!")
            return
        series = self.df[column].dropna()
        if pd.api.types.is_numeric_dtype(series):
            try:
                from_val = self.from_edit.text().strip()
                to_val = self.to_edit.text().strip() if hasattr(self, 'to_edit') else ""

                if not from_val:
                    QMessageBox.warning(self, "Ошибка", "Введите значение 'от'.")
                    return

                low = float(from_val)

                # Если заполнено 'до' → диапазон
                if to_val.strip():
                    high = float(to_val.strip())
                    if low > high:
                        QMessageBox.warning(self, "Ошибка", "Значение 'от' больше 'до'.")
                        return
                    mask = (self.df[column] >= low) & (self.df[column] <= high)
                    result_text = f"Найдено: <b>{mask.sum()}</b> строк ({low} ≤ x ≤ {high})"
                else:
                    # Только 'от' → точное совпадение
                    mask = self.df[column] == low
                    result_text = f"Найдено: <b>{mask.sum()}</b> строк (x = {low})"

            except ValueError:
                QMessageBox.critical(self, "Ошибка", "Введите корректные числовые значения!")
                return

        else:
            if not hasattr(self, 'str_edit'):
                return
            str_vals = self.str_edit.text().strip()
            if not str_vals:
                QMessageBox.warning(self, "Ошибка", "Введите хотя бы одно строковое значение!")
                return
            values = [v.strip() for v in str_vals.split(",") if v.strip()]
            mask = self.df[column].astype(str).isin(values)
            result_text = f"Найдено: <b>{mask.sum()}</b> строк (входит в {values})"

        self.class_filtered_df = self.df[mask].copy()
        self.class_result_label.setText(f"<b>Фильтрация по '{column}'</b><br>{result_text}")
        self.save_class_btn.setEnabled(True)

    def save_class_dataset(self):
        """Сохранить отфильтрованный датасет"""
        if self.class_filtered_df is None:
            return
        suffix = f"filtered_by_{self.class_combo.currentText()}.csv"
        self.save_dataframe(self.class_filtered_df, suffix, "Отфильтрованный датасет")

    def split_by_target(self):
        """Разделение по пропускам в целевой переменной"""
        if not self.target_column:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите целевую переменную!")
            return

        try:
            self.complete_df = self.df[self.df[self.target_column].notna()].copy()
            self.missing_df = self.df[self.df[self.target_column].isna()].copy()

            total = len(self.df)
            complete_count = len(self.complete_df)
            missing_count = len(self.missing_df)

            missing_stats = self.missing_df.isnull().sum()
            missing_cols_with_nan = missing_stats[missing_stats > 0]

            result_text = f"""
            <b>Разделение по целевой переменной: '{self.target_column}'</b><br><br>
            ✅ <b>Полный набор</b> (где '{self.target_column}' заполнена):<br>
            &nbsp;&nbsp;• Строк: {complete_count} ({complete_count/total*100:.1f}%)<br><br>
            
            ⚠️ <b>Набор с пропущенной целевой переменной</b> (где '{self.target_column}' = NaN):<br>
            &nbsp;&nbsp;• Строк: {missing_count} ({missing_count/total*100:.1f}%)<br>
            """

            if len(missing_cols_with_nan) > 0:
                result_text += "&nbsp;&nbsp;• Столбцы с пропусками:<br>"
                for col, count in missing_cols_with_nan.items():
                    result_text += f"&nbsp;&nbsp;&nbsp;&nbsp;• {col}: {count}<br>"
            else:
                result_text += "&nbsp;&nbsp;• Других пропусков нет<br>"

            self.result_label.setText(result_text)
            self.save_complete_btn.setEnabled(True)
            self.save_missing_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при разделении:\n{str(e)}")

    def save_complete(self):
        self.save_dataframe(self.complete_df, f"with_{self.target_column}_filled.csv",
                            f"Полный набор (с заполненной '{self.target_column}')")

    def save_with_missing(self):
        self.save_dataframe(self.missing_df, f"with_{self.target_column}_missing.csv",
                            f"Набор с пропущенной '{self.target_column}'")

    def save_dataframe(self, df, suffix, name):
        """Сохранение DataFrame в папку dataset/split"""
        try:
            output_dir = "dataset/split"
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(self.df_path))[0] if self.df_path else "dataset"
            filename = f"{output_dir}/{base_name}_{suffix}"
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "Успех", f"{name} сохранён:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
