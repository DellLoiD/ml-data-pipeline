#splitting_dataset.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
    QScrollArea, QComboBox, QHBoxLayout, QFrame, QLineEdit,
    QListWidget, QListWidgetItem, QGroupBox
)
from PySide6.QtCore import Qt
import os
import pandas as pd

# Импорт нового трекера
from utils.meta_tracker import MetaTracker


class SplittingDatasetWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.target_column = None
        self.complete_df = None  # где target НЕ NaN
        self.missing_df = None   # где target NaN
        self.class_filtered_df = None
        self.df_path = None      # путь к исходному файлу
        self._last_loaded_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)  # Управление историей и версиями
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # === Кнопка загрузки — без заголовка ===
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

        # === История изменений ===
        history_group = QGroupBox("История изменений")
        history_layout = QVBoxLayout()

        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                font-family: 'Courier';
                font-size: 12px;
                background: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background: #e0f0ff;
                color: #000;
            }
        """)
        self.history_list.setFixedHeight(120)
        history_layout.addWidget(self.history_list)

        self.label_detail = QLabel("Выберите версию, чтобы посмотреть изменения.")
        self.label_detail.setWordWrap(True)
        self.label_detail.setStyleSheet("font-size: 11px; color: #555;")
        history_layout.addWidget(self.label_detail)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        # === Финал ===
        self.setLayout(layout)
        self.reset_state()

        # Подключаем клик по истории
        self.history_list.itemClicked.connect(self.on_history_item_clicked)

    def on_history_item_clicked(self, item):
        """Показывает детали выбранной версии"""
        version = item.text().split(" ")[0]  # v1
        changes = self.meta_tracker.get_change_description(version)
        self.label_detail.setText(f"🔸 {changes}")

    def update_history_display(self):
        """Обновляет список истории"""
        self.history_list.clear()
        for version, changes in self.meta_tracker.history.items():
            item = QListWidgetItem(f"{version} – {changes}")
            self.history_list.addItem(item)

    def add_section_separator(self, layout, text):
        """Добавляет визуальный заголовок-разделитель"""
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
        self._last_loaded_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)  # Восстанавливаем трекер

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

        if hasattr(self, 'from_edit'):
            delattr(self, 'from_edit')
        if hasattr(self, 'to_edit'):
            delattr(self, 'to_edit')
        if hasattr(self, 'str_edit'):
            delattr(self, 'str_edit')

        self.update_history_display()

    def clear_input_fields(self):
        """Очищает поля ввода"""
        while self.input_container.count():
            child = self.input_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def load_dataset(self):
        """Загрузка датасета с использованием MetaTracker"""
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
            # Загружаем мета-информацию
            self.meta_tracker.load_from_file(file_path)

            # Читаем данные
            self.df = pd.read_csv(file_path, comment='#', skipinitialspace=True)
            self.df_path = file_path
            self._last_loaded_path = file_path

            filename = os.path.basename(file_path)
            rows, cols = self.df.shape

            self.info_label.setText(f"✅ Загружен: <b>{filename}</b><br>"
                                    f"Размер: <b>{rows} строк × {cols} столбцов</b><br>"
                                    f"Общее количество пропусков: <b>{self.df.isnull().sum().sum()}</b>")

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

            # Обновляем историю
            self.meta_tracker.add_change("загружен датасет для разделения")
            self.update_history_display()

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
                                  f"Нажмите 'Выполнить разделение'.")

        self.target_column = column

    def on_class_column_changed(self, column):
        self.clear_input_fields()
        self.type_label.setText("")
        self.class_result_label.setText("")
        self.save_class_btn.setEnabled(False)

        self.split_class_btn.setEnabled(column not in ["", "Выберите категорию"])

        if self.df is None or column == "Выберите категорию" or column not in self.df.columns:
            return

        series = self.df[column].dropna()
        if series.empty:
            self.type_label.setText("⚠️ Нет данных для анализа")
            return

        if pd.api.types.is_numeric_dtype(series):
            min_val, max_val = series.min(), series.max()
            self.type_label.setText(f"Тип: числовая. Диапазон: от {min_val} до {max_val}")

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

        else:
            self.type_label.setText("Тип: строка. Введите значения через запятую.")
            row = QHBoxLayout()
            row.addWidget(QLabel("Значения (через запятую):"))
            self.str_edit = QLineEdit()
            self.str_edit.setPlaceholderText("напр. Northern, Western")
            row.addWidget(self.str_edit)
            self.input_container.addLayout(row)

    def split_by_class(self):
        """Фильтрация по выбранной колонке"""
        if self.df is None:
            return
        column = self.class_combo.currentText()
        if not column or column == "Выберите категорию":
            QMessageBox.warning(self, "Предупреждение", "Выберите колонку!")
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

                if to_val.strip():
                    high = float(to_val.strip())
                    if low > high:
                        QMessageBox.warning(self, "Ошибка", "'От' больше 'до'.")
                        return
                    mask = (self.df[column] >= low) & (self.df[column] <= high)
                    result_text = f"Найдено: <b>{mask.sum()}</b> строк ({low} ≤ x ≤ {high})"
                    change_text = f"фильтрация по '{column}' от {low} до {high}"
                else:
                    mask = self.df[column] == low
                    result_text = f"Найдено: <b>{mask.sum()}</b> строк (x = {low})"
                    change_text = f"фильтрация по '{column}' = {low}"

            except ValueError:
                QMessageBox.critical(self, "Ошибка", "Введите корректные числовые значения!")
                return

        else:
            if not hasattr(self, 'str_edit'):
                return
            str_vals = self.str_edit.text().strip()
            if not str_vals:
                QMessageBox.warning(self, "Ошибка", "Введите строковые значения!")
                return
            values = [v.strip() for v in str_vals.split(",") if v.strip()]
            mask = self.df[column].astype(str).isin(values)
            result_text = f"Найдено: <b>{mask.sum()}</b> строк (входит в {values})"
            change_text = f"фильтрация по '{column}' в {values}"

        self.class_filtered_df = self.df[mask].copy()
        self.class_result_label.setText(f"<b>Фильтрация по '{column}'</b><br>{result_text}")
        self.save_class_btn.setEnabled(True)

        # Добавляем изменение
        self.meta_tracker.add_change(change_text)

    def split_by_target(self):
        """Разделение по пропускам в целевой переменной"""
        if not self.target_column:
            QMessageBox.warning(self, "Предупреждение", "Выберите целевую переменную!")
            return

        try:
            self.complete_df = self.df[self.df[self.target_column].notna()].copy()
            self.missing_df = self.df[self.df[self.target_column].isna()].copy()

            total = len(self.df)
            complete_count = len(self.complete_df)
            missing_count = len(self.missing_df)

            change_text = f"разделён по пропускам в '{self.target_column}'"
            self.meta_tracker.add_change(change_text)

            result_text = f"""
            <b>Разделение по целевой переменной: '{self.target_column}'</b><br><br>
            ✅ <b>Полный набор</b> (заполнена):<br>
            &nbsp;&nbsp;• {complete_count} строк ({complete_count/total*100:.1f}%)<br><br>
            ⚠️ <b>Набор с пропущенной</b>:<br>
            &nbsp;&nbsp;• {missing_count} строк ({missing_count/total*100:.1f}%)
            """

            self.result_label.setText(result_text)
            self.save_complete_btn.setEnabled(True)
            self.save_missing_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при разделении:\n{str(e)}")

    def save_complete(self):
        self.save_dataframe(self.complete_df, f"with_{self.target_column}_filled", "Полный набор")

    def save_with_missing(self):
        self.save_dataframe(self.missing_df, f"with_{self.target_column}_missing", "Набор с пропусками")

    def save_class_dataset(self):
        if self.class_filtered_df is None:
            return
        suffix = f"filtered_by_{self.class_combo.currentText()}"
        self.save_dataframe(self.class_filtered_df, suffix, "Отфильтрованный датасет")

    def save_dataframe(self, df, suffix, name):
        """Сохранение DataFrame с использованием MetaTracker"""
        try:
            output_dir = "dataset/split"
            os.makedirs(output_dir, exist_ok=True)

            base_name = "dataset"
            if self._last_loaded_path:
                base_name = os.path.splitext(os.path.basename(self._last_loaded_path))[0]
                base_name = base_name.split("_v")[0]

            save_path = os.path.join(output_dir, f"{base_name}_{suffix}_v{self.meta_tracker.version}.csv")

            success = self.meta_tracker.save_to_file(save_path, df)
            if success:
                self._last_loaded_path = save_path
                self.meta_tracker.version += 1
                self.update_history_display()
                self.label_detail.setText(f"✅ Последнее изменение сохранено (v{self.meta_tracker.version - 1})")

                QMessageBox.information(
                    self, "Сохранено",
                    f"{name} сохранён:\n{os.path.basename(save_path)}\n\nВерсия: v{self.meta_tracker.version - 1}"
                )
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
