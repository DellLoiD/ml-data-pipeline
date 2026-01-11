# preprocessing/feature_selector_ui.py
import os
import pandas as pd
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

# Импорт нового трекера
from utils.meta_tracker import MetaTracker


class FeatureSelector(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Выбор признаков датасета")
        layout = QVBoxLayout(self)

        # === Кнопка: Загрузить датасет ===
        load_data_btn = QPushButton("Выбрать датасет")
        load_data_btn.clicked.connect(self.load_dataset)
        layout.addWidget(load_data_btn)

        # === Выбор признака ===
        self.feature_combo_box = QComboBox()
        self.feature_combo_box.currentIndexChanged.connect(self.update_feature_info)
        layout.addWidget(self.feature_combo_box)

        # === Информация о признаке ===
        self.feature_info_label = QLabel("", alignment=Qt.AlignLeft)
        layout.addWidget(self.feature_info_label)

        # === Кнопка: Удалить признак ===
        delete_feature_btn = QPushButton("Удалить колонку")
        delete_feature_btn.clicked.connect(self.delete_selected_feature)
        layout.addWidget(delete_feature_btn)

        # === Кнопка: Сохранить датасет ===
        save_data_btn = QPushButton("Сохранить датасет")
        save_data_btn.clicked.connect(self.save_dataset)
        layout.addWidget(save_data_btn)

        # === Переменные ===
        self.df = None
        self._last_loaded_path = None
        self.selected_feature = None
        self.meta_tracker = MetaTracker(max_line_length=150)  # Управление историей

        self.setLayout(layout)

    def load_dataset(self):
        """Загрузка датасета с использованием MetaTracker"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выбор датасета", "./dataset", "CSV Files (*.csv)"
        )
        if not file_name:
            return

        try:
            # Загружаем мета-информацию
            self.meta_tracker.load_from_file(file_name)

            self.df = pd.read_csv(file_name, comment='#')
            self._last_loaded_path = file_name

            # Обновляем интерфейс
            features = self.df.columns.tolist()
            self.feature_combo_box.clear()
            self.feature_combo_box.addItems(features)
            self.feature_info_label.clear()
            self.selected_feature = None

            basename = os.path.basename(file_name)
            self.meta_tracker.add_change("загружен датасет для удаления признаков")

            QMessageBox.information(self, "Загружено", f"Файл: {basename}\nКолонок: {len(features)}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def update_feature_info(self):
        """Обновление статистики по выбранному признаку"""
        selected_feature = self.feature_combo_box.currentText()
        if not selected_feature or self.df is None or selected_feature not in self.df.columns:
            return

        self.selected_feature = selected_feature
        class_counts = self.df[selected_feature].value_counts()

        total_classes = len(class_counts)
        first_10 = class_counts.head(10)
        last_2 = class_counts.tail(2)

        # Комбинируем, избегая дубликатов
        combined_info = pd.concat([first_10, last_2], ignore_index=False).drop_duplicates()

        info = []
        for class_val, count in combined_info.items():
            info.append(f"Класс {class_val}: {count} экземпляров")

        # Добавляем разделитель, если много классов
        if total_classes > 12:
            skipped = total_classes - 12
            insert_pos = len(first_10)
            info.insert(insert_pos, f"... (ещё {skipped} классов)")

        self.feature_info_label.setText("\n".join(info))

    def delete_selected_feature(self):
        """Удаление выбранной колонки"""
        if not self.selected_feature:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите признак.")
            return

        reply = QMessageBox.question(
            self, "Подтверждение", f"Удалить колонку '{self.selected_feature}'?"
        )
        if reply != QMessageBox.Yes:
            return

        try:
            del self.df[self.selected_feature]
            self.meta_tracker.add_change(f"удалён признак '{self.selected_feature}'")

            # Обновляем интерфейс
            remaining_features = self.df.columns.tolist()
            self.feature_combo_box.clear()
            if remaining_features:
                self.feature_combo_box.addItems(remaining_features)
            else:
                self.feature_combo_box.addItem("Нет признаков")

            self.feature_info_label.clear()
            self.selected_feature = None

            QMessageBox.information(self, "Готово", f"Признак '{self.selected_feature}' удалён.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось удалить признак:\n{e}")

    def save_dataset(self):
        """Сохранение датасета с использованием MetaTracker"""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите датасет.")
            return

        # Определяем имя файла
        base_name = "dataset"
        if self._last_loaded_path:
            name = os.path.splitext(os.path.basename(self._last_loaded_path))[0]
            base_name = name.split("_v")[0]  # Убираем версию

        save_path = os.path.join("dataset", f"{base_name}_v{self.meta_tracker.version}.csv")

        try:
            # Сохраняем через MetaTracker
            success = self.meta_tracker.save_to_file(save_path, self.df)
            if success:
                self._last_loaded_path = save_path
                self.meta_tracker.version += 1  # Увеличиваем для следующего сохранения

                QMessageBox.information(
                    self, "Сохранено",
                    f"✅ Датасет сохранён:\n{os.path.basename(save_path)}\n\n"
                    f"Версия: v{self.meta_tracker.version - 1}"
                )
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")
