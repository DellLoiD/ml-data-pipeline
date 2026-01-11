# preprocessing/data_balancing/dataset_trim/dataset_trim_window_ui.py
from PySide6.QtWidgets import QDialog, QPushButton, QLabel, QVBoxLayout, QLineEdit, QMessageBox, QFileDialog, QInputDialog
import os
import pandas as pd
from preprocessing.data_balancing.dataset_trim.dataset_trim_window_logic import DatasetTrimLogic
from utils.meta_tracker import MetaTracker


class DatasetTrimWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.logic = DatasetTrimLogic()
        self._last_loaded_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)  # Управление историей и версиями
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.load_button = QPushButton("Загрузить датасет")
        self.load_button.clicked.connect(self.on_load_dataset_clicked)
        layout.addWidget(self.load_button)

        self.file_name_label = QLabel("Файл не выбран")
        layout.addWidget(self.file_name_label)

        self.trim_input = QLineEdit()
        self.trim_input.setPlaceholderText("число записей")
        layout.addWidget(self.trim_input)

        self.trim_button = QPushButton("Обрезать датасет")
        self.trim_button.clicked.connect(self.on_trim_dataset_clicked)
        layout.addWidget(self.trim_button)

        self.before_label = QLabel("Статистика до обработки:")
        layout.addWidget(self.before_label)

        self.after_label = QLabel("Статистика после обработки:")
        layout.addWidget(self.after_label)

        self.save_button = QPushButton("Сохранить датасет")
        self.save_button.clicked.connect(self.on_save_button_clicked)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.setWindowTitle("Обрезка датасета")
        self.resize(400, 400)

    def on_load_dataset_clicked(self):
        """Загрузка датасета с использованием MetaTracker"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите датасет", "./dataset", "CSV Files (*.csv)"
        )
        if not file_name:
            return

        try:
            # Загружаем мета-информацию
            self.meta_tracker.load_from_file(file_name)

            df = pd.read_csv(file_name, comment='#')
            self._last_loaded_path = file_name

            # Предположим, что есть target-колонка (настройте под ваш случай)
            numeric_columns = df.select_dtypes(include=['number', 'bool']).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Ошибка", "Нет числовых колонок для балансировки.")
                return

            item, ok = QInputDialog.getItem(
                self, "Целевая переменная", "Выберите target-колонку:", numeric_columns, 0, False
            )
            if not ok or not item:
                return

            target_col = item
            feature_cols = list(set(numeric_columns) - {target_col})

            X = df[feature_cols].values
            y = df[target_col].values

            # Передаём данные в логику
            self.logic.X_original = X
            self.logic.y_original = y
            self.logic.feature_cols = feature_cols
            self.logic.target_col = target_col

            # Обновляем интерфейс
            self.file_name_label.setText(f"✅ {os.path.basename(file_name)}")
            before_stats = pd.Series(y).value_counts().to_string()
            self.before_label.setText(f"До обрезки:\n{before_stats}")

            # Фиксируем операцию
            self.meta_tracker.add_change("загружен датасет для обрезки")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def on_trim_dataset_clicked(self):
        """Обрезка датасета"""
        try:
            samples_count = int(self.trim_input.text())
            if samples_count <= 0:
                raise ValueError("Количество записей должно быть положительным")

            X_trimmed, y_trimmed = self.logic.trim_dataset(samples_count)
            after_stats = pd.Series(y_trimmed).value_counts().to_string()
            self.after_label.setText(f"После обрезки:\n{after_stats}")

            # Фиксируем операцию
            self.meta_tracker.add_change(f"обрезан до {samples_count} записей")

        except ValueError as e:
            QMessageBox.critical(self, 'Ошибка', str(e))
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f"Не удалось обрезать: {e}")

    def on_save_button_clicked(self):
        """Сохранение обрезанного датасета"""
        if not hasattr(self.logic, 'X_resampled') or self.logic.X_resampled is None:
            QMessageBox.warning(self, "Ошибка", "Датасет сначала нужно обрезать.")
            return

        try:
            # Извлекаем оригинальное имя
            base_name = "trimmed_dataset"
            if self._last_loaded_path:
                name = os.path.basename(self._last_loaded_path)
                base_name = os.path.splitext(name)[0].split("_v")[0]

            save_path = os.path.join("dataset", f"{base_name}_v{self.meta_tracker.version}.csv")

            # Создаём DataFrame для сохранения
            df_to_save = pd.DataFrame(self.logic.X_resampled, columns=self.logic.feature_cols)
            df_to_save[self.logic.target_col] = self.logic.y_resampled

            # Сохраняем с помощью MetaTracker
            success = self.meta_tracker.save_to_file(save_path, df_to_save)
            if success:
                self._last_loaded_path = save_path
                self.meta_tracker.version += 1
                QMessageBox.information(
                    self, "Сохранено",
                    f"✅ Датасет сохранён:\n{os.path.basename(save_path)}\n\n"
                    f"Версия: v{self.meta_tracker.version - 1}"
                )
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")
