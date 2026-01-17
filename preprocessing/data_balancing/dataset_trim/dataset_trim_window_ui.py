# preprocessing/data_balancing/dataset_trim/dataset_trim_window_ui.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QMessageBox, QHBoxLayout, QToolButton, QInputDialog
)
from PySide6.QtCore import Qt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.meta_tracker import MetaTracker
from .dataset_trim_window_logic import DatasetTrimLogic


class DatasetTrimWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.logic = DatasetTrimLogic()
        self.df = None
        self.df_train = None
        self.df_test = None
        self._last_loaded_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)
        self.split_mode = False  # True если выбрано "разделить"
        self.target_col = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # === Загрузка датасета ===
        self.load_button = QPushButton("Загрузить датасет")
        self.load_button.clicked.connect(self.on_load_dataset_clicked)
        layout.addWidget(self.load_button)

        self.file_name_label = QLabel("Файл не выбран")
        layout.addWidget(self.file_name_label)

        # === Выбор целевой переменной ===
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Целевая переменная:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)

        # === Стратегия обрезки ===
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Стратегия:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Равное количество",
            "Пропорционально",
            "Только мажоритарный"
        ])
        strategy_layout.addWidget(self.strategy_combo)

        self.help_button = QToolButton()
        self.help_button.setText("?")
        self.help_button.setFixedSize(20, 20)
        self.help_button.clicked.connect(self.show_strategy_help)
        strategy_layout.addWidget(self.help_button)
        layout.addLayout(strategy_layout)

        # === Число записей ===
        trim_layout = QHBoxLayout()
        trim_layout.addWidget(QLabel("Число записей:"))
        self.trim_input = QLineEdit()
        self.trim_input.setPlaceholderText("например: 2000")
        trim_layout.addWidget(self.trim_input)
        layout.addLayout(trim_layout)

        # === Кнопка обрезки ===
        self.trim_button = QPushButton("Обрезать датасет")
        self.trim_button.clicked.connect(self.on_trim_dataset_clicked)
        self.trim_button.setEnabled(False)
        layout.addWidget(self.trim_button)

        # === Статистика ===
        self.before_label = QLabel("Статистика до обработки:")
        self.before_label.setWordWrap(True)
        layout.addWidget(self.before_label)

        self.after_label = QLabel("Статистика после обработки:")
        self.after_label.setWordWrap(True)
        layout.addWidget(self.after_label)

        # === Сохранение ===
        self.save_button = QPushButton("Сохранить датасет")
        self.save_button.clicked.connect(self.on_save_button_clicked)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.setWindowTitle("Обрезка датасета")
        self.resize(500, 600)

    def show_strategy_help(self):
        strategy = self.strategy_combo.currentText()
        help_text = ""

        if strategy == "Равное количество":
            help_text = (
                "Оставляет одинаковое количество записей в каждом классе.\n"
                "Например: если выбрать 1000, то будет по 1000 записей для\n"
                "«Здоровый», «Преддиабет», «Диабет»."
            )
        elif strategy == "Пропорционально":
            help_text = (
                "Сохраняет оригинальное соотношение классов, но уменьшает\n"
                "общее количество записей. Например: если было 90% здоровых,\n"
                "то после обрезки их останется ~90%."
            )
        elif strategy == "Только мажоритарный":
            help_text = (
                "Обрезает только самый большой класс (например, «Здоровые»),\n"
                "оставляя малые классы («Диабет») без изменений.\n"
                "Полезно, когда миноритарные классы нельзя терять."
            )

        QMessageBox.information(self, f"Справка: {strategy}", help_text)

    def on_load_dataset_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите датасет", "./dataset", "CSV Files (*.csv)"
        )
        if not file_name:
            return

        try:
            self.meta_tracker.load_from_file(file_name)
            df = pd.read_csv(file_name, comment='#')
            self._last_loaded_path = file_name

            # Спросить: разделить на train/test?
            reply = QMessageBox.question(
                self, "Разделение",
                "Разделить датасет на train и test?",
                QMessageBox.Yes | QMessageBox.No
            )
            self.split_mode = reply == QMessageBox.Yes

            numeric_columns = df.select_dtypes(include=['number', 'bool']).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Ошибка", "Нет числовых колонок.")
                return

            item, ok = QInputDialog.getItem(
                self, "Целевая переменная", "Выберите target-колонку:",
                numeric_columns, editable=False
            )
            if not ok or not item:
                return

            self.target_col = item
            self.target_combo.clear()
            self.target_combo.addItems(numeric_columns)
            self.target_combo.setCurrentText(item)

            if self.split_mode:
                # Разделяем
                X = df.drop(columns=[item]).values
                y = df[item].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                self.df_train = pd.DataFrame(X_train, columns=df.drop(columns=[item]).columns)
                self.df_train[item] = y_train
                self.df_test = pd.DataFrame(X_test, columns=df.drop(columns=[item]).columns)
                self.df_test[item] = y_test
                self.df = self.df_train.copy()  # Работаем с train
            else:
                self.df = df.copy()

            self.file_name_label.setText(f"✅ {os.path.basename(file_name)}")
            self.update_before_stats()

            self.trim_button.setEnabled(True)
            self.meta_tracker.add_change(f"загружен датасет для обрезки ({'разделён' if self.split_mode else 'полный'})")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def update_before_stats(self):
        if self.df is None or self.target_col is None:
            return
        counts = self.df[self.target_col].value_counts().to_string()
        self.before_label.setText(f"До обрезки:\n{counts}")

    def on_trim_dataset_clicked(self):
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет.")
            return

        try:
            n = int(self.trim_input.text())
            if n <= 0:
                raise ValueError("Число должно быть > 0")

            self.logic.set_data(self.df, self.target_col)
            strategy = self.strategy_combo.currentText()

            if strategy == "Равное количество":
                trimmed_df = self.logic.trim_equal(n)
            elif strategy == "Пропорционально":
                trimmed_df = self.logic.trim_proportional(n)
            elif strategy == "Только мажоритарный":
                trimmed_df = self.logic.trim_majority_only(n)
            else:
                raise ValueError("Неизвестная стратегия")

            self.logic.df_trimmed = trimmed_df
            self.update_after_stats(trimmed_df)
            self.save_button.setEnabled(True)
            self.meta_tracker.add_change(f"обрезан ({strategy}) до {n} записей")

        except ValueError as e:
            QMessageBox.critical(self, 'Ошибка', str(e))
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f"Не удалось обрезать: {e}")

    def update_after_stats(self, df):
        counts = df[self.target_col].value_counts().to_string()
        self.after_label.setText(f"После обрезки:\n{counts}")

    def on_save_button_clicked(self):
        if not hasattr(self.logic, 'df_trimmed') or self.logic.df_trimmed is None:
            QMessageBox.warning(self, "Ошибка", "Сначала обрежьте датасет.")
            return

        try:
            base_name = "dataset"
            if self._last_loaded_path:
                name = os.path.basename(self._last_loaded_path)
                base_name = os.path.splitext(name)[0].split("_v")[0]

            version = self.meta_tracker.version

            if self.split_mode:
                # Сохраняем train и test
                train_path = os.path.join("dataset", f"{base_name}_train_v{version}.csv")
                test_path = os.path.join("dataset", f"{base_name}_test_v{version}.csv")

                # Сохраняем train
                success_train = self.meta_tracker.save_to_file(train_path, self.logic.df_trimmed)
                # Сохраняем test (без изменения версии)
                success_test = self.meta_tracker.save_to_file(test_path, self.df_test, preserve_version=True)

                if success_train and success_test:
                    self.meta_tracker.version += 1
                    self._last_loaded_path = train_path
                    QMessageBox.information(
                        self, "Сохранено",
                        f"✅ Сохранено:\n"
                        f"• {os.path.basename(train_path)}\n"
                        f"• {os.path.basename(test_path)}\n\n"
                        f"Версия: v{version}"
                    )
                else:
                    QMessageBox.critical(self, "Ошибка", "Не удалось сохранить один из файлов.")

            else:
                # Сохраняем полный
                save_path = os.path.join("dataset", f"{base_name}_v{version}.csv")
                success = self.meta_tracker.save_to_file(save_path, self.logic.df_trimmed)
                if success:
                    self._last_loaded_path = save_path
                    self.meta_tracker.version += 1
                    QMessageBox.information(
                        self, "Сохранено",
                        f"✅ Датасет сохранён:\n{os.path.basename(save_path)}\n\n"
                        f"Версия: v{version}"
                    )
                else:
                    QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")
