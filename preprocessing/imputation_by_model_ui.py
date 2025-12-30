# imputation_by_model_ui.py
import os
import joblib  # Только joblib — он умеет в .pkl от sklearn
import pandas as pd
import traceback
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QGroupBox, QTextEdit, QDialog, QDialogButtonBox, QApplication
)
from PySide6.QtGui import QFont, QClipboard


class CopyableMessageBox(QDialog):
    """Диалог с возможностью копирования текста ошибки"""

    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 300)

        layout = QVBoxLayout(self)

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(message)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)

        copy_button = QPushButton("Копировать")
        copy_button.clicked.connect(self.copy_to_clipboard)
        buttons.addButton(copy_button, QDialogButtonBox.ActionRole)

        layout.addWidget(buttons)

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())

    @classmethod
    def question(cls, parent, title, text):
        msg_box = QDialog(parent)
        msg_box.setWindowTitle(title)
        msg_box.resize(400, 150)

        layout = QVBoxLayout(msg_box)
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)

        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        buttons.accepted.connect(msg_box.accept)
        buttons.rejected.connect(msg_box.reject)
        layout.addWidget(buttons)

        result = msg_box.exec()
        return result == QDialog.Accepted

class ImputationByModelApp(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.model = None
        self.dataset_file_name = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Восстановление пропущенных значений моделью")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        self.load_data_btn = QPushButton("📁 Выбрать датасет с пропусками")
        self.load_data_btn.clicked.connect(self.load_dataset_with_nan)
        layout.addWidget(self.load_data_btn)

        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Целевая колонка (с NaN):"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.setPlaceholderText("Выберите колонку")
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)

        self.load_model_btn = QPushButton("🧠 Выбрать модель (.pkl или .joblib)")
        self.load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_btn)

        self.model_info_label = QLabel("Модель не загружена")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.model_info_label)

        warning_group = QGroupBox("⚠️ Важное предупреждение")
        warning_layout = QVBoxLayout()
        warning_text = QTextEdit()
        warning_text.setReadOnly(True)
        warning_text.setHtml(
            "Вам нужно убедиться, что данные имеют тот же формат, что и при обучении модели.<br><br>"
            "<b>Требования:</b><br>"
            "• Те же колонки (в том же порядке)<br>"
            "• Те же типы данных<br>"
            "• Те же преобразования (OHE, StandardScaler и т.д.)"
        )
        warning_layout.addWidget(warning_text)
        warning_group.setLayout(warning_layout)
        layout.addWidget(warning_group)

        self.run_btn = QPushButton("▶️ Запустить восстановление")
        self.run_btn.clicked.connect(self.run_imputation)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)

        results_group = QGroupBox("Результат")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Здесь появится отчёт...")
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)
        self.resize(700, 600)
        self.setWindowTitle("Восстановление пропущенных значений моделью")
        self.show()

    def show_critical(self, title, message):
        msg_box = CopyableMessageBox(title, message, self)
        msg_box.exec()

    def show_info(self, title, message):
        msg_box = CopyableMessageBox(title, message, self)
        msg_box.exec()

    def load_dataset_with_nan(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV файл", "./dataset/", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            self.dataset_file_name = os.path.basename(file_path)
            self.load_data_btn.setText(f"✅ {self.dataset_file_name}")

            self.target_combo.clear()
            nan_columns = [col for col in self.df.columns if self.df[col].isna().sum() > 0]

            if nan_columns:
                self.target_combo.addItems(nan_columns)
            else:
                self.target_combo.addItem("Нет колонок с NaN")

            self.target_combo.setEnabled(bool(nan_columns))

            self.results_text.setText(
                f"Загружен: {self.dataset_file_name}\n"
                f"Размер: {self.df.shape[0]}×{self.df.shape[1]}\n"
                f"Колонок с NaN: {len(nan_columns)}"
            )

            self.check_run_button_state()

        except Exception as e:
            self.show_critical("Ошибка загрузки CSV", f"Не удалось загрузить датасет:\n\n{type(e).__name__}: {e}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель", "./models/",
            "Joblib/Pickle Files (*.pkl *.joblib *.pickle);;All Files (*)"
        )
        if not file_path:
            return

        try:
            # Проверяем существование и размер
            if not os.path.exists(file_path):
                raise FileNotFoundError("Файл не найден")

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError("Файл пуст")

            # Используем joblib.load() — он работает и с .pkl от sklearn
            self.show_info("Загрузка", f"Попытка загрузить модель...\nФайл: {os.path.basename(file_path)}\nРазмер: {file_size} байт")

            loaded_obj = joblib.load(file_path)

            # Если это словарь — извлекаем модель
            if isinstance(loaded_obj, dict):
                if 'model' in loaded_obj:
                    self.model = loaded_obj['model']
                    self.show_info("Извлечение", "Найден ключ 'model' — модель извлечена.")
                else:
                    found = False
                    for key, val in loaded_obj.items():
                        if hasattr(val, 'predict'):
                            self.model = val
                            self.show_info("Извлечение", f"Модель найдена по ключу: '{key}'")
                            found = True
                            break
                    if not found:
                        raise ValueError("В словаре не найдена модель с методом .predict()")
            else:
                self.model = loaded_obj
                self.show_info("Загрузка", "Модель загружена напрямую.")

            # Проверка
            if not hasattr(self.model, 'predict'):
                raise AttributeError(f"Объект типа {type(self.model)} не имеет метода .predict()")

            model_name = os.path.basename(file_path)
            self.load_model_btn.setText(f"✅ {model_name}")
            self.model_info_label.setText(f"Загружена модель: {model_name}")
            self.check_run_button_state()

            self.show_info("Успех", f"Модель '{model_name}' успешно загружена!")

        except Exception as e:
            tb_lines = traceback.format_exception_only(type(e), e)
            tb_str = ''.join(tb_lines)
            self.show_critical("Ошибка загрузки модели", f"Не удалось загрузить модель:\n\n{tb_str}")

    def check_run_button_state(self):
        has_data = self.df is not None
        has_target = self.target_combo.isEnabled() and self.target_combo.currentText()
        has_model = self.model is not None
        self.run_btn.setEnabled(has_data and has_target and has_model)

    def run_imputation(self):
        if self.df is None or self.model is None:
            self.show_critical("Ошибка", "Сначала загрузите датасет и модель!")
            return

        target_col = self.target_combo.currentText()
        if not target_col or target_col not in self.df.columns:
            self.show_critical("Ошибка", "Выберите корректную целевую колонку!")
            return

        nan_mask = self.df[target_col].isna()
        num_missing = nan_mask.sum()
        if num_missing == 0:
            self.show_info("Готово", f"В колонке '{target_col}' нет пропусков.")
            return

        feature_cols = [col for col in self.df.columns if col != target_col]
        X_missing = self.df.loc[nan_mask, feature_cols]

        if X_missing.empty:
            self.show_critical("Ошибка", "Нет данных для предсказания.")
            return

        try:
            expected_features = getattr(self.model, 'n_features_in_', None)
            if expected_features and X_missing.shape[1] != expected_features:
                self.show_critical("Ошибка",
                                   f"Количество признаков не совпадает!\n"
                                   f"Ожидалось: {expected_features}, получено: {X_missing.shape[1]}\n"
                                   "Убедитесь, что колонки совпадают и предобработаны.")
                return
        except Exception as ex:
            self.show_info("Проверка", f"Не удалось проверить количество признаков: {ex}")

        try:
            predictions = self.model.predict(X_missing)
            self.df.loc[nan_mask, target_col] = predictions

            sample_preds = predictions[:10]
            result_text = f"<b>✅ Восстановлено {num_missing} значений!</b><br><br>"
            result_text += f"Целевая колонка: <b>{target_col}</b><br>"
            result_text += "Первые 10 значений:<br><pre>"
            for i, pred in enumerate(sample_preds):
                result_text += f"{i+1:2d}. {pred:.4f}\n"
            result_text += "</pre>"

            self.results_text.setHtml(result_text)
            self.ask_save_result()

        except Exception as e:
            self.show_critical("Ошибка предсказания", f"Не удалось выполнить предсказание:\n{e}")

    def ask_save_result(self):
        reply = CopyableMessageBox.question(self, "Сохранить результат?", "Сохранить обновлённый датасет?")
        if reply:
            self.save_dataset()

    def save_dataset(self):
        if self.df is None:
            return

        default_name = f"imputed_{self.dataset_file_name}"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить датасет", f"./dataset/{default_name}", "CSV Files (*.csv)"
        )
        if not save_path:
            return

        try:
            self.df.to_csv(save_path, index=False)
            self.show_info("Сохранено", f"✅ Датасет сохранён:\n{save_path}")
        except Exception as e:
            self.show_critical("Ошибка", f"Не удалось сохранить файл:\n{e}")
