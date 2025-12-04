# preprocessing/dataset_processing_check_nan.py
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox
)
import os
import sys

class MissingValuesDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.df = None
        self.selected_file_path = None

        # Основной макет
        main_layout = QVBoxLayout()

        # Кнопка выбора датасета
        self.btn_select_dataset = QPushButton('Выбрать датасет')
        self.btn_select_dataset.clicked.connect(self.select_raw_dataset)
        main_layout.addWidget(self.btn_select_dataset)

        # === НОВОЕ: Отображение общего количества строк ===
        self.label_total_rows = QLabel("Всего количество строк в датасете: —")
        self.label_total_rows.setStyleSheet("font-weight: bold; color: #0066cc;")
        main_layout.addWidget(self.label_total_rows)

        # Информация о пропусках
        self.label_missing_values = QLabel("Нет загруженного датасета")
        main_layout.addWidget(self.label_missing_values)

        # Кнопка показа пропусков
        self.btn_show_missing = QPushButton('Показать пропуски')
        self.btn_show_missing.clicked.connect(self.show_missing_values)
        main_layout.addWidget(self.btn_show_missing)

        # Кнопки "Очистить и сохранить" + "Закрыть"
        buttons_layout = QHBoxLayout()
        self.clean_and_save_button = QPushButton("Очистить пропуски и сохранить")
        self.clean_and_save_button.clicked.connect(self.clear_and_save)
        buttons_layout.addWidget(self.clean_and_save_button)

        close_button = QPushButton("Закрыть окно")
        close_button.clicked.connect(self.close)
        buttons_layout.addWidget(close_button)

        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

        # Настройки окна
        self.setWindowTitle('Проверка пропусков')
        self.resize(400, 300)

    def select_raw_dataset(self):
        """Выбор датасета через диалоговое окно."""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Выбрать датасет', './dataset', 'CSV Files (*.csv)'
        )
        if not filename:
            return

        try:
            self.df = pd.read_csv(filename)
            basename = os.path.basename(filename)
            self.btn_select_dataset.setText(f'Файл загружен: {basename}')
            self.selected_file_path = filename

            # ✅ Обновляем общее количество строк
            total_rows = len(self.df)
            self.label_total_rows.setText(f"Всего количество строк в датасете: {total_rows}")

            # Сбрасываем отображение пропусков
            self.label_missing_values.setText("Нажмите 'Показать пропуски', чтобы увидеть детали.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при чтении датасета:\n{e}")
            self.label_total_rows.setText("Всего количество строк в датасете: —")

    def show_missing_values(self):
        """Отображает количество пропусков в каждой колонке."""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите датасет!")
            return

        missing_data = self.df.isnull().sum()
        result_text = "<html><b>Количество пропусков по колонкам:</b><br>"
        for column, count in missing_data.items():
            result_text += f'{column}: <span style="color:red;">{count}</span><br>'
        result_text += "</html>"
        self.label_missing_values.setText(result_text)

    def clear_and_save(self):
        """Очищает пропуски и сохраняет обновленный датасет."""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите датасет!")
            return

        cleaned_df = self.df.dropna()
        file_name_without_extension = os.path.splitext(os.path.basename(self.selected_file_path))[0]
        new_filename = f'dataset/{file_name_without_extension}_cleaned.csv'
        cleaned_df.to_csv(new_filename, index=False)
        rows_deleted = len(self.df) - len(cleaned_df)
        QMessageBox.information(
            self, "Готово",
            f"Датасет успешно очищен и сохранён в:\n{new_filename}\n\n"
            f"Количество удалённых строк: {rows_deleted}"
        )


