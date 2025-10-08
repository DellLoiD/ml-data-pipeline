#from preprocessing.dataset_processing_ui import DatasetProcessingWindow
import pandas as pd
from PySide6.QtWidgets import QFileDialog, QPushButton, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox
import os

def select_raw_dataset(window):
    file_dialog = QFileDialog()
    filename, _ = file_dialog.getOpenFileName(None, 'Выбрать датасет', './dataset', 'CSV Files (*.csv)')
    if not filename:
        return
    try:
        window.df = pd.read_csv(filename)
        basename = os.path.basename(filename)
        window.btn_select_dataset.setText(f'Файл загружен: {basename}')
        window.selected_file_path = filename
    except Exception as e:
        QMessageBox.critical(None, "Ошибка", f"Произошла ошибка при чтении датасета:\n{e}")

def show_missing_values_dialog(df, parent_widget):
    class MissingValuesDialog(QDialog):
        def __init__(self, df, selected_file_path, parent=None):
            super().__init__(parent)
            self.parent_widget = parent
            self.df = df
            self.selected_file_path = selected_file_path
            
            layout = QVBoxLayout()
            self.label = QLabel()
            missing_data = df.isnull().sum()
            result_text = "\n".join([f"{col}: {val}" for col, val in zip(missing_data.index, missing_data)])
            self.label.setText(result_text)
            layout.addWidget(self.label)
            
            button_layout = QHBoxLayout()
            clear_and_save_button = QPushButton("Очистить пропуски и сохранить")
            cancel_button = QPushButton("Закрыть окно")
            button_layout.addWidget(clear_and_save_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            clear_and_save_button.clicked.connect(self.clear_and_save)
            cancel_button.clicked.connect(self.reject)
            
            self.setLayout(layout)
            self.setWindowTitle('Проверка пропусков')
        
        def clear_and_save(self):
            # Удаляем пропуски
            cleaned_df = self.df.dropna()
            
            # Получаем оригинальное имя файла
            original_basename = os.path.splitext(os.path.basename(self.selected_file_path))[0]
            
            # Формируем новое имя файла
            new_filename = f"./dataset/{original_basename}_balancing.csv"
            
            # Сохраняем обработанный датасет
            cleaned_df.to_csv(new_filename, index=False)
            
            # Подсчет количества удалённых строк
            deleted_rows_count = len(self.df) - len(cleaned_df)
            
            # Показываем сообщение пользователю
            message = f"Датасет очищен и сохранён в {new_filename}.\nУдалено пропусков: {deleted_rows_count} строки."
            QMessageBox.information(self.parent_widget, "Подтверждение", message)
            self.accept()

    dialog = MissingValuesDialog(df, parent_widget)
    dialog.exec_()