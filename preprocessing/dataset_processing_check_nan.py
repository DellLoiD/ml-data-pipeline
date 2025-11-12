import pandas as pd
from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox, QScrollArea, QVBoxLayout, QWidget
import os, sys  
 
# Класс окна проверки пропусков теперь унаследован от QWidget
class MissingValuesDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.df = None
        self.selected_file_path = None
        
        # Основной макет виджетов
        main_layout = QVBoxLayout()
        
        # Кнопка для выбора датасета
        self.btn_select_dataset = QPushButton('Выбрать датасет')
        self.btn_select_dataset.clicked.connect(self.select_raw_dataset)
        main_layout.addWidget(self.btn_select_dataset)
        
        # Информация о пропусках
        self.label_missing_values = QLabel("Нет загруженного датасета")
        main_layout.addWidget(self.label_missing_values)
        
        # Кнопка показать пропуски
        self.btn_show_missing = QPushButton('Показать пропуски')
        self.btn_show_missing.clicked.connect(self.show_missing_values)
        main_layout.addWidget(self.btn_show_missing)
        
        # Горизонтальная панель с двумя кнопками
        buttons_layout = QHBoxLayout()
        self.clean_and_save_button = QPushButton("Очистить пропуски и сохранить")
        self.clean_and_save_button.clicked.connect(self.clear_and_save)
        buttons_layout.addWidget(self.clean_and_save_button)
        
        close_button = QPushButton("Закрыть окно")
        close_button.clicked.connect(self.close)
        buttons_layout.addWidget(close_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Установим основной макет
        self.setLayout(main_layout)
        self.setWindowTitle('Проверка пропусков')
        self.resize(300, 200)

    def select_raw_dataset(self):
        """Выбор датасета через диалоговое окно."""
        dialog = QFileDialog()
        filename, _ = dialog.getOpenFileName(self, 'Выбрать датасет', './dataset', 'CSV Files (*.csv)')
        if not filename:
            return
        try:
            self.df = pd.read_csv(filename)
            basename = os.path.basename(filename)
            self.btn_select_dataset.setText(f'Файл загружен: {basename}')
            self.selected_file_path = filename
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при чтении датасета:\\n{e}")

    def show_missing_values(self):
        """Отображает количество пропусков в каждой колонке."""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите датасет!")
            return
        
        missing_data = self.df.isnull().sum()
        result_text = "<html>"
        for column, count in missing_data.items():
            result_text += f'<p>{column}: {count}</p>'
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
        QMessageBox.information(self, "Готово", f"Датасет успешно очищен и сохранён в:\n{new_filename}\n\nКоличество удалённых строк: {rows_deleted}")

# Пример запуска окна
if __name__ == "__main__":
    app = QApplication(sys.argv)
    df = pd.DataFrame({'A': [None, 2], 'B': [3, None]})  # пример датасета с пропусками
    dialog = MissingValuesDialog(df=df, selected_file_path="example.csv")
    dialog.show()
    sys.exit(app.exec())