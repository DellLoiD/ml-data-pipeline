from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox

from preprocessing.dataset_processing_fix_non_numeric_ui import OneHotEncodingWindow
from preprocessing.dataset_processing_check_nan import MissingValuesDialog

One_Hot_Encoding_Window_instance = None
Missing_Values_Dialog_instance = None
class DatasetProcessingWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Настройка размеров окна
        self.setMinimumSize(400, 300)     
        self.resize(400, 300)          
        self.setWindowTitle("Обработка датасета") 
        # Кнопка проверки на пропуски
        self.btn_check_missing_values = QPushButton("Проверка на пропуски")
        self.btn_check_missing_values.clicked.connect(self.on_check_missing_values_clicked)
        # Другие кнопки оставляем без изменений
        self.btn_check_data_values = QPushButton("Проверка значений")
        self.btn_check_data_values.clicked.connect(self.on_check_data_values_clicked)
        self.btn_finish_processing = QPushButton("Закончить обработку датасета")
        self.btn_finish_processing.clicked.connect(self.close) 
        # Создаем вертикальное расположение элементов интерфейса
        layout = QVBoxLayout()       
        layout.addWidget(self.btn_check_missing_values)
        layout.addWidget(self.btn_check_data_values)
        layout.addWidget(self.btn_finish_processing)        
        self.setLayout(layout)   
    
    # Метод обработки нажатия на кнопку проверки пропусков
    def on_check_missing_values_clicked(self):
        global Missing_Values_Dialog_instance
        if not Missing_Values_Dialog_instance or not Missing_Values_Dialog_instance.isVisible():
            Missing_Values_Dialog_instance = MissingValuesDialog()
            Missing_Values_Dialog_instance.show()

    # Метод обработки нажатия на кнопку проверки значений остается прежним
    def on_check_data_values_clicked(self):
        global One_Hot_Encoding_Window_instance
        if not One_Hot_Encoding_Window_instance or not One_Hot_Encoding_Window_instance.isVisible():
            One_Hot_Encoding_Window_instance = OneHotEncodingWindow()
            One_Hot_Encoding_Window_instance.show()

if __name__ == "__main__":
    app = QApplication([])
    
    window = DatasetProcessingWindow()
    window.show()
    
    app.exec()