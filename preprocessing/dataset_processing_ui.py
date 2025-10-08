from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox
from preprocessing.dataset_processing_logic import select_raw_dataset, show_missing_values_dialog

class DatasetProcessingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)     
        self.resize(400, 300)          
        self.setWindowTitle("Обработка датасета")        
        self.btn_select_dataset = QPushButton("Выбор исходного датасета")
        self.btn_select_dataset.clicked.connect(lambda: select_raw_dataset(self))
        btn_check_missing_values = QPushButton("Проверка на пропуски")
        btn_check_missing_values.clicked.connect(self.on_check_missing_values_clicked)        
        btn_finish_processing = QPushButton("Закончить обработку датасета")
        # Подключение обработчика закрытия окна
        btn_finish_processing.clicked.connect(self.close)                
        layout = QVBoxLayout()
        layout.addWidget(self.btn_select_dataset)
        layout.addWidget(btn_check_missing_values)        
        layout.addWidget(btn_finish_processing)        
        self.setLayout(layout)        

    def on_check_missing_values_clicked(self):
        show_missing_values_dialog(self.df, self.selected_file_path)

if __name__ == "__main__":
    app = QApplication([])
    
    window = DatasetProcessingWindow()
    window.show()
    
    app.exec()