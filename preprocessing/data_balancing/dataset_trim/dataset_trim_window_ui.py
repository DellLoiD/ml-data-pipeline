from PySide6.QtWidgets import QDialog, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QFileDialog, QInputDialog
import pandas as pd
from preprocessing.data_balancing.dataset_trim.dataset_trim_window_logic import DatasetTrimLogic

class DatasetTrimWindow(QDialog):
    def __init__(self):  
        super().__init__()
        self.logic = DatasetTrimLogic()  
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
        self.resize(400, 400)
        self.show()

    def on_load_dataset_clicked(self):
        self.logic.load_dataset(self)
    
    def on_save_button_clicked(self):
        try:
            if not hasattr(self.logic, 'X_resampled'):
                QMessageBox.warning(self, "Ошибка", "Датасет сначала нужно обработать и обрезать.")
                return
            
            target_samples = int(self.trim_input.text())
            self.logic.save_trimmed_dataset(target_samples)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Некорректное значение количества выборок")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def on_trim_dataset_clicked(self):
        try:
            samples_count = int(self.trim_input.text())
            X_trimmed, y_trimmed = self.logic.trim_dataset(samples_count)
            after_stats = pd.Series(y_trimmed).value_counts().to_string()
            self.after_label.setText(f"После обрезки:\n{after_stats}") 
        except ValueError as e:
            QMessageBox.critical(self, 'Ошибка', str(e))       