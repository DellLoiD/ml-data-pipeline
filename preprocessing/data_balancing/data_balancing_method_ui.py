import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QApplication, QDialog)
from preprocessing.data_balancing.data_balancing_list_method_ui import BalancingMethodsWindow
from preprocessing.data_balancing.dataset_trim.dataset_trim_window_ui import DatasetTrimWindow
from preprocessing.data_balancing.data_balancing_operaiting_classes import FeatureSelector


class DataBalancingApp(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Балансировка датасета")
        layout = QVBoxLayout(self)
        # Кнопка для выбора метода балансировки
        balance_button = QPushButton("Выбрать метод балансировки")
        balance_button.clicked.connect(self._open_balancing_window) 
        layout.addWidget(balance_button)   
        # Кнопка для выбора метода балансировки
        trim_dataset_button = QPushButton("Обрезать датасет")
        trim_dataset_button.clicked.connect(self._open_trim_window) 
        layout.addWidget(trim_dataset_button)        
        # Кнопка для удаления колонок датасета
        operaiting_classes_button = QPushButton("Удалить колонку")
        operaiting_classes_button.clicked.connect(self._open_operaiting_classes) 
        layout.addWidget(operaiting_classes_button)        

    def _open_balancing_window(self):    
        balancing_window = BalancingMethodsWindow()    
        balancing_window.exec()
    
    def _open_trim_window(self):
        dataset_trim_window = DatasetTrimWindow()
        dataset_trim_window.exec() 
        
    def _open_operaiting_classes(self):
        feature_selector_window = FeatureSelector()
        feature_selector_window.exec() 
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataBalancingApp()
    window.show()
    sys.exit(app.exec())