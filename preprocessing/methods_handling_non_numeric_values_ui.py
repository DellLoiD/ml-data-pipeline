from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QMessageBox
from preprocessing.dataset_processing_logic  import (
    process_one_hot_encoding,
    process_label_encoding,
    process_target_encoding,
    process_frequency_encoding,
    process_binary_encoding
)

class DataEncodingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработка данных")
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Список методов и соответствующих обработчиков
        methods = [
            ("One-Hot Encoding", process_one_hot_encoding),
            ("Label Encoding", process_label_encoding),
            ("Target Encoding", process_target_encoding),
            ("Frequency Encoding", process_frequency_encoding),
            ("Binary Encoding", process_binary_encoding)
        ]
        
        # Добавление кнопок для каждого метода
        for label, func in methods:
            row_layout = QVBoxLayout()
            # Главная кнопка запуска метода
            main_button = QPushButton(label)
            main_button.clicked.connect(lambda checked=False, f=func: self.run_function(f))
            row_layout.addWidget(main_button)
            
            # Кнопка помощи
            help_button = QPushButton("?")
            help_button.setMaximumWidth(30)
            help_button.clicked.connect(lambda checked=False, lbl=label: self.show_help(lbl))
            row_layout.addWidget(help_button)
            
            layout.addLayout(row_layout)
    
    def run_function(self, function):
        """
        Запуск выбранного метода обработки данных.
        """
        try:
            result = function()
            QMessageBox.information(self, "Результат", f"Метод '{function.__name__}' успешно выполнен!")
        except Exception as err:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {err}")
    
    def show_help(self, label):
        """
        Показывает окно справки для конкретного метода.
        """
        helps = {
            "One-Hot Encoding": "Преобразование категорий в бинарные признаки (каждая категория становится отдельной бинарной колонкой)",
            "Label Encoding": "Присваивает категориям уникальные целые числа (A→0, B→1...)",
            "Target Encoding": "Категория замещается средней величиной целевой переменной, связанной с ней",
            "Frequency Encoding": "Категория заменяется частотой своего появления в наборе данных",
            "Binary Encoding": "Представляет категорию в виде последовательности бит"
        }
        QMessageBox.information(self, "Справка", helps.get(label, "Нет доступной справки."))

if __name__ == '__main__':
    app = QApplication([])
    window = DataEncodingWindow()
    window.resize(400, 300)
    window.show()
    app.exec()