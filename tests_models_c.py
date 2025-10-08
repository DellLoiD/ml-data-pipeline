import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QListWidget, QVBoxLayout, QMessageBox
from PySide6.QtCore import Qt
import pandas as pd

class DiabetesFeatureViewer(QWidget):
    def __init__(self):
        super().__init__()        
        # Загрузка CSV-файла
        self.df = pd.read_csv('dataset/diabetes_BRFSS2015-balanced-Diabetes_012-size50000.csv')        
        # Определим список признаков
        self.feature_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
            'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]        
        # Создаем интерфейс
        layout = QVBoxLayout()        
        # Виджет-список для выбора признака
        self.list_widget = QListWidget(self)
        for feature in self.feature_cols:
            self.list_widget.addItem(feature)            
        # Кнопка для отображения распределения классов
        button_show_distribution = QPushButton("Показать распределение классов", self)
        button_show_distribution.clicked.connect(self.show_class_distribution)        
        # Добавляем элементы в компоновщик
        layout.addWidget(QLabel("Выбор признака:"))
        layout.addWidget(self.list_widget)
        layout.addWidget(button_show_distribution)
        
        self.setLayout(layout)
        self.setWindowTitle("Анализ диабета BRFSS 2015")
        self.resize(400, 300)
    
    def show_class_distribution(self):
        selected_feature = self.list_widget.currentItem().text() if self.list_widget.currentItem() else None
        if not selected_feature:
            QMessageBox.warning(self, "Ошибка", "Выберите признак!")
            return
        
        class_counts = self.df[selected_feature].value_counts()
        message = f'Распределение классов признака "{selected_feature}"\n\n'

        # Преобразование списка значений в список пар ключ-значение
        values_list = list(class_counts.items())
        half = len(values_list) // 2

        # Создание двух колонок
        left_column = values_list[:half]
        right_column = values_list[half:]

        # Объединение элементов обеих колонок
        for i in range(max(len(left_column), len(right_column))):
            # Проверка наличия элемента в левой колонке
            if i < len(left_column):
                left_value = left_column[i]
            else:
                left_value = ('', '')  # Пустые значения для заполнения второй колонки
                
            # Проверка наличия элемента в правой колонке
            if i < len(right_column):
                right_value = right_column[i]
            else:
                right_value = ('', '')  # Пустые значения для заполнения второй колонки
                
            # Добавление строки с двумя значениями
            message += f"{left_value[0]:<15}{left_value[1]:>5}\t\t{right_value[0]:<15}{right_value[1]:>5}\n"

        QMessageBox.information(self, "Результат", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiabetesFeatureViewer()
    window.show()
    sys.exit(app.exec())
