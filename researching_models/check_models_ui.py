import sys
from .check_models_logic import DataModelHandler #select_dataset, evaluate_models, show_feature_importance
import pandas as pd
import os
from PySide6.QtWidgets import *
from PySide6.QtGui import QPixmap

class ClassificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_file_name = ""
        self.init_ui()
    
        # Инициализация обработчика данных после UI
        self.data_handler = DataModelHandler(parent=self, df=None, 
                                             combobox=self.target_var_combobox, 
                                             checkboxes=self.checkboxes, 
                                             labels_and_lines=self.labels_and_lines, 
                                             accuracy_label=self.accuracy_label, 
                                             time_label=self.time_label)

    def init_ui(self):
        main_layout = QVBoxLayout()
        title_label = QLabel('Выбор модели и обучение')
        main_layout.addWidget(title_label)

        # 1. Выбор датасета
        self.select_dataset_btn = QPushButton("Выбрать датасет")
        self.select_dataset_btn.clicked.connect(self.on_select_dataset_clicked)
        main_layout.addWidget(self.select_dataset_btn)

        # 2. Выбор целевой переменной
        self.target_var_combobox = QComboBox()
        self.target_var_combobox.setEnabled(False)
        main_layout.addWidget(self.target_var_combobox)

        # 3. Модель классификации
        model_group_box = QGroupBox("Модель классификации")
        model_vlayout = QVBoxLayout()
        self.checkboxes = []
        self.labels_and_lines = {}
        models_params = {
            'Random Forest': ['Количество деревьев', 'Test Size', 'Random State'],
            'Gradient Boosting': ['Количество деревьев', 'Test Size', 'Random State'],
            'Logistic Regression': ['C', 'Max Iterations', 'Penalty']
        }
        for model_name, params_list in models_params.items():
            hbox = QHBoxLayout()
            cb = QCheckBox(model_name)
            cb.setChecked(True if model_name == "Random Forest" else False)
            self.checkboxes.append(cb)
            hbox.addWidget(cb)
            
            # Добавляем текстовые поля для каждого параметра конкретной модели
            lines = {}
            for i, param_name in enumerate(params_list):
                lbl = QLabel(param_name)
                le = QLineEdit()
                default_value = ''  # Значения по умолчанию
                if param_name == 'Количество деревьев':
                    default_value = '100'
                elif param_name == 'C':
                    default_value = '0.01'
                elif param_name == 'Max Iterations':
                    default_value = '100'
                elif param_name == 'Penalty':
                    default_value = 'l2'
                elif param_name == 'Test Size':
                    default_value = '0.2'  # Установили стандартное значение 0.2
                elif param_name == 'Random State':
                    default_value = '42'   # Установили стандартное значение 42
                le.setText(default_value)
                hbox.addWidget(lbl)
                hbox.addWidget(le)
                lines[param_name] = le
            
            self.labels_and_lines[model_name] = lines
            model_vlayout.addLayout(hbox)
        
        model_group_box.setLayout(model_vlayout)
        main_layout.addWidget(model_group_box)

        # 4. Оценка моделей
        evaluate_models_btn = QPushButton('Оценить модели')
        evaluate_models_btn.clicked.connect(self.on_evaluate_models_clicked)
        main_layout.addWidget(evaluate_models_btn)

        # 5. Результаты оценки
        self.accuracy_label = QLabel('')
        self.time_label = QLabel('')
        main_layout.addWidget(self.accuracy_label)
        main_layout.addWidget(self.time_label)

        # 6. Показать важность признаков
        # Чекбоксы для выбора моделей
        self.rf_checkbox = QCheckBox('Random Forest')
        self.gb_checkbox = QCheckBox('Gradient Boosting')
        self.lr_checkbox = QCheckBox('Logistic Regression')

        # Кнопка показа важности признаков
        show_importance_btn = QPushButton("Показать важность признаков")
        show_importance_btn.clicked.connect(self.on_show_feature_importance)
        # Добавляем чекбоксы и кнопку в макет
        models_group_box = QGroupBox("Выберите модель для анализа важности признаков")
        models_layout = QVBoxLayout()
        # Создаем горизонтальный макет для чекбоксов
        checkboxes_layout = QHBoxLayout()
        checkboxes_layout.addWidget(self.rf_checkbox)
        checkboxes_layout.addWidget(self.gb_checkbox)
        checkboxes_layout.addWidget(self.lr_checkbox)
        
        models_layout.addLayout(checkboxes_layout)
        models_layout.addWidget(show_importance_btn)

        models_group_box.setLayout(models_layout)
        main_layout.addWidget(models_group_box)

        self.setLayout(main_layout)
        self.resize(800, 600)
        self.show()
        
    def on_select_dataset_clicked(self):
        """Обработчик нажатия кнопки 'Выбрать датасет'."""
        file_path, _ = QFileDialog.getOpenFileName(None, "Выберите файл CSV", "./dataset/", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.dataset_file_name = os.path.basename(file_path)
                self.select_dataset_btn.setText(self.dataset_file_name)                
                # Передача фрейма данных обработчику
                self.data_handler.update_dataframe(df)
            except Exception as e:
                QMessageBox.critical(
                    None,
                    "Ошибка",
                    f"Невозможно загрузить файл:\n{e}",
                    QMessageBox.StandardButton.Ok
                )
        else:
            print("Файл не выбран.")
        
    def on_evaluate_models_clicked(self):
        self.data_handler.evaluate_models()
        
    def on_show_feature_importance(self):
        selected_models = {}
        
        if self.rf_checkbox.isChecked():
            selected_models['Random Forest'] = True
        if self.gb_checkbox.isChecked():
            selected_models['Gradient Boosting'] = True
        if self.lr_checkbox.isChecked():
            selected_models['Logistic Regression'] = True
        
        if len(selected_models) > 0:
            self.data_handler.calculate_feature_importances(selected_models)
        else:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну модель!")    

# Главное окно приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ClassificationApp()
    sys.exit(app.exec())