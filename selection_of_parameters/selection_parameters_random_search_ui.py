import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import QIcon
from .selection_of_parameters_logic import get_hyperparameters, save_hyperparameters, get_random_search_params, save_random_search_params

class RandomSearchConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.text_fields = {}
        self.initUI()

    def initUI(self):
        # Макет главного окна
        main_layout = QVBoxLayout()
        self.setWindowTitle("Конфигурация RandomizedSearchCV")
        # Конфигурационные параметры и их описания
        config_params = {
            'estimator': {
                'value': 'RandomForestClassifier(random_state=42)',
                'tooltip': 'Базовый классификатор, используемый для подбора параметров.'
                           '\nЗдесь указан RandomForestClassifier с фиксированным seed для воспроизводимости результатов.',
            },
            'param_distributions': {
                'value': 'random_grid',
                'tooltip': 'Диапазоны возможных значений гиперпараметров.'
                           '\nЭто сетка, определяющая возможные комбинации гиперпараметров для перебора.',
            },
            'n_iter': {
                'value': '100',
                'tooltip': 'Количество итераций поиска лучших параметров.'
                           '\nБольшее значение обеспечивает лучшее покрытие пространства гиперпараметров, но требует больше вычислительных ресурсов.',
            },
            'cv': {
                'value': '6',
                'tooltip': 'Количество фолдов для кросс-валидации.'
                           '\nПозволяет оценить стабильность модели и уменьшить смещение оценки производительности.',
            },
            'scoring': {
                'value': "['accuracy', 'f1_macro']",
                'tooltip': 'Метрики, используемые для оценки моделей.'
                           '\nНесколько показателей позволяют выбрать лучшую комбинацию параметров на основе компромисса между точностью и полнотой.',
            },
            'refit': {
                'value': "'accuracy'",
                'tooltip': 'Выбор лучшей модели производится по указанной метрике.'
                           '\nAccuracy наиболее распространенная мера точности классификации.',
            },
            'random_state': {
                'value': '42',
                'tooltip': 'Фиксированное начальное состояние генератора случайных чисел.'
                           '\nУстанавливает воспроизведение эксперимента при разных запусках.',
            },
            'verbose': {
                'value': '2',
                'tooltip': 'Уровень детализации печати прогресса.'
                           '\nЧем выше значение, тем подробнее информация о ходе процесса подбора параметров.',
            },
            'n_jobs': {
                'value': '-1',
                'tooltip': 'Количество ядер CPU, используемых для параллельных вычислений.'
                           '\n-1 означает задействование всех доступных ядер.',
            },
        }

        # Генерируем элементы графического интерфейса
        for param_name, details in config_params.items():
            # Группа элементов для текущего параметра
            group_layout = QHBoxLayout()
            # Название параметра
            label = QLabel(param_name + ": ")
            group_layout.addWidget(label)
            # Поле ввода
            edit_field = QLineEdit(details['value'])
            group_layout.addWidget(edit_field)            
            # Store text field reference
            self.text_fields[param_name] = edit_field
            # Кнопка справки
            help_button = QPushButton()
            help_button.setFixedSize(24, 24)
            help_button.setIcon(QIcon.fromTheme("dialog-question"))
            help_button.clicked.connect(lambda _, tip=details['tooltip']: self.show_help_message(tip))
            group_layout.addWidget(help_button)
            # Добавляем группу в основной макет
            main_layout.addLayout(group_layout)
        # Add Save Parameters button
        save_button = QPushButton('Save Parameters')
        save_button.clicked.connect(self.save_parameters)
        main_layout.addWidget(save_button)
        # Устанавливаем основной макет окна
        self.setLayout(main_layout)        
        # Load current parameters into fields
        self.load_current_parameters()

    def show_help_message(self, message):
        """Отображение окна с подсказкой по выбранному параметру."""
        QMessageBox.information(self, "Справка", message)
    
    def load_current_parameters(self):
        """
        Load current parameters from the logic file into the text fields.
        """
        try:
            # Get current hyperparameters from the logic file
            current_hyperparams = get_hyperparameters()
            current_search_params = get_random_search_params()
            
            # Update param_distributions field with current random_grid
            if 'param_distributions' in self.text_fields:
                self.text_fields['param_distributions'].setText(str(current_hyperparams))
            
            # Update other RandomizedSearchCV parameters
            for param_name, text_field in self.text_fields.items():
                if param_name in current_search_params and param_name != 'param_distributions':
                    text_field.setText(str(current_search_params[param_name]))
                
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load current parameters: {str(e)}")
    
    def save_parameters(self):
        """
        Save the parameters from text fields to the logic file.
        """
        try:
            # Get values from text fields
            param_values = {}
            for param_name, text_field in self.text_fields.items():
                param_values[param_name] = text_field.text()
            
            # Special handling for param_distributions - it should be a dict
            if 'param_distributions' in param_values:
                try:
                    # Try to evaluate the string as a dictionary
                    param_dict = eval(param_values['param_distributions'])
                    if isinstance(param_dict, dict):
                        save_hyperparameters(param_dict)
                    else:
                        QMessageBox.warning(self, "Error", "param_distributions must be a valid dictionary")
                        return
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Invalid dictionary format for param_distributions: {str(e)}")
                    return
            
            # Save RandomizedSearchCV parameters (excluding param_distributions)
            search_params = {}
            for param_name, value in param_values.items():
                if param_name != 'param_distributions':
                    # Try to convert string values to appropriate types
                    try:
                        # Handle numeric values
                        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                            search_params[param_name] = int(value)
                        elif value.replace('.', '').replace('-', '').isdigit():
                            search_params[param_name] = float(value)
                        # Handle list-like strings
                        elif value.startswith('[') and value.endswith(']'):
                            search_params[param_name] = eval(value)
                        # Handle quoted strings
                        elif value.startswith("'") and value.endswith("'"):
                            search_params[param_name] = eval(value)
                        else:
                            search_params[param_name] = value
                    except:
                        search_params[param_name] = value
            
            save_random_search_params(search_params)
            
            # Store parameters for later use
            self.saved_params = param_values
            
            QMessageBox.information(self, "Success", "Parameters saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {str(e)}")
    
    def get_saved_parameters(self):
        """
        Return the saved parameters for use by the logic file.
        """
        return getattr(self, 'saved_params', {})


# Запуск приложения
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RandomSearchConfigGUI()
    gui.show()
    sys.exit(app.exec())