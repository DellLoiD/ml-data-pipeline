import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import QIcon
import ast
from selection_of_parameters.selection_of_parameters_logic import save_hyperparameters, get_hyperparameters


class HyperParameterOptimizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Основная структура окна
        main_layout = QVBoxLayout()
        self.setWindowTitle("Настройки гиперпараметров")
        # Получаем текущие значения из логики
        current_grid = get_hyperparameters()
        # Определение параметров и их пояснений
        hyperparameters = {
            'n_estimators': {
                'value': str(current_grid.get('n_estimators', [50, 100, 200, 300, 500])),
                'tooltip': 'Количество деревьев в случайном лесе.'
                          '\nБольше деревьев повышает точность, но замедляет процесс обучения.',
            },
            'max_depth': {
                'value': str(current_grid.get('max_depth', [None, 10, 20, 30, 40, 60])),
                'tooltip': 'Максимальная глубина дерева.'
                          '\nЗначение None означает неограниченную глубину.',
            },
            'min_samples_split': {
                'value': str(current_grid.get('min_samples_split', [2, 5, 10, 15])),
                'tooltip': 'Минимальное количество примеров, необходимое для разделения узла.'
                          '\nПовышение этого значения снижает вероятность переобучения.',
            },
            'min_samples_leaf': {
                'value': str(current_grid.get('min_samples_leaf', [1, 2, 4, 8])),
                'tooltip': 'Минимальное количество примеров в листовом узле.'
                          '\nНизкое значение помогает избегать чрезмерного дробления.',
            },
            'bootstrap': {
                'value': str(current_grid.get('bootstrap', [True, False])),
                'tooltip': 'Использование Bootstrapping при обучении каждого дерева.'
                          '\nTrue включает bootstrapping, False — использует весь датасет целиком.',
            },
            'criterion': {
                'value': str(current_grid.get('criterion', ['gini', 'entropy'])),
                'tooltip': 'Метод расчета критерия разделения узлов:'
                          '\n- gini: индекс Джини (для измерения неопределенности)'
                          '\n- entropy: энтропия (для минимизации нечистоты)',
            },
            'class_weight': {
                'value': str(current_grid.get('class_weight', ['balanced', None])),
                'tooltip': 'Весовые коэффициенты классов:'
                          '\n- balanced: классы уравновешены, веса вычисляются автоматически'
                          '\n- None: классам присваиваются равные веса.',
            },
            'max_features': {
                'value': str(current_grid.get('max_features', [None, 'sqrt', 'log2'])),
                'tooltip': 'Максимальное количество признаков для рассчета разделения узлов:'
                          '\n- None: используются все признаки'
                          '\n- sqrt: квадратный корень от общего количества признаков'
                          '\n- log2: логарифм от общего количества признаков',
            },
            'ccp_alpha': {
                'value': str(current_grid.get('ccp_alpha', [0.0, 0.01, 0.1])),
                'tooltip': 'Коэффициент регуляризации Cost Complexity Pruning (CCP Alpha).'
                          '\nЗначения ближе к нулю приводят к менее строгой обрезке дерева.',
            },
            'verbose': {
                'value': str(current_grid.get('verbose', [0])),
                'tooltip': 'Подробность логгирования процесса обучения:'
                          '\n- 0: ничего не выводить'
                          '\n- >0: уровень детальности вывода увеличивается пропорционально значению.',
            },
        }

        # Динамическая генерация элементов
        self.param_fields = {}
        for param_name, details in hyperparameters.items():
            # Группа элементов для каждого параметра
            group_layout = QHBoxLayout()

            # Имя параметра
            label = QLabel(param_name + ": ")
            group_layout.addWidget(label)

            # Поле ввода
            edit_field = QLineEdit(details['value'])
            group_layout.addWidget(edit_field)
            # Сохраняем ссылку на поле, чтобы позже прочитать значение
            self.param_fields[param_name] = edit_field

            # Кнопка справки
            help_button = QPushButton()
            help_button.setFixedSize(24, 24)
            help_button.setIcon(QIcon.fromTheme("dialog-question"))
            help_button.clicked.connect(lambda _, tip=details['tooltip']: self.show_help_message(tip))
            group_layout.addWidget(help_button)

            # Добавляем группу в основной макет
            main_layout.addLayout(group_layout)
        
        # Кнопка "Сохранить параметры"
        save_button = QPushButton('Сохранить параметры')
        save_button.clicked.connect(self.on_save_clicked)
        main_layout.addWidget(save_button)
        # Устанавливаем основной макет окна
        self.setLayout(main_layout)       


    def show_help_message(self, message):
        QMessageBox.information(self, "Справка", message)
        
    def update_random_grid(self):
        random_grid = {}
        for param_name, edit in self.param_fields.items():
            raw = edit.text()
            try:
                random_grid[param_name] = ast.literal_eval(raw)
            except Exception:
                random_grid[param_name] = raw
        return random_grid

    def on_save_clicked(self):
        random_grid = self.update_random_grid()
        save_hyperparameters(random_grid)
        QMessageBox.information(self, "Параметры сохранены", str(random_grid))

# Запуск приложения
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = HyperParameterOptimizerGUI()
    gui.show()
    sys.exit(app.exec())
