import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import QIcon


class RandomSearchConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
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

            # Кнопка справки
            help_button = QPushButton()
            help_button.setFixedSize(24, 24)
            help_button.setIcon(QIcon.fromTheme("dialog-question"))
            help_button.clicked.connect(lambda _, tip=details['tooltip']: self.show_help_message(tip))
            group_layout.addWidget(help_button)

            # Добавляем группу в основной макет
            main_layout.addLayout(group_layout)

        # Устанавливаем основной макет окна
        self.setLayout(main_layout)

    def show_help_message(self, message):
        """
        Отображение окна с подсказкой по выбранному параметру.
        """
        QMessageBox.information(self, "Справка", message)


# Запуск приложения
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RandomSearchConfigGUI()
    gui.show()
    sys.exit(app.exec())