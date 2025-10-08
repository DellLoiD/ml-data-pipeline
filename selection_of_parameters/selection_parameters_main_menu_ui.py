import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
# Импортируем ваши оконные модули
from selection_of_parameters.selection_of_parameters_ui import HyperParameterOptimizerGUI
from selection_of_parameters.selection_parameters_random_search_ui import RandomSearchConfigGUI

class MainWindow_selection_parameters(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Настройка главного окна
        self.setWindowTitle("Главное меню")
        layout = QVBoxLayout()

        # Первая кнопка для открытия окна выбора параметров
        btn_select_params = QPushButton("Указать параметры для подбора")
        btn_select_params.clicked.connect(self.open_selection_of_parameters)
        layout.addWidget(btn_select_params)

        # Вторая кнопка для открытия окна настройки условий подбора
        btn_configure_search = QPushButton("Настроить условия подбора параметров")
        btn_configure_search.clicked.connect(self.open_selection_parameters_random_search)
        layout.addWidget(btn_configure_search)

        # Применяем макет
        self.setLayout(layout)

    def open_selection_of_parameters(self):
        # Открываем первое окно
        win = HyperParameterOptimizerGUI()
        win.show()

    def open_selection_parameters_random_search(self):
        # Открываем второе окно
        win = RandomSearchConfigGUI()
        win.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow_selection_parameters()
    main_win.show()
    sys.exit(app.exec())