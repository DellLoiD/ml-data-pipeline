import sys
import os
import pandas as pd
from PySide6.QtCore import *
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                               QFileDialog, QMessageBox, QDialog, QLabel, QInputDialog)
import logging
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parameter_tuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Импортируем ваши оконные модули
from selection_of_parameters.selection_of_parameters_ui import HyperParameterOptimizerGUI
from selection_of_parameters.selection_parameters_random_search_ui import RandomSearchConfigGUI
from .selection_of_parameters_logic import (get_hyperparameters, get_random_search_params)
from .selection_parameters_parameter_tuning_window import ParameterTuningWindow

class MainWindow_selection_parameters(QWidget):
    def __init__(self):
        super().__init__()
        self.parameter_window = None
        self.selected_dataset_path = None
        self.target_variable = None
        logger.info("Инициализирован MainWindow_selection_parameters")
        self.initUI()

    def initUI(self):
        # Настройка главного окна
        self.setWindowTitle("Главное меню")
        layout = QVBoxLayout()        
        # Кнопка для выбора датасета сверху
        self.btn_choose_dataset = QPushButton("Выбрать датасет")  
        self.btn_choose_dataset.clicked.connect(self.choose_dataset) 
        layout.addWidget(self.btn_choose_dataset) 
        # Кнопка для отображения текущих параметров модели
        btn_show_params = QPushButton("Показать текущие параметры")
        btn_show_params.clicked.connect(self.show_current_parameters)
        layout.addWidget(btn_show_params)
        # Кнопка для задания параметров подбора моделей
        btn_select_params = QPushButton("Указать параметры для подбора")
        btn_select_params.clicked.connect(self.open_selection_of_parameters)
        layout.addWidget(btn_select_params)
        # Кнопка для настройки условий подбора параметров
        btn_configure_search = QPushButton("Настроить условия подбора параметров")
        btn_configure_search.clicked.connect(self.open_selection_parameters_random_search)
        layout.addWidget(btn_configure_search)
        # Кнопка для запуска процесса подбора лучших параметров
        btn_tune_params = QPushButton("Подобрать лучшие параметры")
        btn_tune_params.clicked.connect(self.tune_best_parameters)
        logger.info("Создана и подключена кнопка 'Подобрать лучшие параметры'")
        layout.addWidget(btn_tune_params)
        # Устанавливаем созданный макет в главное окно
        self.setLayout(layout)

    def open_selection_of_parameters(self):
        # Открываем первое окно
        win = HyperParameterOptimizerGUI()
        win.show()

    def open_selection_parameters_random_search(self):
        # Открываем второе окно
        win = RandomSearchConfigGUI()
        win.show()
#------------------------------------------------------
    def choose_dataset(self):
        # Папка с датасетами
        dataset_folder = "dataset"
        if not os.path.exists(dataset_folder):
            QMessageBox.warning(self, "Предупреждение", f"Папка с датасетом '{dataset_folder}' не найдена!")
            return          
        # Выбор файла датасета
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите датасет",
            dataset_folder,
            "Файлы CSV (*.csv);;Все файлы (*)"
        )        
        if file_path:
            try:
                # Чтение датасета
                df = pd.read_csv(file_path)                
                # Список столбцов датасета
                column_names = df.columns.tolist()                
                # Диалог выбора целевой переменной
                chosen_column, ok_pressed = QInputDialog.getItem(
                    self,
                    "Выбор целевой переменной",
                    "Выберите целевую переменную:",
                    column_names,
                    current=0, editable=False
                )
                
                if ok_pressed:
                    self.target_variable = chosen_column
                    QMessageBox.information(self, "Успех", f"Ваш выбор целевой переменной: {chosen_column}")                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка чтения файла: {str(e)}")
                return            
            # Сохраняем путь к файлу датасета
            self.selected_dataset_path = file_path
            filename = os.path.basename(file_path)            
            # Обновляем надпись на кнопке выбором имени выбранного датасета
            self.btn_choose_dataset.setText(filename)
            QMessageBox.information(self, "Успех", f"Выполнен выбор датасета: {filename}")

    def show_current_parameters(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Текущие настроечные параметры")
        dialog.setModal(True)
        dialog.setMinimumSize(600, 500)        
        layout = QVBoxLayout()
        try:
            hyperparams = get_hyperparameters()
            search_params = get_random_search_params()            
            # Добавление заголовка для случайных сеточных параметров
            title1 = QLabel("=== Параметры случайной сетки ===")
            title1.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
            layout.addWidget(title1)            
            # Отображение каждого параметра отдельно с помощью QLabel
            for key, value in hyperparams.items():
                param_label = QLabel(f"{key}: {value}")
                param_label.setWordWrap(True)
                param_label.setStyleSheet("margin-left: 20px; margin-bottom: 5px;")
                layout.addWidget(param_label)            
            # Пространство и заголовок для параметров RandomizedSearchCV
            spacer1 = QLabel("")
            spacer1.setMinimumHeight(10)
            layout.addWidget(spacer1)            
            title2 = QLabel("=== Параметры RandomizedSearchCV ===")
            title2.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
            layout.addWidget(title2)            
            # Отображение каждого параметра поиска отдельно с помощью QLabel
            for key, value in search_params.items():
                param_label = QLabel(f"{key}: {value}")
                param_label.setWordWrap(True)
                param_label.setStyleSheet("margin-left: 20px; margin-bottom: 5px;")
                layout.addWidget(param_label)        
        except Exception as e:
            error_label = QLabel(f"Ошибка загрузки параметров: {str(e)}")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(error_label)        
        # Добавляем растяжку, чтобы кнопка закрытия была внизу
        layout.addStretch()        
        # Кнопка Закрыть
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("font-size: 12px; padding: 8px;")
        layout.addWidget(close_btn)        
        dialog.setLayout(layout)
        dialog.exec()
        
    def tune_best_parameters(self):
        logger.info("Метод tune_best_parameters вызван")        
        # Проверяем наличие выбранного датасета
        if not self.selected_dataset_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите датасет!")
            return
        # Создаем абсолютно новое окно настроек
        self.parameter_window = ParameterTuningWindow(
            parent=None,
            dataset_path=self.selected_dataset_path,
            target_variable=self.target_variable
        )
        self.parameter_window.setGeometry(100, 100, 700, 600)
        self.parameter_window.setWindowModality(Qt.NonModal)
        self.parameter_window.show()        
        self.parameter_window.start_tuning()
        # Привлекаем внимание пользователя к новому окну
        print("Видимо:", self.parameter_window.isVisible())
        print("Активно:", self.parameter_window.isActiveWindow())
        QApplication.processEvents()
        
if __name__ == '__main__':
    logger.info("Starting application...")
    app = QApplication(sys.argv)
    logger.info("QApplication created")
    main_win = MainWindow_selection_parameters()
    logger.info("MainWindow created")
    main_win.show()
    logger.info("MainWindow shown")
    logger.info("Starting application event loop...")
    sys.exit(app.exec())