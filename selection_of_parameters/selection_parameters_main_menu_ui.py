import sys
import os
import pandas as pd
from PySide6.QtCore import *
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, 
                               QFileDialog, QMessageBox, QDialog, QLabel, QInputDialog, QScrollArea, QHBoxLayout)
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
from .selection_of_parameters_logic import (get_random_grid, get_random_search_params)
from .selection_parameters_parameter_tuning_window import ParameterTuningWindow

class MainWindow_selection_parameters(QWidget):
    def __init__(self):
        super().__init__()
        self.parameter_window = None
        self.selected_dataset_path = None
        self.target_variable = None
        self.selected_model = ""
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
        # Выбираем модель
        label_model_choice = QLabel("Выбор модели:", font=QFont('Arial', 12))
        layout.addWidget(label_model_choice)
        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(["RandomForest", "GradientBoosting", "LogisticRegression"])
        layout.addWidget(self.model_combo_box)        
        # Подключение сигнала для отслеживания изменений в ComboBox
        self.model_combo_box.currentTextChanged.connect(self.on_model_change)        
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

    @Slot(str)
    def on_model_change(self, new_value):
        """
        Сохраняет выбранную модель при смене в ComboBox
        """
        self.selected_model = new_value
        print(f"Выбранная модель: {self.selected_model}")

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
    def format_parameter_value(self, value):
        if isinstance(value, (list, tuple)):
            # Делаем красивый многострочный вид или просто читаемый
            items = [str(x) for x in value]
            return "[" + ", ".join(items) + "]"
        elif hasattr(value, 'rvs'):  # loguniform и т.п.
            return f"scipy.stats.{type(value).__name__} (distribution)"
        elif isinstance(value, range):
            return f"range(start={value.start}, stop={value.stop}, step={value.step})"
        elif isinstance(value, str):
            return f'"{value}"'
        elif value is None:
            return "None"
        elif isinstance(value, bool):
            return "True" if value else "False"
        else:
            return str(value)
            
    def show_current_parameters(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Текущие настроечные параметры")
        dialog.setModal(True)
        dialog.resize(850, 700)

        # Главной будет QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Внутренний виджет растягивается
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Виджет внутри QScrollArea
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        try:
            hyperparams = get_random_grid()
            search_params = get_random_search_params()

            # === Параметры случайной сетки ===
            title1 = QLabel("=== Параметры случайной сетки (random_grid) ===")
            title1.setStyleSheet("font-weight: bold; font-size: 14px;")
            layout.addWidget(title1)

            # Собираем весь текст как одну HTML-строку
            grid_text = ""
            for model_name, model_params in hyperparams.items():
                grid_text += f"<b>{model_name}:</b><br>"
                if isinstance(model_params, dict):
                    for param_key, param_value in model_params.items():
                        formatted_value = self.format_parameter_value(param_value)
                        grid_text += f"&nbsp;&nbsp;&nbsp;• <b>{param_key}:</b> {formatted_value}<br>"
                else:
                    formatted_value = self.format_parameter_value(model_params)
                    grid_text += f"&nbsp;&nbsp;&nbsp;{formatted_value}<br>"
                grid_text += "<br>"

            label1 = QLabel(grid_text)
            label1.setTextFormat(Qt.RichText)
            label1.setWordWrap(True)
            label1.setStyleSheet("font-family: 'Courier New'; font-size: 11px; padding: 8px; background-color: #f9f9f9;")
            layout.addWidget(label1)

            # === Параметры RandomizedSearchCV ===
            title2 = QLabel("=== Параметры RandomizedSearchCV ===")
            title2.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
            layout.addWidget(title2)

            search_text = ""
            for key, value in search_params.items():
                formatted_value = self.format_parameter_value(value)
                search_text += f"<b>{key}:</b> {formatted_value}<br>"

            label2 = QLabel(search_text)
            label2.setTextFormat(Qt.RichText)
            label2.setWordWrap(True)
            label2.setStyleSheet("font-family: 'Courier New'; font-size: 11px; padding: 8px; background-color: #f9f9f9;")
            layout.addWidget(label2)

        except Exception as e:
            error_label = QLabel(f"Ошибка загрузки параметров:\n{str(e)}")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            error_label.setWordWrap(True)
            layout.addWidget(error_label)

        layout.addStretch()

        # Добавляем кнопку "Закрыть" НА ДИАЛОГ, а не во внутренний layout!
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Закрыть")
        close_btn.setStyleSheet("font-size: 12px; padding: 8px;")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)

        # Устанавливаем layout в content_widget
        content_widget.setLayout(layout)

        # Устанавливаем content_widget в scroll_area
        scroll_area.setWidget(content_widget)

        # Главный layout диалога
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(button_layout)  # кнопка снизу, вне прокрутки

        dialog.setLayout(main_layout)
        dialog.exec()
        
    def tune_best_parameters(self):
        """
        Метод, вызываемый при нажатии кнопки 'Подобрать лучшие параметры'.
        Проверяет наличие выбранного датасета и открывает окно подбора параметров.
        """
        if not hasattr(self, 'selected_dataset_path') or not self.selected_dataset_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите датасет!")
            return
        
        # Получение выбранной модели из комбобокса
        selected_model = self.model_combo_box.currentText()
        
        # Создание экземпляра окна подбора параметров
        self.parameter_window = ParameterTuningWindow(
            parent=None,
            dataset_path=self.selected_dataset_path,
            target_variable=self.target_variable,
            chosen_model=selected_model  # передача выбранной модели
        )
        
        # Конфигурация окна и старт подбора параметров
        self.parameter_window.setGeometry(100, 100, 700, 600)
        self.parameter_window.setWindowModality(Qt.NonModal)
        self.parameter_window.show()
        self.parameter_window.start_tuning()
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