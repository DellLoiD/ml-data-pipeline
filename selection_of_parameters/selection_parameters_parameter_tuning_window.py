from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from pprint import pformat
import logging
import os
from .selection_of_parameters_logic import random_grid, random_search_params
from .selection_parameters_parameter_tuning_worker import ParameterTuningWorker
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from PySide6.QtCore import QObject, Signal

class LogStream(QObject):
    message_written = Signal(str)

    def write(self, text):
        if not self.signalsBlocked():
            self.message_written.emit(text)

class ParameterTuningWindow(QWidget):
    def __init__(self, parent=None, dataset_path=None, target_variable=None):
        """Инициализация окна выбора параметров"""
        logger.info(f"ПараметрTuningWindow.__init__ вызван с dataset_path: {dataset_path}")
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.best_model = None
        self.best_params = None
        self.accuracy = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Выбор параметров")
        self.setGeometry(300, 300, 700, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        main_layout = QVBoxLayout()
        title = QLabel("Текущие параметры подбора")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)
        # Горизонтальная компоновка для размещения двух блоков рядом
        horizontal_layout = QHBoxLayout()

        # Блок справа: Гиперпараметры модели (будет теперь слева)
        self.hyperparameters_label = QLabel("Гиперпараметры модели:")
        hyperparameters_text = QLabel()
        hyperparameters_text.setWordWrap(True)
        hyperparameters_text.setText(pformat(random_grid))

        # Добавляем элементы блока справа в горизонтальную компоновку
        vertical_right_layout = QVBoxLayout()  # Внутренняя вертикальная компоновка для нового левого блока
        vertical_right_layout.addWidget(self.hyperparameters_label)
        vertical_right_layout.addWidget(hyperparameters_text)
        horizontal_layout.addLayout(vertical_right_layout)  # Сначала добавляем правую секцию

        # Блок слева: Параметры случайного поиска (переносим вправо)
        self.random_search_params_label = QLabel("Параметры случайного поиска:")
        random_search_params_text = QLabel()
        random_search_params_text.setWordWrap(True)
        random_search_params_text.setText(pformat(random_search_params))

        # Добавляем элементы блока слева в горизонтальную компоновку
        vertical_left_layout = QVBoxLayout()  # Внутренняя вертикальная компоновка для нового правого блока
        vertical_left_layout.addWidget(self.random_search_params_label)
        vertical_left_layout.addWidget(random_search_params_text)
        horizontal_layout.addLayout(vertical_left_layout)  # Затем добавляем левую секцию

        # Добавляем горизонтальную компоновку в основную вертикальную компоновку
        main_layout.addLayout(horizontal_layout)

        # Горизонтальная линия-разделитель
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator_line)

        # Заголовок результата оптимизации
        self.results_title = QLabel("Результаты оптимизации")
        self.results_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        self.results_title.setVisible(False)
        main_layout.addWidget(self.results_title)

        # Сообщение о завершении настройки
        self.completion_label = QLabel("")
        self.completion_label.setStyleSheet("font-size: 14px; font-weight: bold; color: green; margin: 10px 0;")
        self.completion_label.setVisible(False)
        main_layout.addWidget(self.completion_label)

        # Точность
        self.accuracy_label = QLabel("")
        self.accuracy_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E8B57; margin: 10px 0;")
        self.accuracy_label.setVisible(False)
        main_layout.addWidget(self.accuracy_label)

        # Параметры
        self.params_title = QLabel("Лучшие параметры:")
        self.params_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        self.params_title.setVisible(False)
        main_layout.addWidget(self.params_title)

        # Контейнер для отображения параметров
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_container.setLayout(self.params_layout)
        self.params_container.setVisible(False)
        main_layout.addWidget(self.params_container)

        main_layout.addStretch()

        # Кнопка для сохранения лучшей модели
        self.save_button = QPushButton("Сохранить лучшую модель")
        self.save_button.clicked.connect(self.save_best_model)
        self.save_button.setVisible(False)
        self.save_button.setStyleSheet(
            "font-size: 12px; padding: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;"
        )
        main_layout.addWidget(self.save_button)

        # Основной макет окна
        self.setLayout(main_layout)
        self.setVisible(True)
    
    def start_tuning(self):
        #self.worker = ParameterTuningWorker(self.dataset_path, self)
        self.worker = ParameterTuningWorker(dataset_path=self.dataset_path, target_variable=self.target_variable,parent=self)
        self.worker.tuning_completed.connect(self.on_tuning_completed)
        self.worker.start()
    
    def on_tuning_completed(self, best_model, best_params, accuracy):
        #Обработчик завершения процедуры подбора параметров
        self.best_model = best_model
        self.best_params = best_params
        self.accuracy = accuracy        
        self.completion_label.setText("Настройка параметров завершена!")  
        # Показываем секцию результатов
        self.results_title.setVisible(True)        
        # Отображаем точность лучшей модели
        self.accuracy_label.setText(f"Точность наилучшей модели: {accuracy:.4f}")
        self.accuracy_label.setVisible(True)        
        # Показываем заголовок и контейнер лучших параметров
        self.params_title.setVisible(True)
        self.params_container.setVisible(True)        
        # Очищаем существующие метки параметров
        for i in reversed(range(self.params_layout.count())):
            self.params_layout.itemAt(i).widget().setParent(None)        
        # Добавляем новые метки для каждого параметра
        for key, value in best_params.items():
            param_label = QLabel(f"{key}: {value}")
            param_label.setStyleSheet("margin-left: 20px; margin-bottom: 3px; font-size: 12px;")
            param_label.setWordWrap(True)
            self.params_layout.addWidget(param_label)        
        # Показываем кнопку сохранения
        self.save_button.setVisible(True)    
    
    def save_best_model(self):
        """Save the best trained model"""
        if not self.best_model:
            QMessageBox.warning(self, "Warning", "No model to save!")
            return        
        try:# Create trained_models directory if it doesn't exist
            models_dir = "trained_models"
            os.makedirs(models_dir, exist_ok=True)
            
            # Generate filename
            dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
            model_name = "random_forest"
            accuracy_str = f"{self.accuracy:.2f}".replace('.', '_')
            filename = f"{model_name}_{dataset_name}_acc_{accuracy_str}_percent.pkl"            
            file_path = os.path.join(models_dir, filename)            
            # Save model
            import joblib
            joblib.dump(self.best_model, file_path)            
            QMessageBox.information(self, "Success", f"Model saved as: {filename}")            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")