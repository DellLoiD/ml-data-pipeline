import sys
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from preprocessing.dataset_processing_check_nan import MissingValuesDialog
from preprocessing.dataset_processing_fix_non_numeric_ui import OneHotEncodingWindow
from preprocessing.correlation_graph_ui import CorrelationGraphUI
from preprocessing.data_balancing.data_balancing_method_ui import DataBalancingApp
from researching_models.check_models_ui import ClassificationApp
from selection_of_parameters.selection_parameters_main_menu_ui import MainWindow_selection_parameters
from inference_models.inference_trained_models import SurveyForm

# Глобальная ссылка на окна
processing_window_instance = None
correlation_graph_instance = None
data_balancing_smote_instance = None
classification_app_instance  = None
selection_of_parameters_instance = None
inference_trained_models_instance = None
    
class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
    
        # Заголовок окна
        self.setWindowTitle("Обучение модели")    
        # Минимальные размеры окна
        self.setMinimumSize(400, 300)    
        # Начальное отображение окна с определённым размером
        self.resize(400, 300)
    
        # Создаем кнопки
        btn_process_nan_value = QPushButton("Удаление пропущеных значений")
        btn_process_nan_value.clicked.connect(self.deleteNanValue)
        btn_process_fix_non_numeric = QPushButton("Обработка не числовых значений")
        btn_process_fix_non_numeric.clicked.connect(self.fixNonNumericValue)
        btn_correlation_plot = QPushButton("Корреляция параметров (график)")
        btn_correlation_plot.clicked.connect(self.openCorrelationGraph)
        btn_edit_dataset = QPushButton("Редактирование датасета (SMOTE, TRIM)")
        btn_edit_dataset.clicked.connect(self.openDataBalancingSmote)
        btn_model_selection = QPushButton("Оценка и выбор модели")
        btn_model_selection.clicked.connect(self.open_classification_app) 
        btn_hyperparameters_tuning = QPushButton("Подбор параметров для модели и обучение")
        btn_hyperparameters_tuning.clicked.connect(self.openHyperParametersTuning)
        btn_inference_models = QPushButton("Инференс модели")
        btn_inference_models.clicked.connect(self.openInferenceTrainedModels)
        
        layout = QVBoxLayout()
        layout.addWidget(btn_process_nan_value)
        layout.addWidget(btn_process_fix_non_numeric)
        layout.addWidget(btn_correlation_plot)
        layout.addWidget(btn_edit_dataset)
        layout.addWidget(btn_model_selection)
        layout.addWidget(btn_hyperparameters_tuning)
        layout.addWidget(btn_inference_models)
    
        # Устанавливаем макет
        self.setLayout(layout)
        
    def open_classification_app(self):
        global classification_app_instance
        if not classification_app_instance or not classification_app_instance.isVisible():
            classification_app_instance = ClassificationApp()
            classification_app_instance.show()        
        
    def deleteNanValue(self):
        global processing_window_instance
        if not processing_window_instance or not processing_window_instance.isVisible():
            processing_window_instance = MissingValuesDialog()
            processing_window_instance.show()
            
    def fixNonNumericValue(self):
        global processing_window_instance
        if not processing_window_instance or not processing_window_instance.isVisible():
            processing_window_instance = OneHotEncodingWindow()
            processing_window_instance.show()
            
    def openCorrelationGraph(self):
        global correlation_graph_instance
        if not correlation_graph_instance or not correlation_graph_instance.isVisible():
            correlation_graph_instance = CorrelationGraphUI()
            correlation_graph_instance.show()
    
    def openDataBalancingSmote(self):
        global data_balancing_smote_instance
        if not data_balancing_smote_instance or not data_balancing_smote_instance.isVisible():
            data_balancing_smote_instance = DataBalancingApp()
            data_balancing_smote_instance.show()
            
    def openHyperParametersTuning(self):
        global selection_of_parameters_instance
        if not selection_of_parameters_instance or not selection_of_parameters_instance.isVisible():
            selection_of_parameters_instance = MainWindow_selection_parameters()
            selection_of_parameters_instance.show()
            
    def openInferenceTrainedModels(self):
        global inference_trained_models_instance
        if not inference_trained_models_instance or not inference_trained_models_instance.isVisible():
            inference_trained_models_instance = SurveyForm()
            inference_trained_models_instance.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = TrainingWindow()
    window.show()
    
    sys.exit(app.exec())


