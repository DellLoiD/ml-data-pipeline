# training_window.py
import sys
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
# === Импорт всех модулей ===
from preprocessing.dataset_processing_check_nan import MissingValuesDialog
from preprocessing.dataset_processing_fix_non_numeric_ui import OneHotEncodingWindow
from preprocessing.correlation_graph_ui import CorrelationGraphUI
from preprocessing.data_balancing.data_balancing_method_ui import DataBalancingApp
from preprocessing.outlier_categories_ui import OutlierCategoriesApp
from researching_models.check_models_ui import ClassificationApp
from selection_of_parameters.selection_parameters_main_menu_ui import MainWindow_selection_parameters
from inference_models.inference_trained_models import SurveyForm
from load_dataset_ui import LoadDatasetWindow
from splitting_dataset_ui import SplittingDatasetWindow
from checking_data_formats_ui import CheckingDataFormatsWindow
from preprocessing.imputation_by_model_ui import ImputationByModelApp
from preprocessing.hashing_methods_ui import HashingMethodsWindow

# === Глобальные ссылки на окна (чтобы не открывалось несколько раз) ===
processing_window_instance = None
correlation_graph_instance = None
data_balancing_smote_instance = None
classification_app_instance = None
selection_of_parameters_instance = None
inference_trained_models_instance = None
load_dataset_window_instance = None
splitting_dataset_window_instance = None
checking_data_formats_window_instance = None
outlier_categories_instance = None
imputation_model_instance = None
hashing_methods_instance = None 

class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Заголовок окна
        self.setWindowTitle("Обучение модели")
        # Минимальные размеры окна
        self.setMinimumSize(400, 300)
        # Начальное отображение окна с определённым размером
        self.resize(300, 400)

        # === Кнопки ===
        btn_load_dataset = QPushButton("📥 Загрузка датасета извне в приложение")
        btn_load_dataset.clicked.connect(self.open_load_dataset)

        btn_check_formats = QPushButton("🔍 Проверка форматов данных")
        btn_check_formats.clicked.connect(self.open_checking_data_formats)

        btn_process_nan_value = QPushButton("Удаление пропущенных значений")
        btn_process_nan_value.clicked.connect(self.deleteNanValue)

        btn_outlier_categories = QPushButton("🔍 Анализ редких классов")
        btn_outlier_categories.clicked.connect(self.open_outlier_categories)

        btn_split_dataset = QPushButton("✂️ Разделение датасета")
        btn_split_dataset.clicked.connect(self.open_splitting_dataset)

        btn_hashing_methods = QPushButton("🔏 Хеширование строковых классов")
        btn_hashing_methods.clicked.connect(self.open_hashing_methods)

        btn_process_fix_non_numeric = QPushButton("🛠️ Обработка нечисловых значений")
        btn_process_fix_non_numeric.clicked.connect(self.fixNonNumericValue)

        btn_correlation_plot = QPushButton("Корреляция параметров (график)")
        btn_correlation_plot.clicked.connect(self.openCorrelationGraph)

        btn_edit_dataset = QPushButton("Редактирование датасета (SMOTE, TRIM)")
        btn_edit_dataset.clicked.connect(self.openDataBalancingSmote)

        btn_model_selection = QPushButton("Оценка и выбор модели")
        btn_model_selection.clicked.connect(self.open_classification_app)

        btn_hyperparameters_tuning = QPushButton("Подбор параметров для модели и обучение")
        btn_hyperparameters_tuning.clicked.connect(self.openHyperParametersTuning)
        
        btn_impute_model = QPushButton("🔧 Восстановить значения моделью")
        btn_impute_model.clicked.connect(self.open_impute_by_model)        

        btn_inference_models = QPushButton("Инференс модели")
        btn_inference_models.clicked.connect(self.openInferenceTrainedModels)

        # === Макет ===
        layout = QVBoxLayout()
        layout.addWidget(btn_load_dataset)
        layout.addWidget(btn_check_formats)
        layout.addWidget(btn_process_nan_value)
        layout.addWidget(btn_outlier_categories)
        layout.addWidget(btn_split_dataset)
        layout.addWidget(btn_hashing_methods)          # ✅ Кнопка вставлена ДО обработки
        layout.addWidget(btn_process_fix_non_numeric)  # ✅ После идёт обработка нечисловых
        layout.addWidget(btn_correlation_plot)
        layout.addWidget(btn_edit_dataset)
        layout.addWidget(btn_model_selection)
        layout.addWidget(btn_hyperparameters_tuning)
        layout.addWidget(btn_impute_model)
        layout.addWidget(btn_inference_models)


        # Устанавливаем макет
        self.setLayout(layout)

    # === Методы открытия окон ===
    def open_impute_by_model(self):
        global imputation_model_instance
        if not imputation_model_instance or not imputation_model_instance.isVisible():
            imputation_model_instance = ImputationByModelApp()
            imputation_model_instance.show()

    def open_load_dataset(self):
        global load_dataset_window_instance
        if not load_dataset_window_instance or not load_dataset_window_instance.isVisible():
            load_dataset_window_instance = LoadDatasetWindow()
            load_dataset_window_instance.show()

    def open_checking_data_formats(self):
        global checking_data_formats_window_instance
        if not checking_data_formats_window_instance or not checking_data_formats_window_instance.isVisible():
            checking_data_formats_window_instance = CheckingDataFormatsWindow()
            checking_data_formats_window_instance.show()

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

    def open_splitting_dataset(self):
        global splitting_dataset_window_instance
        if not splitting_dataset_window_instance or not splitting_dataset_window_instance.isVisible():
            splitting_dataset_window_instance = SplittingDatasetWindow()
            splitting_dataset_window_instance.show()

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

    def open_classification_app(self):
        global classification_app_instance
        if not classification_app_instance or not classification_app_instance.isVisible():
            classification_app_instance = ClassificationApp()
            classification_app_instance.show()

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

    def open_outlier_categories(self):
        global outlier_categories_instance
        if not outlier_categories_instance or not outlier_categories_instance.isVisible():
            outlier_categories_instance = OutlierCategoriesApp()
            outlier_categories_instance.show()

    # ✅ НОВЫЙ МЕТОД: Открытие окна хеширования строковых классов
    def open_hashing_methods(self):
        global hashing_methods_instance
        if not hashing_methods_instance or not hashing_methods_instance.isVisible():
            hashing_methods_instance = HashingMethodsWindow()
            hashing_methods_instance.show()


# === Запуск приложения ===
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = TrainingWindow()
    window.show()

    sys.exit(app.exec())
