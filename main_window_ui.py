# main_window_ui.py

import sys
import logging

# === НАСТРОЙКА ЛОГИРОВАНИЯ — ДО ВСЕХ ИМПОРТОВ МОДУЛЕЙ ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parameter_tuning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QGroupBox
)

# === Импорт всех модулей ===
from preprocessing.dataset_processing_check_nan import MissingValuesDialog
from preprocessing.dataset_processing_fix_non_numeric_ui import FixNonNumericWindow
from preprocessing.correlation_graph_ui import CorrelationGraphUI
from preprocessing.data_balancing.data_balancing_method_ui import DataBalancingApp
from researching_models.model_evaluation_ui import ModelEvaluationUI
from researching_models.feature_importance_ui import FeatureImportanceUI
from researching_models.learning_curve_ui import LearningCurveUI
from researching_models.cross_validation_ui import CrossValidationUI 
from selection_of_parameters.selection_parameters_main_menu_ui import MainWindow_selection_parameters
from inference_models.inference_trained_models import SurveyForm
from splitting_dataset_ui import SplittingDatasetWindow
from checking_data_formats_ui import CheckingDataFormatsWindow
from preprocessing.imputation_by_model_ui import ImputationByModelApp
from preprocessing.hashing_methods_ui import HashingMethodsWindow
from load_params_and_train_final_model import FinalTrainingWindow

# === Глобальные ссылки на окна ===
missing_values_window_instance = None
onehot_window_instance = None
correlation_graph_instance = None
data_balancing_smote_instance = None
model_evaluation_instance = None
feature_importance_instance = None
learning_curve_instance = None
cross_validation_instance = None
selection_of_parameters_instance = None
inference_trained_models_instance = None
splitting_dataset_window_instance = None
checking_data_formats_window_instance = None
imputation_model_instance = None
hashing_methods_instance = None
final_training_instance = None


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обучение модели")
        self.resize(320, 680)
        self.setMinimumSize(300, 680)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === 1. Предобработка данных ===
        preprocessing_group = QGroupBox("🧹 Предобработка данных")
        preprocessing_layout = QVBoxLayout()

        btn_check_formats = QPushButton("🔍 Загрузка и анализ датасета")
        btn_check_formats.clicked.connect(self.open_checking_data_formats)
        preprocessing_layout.addWidget(btn_check_formats)

        btn_process_nan_value = QPushButton("Обработка пропущенных значений")
        btn_process_nan_value.clicked.connect(self.deleteNanValue)
        preprocessing_layout.addWidget(btn_process_nan_value)

        btn_hashing_methods = QPushButton("🔏 Хеширование строковых классов")
        btn_hashing_methods.clicked.connect(self.open_hashing_methods)
        preprocessing_layout.addWidget(btn_hashing_methods)

        btn_process_fix_non_numeric = QPushButton("🛠️ Обработка нечисловых значений")
        btn_process_fix_non_numeric.clicked.connect(self.fixNonNumericValue)
        preprocessing_layout.addWidget(btn_process_fix_non_numeric)

        btn_edit_dataset = QPushButton("Редактирование датасета (SMOTE, TRIM)")
        btn_edit_dataset.clicked.connect(self.openDataBalancingSmote)
        preprocessing_layout.addWidget(btn_edit_dataset)

        preprocessing_group.setLayout(preprocessing_layout)
        main_layout.addWidget(preprocessing_group)

        # === 2. Разделение и восстановление ===
        engineering_group = QGroupBox("⚙️ Разделение и восстановление")
        engineering_layout = QVBoxLayout()

        btn_split_dataset = QPushButton("✂️ Разделение датасета")
        btn_split_dataset.clicked.connect(self.open_splitting_dataset)
        engineering_layout.addWidget(btn_split_dataset)

        btn_impute_model = QPushButton("🔧 Восстановить значения моделью")
        btn_impute_model.clicked.connect(self.open_impute_by_model)
        engineering_layout.addWidget(btn_impute_model)

        engineering_group.setLayout(engineering_layout)
        main_layout.addWidget(engineering_group)

        # === 3. Анализ и визуализация ===
        analysis_group = QGroupBox("🔍 Анализ и визуализация")
        analysis_layout = QVBoxLayout()

        btn_correlation_plot = QPushButton("Корреляция параметров (график)")
        btn_correlation_plot.clicked.connect(self.openCorrelationGraph)
        analysis_layout.addWidget(btn_correlation_plot)

        btn_feature_importance = QPushButton("Важность признаков")
        btn_feature_importance.clicked.connect(self.open_feature_importance)
        analysis_layout.addWidget(btn_feature_importance)

        analysis_group.setLayout(analysis_layout)
        main_layout.addWidget(analysis_group)

        # === 4. Оценка и выбор модели ===
        evaluation_group = QGroupBox("📊 Оценка и выбор модели")
        evaluation_layout = QVBoxLayout()

        btn_model_evaluation = QPushButton("Оценка модели")
        btn_model_evaluation.clicked.connect(self.open_model_evaluation)
        evaluation_layout.addWidget(btn_model_evaluation)

        btn_learning_curve = QPushButton("Кривая обучения")
        btn_learning_curve.clicked.connect(self.open_learning_curve)
        evaluation_layout.addWidget(btn_learning_curve)

        btn_cross_validation = QPushButton("Кросс-валидация")
        btn_cross_validation.clicked.connect(self.open_cross_validation)
        evaluation_layout.addWidget(btn_cross_validation)

        evaluation_group.setLayout(evaluation_layout)
        main_layout.addWidget(evaluation_group)
        
        # === 5. Моделирование и инференс ===
        modeling_group = QGroupBox("🧠 Моделирование и инференс")
        modeling_layout = QVBoxLayout()

        btn_hyperparameters_tuning = QPushButton("Подбор гиперпараметров")
        btn_hyperparameters_tuning.clicked.connect(self.openHyperParametersTuning)
        modeling_layout.addWidget(btn_hyperparameters_tuning)
        
        btn_final_train = QPushButton("🎯 Финальное обучение модели")
        btn_final_train.clicked.connect(self.open_final_training)
        modeling_layout.addWidget(btn_final_train)

        btn_inference_models = QPushButton("Инференс модели")
        btn_inference_models.clicked.connect(self.openInferenceTrainedModels)
        modeling_layout.addWidget(btn_inference_models)

        modeling_group.setLayout(modeling_layout)
        main_layout.addWidget(modeling_group)

        self.setLayout(main_layout)


    # === Методы открытия окон ===
    def open_final_training(self):
        """Открывает окно финального обучения модели"""
        global final_training_instance
        if not final_training_instance or not final_training_instance.isVisible():
            final_training_instance = FinalTrainingWindow()
            final_training_instance.show()
        else:
            final_training_instance.raise_()
            final_training_instance.activateWindow()

    def open_impute_by_model(self):
        global imputation_model_instance
        if not imputation_model_instance or not imputation_model_instance.isVisible():
            imputation_model_instance = ImputationByModelApp()
            imputation_model_instance.show()

    def open_checking_data_formats(self):
        global checking_data_formats_window_instance
        if not checking_data_formats_window_instance or not checking_data_formats_window_instance.isVisible():
            checking_data_formats_window_instance = CheckingDataFormatsWindow()
            checking_data_formats_window_instance.show()

    def deleteNanValue(self):
        global missing_values_window_instance
        if not missing_values_window_instance or not missing_values_window_instance.isVisible():
            missing_values_window_instance = MissingValuesDialog()
            missing_values_window_instance.show()

    def fixNonNumericValue(self):
        global onehot_window_instance
        if not onehot_window_instance or not onehot_window_instance.isVisible():
            onehot_window_instance = FixNonNumericWindow()
            onehot_window_instance.show()

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

    def open_model_evaluation(self):
        global model_evaluation_instance
        if not model_evaluation_instance or not model_evaluation_instance.isVisible():
            model_evaluation_instance = ModelEvaluationUI()
            model_evaluation_instance.show()

    def open_feature_importance(self):
        global feature_importance_instance
        if not feature_importance_instance or not feature_importance_instance.isVisible():
            feature_importance_instance = FeatureImportanceUI()
            feature_importance_instance.show()
        else:
            feature_importance_instance.raise_()
            feature_importance_instance.activateWindow()

    def open_learning_curve(self):
        global learning_curve_instance
        if not learning_curve_instance or not learning_curve_instance.isVisible():
            learning_curve_instance = LearningCurveUI()
            learning_curve_instance.show()
        else:
            learning_curve_instance.raise_()
            learning_curve_instance.activateWindow()

    def open_cross_validation(self):
        global cross_validation_instance
        if not cross_validation_instance or not cross_validation_instance.isVisible():
            cross_validation_instance = CrossValidationUI()
            cross_validation_instance.show()
        else:
            cross_validation_instance.raise_()
            cross_validation_instance.activateWindow()

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
