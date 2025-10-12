from PySide6.QtWidgets import *
from PySide6.QtCore import *
import logging
import os
from .selection_parameters_parameter_tuning_worker import ParameterTuningWorker
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ParameterTuningWindow(QWidget):
    def __init__(self, dataset_path, parent=None):
        """Инициализация окна выбора параметров"""
        logger.info(f"ПараметрTuningWindow.__init__ вызван с dataset_path: {dataset_path}")
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.best_model = None
        self.best_params = None
        self.accuracy = None
        logger.info("Вызываем initUI...")
        self.initUI()
        logger.info("Инициализация окна завершена - пока ещё не запускаем оптимизацию параметров")
        
        # Запуск оптимизации параметров задерживается, чтобы сначала появилось окно
        def delayed_start():
            logger.info("Задержанный старт - вызываем simulate_tuning...")
            self.simulate_tuning()
            
        QTimer.singleShot(1000, delayed_start)  # Задержка в 1 секунду
        logger.info("Задержанный запуск оптимизации запланирован")
    
    def initUI(self):
        """Настройка интерфейса окна"""
        logger.info("ПараметрTuningWindow.initUI вызван")
        self.setWindowTitle("Выбор параметров")
        self.setGeometry(300, 300, 700, 600)
        logger.info("Заголовок окна и размеры установлены")
        
        # Настраиваем флаги окна и поведение закрытия
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        logger.info("Флаги окна и атрибуты настроены")
        
        # Макет окна
        layout = QVBoxLayout()
        
        # Заголовок
        title = QLabel("Прогресс оптимизации параметров")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Полоса прогресса
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Метка статуса
        self.progress_label = QLabel("Начинаем оптимизацию параметров...")
        self.progress_label.setStyleSheet("font-size: 14px; margin: 10px 0;")
        layout.addWidget(self.progress_label)
        
        # Статус метки
        self.status_label = QLabel("Инициализация...")
        self.status_label.setStyleSheet("font-size: 12px; color: #666; margin-bottom: 10px;")
        layout.addWidget(self.status_label)
        
        # Результаты оптимизации
        self.results_title = QLabel("=== Результаты оптимизации ===")
        self.results_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        self.results_title.setVisible(False)
        layout.addWidget(self.results_title)
        
        # Точность модели
        self.accuracy_label = QLabel("")
        self.accuracy_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E8B57; margin: 10px 0;")
        self.accuracy_label.setVisible(False)
        layout.addWidget(self.accuracy_label)
        
        # Лучшие параметры
        self.params_title = QLabel("Лучшие параметры:")
        self.params_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        self.params_title.setVisible(False)
        layout.addWidget(self.params_title)
        
        # Контейнер для лучших параметров
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_container.setLayout(self.params_layout)
        self.params_container.setVisible(False)
        layout.addWidget(self.params_container)
        
        # Растягивающийся элемент для заполнения свободного пространства
        layout.addStretch()
        
        # Кнопка сохранить
        self.save_button = QPushButton("Сохранить лучшую модель")
        self.save_button.clicked.connect(self.save_best_model)
        self.save_button.setVisible(False)
        self.save_button.setStyleSheet(
            "font-size: 12px; padding: 10px; "
            "background-color: #4CAF50; color: white; "
            "border: none; border-radius: 5px;"
        )
        layout.addWidget(self.save_button)
        
        # Установка макета окна
        self.setLayout(layout)
        logger.info("Макет интерфейса окна завершён")
        self.setVisible(True)
        logger.info("Окно установлено как видимое в initUI")
        
    def closeEvent(self, event):
        """Handle window close event"""
        logger.info("ParameterTuningWindow.closeEvent called")
        logger.info(f"Close event type: {type(event)}")
        logger.info("Window is being closed!")
        super().closeEvent(event)
    
    def simulate_tuning(self):
        """Simulate tuning process for testing"""
        logger.info("simulate_tuning method called")
        
        # Simulate progress updates
        total_iterations = 10  # Reduced for testing
        for i in range(total_iterations + 1):
            QTimer.singleShot(i * 200, lambda iter=i: self.update_progress(iter, total_iterations))
        
        # Simulate completion after 3 seconds
        def simulate_completion():
            logger.info("Simulating tuning completion")
            fake_best_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'criterion': 'gini'
            }
            fake_accuracy = 0.9289
            self.on_tuning_completed(None, fake_best_params, fake_accuracy)
        
        QTimer.singleShot(3000, simulate_completion)
        logger.info("Simulation timers scheduled")
    
    def start_tuning(self):
        """Start the parameter tuning process"""
        logger.info("start_tuning method called")
        logger.info(f"Creating ParameterTuningWorker with dataset: {self.dataset_path}")
        
        try:
            self.worker = ParameterTuningWorker(self.dataset_path)
            logger.info("ParameterTuningWorker created successfully")
            
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.tuning_completed.connect(self.on_tuning_completed)
            self.worker.error_occurred.connect(self.on_error)
            logger.info("Worker signals connected")
            
            logger.info("Starting worker thread...")
            self.worker.start()
            logger.info("Worker thread started successfully")
            
        except Exception as e:
            logger.error(f"Error in start_tuning: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def update_progress(self, current_iter, total_iter):
        """Update progress bar and status"""
        progress = int((current_iter / total_iter) * 100)
        self.progress_bar.setValue(progress)
        
        # Update progress label with iteration count
        self.progress_label.setText(f"Progress: {current_iter} / {total_iter} iterations completed")
        self.status_label.setText(f"Optimization in progress... {progress}% complete")
    
    def on_tuning_completed(self, best_model, best_params, accuracy):
        """Handle tuning completion"""
        self.best_model = best_model
        self.best_params = best_params
        self.accuracy = accuracy
        
        # Update UI
        self.progress_bar.setValue(100)
        self.progress_label.setText("Parameter optimization completed!")
        self.status_label.setText("All iterations finished successfully.")
        
        # Show results section
        self.results_title.setVisible(True)
        
        # Display accuracy using QLabel
        self.accuracy_label.setText(f"Best Model Accuracy: {accuracy:.4f}")
        self.accuracy_label.setVisible(True)
        
        # Display best parameters using QLabel widgets
        self.params_title.setVisible(True)
        self.params_container.setVisible(True)
        
        # Clear any existing parameter labels
        for i in reversed(range(self.params_layout.count())):
            self.params_layout.itemAt(i).widget().setParent(None)
        
        # Add each parameter as a separate QLabel
        for key, value in best_params.items():
            param_label = QLabel(f"  {key}: {value}")
            param_label.setStyleSheet("margin-left: 20px; margin-bottom: 3px; font-size: 12px;")
            param_label.setWordWrap(True)
            self.params_layout.addWidget(param_label)
        
        # Show save button
        self.save_button.setVisible(True)
    
    def on_error(self, error_message):
        """Handle errors during tuning"""
        self.progress_label.setText("Parameter optimization failed!")
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("font-size: 12px; color: #DC143C; margin-bottom: 10px;")
    
    def save_best_model(self):
        """Save the best trained model"""
        if not self.best_model:
            QMessageBox.warning(self, "Warning", "No model to save!")
            return
        
        try:
            # Create trained_models directory if it doesn't exist
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