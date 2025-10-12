import sys
import os
import pandas as pd
import logging
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                               QFileDialog, QMessageBox, QDialog, QTextEdit, 
                               QProgressBar, QLabel, QHBoxLayout)
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from sklearn.model_selection import train_test_split

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
        self.selected_dataset_path = None
        logger.info("MainWindow_selection_parameters initialized")
        self.initUI()

    def initUI(self):
        # Настройка главного окна
        self.setWindowTitle("Главное меню")
        layout = QVBoxLayout()

        # Choose Dataset button at the top
        self.btn_choose_dataset = QPushButton("Выбрать датасет")
        self.btn_choose_dataset.clicked.connect(self.choose_dataset)
        layout.addWidget(self.btn_choose_dataset)

        # Show Current Parameters button
        btn_show_params = QPushButton("Показать текущие параметры")
        btn_show_params.clicked.connect(self.show_current_parameters)
        layout.addWidget(btn_show_params)
        
        # Первая кнопка для открытия окна выбора параметров
        btn_select_params = QPushButton("Указать параметры для подбора")
        btn_select_params.clicked.connect(self.open_selection_of_parameters)
        layout.addWidget(btn_select_params)

        # Вторая кнопка для открытия окна настройки условий подбора
        btn_configure_search = QPushButton("Настроить условия подбора параметров")
        btn_configure_search.clicked.connect(self.open_selection_parameters_random_search)
        layout.addWidget(btn_configure_search)

        # Tune Best Parameters button
        btn_tune_params = QPushButton("Подобрать лучшие параметры")
        btn_tune_params.clicked.connect(self.tune_best_parameters)
        logger.info("Tune Best Parameters button created and connected")
        layout.addWidget(btn_tune_params)

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

    def choose_dataset(self):
        """Open dataset selection dialog"""
        logger.info("choose_dataset method called")
        dataset_folder = "dataset"
        logger.info(f"Checking dataset folder: {dataset_folder}")
        
        if not os.path.exists(dataset_folder):
            logger.warning(f"Dataset folder '{dataset_folder}' not found!")
            QMessageBox.warning(self, "Warning", f"Dataset folder '{dataset_folder}' not found!")
            return
        
        logger.info(f"Opening file dialog for dataset selection")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Dataset",
            dataset_folder,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.selected_dataset_path = file_path
            filename = os.path.basename(file_path)
            logger.info(f"Dataset selected: {file_path}")
            logger.info(f"Dataset filename: {filename}")
            # Update button label to show selected dataset name
            self.btn_choose_dataset.setText(filename)
            QMessageBox.information(self, "Success", f"Selected dataset: {filename}")
        else:
            logger.info("No dataset selected (user cancelled or closed dialog)")

    def show_current_parameters(self):
        """Show current parameters in a dialog window using QLabel widgets"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Current Tuning Parameters")
        dialog.setModal(True)
        dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        
        # Get current parameters
        try:
            hyperparams = get_hyperparameters()
            search_params = get_random_search_params()
            
            # Add title for Random Grid Parameters
            title1 = QLabel("=== Random Grid Parameters ===")
            title1.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
            layout.addWidget(title1)
            
            # Display each hyperparameter on its own line using QLabel
            for key, value in hyperparams.items():
                param_label = QLabel(f"{key}: {value}")
                param_label.setWordWrap(True)
                param_label.setStyleSheet("margin-left: 20px; margin-bottom: 5px;")
                layout.addWidget(param_label)
            
            # Add spacing and title for RandomizedSearchCV Parameters
            spacer1 = QLabel("")
            spacer1.setMinimumHeight(10)
            layout.addWidget(spacer1)
            
            title2 = QLabel("=== RandomizedSearchCV Parameters ===")
            title2.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
            layout.addWidget(title2)
            
            # Display each search parameter on its own line using QLabel
            for key, value in search_params.items():
                param_label = QLabel(f"{key}: {value}")
                param_label.setWordWrap(True)
                param_label.setStyleSheet("margin-left: 20px; margin-bottom: 5px;")
                layout.addWidget(param_label)
            
        except Exception as e:
            error_label = QLabel(f"Error loading parameters: {str(e)}")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(error_label)
        
        # Add stretch to push close button to bottom
        layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("font-size: 12px; padding: 8px;")
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()

    def tune_best_parameters(self):
        """Launch parameter tuning with progress visualization"""
        logger.info("tune_best_parameters method called")
        logger.info(f"Current selected_dataset_path: {self.selected_dataset_path}")
        
        if not self.selected_dataset_path:
            logger.warning("No dataset selected - showing warning message")
            QMessageBox.warning(self, "Warning", "Please select a dataset first!")
            return
        
        logger.info(f"Dataset path is valid: {self.selected_dataset_path}")
        logger.info("Creating ParameterTuningWindow...")
        
        try:
            # Create and show parameter tuning window
            # Try without parent first to see if parent relationship is causing issues
            tuning_window = ParameterTuningWindow(self.selected_dataset_path, None)
            logger.info("ParameterTuningWindow created successfully")
            
            # Try multiple approaches to make window visible
            logger.info("Attempting to show window...")
            
            # Method 1: Standard show()
            tuning_window.show()
            logger.info("ParameterTuningWindow.show() called successfully")
            
            # Method 2: Force visibility
            tuning_window.setVisible(True)
            logger.info("setVisible(True) called")
            
            # Method 3: Bring to front and activate
            tuning_window.raise_()
            tuning_window.activateWindow()
            logger.info("raise() and activateWindow() called")
            
            # Method 4: Process events to ensure UI updates
            QApplication.processEvents()
            logger.info("QApplication.processEvents() called")
            
            # Additional debugging for window visibility
            logger.info(f"Window is visible: {tuning_window.isVisible()}")
            logger.info(f"Window is hidden: {tuning_window.isHidden()}")
            logger.info(f"Window geometry: {tuning_window.geometry()}")
            logger.info(f"Window size: {tuning_window.size()}")
            logger.info(f"Window position: {tuning_window.pos()}")
            
            # Method 5: Use QTimer with error handling
            def ensure_visibility():
                try:
                    logger.info("QTimer callback - ensuring window visibility")
                    tuning_window.setVisible(True)
                    tuning_window.raise_()
                    tuning_window.activateWindow()
                    QApplication.processEvents()
                    logger.info(f"Timer: Window is visible: {tuning_window.isVisible()}")
                    logger.info("QTimer callback completed successfully")
                except Exception as e:
                    logger.error(f"Error in QTimer callback: {str(e)}")
                    import traceback
                    logger.error(f"Timer callback traceback: {traceback.format_exc()}")
            
            QTimer.singleShot(100, ensure_visibility)  # 100ms delay
            logger.info("QTimer scheduled for window visibility check")
            
            # Additional test: Try creating a simple test window to see if Qt windows work at all
            def test_simple_window():
                try:
                    logger.info("Creating test window...")
                    test_window = QWidget()
                    test_window.setWindowTitle("Test Window")
                    test_window.setGeometry(500, 500, 300, 200)
                    test_window.show()
                    logger.info(f"Test window created and shown. Visible: {test_window.isVisible()}")
                    
                    # Close test window after 2 seconds
                    def close_test():
                        test_window.close()
                        logger.info("Test window closed")
                    QTimer.singleShot(2000, close_test)
                    
                except Exception as e:
                    logger.error(f"Error creating test window: {str(e)}")
            
            QTimer.singleShot(500, test_simple_window)  # 500ms delay
            logger.info("Test window scheduled")
            
        except Exception as e:
            logger.error(f"Error creating or showing ParameterTuningWindow: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to open parameter tuning window: {str(e)}")

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