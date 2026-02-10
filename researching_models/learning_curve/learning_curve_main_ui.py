from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QInputDialog,
    QGroupBox, QButtonGroup, QRadioButton, QLineEdit, QScrollArea, QDialog, QApplication, QComboBox, QFormLayout,
    QTabWidget
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import os
import gc
import psutil
import matplotlib.pyplot as plt

from researching_models.learning_curve.learning_curve_optuna_logic import OptunaAnalyzer
from researching_models.learning_curve.learning_curve_random_search_logic import RandomSearchAnalyzer

class HelpDialog(QDialog):
    """Справка по метрикам и параметрам"""
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.setModal(True)
        self.resize(400, 300)
        layout = QVBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
        self.setLayout(layout)

class LearningCurveMainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.optuna_analyzer = OptunaAnalyzer()
        self.random_search_analyzer = RandomSearchAnalyzer()
        self.results_layout = QVBoxLayout()
        self.curve_params = {}
        self.process = psutil.Process(os.getpid())
        self.tab_widget = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Кривые обучения - Сравнение методов")
        main_layout = QVBoxLayout()

        title_label = QLabel("Подбор гиперпараметров и кривые обучения")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)

        # === СТРОКА: Задача + Кнопка загрузки ===
        task_load_layout = QHBoxLayout()

        # Группа для типа задачи (чтобы не растягивалась)
        task_widget = QWidget()
        task_layout = QHBoxLayout(task_widget)
        task_layout.addWidget(QLabel("Задача:"))
        self.classification_radio = QRadioButton("Классификация")
        self.regression_radio = QRadioButton("Регрессия")
        self.classification_radio.setChecked(True)
        self.task_group = QButtonGroup()
        self.task_group.addButton(self.classification_radio, 1)
        self.task_group.addButton(self.regression_radio, 2)
        self.task_group.buttonClicked.connect(self.on_task_selected)
        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.regression_radio)
        task_layout.addStretch() 

        # Кнопка загрузки
        self.load_btn = QPushButton("📁 Загрузить датасет")
        self.load_btn.clicked.connect(self.on_load_dataset)

        # Добавляем в основную строку: сначала группу задач, потом кнопку
        task_load_layout.addWidget(task_widget) 
        task_load_layout.addWidget(self.load_btn) 
        task_load_layout.setStretch(0, 0)  
        task_load_layout.setStretch(1, 1)  

        main_layout.addLayout(task_load_layout)
        # Целевая переменная и метка памяти в одной строке
        target_memory_layout = QHBoxLayout()
        self.target_label = QLabel("Целевая переменная: не выбрана")
        self.target_label.setStyleSheet("font-weight: bold;")
        self.memory_label = QLabel("📊 Память: ? МБ")
        self.memory_label.setStyleSheet("color: #555; font-size: 11px;")
        target_memory_layout.addWidget(self.target_label)
        target_memory_layout.addWidget(self.memory_label)
        target_memory_layout.addStretch() 
        main_layout.addLayout(target_memory_layout)

        # === Вкладки ===
        self.tab_widget = QTabWidget()

        # Вкладка Optuna
        from .learning_curve_optuna_ui import LearningCurveOptunaTab
        self.optuna_tab = LearningCurveOptunaTab(analyzer=self.optuna_analyzer, main_window=self)
        self.tab_widget.addTab(self.optuna_tab, "Optuna")

        # Вкладка Random Search
        #from .learning_curve_random_search_ui import LearningCurveRandomSearchTab
        #self.random_search_tab = LearningCurveRandomSearchTab(analyzer=self.random_search_analyzer, main_window=self)
        #self.tab_widget.addTab(self.random_search_tab, "Random Search")
        main_layout.addWidget(self.tab_widget)

        # Удаляем проверку загрузки датасета в LearningCurveOptunaTab
        self.optuna_tab.check_dataset_loaded = lambda: True

        self.setLayout(main_layout)
        self.adjustSize() 
        self.show()
        self.update_memory_usage()
        
        # Принудительно обновляем виджеты результатов
        QApplication.processEvents()

    def on_task_selected(self):
        # Передаем выбор задачи во все вкладки
        task_type = "classification" if self.classification_radio.isChecked() else "regression"
        self.optuna_analyzer.task_type = task_type
        self.random_search_analyzer.task_type = task_type
        if hasattr(self, 'optuna_tab'):
            self.optuna_tab.analyzer.task_type = task_type
            self.optuna_tab.update_scoring_options()
        if hasattr(self, 'random_search_tab'):
            self.random_search_tab.analyzer.task_type = task_type
            self.random_search_tab.update_scoring_options()

    def on_model_changed(self):
        model = self.model_combo.currentText()
        if hasattr(self, 'optuna_tab'):
            self.optuna_tab.model_combo.setCurrentText(model)
            self.optuna_tab.on_model_changed()
        if hasattr(self, 'random_search_tab'):
            self.random_search_tab.model_combo.setCurrentText(model)
            self.random_search_tab.on_model_changed()

    def closeEvent(self, event):
        plt.close('all')
        self.optuna_analyzer = None
        self.random_search_analyzer = None
        gc.collect()
        super().closeEvent(event)

    def on_load_dataset(self):
        reply = QMessageBox.question(
            self, "Режим загрузки",
            "Разделить датасет на train/test?\n"
            "• Да — загрузить два файла\n"
            "• Нет — один файл, разделю автоматически",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.load_separate_datasets()
        else:
            self.load_single_dataset()


    def load_single_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV", "./dataset/", "CSV (*.csv)")
        if not path: return
        try:
            import pandas as pd
            df = pd.read_csv(path, comment='#')
            target, ok = QInputDialog.getItem(self, "Целевая", "Выберите:", df.columns, 0, False)
            if not ok: return
            
            # Загружаем данные в оба анализатора
            self.optuna_analyzer.load_from_dataframe(df, target, self.optuna_analyzer.task_type)
            self.random_search_analyzer.load_from_dataframe(df, target, self.random_search_analyzer.task_type)
            
            self.target_label.setText(f"Целевая: {target}")
            # Кнопка анализа больше не нужна, анализ запускается из вкладок
            self.load_btn.setText(f"📁 {os.path.basename(path)}")
            self.update_memory_usage()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не загружен:\n{e}")

    def load_separate_datasets(self):
        train_path, _ = QFileDialog.getOpenFileName(self, "Train", "./dataset/", "CSV (*.csv)")
        if not train_path: return
        test_path, _ = QFileDialog.getOpenFileName(self, "Test", "./dataset/", "CSV (*.csv)")
        if not test_path: return
        try:
            import pandas as pd
            df_train = pd.read_csv(train_path, comment='#')
            df_test = pd.read_csv(test_path, comment='#')

            common_cols = set(df_train.columns) & set(df_test.columns)
            if not common_cols:
                QMessageBox.critical(self, "Ошибка", "Нет общих колонок!")
                return

            possible_targets = [col for col in common_cols if df_train[col].nunique() < 0.9 * len(df_train)]
            if not possible_targets:
                possible_targets = list(common_cols)

            target, ok = QInputDialog.getItem(self, "Целевая", "Выберите:", sorted(possible_targets), 0, False)
            if not ok or not target:
                return

            # Загружаем данные в оба анализатора
            self.optuna_analyzer.load_separate_datasets(train_path, test_path, target, self.optuna_analyzer.task_type)
            self.random_search_analyzer.load_separate_datasets(train_path, test_path, target, self.random_search_analyzer.task_type)
            
            self.target_label.setText(f"Целевая: {target}")
            # Кнопка анализа больше не нужна, анализ запускается из вкладок

            train_name = os.path.basename(train_path)
            test_name = os.path.basename(test_path)
            self.load_btn.setText(f"📁 train: {train_name}\n   test: {test_name}")
            self.update_memory_usage()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки:\n{e}")






    def update_memory_usage(self):
        try:
            mem_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"📊 Память: {mem_mb:.1f} МБ")
        except:
            self.memory_label.setText("📊 Память: ошибка")


