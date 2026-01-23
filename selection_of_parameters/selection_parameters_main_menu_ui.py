# selection_of_parameters/main_window_selection_parameters.py
import sys
import os
import pandas as pd
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QDialog, QLabel, QInputDialog,
    QScrollArea, QHBoxLayout, QRadioButton, QButtonGroup
)

# Импорты
from utils.dataset_version_checker import check_train_test_versions, extract_version
from .selection_of_parameters_logic import get_random_grid, get_random_search_params, save_random_search_params
from .selection_parameters_parameter_tuning_window import ParameterTuningWindow
from selection_of_parameters.selection_of_parameters_ui import HyperParameterOptimizerGUI
from selection_of_parameters.selection_parameters_random_search_ui import RandomSearchConfigGUI


class MainWindow_selection_parameters(QWidget):
    def __init__(self):
        super().__init__()
        self.parameter_window = None
        self.selected_dataset_path = None  
        self.train_path = None      
        self.test_path = None
        self.df = None
        self.df_train = None
        self.df_test = None
        self.target_variable = None
        self.selected_model = ""
        
        # Выбор типа задачи при старте
        task, ok = QInputDialog.getItem(
            self, "Тип задачи", "Выберите тип задачи:",
            ["Классификация", "Регрессия"],
            current=0, editable=False
        )
        if not ok:
            self.close()
            return
        selected_task = "classification" if task == "Классификация" else "regression"
        self._set_refit_for_task(selected_task)
        
        self.initUI()
        
        # Установка радиокнопки
        if selected_task == "classification":
            self.classification_radio.setChecked(True)
        else:
            self.regression_radio.setChecked(True)
        self.update_model_list()

    def _set_refit_for_task(self, task_type):
        params = get_random_search_params()
        new_refit = "f1_macro" if task_type == "classification" else "r2"
        if params.get('refit') != new_refit:
            updated_params = params.copy()
            updated_params['refit'] = new_refit
            save_random_search_params(updated_params)

    def initUI(self):
        self.setWindowTitle("Настройка параметров моделей")
        layout = QVBoxLayout()

        # === Тип задачи ===
        task_layout = QHBoxLayout()
        task_label = QLabel("Тип задачи:")
        task_label.setStyleSheet("font-weight: bold;")
        task_layout.addWidget(task_label)

        self.classification_radio = QRadioButton("1. Классификация")
        self.regression_radio = QRadioButton("2. Регрессия")
        self.classification_radio.setChecked(True)

        self.task_group = QButtonGroup()
        self.task_group.addButton(self.classification_radio, 1)
        self.task_group.addButton(self.regression_radio, 2)

        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.regression_radio)
        task_layout.addStretch()
        layout.addLayout(task_layout)

        # === Кнопка выбора датасета ===
        self.btn_choose_dataset = QPushButton("Выбрать датасет")
        self.btn_choose_dataset.clicked.connect(self.choose_dataset)
        layout.addWidget(self.btn_choose_dataset)

        # === Показать параметры ===
        btn_show_params = QPushButton("Показать текущие параметры")
        btn_show_params.clicked.connect(self.show_current_parameters)
        layout.addWidget(btn_show_params)

        # === Выбор модели ===
        label_model_choice = QLabel("Выбор модели:", font=QFont('Arial', 12))
        layout.addWidget(label_model_choice)

        self.model_combo_box = QComboBox()
        self.model_combo_box.currentTextChanged.connect(self.on_model_change)
        layout.addWidget(self.model_combo_box)

        self.update_model_list()
        self.selected_model = self.model_combo_box.currentText()

        self.classification_radio.toggled.connect(self.on_task_changed)
        self.regression_radio.toggled.connect(self.on_task_changed)

        # === Кнопки конфигурации ===
        btn_select_params = QPushButton("Указать параметры для подбора")
        btn_select_params.clicked.connect(self.open_selection_of_parameters)
        layout.addWidget(btn_select_params)

        btn_configure_search = QPushButton("Настроить условия подбора параметров")
        btn_configure_search.clicked.connect(self.open_selection_parameters_random_search)
        layout.addWidget(btn_configure_search)

        # === Кнопка подбора ===
        self.btn_tune_params = QPushButton("Подобрать лучшие параметры")
        self.btn_tune_params.clicked.connect(self.tune_best_parameters)
        layout.addWidget(self.btn_tune_params)

        self.setLayout(layout)

    def get_task_type(self):
        return "classification" if self.classification_radio.isChecked() else "regression"
    
    def open_selection_of_parameters(self):
        """
        Открывает окно настройки гиперпараметров для подбора
        """
        win = HyperParameterOptimizerGUI()
        win.show()

    def open_selection_parameters_random_search(self):
        """
        Открывает окно настройки параметров RandomizedSearch
        """
        win = RandomSearchConfigGUI()
        win.show()

    def update_model_list(self):
        self.model_combo_box.clear()
        task = self.get_task_type()
        models = (
            ["RandomForestClassifier", "GradientBoostingClassifier", "LinearClassifier"]
            if task == "classification"
            else ["RandomForestRegressor", "GradientBoostingRegressor"]
        )
        self.model_combo_box.addItems(models)
        self.selected_model = self.model_combo_box.currentText()

    @Slot()
    def on_task_changed(self):
        task_type = self.get_task_type()
        new_refit = "f1_macro" if task_type == "classification" else "r2"
        params = get_random_search_params()
        if params.get('refit') != new_refit:
            updated_params = params.copy()
            updated_params['refit'] = new_refit
            save_random_search_params(updated_params)
        self.update_model_list()
        self.selected_model = self.model_combo_box.currentText()

    @Slot(str)
    def on_model_change(self, new_value):
        self.selected_model = new_value

    def choose_dataset(self):
        reply = QMessageBox.question(
            self, "Режим загрузки",
            "Разделить датасет на train и test?\n\n"
            "• Да → загрузить train и test отдельно\n"
            "• Нет → загрузить один датасет, разделю при обучении",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.load_separate_datasets()
        else:
            self.load_single_dataset()

    def load_single_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите датасет", "dataset", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path, comment='#')
            column_names = df.columns.tolist()

            chosen_column, ok = QInputDialog.getItem(
                self, "Целевая переменная", "Выберите целевую переменную:",
                column_names, current=0, editable=False
            )
            if not ok:
                return

            self.target_variable = chosen_column
            self.df = df
            self.train_path = self.test_path = None
            self.df_train = self.df_test = None

            filename = os.path.basename(file_path)
            self.selected_dataset_path = file_path
            self.btn_choose_dataset.setText(f"📁 {filename}")
            QMessageBox.information(self, "Успех", f"Датасет загружен: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка чтения файла: {str(e)}")

    def load_separate_datasets(self):
        train_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите train-файл", "dataset", "CSV Files (*.csv)"
        )
        if not train_path:
            return

        test_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите test-файл", "dataset", "CSV Files (*.csv)"
        )
        if not test_path:
            return

        try:
            # Проверка версий
            if not check_train_test_versions(train_path, test_path, self):
                return

            df_train = pd.read_csv(train_path, comment='#')
            df_test = pd.read_csv(test_path, comment='#')

            # Проверка колонок
            target_col = None
            feature_cols = [c for c in df_train.columns if c != 'Unnamed: 0']
            if not feature_cols:
                QMessageBox.critical(self, "Ошибка", "Нет признаков в train.")
                return

            target_col = feature_cols[-1]  # Предположим, что target — последний
            for col in df_train.columns:
                if col in df_test.columns and col != 'Unnamed: 0':
                    if col != target_col:
                        continue
                    # Проверим тип
                    if df_train[col].dtype != df_test[col].dtype:
                        QMessageBox.critical(self, "Ошибка", f"Колонка '{col}' имеет разные типы в train и test.")
                        return
                    target_col = col
                    break

            if not target_col:
                QMessageBox.critical(self, "Ошибка", "Не удалось определить целевую переменную.")
                return

            self.target_variable = target_col
            self.df_train = df_train
            self.df_test = df_test
            self.train_path = train_path
            self.test_path = test_path
            self.df = None
            self.selected_dataset_path = None

            train_name = os.path.basename(train_path)
            test_name = os.path.basename(test_path)
            self.btn_choose_dataset.setText(f"📁 train: {train_name}\n   test: {test_name}")
            QMessageBox.information(self, "Успех", "Train и test загружены и проверены.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки файлов:\n{str(e)}")

    def format_parameter_value(self, value):
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(str(x) for x in value) + "]"
        elif hasattr(value, 'rvs'):
            return f"scipy.stats.{type(value).__name__}"
        elif isinstance(value, range):
            return f"range({value.start}, {value.stop}, {value.step})"
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

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        try:
            hyperparams = get_random_grid()
            search_params = get_random_search_params()

            # === Параметры случайной сетки ===
            title1 = QLabel("=== Параметры случайной сетки (random_grid) ===")
            title1.setStyleSheet("font-weight: bold; font-size: 14px;")
            layout.addWidget(title1)

            grid_text = ""
            for model_name, model_params in hyperparams.items():
                grid_text += f"<b>{model_name}:</b><br>"
                if isinstance(model_params, dict):
                    for param_key, param_value in model_params.items():
                        formatted_value = self.format_parameter_value(param_value)
                        grid_text += f"&nbsp;&nbsp;&nbsp;• <b>{param_key}:</b> {formatted_value}<br>"
                else:
                    grid_text += f"&nbsp;&nbsp;&nbsp;{self.format_parameter_value(model_params)}<br>"
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

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Закрыть")
        close_btn.setStyleSheet("font-size: 12px; padding: 8px;")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)

        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(button_layout)
        dialog.setLayout(main_layout)
        dialog.exec()

    def tune_best_parameters(self):
        if not self.target_variable:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите целевую переменную!")
            return

        selected_model = self.model_combo_box.currentText()
        if not selected_model:
            QMessageBox.warning(self, "Предупреждение", "Выберите модель!")
            return

        # Передаём либо один df, либо train/test
        self.parameter_window = ParameterTuningWindow(
            parent=None,
            dataset_path=self.selected_dataset_path,
            df=self.df,
            df_train=self.df_train,
            df_test=self.df_test,
            target_variable=self.target_variable,
            chosen_model=selected_model,
            task_type=self.get_task_type()
        )

        self.parameter_window.setGeometry(100, 100, 800, 700)
        self.parameter_window.setWindowModality(Qt.NonModal)
        self.parameter_window.show()
        self.parameter_window.start_tuning()
        QApplication.processEvents()
