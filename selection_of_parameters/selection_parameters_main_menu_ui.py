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
from .selection_of_parameters_logic import get_random_grid, get_random_search_params, save_random_search_params
from .selection_parameters_parameter_tuning_window import ParameterTuningWindow

class MainWindow_selection_parameters(QWidget):
    def __init__(self):
        super().__init__()
        self.parameter_window = None
        self.selected_dataset_path = None
        self.target_variable = None
        self.selected_model = ""
        logger.info("Инициализирован MainWindow_selection_parameters")
        # === 🔥 Показываем диалог выбора типа задачи СРАЗУ при открытии ===
        task, ok = QInputDialog.getItem(
            self, "Тип задачи", "Выберите тип задачи:",
            ["Классификация", "Регрессия"],
            current=0,  # по умолчанию — классификация
            editable=False
        )
        if not ok:
            # Если отменили — закрываем окно
            logger.warning("Пользователь отменил выбор типа задачи. Завершение.")
            # Можно и завершить, но лучше дать продолжить
            task = "Классификация"

        # Устанавливаем refit в зависимости от выбора
        selected_task = "classification" if task == "Классификация" else "regression"
        self._set_refit_for_task(selected_task)
        
        # Теперь инициализируем интерфейс
        self.initUI()

        # Устанавливаем правильный тип задачи в UI
        if selected_task == "classification":
            self.classification_radio.setChecked(True)
        else:
            self.regression_radio.setChecked(True)
        # Обновляем список моделей
        self.update_model_list()
        
    def _set_refit_for_task(self, task_type):
        """Устанавливает правильный refit в зависимости от типа задачи"""
        params = get_random_search_params()
        new_refit = "f1_macro" if task_type == "classification" else "r2"

        # Обновляем только refit, остальное без изменений
        if params.get('refit') != new_refit:
            updated_params = params.copy()
            updated_params['refit'] = new_refit
            save_random_search_params(updated_params)
            logger.info(f"[INIT] refit обновлён на: {new_refit} (для {task_type})")
        
    def initUI(self):
        self.setWindowTitle("Настройка параметров моделей")
        layout = QVBoxLayout()

        # === Тип задачи: Классификация / Регрессия ===
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

        # === Кнопка "Показать текущие параметры" ===
        btn_show_params = QPushButton("Показать текущие параметры")
        btn_show_params.clicked.connect(self.show_current_parameters)
        layout.addWidget(btn_show_params)

        # === Выбор модели ===
        label_model_choice = QLabel("Выбор модели:", font=QFont('Arial', 12))
        layout.addWidget(label_model_choice)

        self.model_combo_box = QComboBox()
        self.model_combo_box.currentTextChanged.connect(self.on_model_change)
        layout.addWidget(self.model_combo_box)

        # Инициализируем список моделей
        self.update_model_list()
        self.selected_model = self.model_combo_box.currentText()

        # Подключаем сигналы переключения задачи
        self.classification_radio.toggled.connect(self.on_task_changed)
        self.regression_radio.toggled.connect(self.on_task_changed)

        # === Кнопки конфигурации ===
        btn_select_params = QPushButton("Указать параметры для подбора")
        btn_select_params.clicked.connect(self.open_selection_of_parameters)
        layout.addWidget(btn_select_params)

        btn_configure_search = QPushButton("Настроить условия подбора параметров")
        btn_configure_search.clicked.connect(self.open_selection_parameters_random_search)
        layout.addWidget(btn_configure_search)

        # === Кнопка запуска обучения ===
        self.btn_tune_params = QPushButton("Подобрать лучшие параметры")
        self.btn_tune_params.clicked.connect(self.tune_best_parameters)
        layout.addWidget(self.btn_tune_params)

        self.setLayout(layout)

    def update_model_list(self):
        """Обновляет список моделей в зависимости от типа задачи"""
        self.model_combo_box.clear()
        task = self.get_task_type()

        if task == "classification":
            models = ["RandomForestClassifier", "GradientBoostingClassifier", "LinearClassifier"]
        else:  # regression
            models = ["RandomForestRegressor", "GradientBoostingRegressor"]

        self.model_combo_box.addItems(models)
        self.selected_model = self.model_combo_box.currentText()

    def get_task_type(self):
        """Возвращает тип задачи: 'classification' или 'regression'"""
        return "classification" if self.classification_radio.isChecked() else "regression"

    @Slot()
    def on_task_changed(self):
        """Обновление списка моделей и параметров при смене типа задачи"""
        task_type = self.get_task_type()
        
        # ✅ Обновляем refit в зависимости от задачи
        params = get_random_search_params()
        new_refit = "f1_macro" if task_type == "classification" else "r2"
        
        if params.get('refit') != new_refit:
            # Обновляем глобальный параметр
            updated_params = params.copy()
            updated_params['refit'] = new_refit
            save_random_search_params(updated_params)
            logger.info(f"refit обновлён на: {new_refit} (для {task_type})")

        self.update_model_list()
        self.selected_model = self.model_combo_box.currentText()

    @Slot(str)
    def on_model_change(self, new_value):
        """Сохраняет выбранную модель"""
        self.selected_model = new_value
        logger.info(f"Выбрана модель: {new_value}")

    def open_selection_of_parameters(self):
        win = HyperParameterOptimizerGUI()
        win.show()

    def open_selection_parameters_random_search(self):
        win = RandomSearchConfigGUI()
        win.show()

    def choose_dataset(self):
        dataset_folder = "dataset"
        if not os.path.exists(dataset_folder):
            QMessageBox.warning(self, "Предупреждение", f"Папка '{dataset_folder}' не найдена!")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите датасет", dataset_folder, "Файлы CSV (*.csv);;Все файлы (*)"
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            column_names = df.columns.tolist()

            chosen_column, ok_pressed = QInputDialog.getItem(
                self, "Выбор целевой переменной", "Выберите целевую переменную:",
                column_names, current=0, editable=False
            )

            if ok_pressed:
                self.target_variable = chosen_column
                QMessageBox.information(self, "Успех", f"Целевая переменная: {chosen_column}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка чтения файла: {str(e)}")
            return

        self.selected_dataset_path = file_path
        filename = os.path.basename(file_path)
        self.btn_choose_dataset.setText(f"✅ {filename}")
        QMessageBox.information(self, "Успех", f"Датасет загружен: {filename}")

    def format_parameter_value(self, value):
        if isinstance(value, (list, tuple)):
            items = [str(x) for x in value]
            return "[" + ", ".join(items) + "]"
        elif hasattr(value, 'rvs'):
            return f"scipy.stats.{type(value).__name__} (distribution)"
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

        # Кнопка "Закрыть"
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
        """Запуск подбора параметров с учётом типа задачи"""
        if not self.selected_dataset_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите датасет!")
            return

        if not self.target_variable:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите целевую переменную!")
            return

        selected_model = self.model_combo_box.currentText()
        if not selected_model:
            QMessageBox.warning(self, "Предупреждение", "Выберите модель!")
            return

        # Передаём в окно и тип задачи
        self.parameter_window = ParameterTuningWindow(
            parent=None,
            dataset_path=self.selected_dataset_path,
            target_variable=self.target_variable,
            chosen_model=selected_model,
            task_type=self.get_task_type()  # ✅ Передаём тип задачи
        )

        self.parameter_window.setGeometry(100, 100, 800, 700)
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
    sys.exit(app.exec())
