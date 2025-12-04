# selection_parameters_parameter_tuning_window.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QProgressBar, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from pprint import pformat
import logging
import os
import re
import joblib

from .selection_of_parameters_logic import get_random_grid, get_random_search_params
from .selection_parameters_parameter_tuning_worker import ParameterTuningWorker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ParameterTuningWindow(QWidget):
    def __init__(self, parent=None, dataset_path=None, target_variable=None, chosen_model=None):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.chosen_model = chosen_model
        self.best_model = None
        self.best_params = None
        self.accuracy = None
        self.metrics_text = ""
        self.worker = None
        self.initUI()

    def format_param_value(self, value):
        if hasattr(value, 'rvs'):
            dist_name = type(value.dist).__name__
            try:
                args = [f"{x:.3e}" if isinstance(x, float) else str(x) for x in value.args]
                return f"{dist_name}({', '.join(args)})"
            except Exception:
                return f"{dist_name}(...)"
        elif isinstance(value, (list, tuple)):
            return "[" + ", ".join([self.format_param_value(x) for x in value]) + "]"
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, range):
            return f"range({value.start}, {value.stop}, {value.step})"
        elif value is None:
            return "None"
        elif isinstance(value, bool):
            return "True" if value else "False"
        else:
            return str(value)

    def initUI(self):
        self.setWindowTitle("Подбор гиперпараметров")
        self.setGeometry(300, 300, 800, 700)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        main_layout = QVBoxLayout()

        title = QLabel("Настройка гиперпараметров")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        self.model_name_label = QLabel(f"<b>Модель:</b> {self.chosen_model}")
        self.model_name_label.setStyleSheet("font-size: 16px; color: #1E90FF;")
        main_layout.addWidget(self.model_name_label)

        h_layout = QHBoxLayout()

        # Параметры модели
        grid = get_random_grid()
        model_params = grid.get(self.chosen_model, {})
        model_text = "\n".join([f"{k}: {self.format_param_value(v)}" for k, v in model_params.items()])
        model_label = QLabel("<b>Гиперпараметры модели:</b>")
        model_value = QLabel(model_text)
        model_value.setWordWrap(True)
        model_value.setFont(QFont("Courier", 10))
        left_layout = QVBoxLayout()
        left_layout.addWidget(model_label)
        left_layout.addWidget(model_value)
        h_layout.addLayout(left_layout)

        # Параметры поиска
        search_params = get_random_search_params()
        search_text = "\n".join([f"{k}: {self.format_param_value(v)}" for k, v in search_params.items()])
        search_label = QLabel("<b>Параметры поиска:</b>")
        search_value = QLabel(search_text)
        search_value.setWordWrap(True)
        search_value.setFont(QFont("Courier", 10))
        right_layout = QVBoxLayout()
        right_layout.addWidget(search_label)
        right_layout.addWidget(search_value)
        h_layout.addLayout(right_layout)

        main_layout.addLayout(h_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        self.status_label = QLabel("Идёт обучение модели...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0066cc;")
        self.status_label.setVisible(False)
        main_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Поиск лучших параметров...")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                height: 20px;}
            QProgressBar::chunk {
                background-color: #05B8CC;}""")
        main_layout.addWidget(self.progress_bar)

        self.results_title = QLabel("Результаты")
        self.results_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        self.results_title.setVisible(False)
        main_layout.addWidget(self.results_title)

        self.metrics_label = QLabel("")
        self.metrics_label.setFont(QFont("Courier", 12))
        self.metrics_label.setStyleSheet("color: #333;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.metrics_label)
        scroll.setVisible(False)
        main_layout.addWidget(scroll)
        self.metrics_scroll = scroll

        self.params_title = QLabel("Лучшие параметры модели:")
        self.params_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        self.params_title.setVisible(False)
        main_layout.addWidget(self.params_title)

        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_container.setLayout(self.params_layout)
        self.params_container.setVisible(False)
        main_layout.addWidget(self.params_container)

        main_layout.addStretch()

        self.save_button = QPushButton("💾 Сохранить лучшую модель")
        self.save_button.clicked.connect(self.save_best_model)
        self.save_button.setVisible(False)
        self.save_button.setStyleSheet("""
            font-size: 14px; padding: 12px;
            background-color: #4CAF50; color: white;
            border: none; border-radius: 6px;""")
        main_layout.addWidget(self.save_button)

        self.setLayout(main_layout)
        self.setVisible(True)
        self.start_tuning()

    def start_tuning(self):
        self.worker = ParameterTuningWorker(
            dataset_path=self.dataset_path,
            target_variable=self.target_variable,
            model_type=self.chosen_model,
            parent=self)
        self.worker.tuning_completed.connect(self.on_tuning_completed)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.progress_updated.connect(self.on_progress_update)
        self.status_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.worker.start()

    @Slot(float, int, int)
    def on_progress_update(self, progress: float, current: int, total: int):
        self.progress_bar.setFormat(f"Итерация: {current}/{total}")

    @Slot(object, dict, float, str)
    def on_tuning_completed(self, best_model, best_params, accuracy, metrics_str):
        """Обработчик завершения подбора"""
        self.best_model = best_model
        self.best_params = best_params
        self.accuracy = accuracy
        self.metrics_text = metrics_str

        # === ИСПРАВЛЕНИЕ: Используем refit из logic.py, а не жёстко roc_auc ===
        search_params = get_random_search_params()
        refit_key = search_params.get('refit', 'accuracy')  # например, 'f1_macro', 'roc_auc'

        # Маппинг refit-ключей → строки в выводе
        metric_key_map = {
            'accuracy': 'Accuracy',
            'f1_macro': 'F1 Score \(Macro\)',
            'precision_macro': 'Precision \(Macro\)',
            'recall_macro': 'Recall \(Macro\)',
            'roc_auc': 'ROC AUC'
            # Добавьте другие, если нужно
        }

        # Извлекаем значение выбранной метрики
        display_name = metric_key_map.get(refit_key, refit_key.replace('_', ' ').title())
        pattern = metric_key_map.get(refit_key, refit_key)

        match = re.search(rf"{pattern}:\s*([0-9.]+)", metrics_str)
        primary_metric_value = float(match.group(1)) if match else accuracy

        # Сохраняем
        self.primary_metric = primary_metric_value
        self.primary_metric_name = refit_key  # ✅ Теперь имя метрики — из refit

        # === Обновление интерфейса ===
        self.status_label.setText("✅ Обучение завершено!")
        self.status_label.setStyleSheet("color: green;")
        self.progress_bar.setVisible(False)
        self.results_title.setVisible(True)
        self.metrics_label.setText(f"<pre>{metrics_str.strip()}</pre>")
        self.metrics_scroll.setVisible(True)
        self.params_title.setVisible(True)
        self.params_container.setVisible(True)

        self.params_layout.addWidget(QLabel(f"<b>Модель:</b> {self.chosen_model}"))
        for key, value in best_params.items():
            self.params_layout.addWidget(QLabel(f"<b>{key}:</b> {self.format_param_value(value)}"))

        self.save_button.setVisible(True)


    @Slot(str)
    def on_error_occurred(self, error_msg: str):
        self.status_label.setText(f"❌ Ошибка: {error_msg}")
        self.status_label.setStyleSheet("color: red;")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Ошибка", f"Подбор параметров прерван:\n{error_msg}")

    def save_best_model(self):
        if not self.best_model:
            QMessageBox.warning(self, "Предупреждение", "Нет обученной модели для сохранения!")
            return
        try:
            models_dir = "trained_models"
            os.makedirs(models_dir, exist_ok=True)
            dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
            model_name = self.chosen_model.lower().replace(" ", "_")
            metric_value = f"{self.primary_metric:.4f}".replace('.', '_') if self.primary_metric else "unknown"
            filename = f"{model_name}_{dataset_name}_{self.primary_metric_name}_{metric_value}.pkl"
            file_path = os.path.join(models_dir, filename)
            joblib.dump(self.best_model, file_path)
            QMessageBox.information(self, "Успех", f"Модель сохранена:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить модель:\n{str(e)}")
