# selection_parameters_parameter_tuning_window.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QApplication,
    QPushButton, QProgressBar, QMessageBox, QToolButton
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont
import logging
import os
import re
import joblib

from .selection_of_parameters_logic import get_random_grid, get_random_search_params
from .selection_parameters_parameter_tuning_worker import ParameterTuningWorker
from .waiting_dialog_stop_worker import WaitingDialog

# ✅ Импорт справок
from .metrics_help import METRICS_DESCRIPTIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ParameterTuningWindow(QWidget):
    def __init__(self, parent=None, dataset_path=None, target_variable=None, chosen_model=None, task_type="classification"):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.chosen_model = chosen_model
        self.task_type = task_type

        self.best_model = None
        self.best_params = None
        self.accuracy = None
        self.metrics_text = ""
        self.primary_metric = None
        self.primary_metric_name = None
        self.worker = None

        self.setAttribute(Qt.WA_DeleteOnClose, True)
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
        self.setGeometry(300, 300, 900, 700)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

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
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # === КНОПКА "ПРЕРВАТЬ ОБУЧЕНИЕ" ===
        self.cancel_button = QPushButton("🛑 Прервать обучение")
        self.cancel_button.setStyleSheet("""
            font-size: 14px; 
            padding: 10px;
            background-color: #d32f2f; 
            color: white;
            border: none; 
            border-radius: 6px;
        """)
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_tuning)
        main_layout.addWidget(self.cancel_button)

        self.results_title = QLabel("Результаты")
        self.results_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        self.results_title.setVisible(False)
        main_layout.addWidget(self.results_title)

        # === КОНТЕЙНЕР МЕТРИК ===
        self.metrics_container = QWidget()
        self.metrics_layout = QVBoxLayout()
        self.metrics_container.setLayout(self.metrics_layout)
        self.metrics_container.setVisible(False)
        main_layout.addWidget(self.metrics_container)

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
            border: none; border-radius: 6px;
        """)
        main_layout.addWidget(self.save_button)

        self.setLayout(main_layout)
        self.setVisible(True)
        self.start_tuning()

    def cancel_tuning(self):
        if self.worker and self.worker.isRunning():
            logger.info("Пользователь нажал 'Прервать обучение'")

            self.status_label.setText("🛑 Прерывание...")
            self.status_label.setStyleSheet("color: #FF6B6B;")
            self.progress_bar.setVisible(False)
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("⛔ Прерывается...")

            self.wait_dialog = WaitingDialog(self)
            self.wait_dialog.show()
            QApplication.processEvents()

            self.worker.terminate()

            self.check_worker_timer = QTimer()
            self.check_worker_timer.setInterval(200)
            self.check_worker_timer.timeout.connect(self.check_worker_stopped)
            self.check_worker_timer.start()

    def check_worker_stopped(self):
        if not self.worker.isRunning():
            self.check_worker_timer.stop()
            self.delay_timer = QTimer()
            self.delay_timer.setSingleShot(True)
            self.delay_timer.timeout.connect(self.on_worker_fully_stopped)
            self.delay_timer.start(24000)

    @Slot()
    def on_worker_fully_stopped(self):
        if hasattr(self, 'wait_dialog'):
            self.wait_dialog.accept()

        if self.worker:
            self.worker.deleteLater()
            self.worker = None

        self.status_label.setText("🛑 Обучение прервано")
        self.cancel_button.setVisible(False)

        QMessageBox.information(self, "Прервано", "Обучение было прервано пользователем.")

    def start_tuning(self):
        if self.worker and self.worker.isRunning():
            logger.warning("Подбор уже запущен — пропуск")
            return

        self.worker = ParameterTuningWorker(
            dataset_path=self.dataset_path,
            target_variable=self.target_variable,
            model_type=self.chosen_model,
            task_type=self.task_type
        )

        self.worker.tuning_completed.connect(self.on_tuning_completed)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.progress_updated.connect(self.on_progress_update)

        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("🛑 Прервать обучение")

        self.status_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.worker.start()

    @Slot(float, int, int)
    def on_progress_update(self, progress: float, current: int, total: int):
        self.progress_bar.setFormat(f"Итерация: {current}/{total}")

    @Slot(object, dict, float, str)
    def on_tuning_completed(self, best_model, best_params, accuracy, metrics_str):
        self.best_model = best_model
        self.best_params = best_params
        self.accuracy = accuracy
        self.metrics_text = metrics_str

        search_params = get_random_search_params()
        refit_key = search_params.get('refit', 'accuracy')

        metric_key_map = {
            'accuracy': 'Accuracy',
            'f1_macro': 'F1 Score \(Macro\)',
            'precision_macro': 'Precision \(Macro\)',
            'recall_macro': 'Recall \(Macro\)',
            'roc_auc': 'ROC AUC',
            'r2': 'R² Score',
            'neg_mean_squared_error': 'Mean Squared Error'
        }

        pattern = metric_key_map.get(refit_key, refit_key.replace('_', ' ').title())
        match = re.search(rf"{pattern}:\s*([0-9.]+)", metrics_str)
        primary_metric_value = float(match.group(1)) if match else accuracy

        self.primary_metric = primary_metric_value
        self.primary_metric_name = refit_key

        # === ОБНОВЛЕНИЕ МЕТРИК ===
        self.status_label.setText("✅ Обучение завершено!")
        self.status_label.setStyleSheet("color: green;")
        self.progress_bar.setVisible(False)
        self.results_title.setVisible(True)
        self.metrics_container.setVisible(True)
        self.params_title.setVisible(True)
        self.params_container.setVisible(True)
        self.cancel_button.setVisible(False)

        # Очистка предыдущих метрик
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
            else:
                layout = item.layout()
                if layout:
                    while layout.count():
                        child_item = layout.takeAt(0)
                        w = child_item.widget()
                        if w:
                            w.setParent(None)

        # Парсим строки метрик
        lines = metrics_str.strip().split('\n')
        for line in lines:
            if ":" not in line:
                continue
            key_part, value = line.split(":", 1)
            key_part = key_part.strip()
            value = value.strip()

            # Поиск ключа метрики
            metric_key = self._find_matching_metric_key(key_part.lower())
            if metric_key and metric_key in METRICS_DESCRIPTIONS:
                desc = METRICS_DESCRIPTIONS[metric_key]
                label_text = f"<b>{key_part}:</b> {value}"
                tooltip = f"Нажмите, чтобы узнать о метрике: {desc['title']}"
                show_help = True
            else:
                label_text = f"{key_part}: {value}"
                show_help = False

            # Горизонтальный layout для строки
            row_layout = QHBoxLayout()
            row_layout.setSpacing(8)
            row_layout.setContentsMargins(0, 2, 0, 2)

            # Метка метрики
            label = QLabel(label_text)
            label.setTextFormat(Qt.RichText)
            row_layout.addWidget(label)

            # Кнопка помощи
            if show_help:
                help_btn = QToolButton()
                help_btn.setText("?")
                help_btn.setFixedSize(20, 20)
                help_btn.setStyleSheet("QToolButton { font: bold; border-radius: 10px; background: #e0e0e0; }")
                help_btn.setToolTip(tooltip)
                # ✅ Сохраняем desc в замыкании
                help_btn.clicked.connect(lambda checked=False, d=desc: self.show_metric_help(d))
                row_layout.addWidget(help_btn)
            else:
                # Пустое пространство для выравнивания
                row_layout.addSpacing(20)

            row_layout.addStretch()
            self.metrics_layout.addLayout(row_layout)

        # Лучшие параметры
        self.params_layout.addWidget(QLabel(f"<b>Модель:</b> {self.chosen_model}"))
        for key, value in best_params.items():
            self.params_layout.addWidget(QLabel(f"<b>{key}:</b> {self.format_param_value(value)}"))

        self.save_button.setVisible(True)

        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def _find_matching_metric_key(self, text: str) -> str:
        """Находит ключ метрики по названию"""
        text = text.lower().strip()
        # Прямо сопоставляем
        for key, desc in METRICS_DESCRIPTIONS.items():
            if key in text:
                return key
        # По алиасам
        mapping = {
            'accuracy': ['accuracy', 'точность'],
            'f1_macro': ['f1', 'f1 score', 'ф1', 'ф1-мера'],
            'precision_macro': ['precision', 'точность precision'],
            'recall_macro': ['recall', 'полнота'],
            'roc_auc': ['roc', 'roc auc', 'auroc'],
            'r2': ['r2', 'r²', 'коэффициент детерминации'],
            'neg_mean_squared_error': ['mean squared error', 'mse', 'средний квадрат ошибки'],
            'neg_mean_absolute_error': ['mean absolute error', 'mae', 'среднее абсолютное отклонение'],
            'explained_variance': ['explained variance', 'объяснённая дисперсия']
        }
        for key, aliases in mapping.items():
            if any(alias in text for alias in aliases):
                return key
        return None

    def show_metric_help(self, desc: dict):
        """Показывает справку по метрике"""
        QMessageBox.information(self, desc["title"], desc["text"])

    @Slot(str)
    def on_error_occurred(self, error_msg: str):
        self.status_label.setText(f"❌ Ошибка: {error_msg}")
        self.status_label.setStyleSheet("color: red;")
        self.progress_bar.setVisible(False)

        if self.worker:
            self.worker.deleteLater()
            self.worker = None

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

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Подождите", "Обучение ещё идёт. Нельзя закрыть окно до завершения.")
            event.ignore()
        else:
            event.accept()
