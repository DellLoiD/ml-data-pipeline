from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from pprint import pformat
import logging
import os

# Импорт логики и воркера
from .selection_of_parameters_logic import get_random_grid, get_random_search_params
from .selection_parameters_parameter_tuning_worker import ParameterTuningWorker

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ParameterTuningWindow(QWidget):
    def __init__(self, parent=None, dataset_path=None, target_variable=None, chosen_model=None):
        """Инициализация окна выбора параметров"""
        logger.info(f"ParameterTuningWindow.__init__ вызван с dataset_path: {dataset_path}")
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.best_model = None
        self.best_params = None
        self.accuracy = None
        self.chosen_model = chosen_model
        self.worker = None  # Сохраняем ссылку на поток
        self.initUI()
        
    def format_param_value(self, value):
        """Форматирует значение для красивого отображения"""
        if hasattr(value, 'rvs'):  # Это объект scipy.stats (например, loguniform)
            dist_name = type(value.dist).__name__ if hasattr(value, 'dist') else type(value).__name__
            try:
                args = [f"{x:.3e}" if isinstance(x, float) else str(x) for x in value.args]
                return f"{dist_name}({', '.join(args)})"
            except Exception:
                return f"{dist_name}(...)"
        
        elif isinstance(value, (list, tuple)):
            items = [self.format_param_value(x) for x in value]
            return "[" + ", ".join(items) + "]"
        
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
        self.setWindowTitle("Выбор параметров")
        self.setGeometry(300, 300, 700, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        main_layout = QVBoxLayout()

        # Заголовок
        title = QLabel("Текущие параметры подбора")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # Горизонтальный блок: параметры модели и поиска
        horizontal_layout = QHBoxLayout()

        # === Левый блок: Гиперпараметры выбранной модели ===
        self.hyperparameters_label = QLabel(f"Гиперпараметры модели: {self.chosen_model}")
        self.hyperparameters_label.setStyleSheet("font-weight: bold;")
        hyperparameters_text = QLabel()
        hyperparameters_text.setWordWrap(True)
        hyperparameters_text.setFont(QFont("Courier", 10))

        # Получаем только параметры выбранной модели
        grid = get_random_grid()
        model_params = grid.get(self.chosen_model, {})

        formatted_grid = ""
        if model_params:
            for k, v in model_params.items():
                formatted_grid += f"{k}: {self.format_param_value(v)}\n"
        else:
            formatted_grid = f"Нет параметров для {self.chosen_model}"

        hyperparameters_text.setText(formatted_grid)

        vertical_left_layout = QVBoxLayout()
        vertical_left_layout.addWidget(self.hyperparameters_label)
        vertical_left_layout.addWidget(hyperparameters_text)
        horizontal_layout.addLayout(vertical_left_layout)

        # === Правый блок: Параметры случайного поиска ===
        self.random_search_params_label = QLabel("Параметры случайного поиска:")
        self.random_search_params_label.setStyleSheet("font-weight: bold;")
        random_search_params_text = QLabel()
        random_search_params_text.setWordWrap(True)
        random_search_params_text.setFont(QFont("Courier", 10))

        search_params = get_random_search_params()
        formatted_search = ""
        for k, v in search_params.items():
            formatted_search += f"{k}: {self.format_param_value(v)}\n"
        random_search_params_text.setText(formatted_search)

        vertical_right_layout = QVBoxLayout()
        vertical_right_layout.addWidget(self.random_search_params_label)
        vertical_right_layout.addWidget(random_search_params_text)
        horizontal_layout.addLayout(vertical_right_layout)

        main_layout.addLayout(horizontal_layout)

        # Разделитель
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator_line)

        # === Прогресс и результаты ===
        # Сообщение о ходе выполнения
        self.completion_label = QLabel("Подождите, идёт обучение модели...")
        self.completion_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0066cc; margin: 10px 0;")
        self.completion_label.setVisible(False)
        main_layout.addWidget(self.completion_label)

        # Прогресс-бар (неопределённый режим)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Неопределённый режим
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Подбор параметров... Пожалуйста, подождите")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # Результаты
        self.results_title = QLabel("Результаты оптимизации")
        self.results_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        self.results_title.setVisible(False)
        main_layout.addWidget(self.results_title)

        self.accuracy_label = QLabel("")
        self.accuracy_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E8B57; margin: 10px 0;")
        self.accuracy_label.setVisible(False)
        main_layout.addWidget(self.accuracy_label)
        # Метрики (F1, Precision, Recall и др.)
        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet("font-size: 14px; margin: 10px 0; color: #333;")
        self.metrics_label.setVisible(False)
        self.metrics_label.setFont(QFont("Courier", 12))  # моноширинный шрифт — для выравнивания
        main_layout.addWidget(self.metrics_label)

        self.params_title = QLabel("Лучшие параметры:")
        self.params_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        self.params_title.setVisible(False)
        main_layout.addWidget(self.params_title)

        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_container.setLayout(self.params_layout)
        self.params_container.setVisible(False)
        main_layout.addWidget(self.params_container)

        main_layout.addStretch()

        # Кнопка сохранения модели
        self.save_button = QPushButton("Сохранить лучшую модель")
        self.save_button.clicked.connect(self.save_best_model)
        self.save_button.setVisible(False)
        self.save_button.setStyleSheet(
            "font-size: 12px; padding: 10px; background-color: #4CAF50; color: white; "
            "border: none; border-radius: 5px;"
        )
        main_layout.addWidget(self.save_button)

        # Устанавливаем макет
        self.setLayout(main_layout)
        self.setVisible(True)

    def start_tuning(self):
        """Запуск подбора параметров в фоновом потоке"""
        self.worker = ParameterTuningWorker(
            dataset_path=self.dataset_path,
            target_variable=self.target_variable,
            model_type=self.chosen_model,
            parent=self
        )

        # Подключаем сигналы
        self.worker.tuning_completed.connect(self.on_tuning_completed)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.progress_updated.connect(self.on_progress_update)

        # Показываем индикатор прогресса
        self.completion_label.setVisible(True)
        self.completion_label.setText("Идёт подбор параметров...")
        self.progress_bar.setVisible(True)

        # Запускаем поток
        self.worker.start()

    @Slot(float, int, int)
    def on_progress_update(self, progress: float, current: int, total: int):
        """Обновление текста прогресс-бара (даже в неопределённом режиме)"""
        # Можно использовать для логирования или обновления текста
        self.progress_bar.setFormat(f"Подбор параметров... {current}/{total} итераций")

    @Slot(object, dict, float, str)
    def on_tuning_completed(self, best_model, best_params, accuracy, metrics_str):
        """Обработчик успешного завершения подбора"""
        self.best_model = best_model
        self.best_params = best_params
        self.accuracy = accuracy

        # Обновляем интерфейс
        self.completion_label.setText("✅ Настройка параметров завершена!")
        self.completion_label.setStyleSheet("color: green;")
        self.progress_bar.setVisible(False)

        # Показываем результаты
        self.results_title.setVisible(True)
        self.accuracy_label.setText(f"Точность наилучшей модели: {accuracy:.4f}")
        self.accuracy_label.setVisible(True)

        # === Добавляем отображение всех метрик ===
        self.metrics_label.setText(f"<pre>{metrics_str.strip()}</pre>")
        self.metrics_label.setVisible(True)  # ← новая строка

        self.params_title.setVisible(True)
        self.params_container.setVisible(True)

    @Slot(str)
    def on_error_occurred(self, error_msg: str):
        """Обработчик ошибки"""
        self.completion_label.setText(f"❌ Ошибка: {error_msg}")
        self.completion_label.setStyleSheet("color: red;")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Ошибка", f"Подбор параметров прерван:\n{error_msg}")

    def save_best_model(self):
        """Сохранение обученной модели"""
        if not self.best_model:
            QMessageBox.warning(self, "Предупреждение", "Нет обученной модели для сохранения!")
            return

        try:
            models_dir = "trained_models"
            os.makedirs(models_dir, exist_ok=True)

            dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
            model_name = self.chosen_model.lower().replace(" ", "_")
            accuracy_str = f"{self.accuracy:.2f}".replace('.', '_')
            filename = f"{model_name}_{dataset_name}_acc_{accuracy_str}_percent.pkl"
            file_path = os.path.join(models_dir, filename)

            import joblib
            joblib.dump(self.best_model, file_path)

            QMessageBox.information(self, "Успех", f"Модель сохранена как:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить модель:\n{str(e)}")
