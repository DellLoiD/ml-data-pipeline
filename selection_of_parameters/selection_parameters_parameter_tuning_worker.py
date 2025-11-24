"""
Модуль: selection_parameters_parameter_tuning_worker.py
Описание:
    Поток для подбора гиперпараметров модели с помощью RandomizedSearchCV.
    Выполняется в фоне, чтобы не блокировать GUI.
"""
import logging
from PySide6.QtCore import Signal, QThread 
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from .selection_of_parameters_logic import (
    get_random_grid,
    get_random_search_params
)

# Настройки логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParameterTuningWorker(QThread):
    # Сигналы
    # Прогресс (%, текущая итерация, общее число)
    progress_updated = Signal(float, int, int)
    # Подбор завершён: лучшая модель, параметры, точность, строка с метриками
    tuning_completed = Signal(object, dict, float, str)
    # Ошибка
    error_occurred = Signal(str)
    
    def __init__(self, parent=None, dataset_path=None, target_variable=None, model_type=""):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.model_type = model_type
    
    def run(self):
        try:
            logger.info("Начало подбора гиперпараметров...")
            logger.info(f"Модель: {self.model_type}, Датасет: {self.dataset_path}")

            # === 1. Загрузка данных ===
            df = pd.read_csv(self.dataset_path)
            feature_cols = list(df.columns.drop(self.target_variable))
            X = df[feature_cols]
            y = df[self.target_variable]
            logger.info(f"Данные загружены: {X.shape[0]} образцов, {X.shape[1]} признаков")

            # === 2. Получение параметров ===
            random_grid = get_random_grid() or {}
            search_params = get_random_search_params() or {}

            # === 3. Выбор модели ===
            if self.model_type == "RandomForest":
                estimator = RandomForestClassifier()
            elif self.model_type == "GradientBoosting":
                estimator = GradientBoostingClassifier()
            elif self.model_type == "LogisticRegression":
                estimator = LogisticRegression(max_iter=1000)
            else:
                raise ValueError(f"Неправильный тип модели: {self.model_type}")

            hyperparams = random_grid.get(self.model_type)
            if not hyperparams:
                raise ValueError(f"Нет гиперпараметров для модели: {self.model_type}")

            # === 4. Параметры поиска ===
            n_iter_value = search_params.get("n_iter", 100)
            cv = search_params.get("cv", 5)
            verbose = search_params.get("verbose", 0)
            n_jobs = search_params.get("n_jobs", -1)
            random_state = search_params.get("random_state", 42)
            test_size = search_params.get("test_size", 0.2)

            # === 5. Разделение данных ===
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Разделение данных: {len(X_train)} обучение, {len(X_test)} тест")

            # === 6. Настройка RandomizedSearchCV ===
            scoring = search_params.get("scoring", ["accuracy"])
            refit = search_params.get("refit", "accuracy")

            random_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=hyperparams,
                n_iter=n_iter_value,
                cv=cv,
                scoring=scoring,
                refit=refit,
                random_state=random_state,
                verbose=max(1, verbose),  # чтобы видеть ход в логах
                n_jobs=n_jobs
            )

            # === 7. Запуск подбора ===
            # Отправляем начальный прогресс (0%) — индикатор активности
            self.progress_updated.emit(0.0, 0, n_iter_value)

            logger.info("Запуск RandomizedSearchCV...")
            random_search.fit(X_train, y_train)
            logger.info("RandomizedSearchCV завершён")

            # Отправляем "завершено"
            self.progress_updated.emit(100.0, n_iter_value, n_iter_value)

            # === 8. Оценка модели ===
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1_macro = f1_score(y_test, predictions, average='macro')
            precision_macro = precision_score(y_test, predictions, average='macro')
            recall_macro = recall_score(y_test, predictions, average='macro')

            metrics_str = (
                f"\nAccuracy: {accuracy:.4f}\n"
                f"F1 Score (Macro): {f1_macro:.4f}\n"
                f"Precision (Macro): {precision_macro:.4f}\n"
                f"Recall (Macro): {recall_macro:.4f}"
            )

            logger.info(f"Лучшие параметры найдены. Accuracy: {accuracy:.4f}")

            # === 9. Отправка результата ===
            self.tuning_completed.emit(best_model, best_params, accuracy, metrics_str)

        except Exception as e:
            logger.error(f"Ошибка в ParameterTuningWorker.run: {str(e)}")
            logger.error(f"Тип исключения: {type(e).__name__}")
            import traceback
            logger.error(f"След исключения:\n{traceback.format_exc()}")
            self.error_occurred.emit(str(e))
            # Не забываем, что сигнал error_occurred должен быть обработан в GUI
