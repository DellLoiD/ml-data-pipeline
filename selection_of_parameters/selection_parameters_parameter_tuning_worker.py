# selection_parameters_parameter_tuning_worker.py
import logging
from PySide6.QtCore import Signal, QThread
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
)
from sklearn.preprocessing import LabelEncoder
from .selection_of_parameters_logic import get_random_grid, get_random_search_params

logger = logging.getLogger(__name__)


class ParameterTuningWorker(QThread):
    progress_updated = Signal(float, int, int)
    tuning_completed = Signal(object, dict, float, str)
    error_occurred = Signal(str)
    info_message = Signal(str)  # ✅ Новый сигнал для уведомлений

    def __init__(self, parent=None, dataset_path=None, target_variable=None, model_type="", task_type="classification"):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.model_type = model_type
        self.task_type = task_type
        self._is_running = False

    def run(self):
        if self._is_running:
            logger.warning("ParameterTuningWorker уже запущен — пропуск.")
            return
        self._is_running = True

        try:
            logger.info("=== Запуск подбора гиперпараметров ===")

            # === 1. Загрузка данных ===
            df = pd.read_csv(self.dataset_path)
            if self.target_variable not in df.columns:
                raise ValueError(f"Целевая переменная '{self.target_variable}' не найдена")

            X = df.drop(columns=[self.target_variable])
            y = df[self.target_variable].copy()

            # === 🔎 ОСТАВЛЯЕМ ТОЛЬКО ЧИСЛОВЫЕ ПРИЗНАКИ ===
            X_numeric = X.select_dtypes(include=['number'])
            dropped_columns = X.columns.difference(X_numeric.columns).tolist()

            if dropped_columns:
                msg = f"Пропущены нечисловые признаки: {', '.join(dropped_columns)}"
                self.info_message.emit(msg)
                logger.info(msg)
            else:
                self.info_message.emit("Все признаки — числовые.")
                logger.info("Нечисловые признаки не найдены.")

            X = X_numeric  # ✅ Только числа

            # Определение типа задачи
            if self.task_type == "classification":
                is_classification = True
            elif self.task_type == "regression":
                is_classification = False
            else:
                is_classification = (
                    y.dtype == 'object' or
                    y.nunique() < 20 or
                    self.model_type in ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression"]
                )
            task_type = "classification" if is_classification else "regression"

            # Кодируем y при необходимости
            if is_classification and y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            # Загрузка параметров
            params = get_random_search_params()
            grid = get_random_grid()
            hyperparams = grid.get(self.model_type)
            if not hyperparams:
                raise ValueError(f"Нет гиперпараметров для: {self.model_type}")

            # Создание модели
            model_classes = {
                'RandomForestClassifier': RandomForestClassifier,
                'GradientBoostingClassifier': GradientBoostingClassifier,
                'LogisticRegression': LogisticRegression,
                'RandomForestRegressor': RandomForestRegressor,
                'GradientBoostingRegressor': GradientBoostingRegressor,
            }
            model_cls = model_classes.get(self.model_type)
            if not model_cls:
                raise ValueError(f"Неподдерживаемая модель: {self.model_type}")
            estimator = model_cls(random_state=params['random_state'])

            # Метрики
            if task_type == "classification":
                scoring = params['scoring']
                refit = params['refit']
                n_classes = len(pd.unique(y))
                if n_classes > 2:
                    scoring = {name: f'roc_auc_ovr' if name == 'roc_auc' else metric for name, metric in scoring.items()}
            else:
                scoring = params.get('scoring_regression', {
                    'r2': 'r2',
                    'neg_mean_squared_error': 'neg_mean_squared_error',
                    'neg_mean_absolute_error': 'neg_mean_absolute_error'
                })
                refit = params['refit']

            # Обучение
            n_iter = params['n_iter']
            cv = params['cv']
            verbose = params['verbose']
            n_jobs = params['n_jobs']
            random_state = params['random_state']
            test_size = params['test_size']

            stratify = y if task_type == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )

            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=hyperparams,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                refit=refit,
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs
            )

            self.progress_updated.emit(0.0, 0, n_iter)
            search.fit(X_train, y_train)
            self.progress_updated.emit(100.0, n_iter, n_iter)

            # Оценка
            model = search.best_estimator_
            pred = model.predict(X_test)
            if task_type == "classification":
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average='macro', zero_division=0)
                prec = precision_score(y_test, pred, average='macro', zero_division=0)
                rec = recall_score(y_test, pred, average='macro', zero_division=0)
                roc_auc = 0.0
                if hasattr(model, "predict_proba"):
                    if len(set(y_test)) == 2:
                        proba = model.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, proba)
                    else:
                        proba = model.predict_proba(X_test)
                        roc_auc = roc_auc_score(y_test, proba, average='weighted', multi_class='ovr')
                metrics = (
                    f"Accuracy: {acc:.4f}\n"
                    f"F1 Macro: {f1:.4f}\n"
                    f"Precision Macro: {prec:.4f}\n"
                    f"Recall Macro: {rec:.4f}\n"
                    f"ROC AUC: {roc_auc:.4f}"
                )
                primary_metric = roc_auc if roc_auc > 0 else acc
            else:
                r2 = r2_score(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                mae = mean_absolute_error(y_test, pred)
                evs = explained_variance_score(y_test, pred)
                metrics = (
                    f"R² Score: {r2:.4f}\n"
                    f"Mean Squared Error: {mse:.4f}\n"
                    f"Mean Absolute Error: {mae:.4f}\n"
                    f"Explained Variance: {evs:.4f}"
                )
                primary_metric = r2

            self.tuning_completed.emit(model, search.best_params_, primary_metric, metrics)

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._is_running = False
