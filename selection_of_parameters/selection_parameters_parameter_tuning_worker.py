# selection_parameters_parameter_tuning_worker.py
import logging
from PySide6.QtCore import Signal, QThread
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from .selection_of_parameters_logic import get_random_grid, get_random_search_params

logger = logging.getLogger(__name__)


class ParameterTuningWorker(QThread):
    progress_updated = Signal(float, int, int)
    tuning_completed = Signal(object, dict, float, str)
    error_occurred = Signal(str)

    def __init__(self, parent=None, dataset_path=None, target_variable=None, model_type=""):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.model_type = model_type
        self._is_running = False

    def run(self):
        if self._is_running:
            logger.warning("ParameterTuningWorker уже запущен — пропуск.")
            return
        self._is_running = True

        try:
            logger.info("=== Запуск подбора ===")

            # === 1. Загрузка данных ===
            df = pd.read_csv(self.dataset_path)
            if self.target_variable not in df.columns:
                raise ValueError(f"Целевая переменная '{self.target_variable}' не найдена")

            X = df.drop(columns=[self.target_variable])
            y = df[self.target_variable].copy()

            # Кодирование строковых меток
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)
            n_classes = len(pd.unique(y))

            # === 2. Получение параметров из logic.py (единственный источник!) ===
            params = get_random_search_params()
            grid = get_random_grid()
            hyperparams = grid.get(self.model_type)

            if not hyperparams:
                raise ValueError(f"Нет гиперпараметров для: {self.model_type}")

            # === 3. Создание модели БЕЗ жёстких параметров ===
            model_classes = {
                'RandomForest': RandomForestClassifier,
                'GradientBoosting': GradientBoostingClassifier,
                'LogisticRegression': LogisticRegression
            }
            model_cls = model_classes.get(self.model_type)
            if not model_cls:
                raise ValueError(f"Неподдерживаемая модель: {self.model_type}")
            estimator = model_cls()  # Все параметры — из scoring, refit и т.д.

            # === 🔹 ОПРЕДЕЛЕНИЕ ТИПА ЗАДАЧИ: бинарная vs многоклассовая === #
            # Параметры из logic.py
            scoring = params['scoring']  # dict: {'accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
            refit = params['refit']      # например, 'roc_auc' — это КЛЮЧ в словаре scoring
            multi_class = params.get('multi_class', 'ovr')  # 'ovr' или 'ovo'

            # 🔸 Параметры для бинарной классификации (n_classes == 2):
            #   - scoring: 'roc_auc', 'f1', 'accuracy'
            #   - refit: совпадает со scoring (ключ!)
            #   - predict_proba: берем [:, 1]
            #   - roc_auc_score: без multi_class
            #
            # 🔹 Параметры для многоклассовой (n_classes > 2):
            #   - scoring: {'roc_auc': 'roc_auc_ovr'} — значение (scorer) меняется, ключ остаётся
            #   - refit: 'roc_auc' — ссылается на ключ, не на scorer!
            #   - multi_class: 'ovr' или 'ovo' — используется в roc_auc_score
            #   - average: 'weighted' или 'macro' в метриках

            # --- ИСПРАВЛЕНИЕ ОШИБКИ: Меняем scorer, но не ключ и не refit! ---
            if n_classes > 2 and isinstance(scoring, dict):
                # Обновляем значения (scorers), но оставляем ключи без изменений
                scoring = {
                    name: (
                        f'roc_auc_{multi_class}' if name == 'roc_auc' else metric
                    )
                    for name, metric in scoring.items()
                }
                # ВАЖНО: refit остаётся ключом, например 'roc_auc', который указывает на обновлённый scorer
                # НЕ МЕНЯЕМ refit на 'roc_auc_ovr' — это НЕ ключ в словаре!

            # === 4. Параметры поиска ===
            n_iter = params['n_iter']
            cv = params['cv']
            verbose = params['verbose']
            n_jobs = params['n_jobs']
            random_state = params['random_state']
            test_size = params['test_size']

            # === 5. Разделение ===
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # === 6. Поиск ===
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=hyperparams,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                refit=refit,  # ✅ 'roc_auc' — это ключ, указывающий на scorer 'roc_auc_ovr'
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs
            )

            # === 7. Обучение ===
            self.progress_updated.emit(0.0, 0, n_iter)
            search.fit(X_train, y_train)
            self.progress_updated.emit(100.0, n_iter, n_iter)

            # === 8. Оценка ===
            model = search.best_estimator_
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='macro', zero_division=0)

            roc_auc = 0.0
            if hasattr(model, "predict_proba"):
                if n_classes == 2:
                    proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, proba)
                else:
                    proba = model.predict_proba(X_test)
                    roc_auc = roc_auc_score(
                        y_test, proba,
                        multi_class=multi_class,
                        average='weighted'
                    )

            metrics = (
                f"Accuracy: {acc:.4f}\n"
                f"F1 Macro: {f1:.4f}\n"
                f"Precision Macro: {precision_score(y_test, pred, average='macro', zero_division=0):.4f}\n"
                f"Recall Macro: {recall_score(y_test, pred, average='macro', zero_division=0):.4f}\n"
                f"ROC AUC: {roc_auc:.4f}"
            )

            self.tuning_completed.emit(model, search.best_params_, acc, metrics)

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._is_running = False
