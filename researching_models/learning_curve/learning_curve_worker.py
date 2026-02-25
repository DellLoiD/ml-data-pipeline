from PySide6.QtCore import QThread, Signal
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor


class LearningCurveWorker(QThread):
    """
    Поток для выполнения анализа кривой обучения с Optuna.
    Отделяет тяжелые вычисления от основного UI-потока.
    """
    result_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(
        self, analyzer, model_name, n_trials, timeout, direction, scoring,
        n_est_range, max_depth_range, learning_rate_range,
        cv, n_jobs_cv, random_state, optuna_n_jobs
    ):
        super().__init__()
        self.analyzer = analyzer
        self.model_name = model_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.scoring = scoring
        self.n_est_range = n_est_range
        self.max_depth_range = max_depth_range
        self.learning_rate_range = learning_rate_range
        self.cv = cv
        self.n_jobs_cv = n_jobs_cv
        self.random_state = random_state
        self.optuna_n_jobs = optuna_n_jobs

    def run(self):
        try:
            # Запуск Optuna
            study = self.analyzer.run_optuna_study(
                model_name=self.model_name,
                n_trials=self.n_trials,
                timeout=self.timeout,
                direction=self.direction,
                scoring=self.scoring,
                n_est_range=self.n_est_range,
                max_depth_range=self.max_depth_range,
                learning_rate_range=self.learning_rate_range,
                cv=self.cv,
                n_jobs_cv=self.n_jobs_cv,
                random_state=self.random_state,
                optuna_n_jobs=self.optuna_n_jobs
            )

            if not study.best_trial:
                self.error_occurred.emit("Оптимизация Optuna не нашла подходящих решений.")
                return

            # Создание лучшей модели
            best_params = study.best_params
            model_cls = self._get_model_class()
            best_model = model_cls(**best_params, random_state=self.random_state)

            # Кривая обучения
            lc_result = self.analyzer.compute_learning_curve(
                best_model,
                scoring=self.scoring,
                cv=self.cv,
                n_points=self._get_n_points(),
                n_jobs_cv=self.n_jobs_cv,
                random_state=self.random_state
            )

            # Сбор результата
            result = {
                'best_model': best_model,
                'lc_result': lc_result,
                'best_params': best_params,
                'scoring': self.scoring,
                'model_name': self.model_name
            }

            self.result_ready.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _get_model_class(self):
        """Возвращает класс модели на основе имени."""
        if self.model_name == "Random Forest":
            return RandomForestRegressor if self.analyzer.task_type == "regression" else RandomForestClassifier
        elif self.model_name == "Gradient Boosting":
            return GradientBoostingRegressor if self.analyzer.task_type == "regression" else GradientBoostingClassifier
        else:
            raise ValueError(f"Модель не поддерживается: {self.model_name}")

    def _get_n_points(self):
        """Возвращает количество точек для кривой обучения."""
        # Временное решение: значение по умолчанию.
        # В будущем можно передавать из UI.
        return 10