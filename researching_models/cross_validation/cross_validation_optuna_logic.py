# researching_models/cross_validation/cross_validation_optuna_logic.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, precision_score, recall_score
from joblib import parallel_backend
import optuna
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parameter_tuning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Логирование запуска анализа
logger.info(f"Запуск анализа Optuna и кросс-валидации...")

class OptunaAnalyzer:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.target_col = None
        self.task_type = "classification"
        self.scaler = StandardScaler()

    def load_from_dataframe(self, df, target_col, task_type="classification", random_state=42):
        """Загрузка из уже загруженного DataFrame"""
        self.task_type = task_type
        self.target_col = target_col
        df_local = df.copy()

        if task_type == "classification" and df_local[target_col].dtype == 'object':
            df_local[target_col] = LabelEncoder().fit_transform(df_local[target_col])

        X = df_local.drop(columns=[target_col]).select_dtypes(include=['number'])
        y = df_local[target_col]

        if X.empty:
            raise ValueError("Нет числовых признаков.")

        # Принудительная конвертация для классификации
        if task_type == "classification":
            y = y.astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y if task_type == "classification" else None
        )
        return True

    def load_separate_datasets(self, train_path, test_path, target_col, task_type="classification"):
        """Загрузка train/test из двух файлов"""
        df_train = pd.read_csv(train_path, comment='#')
        df_test = pd.read_csv(test_path, comment='#')

        if target_col not in df_train.columns or target_col not in df_test.columns:
            raise ValueError(f"Целевая переменная '{target_col}' отсутствует в одном из файлов")

        X_train = df_train.drop(columns=[target_col]).select_dtypes(include=['number'])
        y_train = df_train[target_col]
        X_test = df_test.drop(columns=[target_col]).select_dtypes(include=['number'])
        y_test = df_test[target_col]

        if X_train.empty or X_test.empty:
            raise ValueError("Нет числовых признаков после очистки")

        if task_type == "classification":
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.target_col = target_col
        self.task_type = task_type
        return True

    def get_scorer(self, scoring_name):
        """Автоматический выбор scorer с учётом числа классов"""
        if self.task_type != "classification":
            return scoring_name

        n_classes = len(np.unique(self.y_train))

        if scoring_name in ['f1', 'precision', 'recall']:
            if n_classes == 2:
                return scoring_name  # Для бинарной классификации используется стандартное поведение
            else:
                # Для многоклассовой задачи используем стратегию 'macro'
                if scoring_name == 'f1':
                    return make_scorer(f1_score, average='macro')
                elif scoring_name == 'precision':
                    return make_scorer(precision_score, average='macro')
                elif scoring_name == 'recall':
                    return make_scorer(recall_score, average='macro')
        
        if scoring_name == 'roc_auc':
            if n_classes == 2:
                return 'roc_auc'
            else:
                return make_scorer(roc_auc_score, multi_class='ovr', response_method='predict_proba')
                
        return scoring_name

    def run_optuna_study(self, model_name, optuna_n_jobs, n_trials, timeout, direction, scoring,
                        n_est_range, max_depth_range, learning_rate_range,
                        cv, n_jobs_cv, random_state, storage_path=None):  
        """Запуск Optuna для подбора гиперпараметров"""
        if self.X_train is None:
            raise ValueError("Данные не загружены")

        # Используем внешнее хранилище (SQLite) для Optuna
        study = optuna.create_study(
            storage=storage_path,
            study_name="cross_validation_optimization",
            direction=direction,
            load_if_exists=True  
        )

        logger.info(f"Запуск Optuna study для {model_name}, n_trials={n_trials}, timeout={timeout}")

        def objective(trial):
            try:
                n_estimators = trial.suggest_int('n_estimators', *n_est_range)
                if 'None' in str(max_depth_range):
                    max_depth = trial.suggest_categorical('max_depth', [3, 5, 7, 10, None])
                else:
                    max_depth = trial.suggest_int('max_depth', *max_depth_range)

                if model_name == "Random Forest":
                    clf = (RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                        if self.task_type == "classification" else
                        RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
                elif model_name == "Gradient Boosting":
                    lr = trial.suggest_float('learning_rate', *learning_rate_range)
                    clf = (GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, random_state=random_state)
                        if self.task_type == "classification" else
                        GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, random_state=random_state))
                else:
                    raise ValueError(f"Модель {model_name} не поддерживается")


                X_train_scaled = self.scaler.fit_transform(self.X_train)
                scorer = self.get_scorer(scoring)
                scores = cross_val_score(clf, X_train_scaled, self.y_train, cv=cv, scoring=scorer, n_jobs=n_jobs_cv)
                return np.mean(scores)
            except Exception as e:
                logger.error(f"[DEBUG] Ошибка в trial: {e}")
                return -np.inf if direction == "maximize" else np.inf

        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=optuna_n_jobs)  
        logger.info(f"Optuna study завершен. Лучшие параметры: {study.best_params}, лучшая метрика: {study.best_value:.4f}")
        return study

    def compute_cross_validation_scores(self, best_model, scoring, cv, n_jobs_cv, random_state):
        """Вычисление метрик кросс-валидации и сохранение оценок для каждого фолда"""
        X_train_scaled = self.scaler.fit_transform(self.X_train)

        # Определяем стратегию разбиения с random_state для воспроизводимости
        if self.task_type == "classification":
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        with parallel_backend('loky', n_jobs=n_jobs_cv):
            scores = cross_val_score(
                best_model, X_train_scaled, self.y_train,
                cv=cv_strategy, scoring=self.get_scorer(scoring), n_jobs=n_jobs_cv
            )
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        logger.info(f"Кросс-валидация завершена. Средняя метрика: {mean_score:.4f} ± {std_score:.4f}")

        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'scores': scores
        }