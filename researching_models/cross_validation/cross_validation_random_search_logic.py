from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np
import random
import gc
import pandas as pd
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
logger.info(f"Запуск анализа Random Search и кросс-валидации...")

class RandomSearchAnalyzer:
    """Класс для анализа моделей с подбором гиперпараметров методом Random Search."""
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

        if scoring_name != 'roc_auc':
            return scoring_name

        if n_classes == 2:
            return 'roc_auc'  
        else:
            return make_scorer(roc_auc_score, multi_class='ovr', response_method='predict_proba')

    def run_random_search(self, n_trials=50, model_name="Random Forest", scoring="accuracy", 
                         direction="maximize", n_est_range=(50, 200), max_depth_range=(2, 5), 
                         lr_range=(0.01, 0.3), cv=5, n_jobs_cv=1, n_jobs_lc=1, n_points=10, random_state=42):
        """Основной метод для выполнения Random Search."""
        best_score = -float('inf') if direction == "maximize" else float('inf')
        best_params = None
        best_model = None
        best_cv_scores = None

        for _ in range(n_trials):
            # Случайные параметры
            n_estimators = random.randint(*n_est_range)
            if 'None' in str(max_depth_range):
                max_depth_val = random.choice([3, 5, 7, 10, None])
            else:
                max_depth_val = random.randint(*max_depth_range)

            learning_rate_val = round(random.uniform(*lr_range), 3)

            if model_name == "Random Forest":
                clf = (RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_val, random_state=random_state)
                       if self.task_type == "classification" else
                       RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth_val, random_state=random_state))
            elif model_name == "Gradient Boosting":
                clf = (GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth_val, learning_rate=learning_rate_val, random_state=random_state)
                       if self.task_type == "classification" else
                       GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth_val, learning_rate=learning_rate_val, random_state=random_state))
            else:
                raise ValueError("Модель не поддерживается")

            # Оценка
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            scorer = self.get_scorer(scoring)
            scores = cross_val_score(clf, X_train_scaled, self.y_train, cv=cv, scoring=scorer, n_jobs=n_jobs_cv)
            score = np.mean(scores)

            # Обновляем лучшее
            if (direction == "maximize" and score > best_score) or (direction == "minimize" and score < best_score):
                best_score = score
                best_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth_val,
                }
                if model_name == "Gradient Boosting":
                    best_params['learning_rate'] = learning_rate_val
                best_model = clf
                best_cv_scores = scores

        if best_params is None:
            return None

        # Проверяем, что best_cv_scores не None
        if best_cv_scores is None:
            logger.warning("Оценки кросс-валидации (cv_scores) отсутствуют, хотя найдены лучшие параметры")
            cv_scores = None
            mean_score = 0.0
            std_score = 0.0
        else:
            cv_scores = best_cv_scores
            mean_score = np.mean(best_cv_scores)
            std_score = np.std(best_cv_scores)

        return {
            'best_params': best_params,
            'model_name': f"{model_name} ({self.task_type})",
            'scoring': scoring,
            'cv_scores': cv_scores,
            'mean_score': mean_score,
            'std_score': std_score
        }