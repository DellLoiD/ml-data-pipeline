from time import perf_counter
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform, randint
import pandas as pd

# === Основная сетка гиперпараметров ===
random_grid = {
    # === Классификация ===
    'RandomForestClassifier': {
        'n_estimators': range(50, 200, 10),
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None],
        'max_features': [None, 'sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.01, 0.1],
        'verbose': [0]
    },
    'GradientBoostingClassifier': {
        'n_estimators': range(50, 200),
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['friedman_mse', 'squared_error'],
        'ccp_alpha': [0.0, 0.01]
    },
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': loguniform(0.01, 100),
        'solver': ['liblinear', 'saga']
    },
    # === Регрессия ===
    'RandomForestRegressor': {
        'n_estimators': range(50, 200, 10),
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
        'max_features': [None, 'sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.01],
        'verbose': [0]
    },
    'GradientBoostingRegressor': {
        'n_estimators': range(50, 200),
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['friedman_mse', 'squared_error'],
        'ccp_alpha': [0.0, 0.01]
    }
}
# === Параметры RandomizedSearchCV ===
random_search_params = {
    'n_iter': 20,
    'cv': 3,
    'scoring': {
        # Классификация
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'roc_auc': 'roc_auc'  # auto-detects binary/multiclass
    },
    'scoring_regression': {
        # Регрессия
        'r2': 'r2',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'explained_variance': 'explained_variance'
    },
    'refit': 'r2',
    'test_size': 0.2,
    'random_state': 42,
    'verbose': 2,
    'n_jobs': -1
}

def get_refit_metric(task_type):
    """Возвращает правильный refit-ключ в зависимости от задачи"""
    if task_type == "classification":
        return "f1_macro"  # или "accuracy", "roc_auc"
    else:  # regression
        return "r2"
    
def get_random_grid():
    return dict(random_grid)

def save_random_grid(new_grid):
    if not isinstance(new_grid, dict):
        raise ValueError("new_grid must be a dict")
    random_grid.clear()
    random_grid.update(new_grid)
    return get_random_grid()

def get_random_search_params():
    return dict(random_search_params)

def save_random_search_params(new_params):
    if not isinstance(new_params, dict):
        raise ValueError("new_params must be a dict")
    random_search_params.clear()
    random_search_params.update(new_params)
    return get_random_search_params()

# === Декораторы ===
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def load_data(filename):
    return pd.read_csv(filename)

@timing_decorator
def split_data(X, y):
    random_state_value = random_search_params.get('random_state')
    test_size_value = random_search_params.get('test_size')
    return train_test_split(X, y, test_size=test_size_value, random_state=random_state_value)
