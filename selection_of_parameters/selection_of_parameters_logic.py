from time import perf_counter
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
import pandas as pd

random_grid = {
    'RandomForest': {
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
    'GradientBoosting': {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': range(10, 200),
        'subsample': [0.8, 1.0],
        'max_depth': [3, 5, 7]
    },
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': loguniform(0.01, 100),
        'solver': ['liblinear']
    }
}

random_search_params = {
    'n_iter': 10,
    'cv': 3,
    'scoring': {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'roc_auc': 'roc_auc'
    },
    'refit': 'roc_auc',
    'test_size': 0.2,
    'random_state': 42,
    'verbose': 2,
    'n_jobs': -1
}

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

# Таймер-декоратор
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Функция загрузки данных с таймером
@timing_decorator
def load_data(filename):
    return pd.read_csv(filename)

# Функция разделения данных с таймером
@timing_decorator
def split_data(X, y):
    random_state_value = random_search_params.get('random_state')
    test_size_value = random_search_params.get('test_size')
    return train_test_split(X, y, test_size=test_size_value, random_state=random_state_value)
