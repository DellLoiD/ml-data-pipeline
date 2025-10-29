import pandas as pd
from time import perf_counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


random_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.01, 0.1],
    'verbose': [0]
}
random_search_params = {
    'n_iter': 10,
    'cv': 3,
    'scoring': ['accuracy', 'f1_macro'], 
    'refit': 'accuracy',
    'test_size': 0.2,
    'random_state': 42, 
    'verbose': 2,
    'n_jobs': -1
}
def get_hyperparameters():
    return dict(random_grid)
def save_hyperparameters(new_grid):
    if not isinstance(new_grid, dict):
        raise ValueError("new_grid must be a dict")
    random_grid.clear()
    random_grid.update(new_grid)
    return get_hyperparameters()
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
    # Извлекаем значение random_state из словаря
    random_state_value = random_search_params.get('random_state')
    test_size_value = random_search_params.get('test_size')
    # Разделение данных с использованием извлеченного значения random_state
    return train_test_split(X, y, test_size=test_size_value, random_state=random_state_value)

# Основная логика
if __name__ == "__main__":
    filename = 'dataset/diabetes_BRFSS2015-balanced-Diabetes_012-size90000.csv'
    df = load_data(filename)
    target_col = 'Diabetes_012'
    feature_cols = list(df.columns.drop(target_col))
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Базовая модель
    @timing_decorator
    def train_base_model(X_train, y_train):
        rf_model = RandomForestClassifier(n_estimators=100)
        rf_model.fit(X_train, y_train)
        return rf_model

    base_model = train_base_model(X_train, y_train)

    # Случайный поиск гиперпараметров
    @timing_decorator
    def perform_random_search(X_train, y_train):
        search_params = random_search_params.copy()
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(),
            param_distributions=random_grid,
            n_iter=search_params.pop('n_iter'),
            cv=search_params.pop('cv'),
            scoring=search_params.pop('scoring'),  # Оценочные метрики передаются списком
            refit=search_params.pop('refit'),
            random_state=search_params.pop('random_state'),
            verbose=search_params.pop('verbose'),
            n_jobs=search_params.pop('n_jobs'))
        random_search.fit(X_train, y_train)
        return random_search

    random_search_result = perform_random_search(X_train, y_train)