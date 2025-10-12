import pandas as pd
from sklearn.model_selection import train_test_split
from time import perf_counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Module-level hyperparameter grid used by UI and tuning logic
random_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40, 60],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.01, 0.1],
    'verbose': [0],
}

# Module-level RandomizedSearchCV parameters
random_search_params = {
    'estimator': 'RandomForestClassifier(random_state=42)',
    'n_iter': 100,
    'cv': 6,
    'scoring': ['accuracy', 'f1_macro'],
    'refit': 'accuracy',
    'random_state': 42,
    'verbose': 2,
    'n_jobs': -1
}

def get_hyperparameters():
    """Return current hyperparameter grid as a plain dict."""
    return dict(random_grid)

def save_hyperparameters(new_grid):
    """Update module-level hyperparameter grid in place from provided dict."""
    if not isinstance(new_grid, dict):
        raise ValueError("new_grid must be a dict")
    # Replace keys present in new_grid; keep unknown keys as provided to allow future extension
    random_grid.clear()
    random_grid.update(new_grid)
    return get_hyperparameters()

def get_random_search_params():
    """Return current RandomizedSearchCV parameters as a plain dict."""
    return dict(random_search_params)

def save_random_search_params(new_params):
    """Update module-level RandomizedSearchCV parameters from provided dict."""
    if not isinstance(new_params, dict):
        raise ValueError("new_params must be a dict")
    random_search_params.clear()
    random_search_params.update(new_params)
    return get_random_search_params()

# Декоратор для измерения времени выполнения функций
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
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Главная программа
if __name__ == "__main__":
    filename = 'dataset/diabetes_BRFSS2015-balanced-Diabetes_012-size90000.csv'
    df = load_data(filename)
    
    # Автоматическое получение списка всех колонок кроме целевой переменной
    target_col = 'Diabetes_012'  # Целевая переменная
    feature_cols = list(df.columns.drop(target_col))
    
    # Преобразование данных
    X = df[feature_cols]
    y = df[target_col]
    
    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Базовая модель классификации (случайный лес)
    @timing_decorator
    def train_base_model(X_train, y_train):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        return rf_model

    base_model = train_base_model(X_train, y_train)

    # Объект RandomizedSearchCV
    @timing_decorator
    def perform_random_search(X_train, y_train):
        # Get current parameters
        params = get_random_search_params()
        
        # Parse estimator string to actual estimator object
        estimator_str = params.get('estimator', 'RandomForestClassifier(random_state=42)')
        if estimator_str == 'RandomForestClassifier(random_state=42)':
            estimator = RandomForestClassifier(random_state=42)
        else:
            # For more complex cases, you might want to use eval() or a more sophisticated parser
            estimator = RandomForestClassifier(random_state=42)  # fallback
        
        # Parse scoring parameter
        scoring = params.get('scoring', ['accuracy', 'f1_macro'])
        if isinstance(scoring, str):
            # Handle string representation of list
            try:
                scoring = eval(scoring) if scoring.startswith('[') else [scoring]
            except:
                scoring = ['accuracy', 'f1_macro']  # fallback
        
        # Parse refit parameter
        refit = params.get('refit', 'accuracy')
        if isinstance(refit, str) and refit.startswith("'"):
            refit = eval(refit)
        
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=random_grid,
            n_iter=params.get('n_iter', 100),
            cv=params.get('cv', 6),
            scoring=scoring,
            refit=refit,
            random_state=params.get('random_state', 42),
            verbose=params.get('verbose', 2),
            n_jobs=params.get('n_jobs', -1)
        )
        random_search.fit(X_train, y_train)
        return random_search

    random_search_result = perform_random_search(X_train, y_train)
