import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from time import perf_counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

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

    # Прогнозирование и оценка базовой модели
    y_pred = base_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on test set: {accuracy:.4f}')

    # Важность признаков
    importances = base_model.feature_importances_
    features_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    features_df.sort_values(by='Importance', ascending=False, inplace=True)

    # Графическое представление важности признаков
    plt.figure(figsize=(10,8))
    sns.barplot(x="Importance", y="Feature", data=features_df)
    plt.title("Важность признаков с новыми параметрами")

    directory = 'plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, 'feature_importance_forest.png'))
    plt.close()

    # Настройки сетки гиперпараметров для рандомизированного поиска
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
        'verbose': [0]      
    }

    # Объект RandomizedSearchCV
    @timing_decorator
    def perform_random_search(X_train, y_train):
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=random_grid,
            n_iter=100,
            cv=6,
            scoring=['accuracy', 'f1_macro'],      
            refit='accuracy',
            random_state=42,
            verbose=2,
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        return random_search

    random_search_result = perform_random_search(X_train, y_train)

    # Лучшие параметры и точность
    print(f'Лучшая комбинация параметров: {random_search_result.best_params_}')
    print(f'Лучшее значение точности: {random_search_result.best_score_}')

    # Новая модель с лучшими параметрами
    best_rf_model = RandomForestClassifier(**random_search_result.best_params_, random_state=42)
    best_rf_model.fit(X_train, y_train)

    # Прогнозирование и оценка точности
    y_pred_best = best_rf_model.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    print(f'Test Accuracy with Best Parameters: {accuracy_best:.4f}')

    # Матрица путаницы
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nМатрица путаницы:")
    print(cm)

    # Дополнительные метрики (Macro F1, Macro Precision, Macro Recall)
    f1_macro = f1_score(y_test, y_pred_best, average='macro')
    precision_macro = precision_score(y_test, y_pred_best, average='macro')
    recall_macro = recall_score(y_test, y_pred_best, average='macro')
    print(f'\nMacro F1 Score: {f1_macro:.4f}')
    print(f'Macro Precision: {precision_macro:.4f}')
    print(f'Macro Recall: {recall_macro:.4f}')

    # Полный отчёт по классификации
    print("\nПолный отчёт по классификации:")
    print(classification_report(y_test, y_pred_best))

    # Сохранение модели
    accuracy_percentage = round(accuracy_best * 100, 2)
    os.makedirs('trained_models', exist_ok=True)
    model_filename = f'random_forest_diabetes_model_acc_{accuracy_percentage:.2f}_percent.pkl'
    model_path = os.path.join('trained_models', model_filename)
    joblib.dump(best_rf_model, model_path)

