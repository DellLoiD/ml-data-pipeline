import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

# Загрузка данных
df = pd.read_csv('dataset/diabetes_BRFSS2015-balanced-Diabetes_012-size90000.csv')
# Выбор набора признаков
feature_cols = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

X = df[feature_cols]
y = df['Diabetes_012']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель классификации (логистическая регрессия)
logreg_model = LogisticRegression(multi_class='ovr', solver='liblinear', C=1, penalty='l2', max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy:.4f}')

# Параметры поиска
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],       # Регуляризационный коэффициент
    'penalty': ['l2'],              # Тип регуляризации ('l1' or 'l2')
    'solver': ['saga']      # Алгоритм обучения
}

# Грид-поиск гиперпараметров
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid=param_grid,
    cv=8,            
    scoring='accuracy',    
    verbose=2,          
    n_jobs=-1           
)

# Поиск лучших параметров
grid_search.fit(X_train, y_train)

# Получение лучших параметров
best_params = grid_search.best_params_
print("Лучшие параметры:", best_params)

# Новая модель с лучшими параметрами
best_logreg_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
best_logreg_model.fit(X_train, y_train)

# Прогнозирование и оценка точности новой модели
y_pred_best = best_logreg_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Test Accuracy with Best Parameters: {accuracy_best:.4f}')

# Сохраняем старую точность для сравнения
old_accuracy = accuracy_score(y_test, y_pred)
print(f'Standard Test Accuracy: {old_accuracy:.4f}')

# ====== СОХРАНЕНИЕ  ОБУЧЕННОЙ МОДЕЛИ ==========
# Преобразуем точность в проценты и округлим до двух десятичных знаков
accuracy_percentage = round(accuracy_best * 100, 2)
# Проверка наличия директории и создание, если отсутствует
os.makedirs('trained_models', exist_ok=True)
# Добавляем точность в название файла
model_filename = f'logistic_regression_diabetes_model_acc_{accuracy_percentage:.2f}_percent.pkl'
# Полный путь файла для сохранения модели
model_path = os.path.join('trained_models', model_filename)
# Сохранение обученной модели
joblib.dump(best_logreg_model, model_path)
