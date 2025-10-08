# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import os
import joblib

# Загрузка данных 
df = pd.read_csv('dataset/diabetes_BRFSS2015-balanced-Diabetes_012-size90000.csv')

# Выбор набора признаков
feature_cols = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

X = df[feature_cols]
y = df['Diabetes_binary']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель классификации (градиентный бустинг)
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy:.4f}')

# Важность признаков
importances = gb_model.feature_importances_
features_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
features_df.sort_values(by='Importance', ascending=False, inplace=True)
plt.figure(figsize=(10,8))
sns.barplot(x="Importance", y="Feature", data=features_df)
plt.title("Важность признаков с новыми параметрами")
# Ваш существующий код построения графика
plt.figure(figsize=(10,8))
sns.barplot(x="Importance", y="Feature", data=features_df)
plt.title("Важность признаков с новыми параметрами")

directory = 'plots'
if not os.path.exists(directory):
    os.makedirs(directory)
# Сохраняем график в файл
plt.savefig(os.path.join(directory, 'feature_importance_gradient_boosting.png'))
# Закрываем окно графика, чтобы оно не мешало выполнению скрипта
plt.close()

# Грид-поиск гиперпараметров
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],     # Шаг обучения
    'n_estimators': [50, 100, 200],       # Количество деревьев
    'max_depth': [3, 5, None],             # Максимальная глубина дерева
    'subsample': [0.8, 1.0],               # Доля образцов для каждого дерева
    'min_samples_split': [2, 5, 10],       # Минимальное количество наблюдений для разделения узла
    'min_samples_leaf': [1, 2, 4]          # Минимальное количество наблюдений в листовом узле
}

# Создание объекта GridSearchCV
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),   # Изменили классификатор
    param_grid=param_grid,
    cv=2,                # n-кратная кросс-валидация
    scoring='accuracy',  # Оптимизация по метрикам точности
    verbose=2,           # Показывать прогресс
    n_jobs=-1            # Использовать все доступные ядра CPU
)
try:
    grid_search.fit(X_train, y_train)
except KeyboardInterrupt:
    print("Прерывание...")

print(f'Лучшая комбинация параметров: {grid_search.best_params_}')
print(f'Лучшее значение точности: {grid_search.best_score_}')

# Поиск лучших параметров
grid_search.fit(X_train, y_train)

# Получение лучших параметров
best_params = grid_search.best_params_
print("Лучшие параметры:", best_params)

# Новая модель с лучшими параметрами
best_gb_model = GradientBoostingClassifier(**best_params, random_state=42)
best_gb_model.fit(X_train, y_train)

# Прогнозирование и оценка точности новой модели
y_pred_best = best_gb_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Test Accuracy with Best Parameters: {accuracy_best:.4f}')

# Важно! Оставляем старую модель и оценку старой точности для сравнения
old_accuracy = accuracy_score(y_test, y_pred)
print(f'Standard Test Accuracy: {old_accuracy:.4f}')

# Преобразуем точность в проценты и округлим до двух десятичных знаков
accuracy_percentage = round(accuracy_best * 100, 2)
# Проверка наличия директории и создание, если отсутствует
os.makedirs('trained_models', exist_ok=True)
# Добавляем точность в название файла
model_filename = f'gradient_boosting_diabetes_model_acc_{accuracy_percentage:.2f}_percent.pkl'
# Полный путь файла для сохранения модели
model_path = os.path.join('trained_models', model_filename)
# Сохранение обученной модели
joblib.dump(best_gb_model, model_path)

