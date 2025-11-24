from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import os
from PySide6.QtGui import QPixmap
from time import perf_counter
from PySide6.QtWidgets import *
from sklearn.model_selection import train_test_split
from .check_models_loading_screen import LoadingScreen
from datetime import datetime
from sklearn.inspection import permutation_importance

from time import perf_counter, time
import logging
# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Декоратор для измерения времени выполнения функций   random_state_input
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        elapsed_time = f"{func.__name__} took {end_time - start_time:.4f} seconds"
        return result, elapsed_time
    return wrapper
    
class DataModelHandler:
    def __init__(self, parent, df=None, combobox=None, test_size_input=None, random_state_input=None, checkboxes=None,
                 labels_and_lines=None, accuracy_label=None, time_label=None):
        self.parent = parent
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.combobox = combobox
        self.test_size_input = test_size_input
        self.random_state_input = random_state_input
        self.checkboxes = checkboxes
        self.labels_and_lines = labels_and_lines
        self.accuracy_label = accuracy_label
        self.time_label = time_label
        
    def set_df(self, dataframe):
        self.df = dataframe
    
    def update_dataframe(self, new_df):        
        self.df = new_df
        if self.combobox is not None:
            columns = new_df.columns.tolist()
            self.combobox.clear()
            self.combobox.addItems(columns)
            self.combobox.setEnabled(True)
            
    def evaluate_models(self):
        if self.df is None or self.df.empty:
            print("Датасет не загружен!")
            return
        
        # Открываем экран загрузки
        splash_screen = LoadingScreen()
        QApplication.instance().processEvents()
        
        # Получаем выбранную целевую переменную
        target_col = self.parent.target_var_combobox.currentText()
        feature_cols = list(self.df.columns.drop(target_col))
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        # Проверяем, какие модели активированы пользователями
        selected_models = []
        for checkbox in self.checkboxes:
            if not checkbox.isChecked():
                continue
            model_name = checkbox.text()
            model_params = self.labels_and_lines.get(model_name, {})
            
            # Определяем параметры каждой модели
            if model_name == 'Random Forest':
                n_estimators = int(model_params['Количество деревьев'].text()) if 'Количество деревьев' in model_params else 100
                test_size = float(model_params['Test Size'].text()) if 'Test Size' in model_params else 0.2
                random_state = int(model_params['Random State'].text()) if 'Random State' in model_params else 42
                clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                selected_models.append((model_name, clf, test_size, random_state))
                
            elif model_name == 'Gradient Boosting':
                n_estimators = int(model_params['Количество деревьев'].text()) if 'Количество деревьев' in model_params else 100
                test_size = float(model_params['Test Size'].text()) if 'Test Size' in model_params else 0.2
                random_state = int(model_params['Random State'].text()) if 'Random State' in model_params else 42
                clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
                selected_models.append((model_name, clf, test_size, random_state))
                
            elif model_name == 'Logistic Regression':
                C = float(model_params['C'].text()) if 'C' in model_params else 1.0
                max_iter = int(model_params['Max Iterations'].text()) if 'Max Iterations' in model_params else 100
                penalty = str(model_params['Penalty'].text()) if 'Penalty' in model_params else 'l2'
                clf = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty)
                selected_models.append((model_name, clf))
        
        # Начинаем оценивать выбранные модели
        results = []  # Список кортежей (название модели, точность, F1-score, ROC-AUC, время вычисления)
        for entry in selected_models:
            if len(entry) == 4:
                model_name, clf, test_size, random_state = entry
            else:
                model_name, clf = entry
                test_size = 0.2
                random_state = 42
            
            # Логика оценки моделей
            start_time = datetime.now()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            probas = clf.predict_proba(X_test)[:, 1]  # Вероятности положительного класса
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            
            # Вычисление метрик
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, probas)
            except ValueError as e:
                auc = "Ошибка при расчете AUC"
            
            # Добавляем результаты
            results.append((model_name, acc, f1, auc, elapsed_time))
        
        # Формирование отчёта
        report = ""
        for result in results:
            model_name, acc, f1, auc, _ = result
            report += f"{model_name}:\nТочность={acc:.4f}, F1-Score={f1:.4f}, ROC-AUC={auc}\n\n"
        
        # Общая статистика по времени выполнения
        total_time = sum(result[-1] for result in results)
        time_text = f"Время выполнения: {total_time:.4f} секунд"
        
        # Закрываем экран загрузки
        splash_screen.close()
        
        # Обновляем интерфейс результатами
        self.accuracy_label.setText(report.strip())
        self.time_label.setText(time_text)
        
    def split_dataset(self):
        logging.info("Начало разделения данных...")
        # Получаем параметры из выбранного интерфейса
        # Предположим, первая выбранная модель задаёт общий Test Size и Random State
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                first_model_name = checkbox.text()
                break
        else:
            raise ValueError("Ни одна модель не была выбрана.")

        # Теперь получаем значения из полей ввода первой выбранной модели
        model_params = self.labels_and_lines.get(first_model_name, {})
        test_size = float(model_params['Test Size'].text()) if 'Test Size' in model_params else 0.2
        random_state = int(model_params['Random State'].text()) if 'Random State' in model_params else 42

        # Остальные строки остаются неизменёнными
        target_column = self.df.columns[-1]
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        logging.debug(f"Датасет: {len(self.df)} строк, {len(self.df.columns)} столбцов.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Информируем о завершении процедуры
    logging.info("Разделение данных завершилось успешно.")
        
    def calculate_feature_importances(self, selected_models=None):
        splash_screen = LoadingScreen()
        splash_screen.show()  # Показываем экран загрузки сразу после входа в метод

        self.split_dataset()
        results = {}

        for model_name, active in selected_models.items():
            if not active:
                continue

            model_params = self.labels_and_lines.get(model_name, {})

            match model_name:
                case 'Random Forest':
                    n_estimators = int(model_params.get('Количество').text() if 'Количество' in model_params else '100')
                    random_state = int(model_params.get('Random State').text() if 'Random State' in model_params else '42')
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                case 'Gradient Boosting':
                    n_estimators = int(model_params.get('Количество').text() if 'Количество' in model_params else '100')
                    random_state = int(model_params.get('Random State').text() if 'Random State' in model_params else '42')
                    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
                case 'Logistic Regression':
                    C = float(model_params.get('C').text() if 'C' in model_params else '1.0')
                    max_iter = int(model_params.get('Max Iterations').text() if 'Max Iterations' in model_params else '100')
                    penalty = str(model_params.get('Penalty').text() if 'Penalty' in model_params else 'l2')
                    clf = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty)
                case _:
                    continue  # Пропускаем неизвестные модели

            clf.fit(self.X_train, self.y_train)
            feature_names = list(self.df.columns[:-1])
            importances = clf.feature_importances_ if hasattr(clf, 'feature_importances_') else np.abs(clf.coef_[0])

            features_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            features_df.sort_values(by='Importance', ascending=False, inplace=True)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=features_df)
            plt.title(f"Важность признаков ({model_name})")
            plt.tight_layout()
            plt.savefig(f"plots/{model_name}_feature_importance.png")
            plt.show()

            results[model_name] = features_df

        splash_screen.close()  # Скрываем экран загрузки после завершения всех операций
        return results
    