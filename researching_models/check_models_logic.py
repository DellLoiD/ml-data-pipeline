# check_models_logic.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PySide6.QtWidgets import QMessageBox, QLineEdit
from PySide6.QtCore import QThread, Signal
from .check_models_loading_screen import LoadingScreen
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# === Поток для оценки моделей ===
class EvaluationThread(QThread):
    finished_signal = Signal(list, str)  # (results, time_text)
    error_signal = Signal(str)

    def __init__(self, parent, models_config, X, y, task_type):
        super().__init__(parent)
        self.models_config = models_config
        self.X = X
        self.y = y
        self.task_type = task_type  # "classification" или "regression"

    def run(self):
        try:
            results = []
            total_time = 0.0

            for model_display_name, clf, test_size, random_state in self.models_config:
                # Разделение и масштабирование
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=random_state
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Обучение
                start_time = datetime.now()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Сбор метрик
                if self.task_type == "classification":
                    n_classes = len(np.unique(self.y))
                    average_mode = 'weighted' if n_classes > 2 else 'binary'

                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
                    rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
                    f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)

                    try:
                        if hasattr(clf, "predict_proba"):
                            probas = clf.predict_proba(X_test)
                            if n_classes == 2:
                                auc = roc_auc_score(y_test, probas[:, 1])
                            else:
                                auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
                        else:
                            auc = "Недоступно"
                    except Exception as e:
                        auc = f"Ошибка: {str(e)[:50]}"

                    line = (f"Точность={acc:.4f}, "
                            f"Precision={prec:.4f}, "
                            f"Recall={rec:.4f}, "
                            f"F1-Score={f1:.4f}, "
                            f"ROC-AUC={auc}")
                else:  # Регрессия
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

                    line = (f"R²={r2:.4f}, "
                            f"MSE={mse:.4f}, "
                            f"MAE={mae:.4f}")

                elapsed = (datetime.now() - start_time).total_seconds()
                total_time += elapsed
                results.append((model_display_name, line))

            time_text = f"Время выполнения: {total_time:.4f} секунд"
            self.finished_signal.emit(results, time_text)

        except Exception as e:
            self.error_signal.emit(str(e))


# === Основной класс обработки данных ===
class DataModelHandler:
    def __init__(self, parent, df=None, combobox=None, checkboxes=None,
                 labels_and_lines=None, accuracy_label=None, time_label=None, task_type="classification"):
        self.parent = parent
        self.df = df
        self.combobox = combobox
        self.checkboxes = checkboxes
        self.labels_and_lines = labels_and_lines
        self.accuracy_label = accuracy_label
        self.time_label = time_label
        self.task_type = task_type

        # Для анализа важности признаков
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.thread = None
        self.splash = None

    def set_df(self, dataframe):
        self.df = dataframe

    def update_dataframe(self, new_df):
        self.df = new_df
        if self.combobox:
            self.combobox.clear()
            self.combobox.addItems(new_df.columns.tolist())
            self.combobox.setEnabled(True)

    def evaluate_models(self):
        if self.df is None or self.df.empty:
            QMessageBox.critical(self.parent, "Ошибка", "Датасет не загружен!")
            return

        target_col = self.parent.target_var_combobox.currentText()
        if not target_col:
            QMessageBox.critical(self.parent, "Ошибка", "Не выбрана целевая переменная!")
            return

        # === Проверка типа целевой переменной ===
        target_dtype = self.df[target_col].dtype
        if self.task_type == "classification":
            if not np.issubdtype(target_dtype, np.number) and target_dtype != 'bool':
                le = LabelEncoder()
                self.df[target_col] = le.fit_transform(self.df[target_col])
        elif self.task_type == "regression":
            if not np.issubdtype(target_dtype, np.number):
                QMessageBox.critical(
                    self.parent, "Ошибка",
                    f"Целевая переменная '{target_col}' не является числовой.\n"
                    "Регрессия требует числовой целевой переменной."
                )
                return

        # Только числовые признаки
        X = self.df.drop(columns=[target_col]).select_dtypes(include=['number', 'Int64'])
        y = self.df[target_col]

        if X.empty:
            QMessageBox.critical(
                self.parent, "Ошибка",
                "После удаления нечисловых колонок не осталось признаков для обучения.\n"
                "Пожалуйста, закодируйте категориальные переменные."
            )
            return

        # Уведомление о проигнорированных колонках
        non_numeric = self.df.drop(columns=[target_col]).select_dtypes(include=['object', 'string', 'category'])
        if not non_numeric.empty:
            ignored_cols = ', '.join(non_numeric.columns)
            msg_box = QMessageBox(self.parent)
            msg_box.setWindowTitle("Информация о признаках")
            msg_box.setText("Следующие колонки не являются числовыми и не будут использованы:")
            msg_box.setInformativeText(f"<b>{ignored_cols}</b>")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()

        # === Сбор моделей ===
        models_config = []
        for checkbox in self.checkboxes:
            if not checkbox.isChecked():
                continue

            model_name = checkbox.text()  # Полное имя: "Random Forest Classification"
            params = self.labels_and_lines.get(model_name, {})

            try:
                test_size = float(params.get('Test Size', QLineEdit('0.2')).text().strip())
                random_state = int(params.get('Random State', QLineEdit('42')).text().strip())

                # === Определение типа модели по имени ===
                if 'Random Forest Classification' in model_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                elif 'Gradient Boosting Classification' in model_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
                elif 'Logistic Regression Classification' in model_name:
                    C = float(params['C'].text())
                    max_iter = int(params['Max Iterations'].text())
                    penalty = params['Penalty'].text().strip()
                    clf = LogisticRegression(
                        C=C, max_iter=max_iter, penalty=penalty, solver='lbfgs', random_state=random_state
                    )
                elif 'Random Forest Regression' in model_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
                elif 'Gradient Boosting Regression' in model_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
                else:
                    continue  # Неизвестная модель

                models_config.append((model_name, clf, test_size, random_state))

            except Exception as e:
                QMessageBox.critical(self.parent, "Ошибка", f"Ошибка в параметрах {model_name}:\n{e}")
                return

        if not models_config:
            QMessageBox.warning(self.parent, "Предупреждение", "Не выбрано ни одной модели!")
            return

        # Запуск в потоке
        self.splash = LoadingScreen()
        self.splash.show()

        self.thread = EvaluationThread(self.parent, models_config, X, y, self.task_type)
        self.thread.finished_signal.connect(self.on_evaluation_finished)
        self.thread.error_signal.connect(self.on_evaluation_error)
        self.thread.start()

    def on_evaluation_finished(self, results, time_text):
        if self.splash:
            self.splash.close()

        report_lines = []
        for model_name, metrics_str in results:
            line = f"<b>{model_name}:</b><br>{metrics_str}"
            report_lines.append(line)

        self.time_label.setText(time_text)
        if hasattr(self.parent, 'update_metrics_display'):
            self.parent.update_metrics_display(report_lines)

    def on_evaluation_error(self, error_msg):
        if self.splash:
            self.splash.close()
        QMessageBox.critical(self.parent, "Ошибка при обучении", f"Произошла ошибка:\n{error_msg}")

    def split_dataset(self):
        """Разделение данных для анализа важности признаков"""
        test_size = 0.2
        random_state = 42

        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                model_name = checkbox.text()
                params = self.labels_and_lines.get(model_name, {})
                try:
                    test_size = float(params.get('Test Size', QLineEdit('0.2')).text().strip())
                    random_state = int(params.get('Random State', QLineEdit('42')).text().strip())
                except:
                    pass
                break

        target_col = self.parent.target_var_combobox.currentText()
        X = self.df.drop(columns=[target_col]).select_dtypes(include=['number', 'Int64'])
        y = self.df[target_col]

        if X.empty:
            raise ValueError("Нет числовых признаков. Закодируйте категориальные переменные.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def calculate_feature_importances(self, selected_models):
        splash_screen = LoadingScreen()
        splash_screen.show()

        try:
            self.split_dataset()
        except Exception as e:
            splash_screen.close()
            QMessageBox.critical(self.parent, "Ошибка", str(e))
            return

        results = {}
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        feature_names = self.X_train.columns.tolist()

        for model_display_name in selected_models:
            try:
                params = self.labels_and_lines.get(model_display_name, {})
                if not params:
                    continue

                # === Извлечение параметров ===
                try:
                    random_state = int(params['Random State'].text().strip())
                except:
                    random_state = 42

                # === Создание модели по полному имени ===
                if 'Random Forest Classification' in model_display_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                elif 'Gradient Boosting Classification' in model_display_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
                elif 'Logistic Regression Classification' in model_display_name:
                    C = float(params['C'].text())
                    max_iter = int(params['Max Iterations'].text())
                    penalty = params['Penalty'].text().strip()
                    clf = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver='liblinear')
                elif 'Random Forest Regression' in model_display_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
                elif 'Gradient Boosting Regression' in model_display_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
                else:
                    continue

                # Обучение
                clf.fit(X_train_scaled, self.y_train)

                # Извлечение важности
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                elif hasattr(clf, 'coef_'):
                    importances = np.abs(clf.coef_.ravel() if clf.coef_.ndim > 1 else clf.coef_)
                else:
                    importances = np.ones(len(feature_names))

                features_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                features_df = features_df.sort_values(by='Importance', ascending=False)

                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=features_df)
                plt.title(f"Важность признаков — {model_display_name}")
                plt.tight_layout()
                os.makedirs("plots", exist_ok=True)
                plt.savefig(f"plots/{model_display_name.replace(' ', '_')}_feature_importance.png")
                plt.show()

                results[model_display_name] = features_df

            except Exception as e:
                QMessageBox.critical(self.parent, "Ошибка", f"Ошибка при построении графика {model_display_name}:\n{e}")

        splash_screen.close()
        return results
