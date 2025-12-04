# check_models_logic.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import QThread, Signal
from .check_models_loading_screen import LoadingScreen
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Поток для оценки моделей ===
class EvaluationThread(QThread):
    # Сигналы
    finished_signal = Signal(list, str)  # (results, time_text)
    error_signal = Signal(str)

    def __init__(self, parent, models_config, X, y, n_classes):
        super().__init__(parent)
        self.models_config = models_config
        self.X = X
        self.y = y
        self.n_classes = n_classes

    def run(self):
        try:
            results = []
            total_time = 0.0

            for model_name, clf, test_size, random_state in self.models_config:
                # Разделение
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=random_state
                )

                # Масштабирование
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Обучение
                start_time = datetime.now()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Метрики
                average_mode = 'weighted' if self.n_classes > 2 else 'binary'
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
                rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)

                # ROC-AUC
                try:
                    if hasattr(clf, "predict_proba"):
                        probas = clf.predict_proba(X_test)

                        if self.n_classes == 2:
                            # Бинарная задача
                            auc = roc_auc_score(y_test, probas[:, 1])
                        else:
                            # Многоклассовая — обязательно указываем multi_class
                            auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
                    else:
                        auc = "Недоступно (нет predict_proba)"
                except ValueError as e:
                    if "multi_class must be in" in str(e):
                        auc = "Ошибка: требуется multi_class='ovr'"
                    else:
                        auc = f"Ошибка: {str(e)[:50]}"
                except Exception as e:
                    auc = f"Ошибка: {str(e)[:50]}"

                elapsed = (datetime.now() - start_time).total_seconds()
                total_time += elapsed

                results.append((model_name, acc, prec, rec, f1, auc))

            time_text = f"Время выполнения: {total_time:.4f} секунд"
            self.finished_signal.emit(results, time_text)

        except Exception as e:
            self.error_signal.emit(str(e))

# === Основной класс обработки данных ===
class DataModelHandler:
    def __init__(self, parent, df=None, combobox=None, checkboxes=None,
                 labels_and_lines=None, accuracy_label=None, time_label=None):
        self.parent = parent
        self.df = df
        self.combobox = combobox
        self.checkboxes = checkboxes
        self.labels_and_lines = labels_and_lines
        self.accuracy_label = accuracy_label
        self.time_label = time_label

        # Для потока
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

        # Удаляем целевую переменную и оставляем ТОЛЬКО числовые колонки
        X = self.df.drop(columns=[target_col]).select_dtypes(include=['number', 'Int64'])
        y = self.df[target_col]

        # Проверка: остались ли признаки
        if X.empty:
            QMessageBox.critical(
                self.parent, "Ошибка",
                "После удаления нечисловых колонок не осталось признаков для обучения.\n"
                "Пожалуйста, закодируйте категориальные переменные (One-Hot, Label и т.д.)."
            )
            return

        # 🔔 Показываем пользователю, какие колонки проигнорированы
        non_numeric = self.df.drop(columns=[target_col]).select_dtypes(include=['object', 'string', 'category'])
        if not non_numeric.empty:
            ignored_cols = ', '.join(non_numeric.columns)
            msg_box = QMessageBox(self.parent)
            msg_box.setWindowTitle("Информация о признаках")
            msg_box.setText("Следующие колонки не являются числовыми и не будут использованы в обучении моделей:")
            msg_box.setInformativeText(f"<b>{ignored_cols}</b>")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()

        # Определение типа задачи
        n_classes = len(y.unique())
        msg = f"Обнаружено {n_classes} классов.\n"
        msg += "Задача: Бинарная классификация" if n_classes == 2 else f"Многоклассовая классификация ({n_classes} класса)"
        QMessageBox.information(self.parent, "Тип задачи", msg)

        # === Сбор моделей ===
        models_config = []
        for checkbox in self.checkboxes:
            if not checkbox.isChecked():
                continue

            model_name = checkbox.text()
            params = self.labels_and_lines.get(model_name, {})

            try:
                # ✅ Инициализация test_size и random_state по умолчанию
                test_size = 0.2      # значение по умолчанию
                random_state = 42    # значение по умолчанию

                # Чтение из полей, если они есть
                if 'Test Size' in params:
                    test_size_val = params['Test Size'].text().strip()
                    if test_size_val:
                        test_size = float(test_size_val)

                if 'Random State' in params:
                    random_state_val = params['Random State'].text().strip()
                    if random_state_val:
                        random_state = int(random_state_val)

                # Создание модели
                if model_name == 'Random Forest':
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

                elif model_name == 'Gradient Boosting':
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)

                elif model_name == 'Logistic Regression':
                    C = float(params['C'].text())
                    max_iter = int(params['Max Iterations'].text())
                    penalty = params['Penalty'].text().strip()
                    clf = LogisticRegression(
                        C=C, max_iter=max_iter, penalty=penalty, solver='lbfgs', random_state=random_state  # ← тоже использует random_state
                    )
                else:
                    continue  # неизвестная модель

                # ✅ Теперь test_size и random_state гарантированно существуют
                models_config.append((model_name, clf, test_size, random_state))

            except Exception as e:
                QMessageBox.critical(self.parent, "Ошибка", f"Ошибка в параметрах {model_name}:\n{e}")
                return

        if not models_config:
            QMessageBox.warning(self.parent, "Предупреждение", "Не выбрано ни одной модели!")
            return

        # === Запуск в потоке ===
        self.splash = LoadingScreen()
        self.splash.show()

        self.thread = EvaluationThread(self.parent, models_config, X, y, n_classes)
        self.thread.finished_signal.connect(self.on_evaluation_finished)
        self.thread.error_signal.connect(self.on_evaluation_error)
        self.thread.start()



    def on_evaluation_finished(self, results, time_text):
        if self.splash:
            self.splash.close()

        # Формирование отчёта
        report_lines = []
        for result in results:
            model_name, acc, prec, rec, f1, auc = result
            line = (f"<b>{model_name}:</b><br>"
                    f"Точность={acc:.4f}, "
                    f"Precision={prec:.4f}, "
                    f"Recall={rec:.4f}, "
                    f"F1-Score={f1:.4f}, "
                    f"ROC-AUC={auc}")
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
                    test_size = float(params['Test Size'].text())
                    random_state = int(params['Random State'].text())
                except:
                    pass
                break

        target_col = self.df.columns[-1]  # или взять из интерфейса

        # Только числовые признаки
        X = self.df.drop(columns=[target_col]).select_dtypes(include=['number', 'Int64'])
        y = self.df[target_col]

        if X.empty:
            raise ValueError(
                "Нет числовых признаков. Закодируйте категориальные переменные перед анализом."
            )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )


    def calculate_feature_importances(self, selected_models=None):
        splash_screen = LoadingScreen()
        splash_screen.show()

        self.split_dataset()
        results = {}

        for model_name, active in selected_models.items():
            if not active:
                continue

            params = self.labels_and_lines.get(model_name, {})
            try:
                if model_name == 'Random Forest':
                    n_estimators = int(params['Количество деревьев'].text())
                    random_state = int(params['Random State'].text())
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

                elif model_name == 'Gradient Boosting':
                    n_estimators = int(params['Количество деревьев'].text())
                    random_state = int(params['Random State'].text())
                    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)

                elif model_name == 'Logistic Regression':
                    C = float(params['C'].text())
                    max_iter = int(params['Max Iterations'].text())
                    penalty = params['Penalty'].text().strip()
                    clf = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver='liblinear')

                else:
                    continue

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(self.X_train)
                X_test_scaled = scaler.transform(self.X_test)

                clf.fit(X_train_scaled, self.y_train)
                feature_names = list(self.df.columns[:-1])
                importances = (
                    clf.feature_importances_ if hasattr(clf, 'feature_importances_')
                    else np.abs(clf.coef_[0])
                )

                features_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                features_df = features_df.sort_values(by='Importance', ascending=False)

                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=features_df)
                plt.title(f"Важность признаков ({model_name})")
                plt.tight_layout()
                os.makedirs("plots", exist_ok=True)
                plt.savefig(f"plots/{model_name}_feature_importance.png")
                plt.show()

                results[model_name] = features_df

            except Exception as e:
                QMessageBox.critical(self.parent, "Ошибка", f"Ошибка при построении графика {model_name}:\n{e}")

        splash_screen.close()
        return results
