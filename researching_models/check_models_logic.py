# check_models_logic.py — Оценка моделей (с прокруткой, кнопкой, макс. 6)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QMessageBox, QVBoxLayout, QGroupBox, QLabel, QPushButton, QTextEdit, QFrame
)
from PySide6.QtCore import QThread, Signal
from .check_models_loading_screen import LoadingScreen
from datetime import datetime


class EvaluationThread(QThread):
    finished_signal = Signal(list, str)
    error_signal = Signal(str)

    def __init__(self, parent, models_config, X_train, X_test, y_train, y_test, task_type):
        super().__init__(parent)
        self.models_config = models_config
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task_type = task_type

    def run(self):
        try:
            results = []
            total_time = 0.0
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)

            for model_display_name, clf in self.models_config:
                start_time = datetime.now()
                clf.fit(X_train_scaled, self.y_train)
                y_pred = clf.predict(X_test_scaled)

                if self.task_type == "classification":
                    n_classes = len(np.unique(self.y_train))
                    avg = 'weighted' if n_classes > 2 else 'binary'
                    acc = accuracy_score(self.y_test, y_pred)
                    prec = precision_score(self.y_test, y_pred, average=avg, zero_division=0)
                    rec = recall_score(self.y_test, y_pred, average=avg, zero_division=0)
                    f1 = f1_score(self.y_test, y_pred, average=avg, zero_division=0)
                    try:
                        if hasattr(clf, "predict_proba"):
                            probas = clf.predict_proba(X_test_scaled)
                            auc = roc_auc_score(self.y_test, probas, multi_class='ovr', average='weighted') if probas.shape[1] > 2 else roc_auc_score(self.y_test, probas[:, 1])
                        else:
                            auc = "Недоступно"
                    except:
                        auc = "Ошибка"
                    metrics = f"Точность={acc:.4f}\nPrecision={prec:.4f}\nRecall={rec:.4f}\nF1-Score={f1:.4f}\nROC-AUC={auc}"
                else:
                    r2 = r2_score(self.y_test, y_pred)
                    mse = mean_squared_error(self.y_test, y_pred)
                    mae = mean_absolute_error(self.y_test, y_pred)
                    metrics = f"R²={r2:.4f}\nMSE={mse:.4f}\nMAE={mae:.4f}"

                elapsed = (datetime.now() - start_time).total_seconds()
                total_time += elapsed
                results.append((model_display_name, metrics, clf))  # ← clf для графика

            time_text = f"Время выполнения: {total_time:.4f} секунд"
            self.finished_signal.emit(results, time_text)
        except Exception as e:
            self.error_signal.emit(str(e))


class DataModelHandler:
    def __init__(self, parent, df=None, combobox=None, checkboxes=None, labels_and_lines=None, accuracy_label=None, time_label=None, task_type="classification", results_layout=None):
        self.parent = parent
        self.df = df
        self.combobox = combobox
        self.checkboxes = checkboxes
        self.labels_and_lines = labels_and_lines
        self.time_label = time_label
        self.task_type = task_type
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_col = None
        self.thread = None
        self.splash = None
        self.results_layout = results_layout  # Для горизонтальной прокрутки

    def update_dataframe(self, new_df):
        self.df = new_df
        if self.combobox:
            self.combobox.clear()
            self.combobox.addItems(new_df.columns.tolist())
            self.combobox.setEnabled(True)

    def set_split_data(self, X_train, X_test, y_train, y_test, target_col):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.target_col = target_col
        self.df = None
        if self.combobox:
            self.combobox.clear()
            self.combobox.addItem(target_col)
            self.combobox.setEnabled(True)

    def evaluate_models(self):
        if self.X_train is not None and self.y_train is not None:
            X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        else:
            if self.df is None:
                QMessageBox.critical(self.parent, "Ошибка", "Датасет не загружен!")
                return
            target_col = self.parent.target_var_combobox.currentText()
            if not target_col:
                QMessageBox.critical(self.parent, "Ошибка", "Не выбрана целевая переменная!")
                return
            if self.task_type == "classification":
                le = LabelEncoder()
                self.df[target_col] = le.fit_transform(self.df[target_col])
            elif self.task_type == "regression" and not np.issubdtype(self.df[target_col].dtype, np.number):
                QMessageBox.critical(self.parent, "Ошибка", "Регрессия требует числовой целевой переменной.")
                return
            X = self.df.drop(columns=[target_col]).select_dtypes(include=['number'])
            y = self.df[target_col]
            if X.empty:
                QMessageBox.critical(self.parent, "Ошибка", "Нет числовых признаков.")
                return
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self._run_evaluation(X_train, X_test, y_train, y_test)

    def _run_evaluation(self, X_train, X_test, y_train, y_test):
        models_config = []
        for checkbox in self.checkboxes:
            if not checkbox.isChecked():
                continue
            model_name = checkbox.text()
            params = self.labels_and_lines.get(model_name, {})
            try:
                if 'Random Forest Classification' in model_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                elif 'Gradient Boosting Classification' in model_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
                elif 'Logistic Regression Classification' in model_name:
                    C = float(params['C'].text())
                    max_iter = int(params['Max Iterations'].text())
                    penalty = params['Penalty'].text().strip()
                    clf = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver='liblinear')
                elif 'Random Forest Regression' in model_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                elif 'Gradient Boosting Regression' in model_name:
                    n_estimators = int(params['Количество деревьев'].text())
                    clf = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
                else:
                    continue
                models_config.append((model_name, clf))
            except Exception as e:
                QMessageBox.critical(self.parent, "Ошибка", f"Ошибка в параметрах {model_name}:\n{e}")
                return

        if not models_config:
            QMessageBox.warning(self.parent, "Предупреждение", "Не выбрано ни одной модели!")
            return

        self.splash = LoadingScreen()
        self.splash.show()
        self.thread = EvaluationThread(self.parent, models_config, X_train, X_test, y_train, y_test, self.task_type)
        self.thread.finished_signal.connect(self.on_evaluation_finished)
        self.thread.error_signal.connect(self.on_evaluation_error)
        self.thread.start()

    def on_evaluation_finished(self, results, time_text):
        if self.splash:
            self.splash.close()

        # ✅ Удаляем самый старый результат, если больше 5 (чтобы 6 стало максимумом)
        while self.results_layout.count() > 5:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        feature_names = self.X_train.columns.tolist()

        for model_name, metrics, clf in results:
            # === Блок для модели ===
            model_group = QGroupBox(f" {model_name} ")
            model_group.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #aaa;
                    border-radius: 6px;
                    margin: 0;
                    padding: 10px;
                    min-width: 240px;
                }
            """)
            model_layout = QVBoxLayout()

            # Метрики (многострочный текст)
            metrics_text = QTextEdit()
            metrics_text.setPlainText(metrics)
            metrics_text.setFixedHeight(120)
            metrics_text.setReadOnly(True)
            model_layout.addWidget(metrics_text)

            # Кнопка "График"
            plot_btn = QPushButton("📊 График важности")
            plot_btn.clicked.connect(
                lambda ch, c=clf, names=feature_names, mn=model_name:
                self.plot_importance(c, names, mn)
            )
            model_layout.addWidget(plot_btn)

            model_group.setLayout(model_layout)
            self.results_layout.addWidget(model_group)

        self.time_label.setText(time_text)

    def on_evaluation_error(self, error_msg):
        if self.splash:
            self.splash.close()
        QMessageBox.critical(self.parent, "Ошибка", f"Произошла ошибка:\n{error_msg}")

    def plot_importance(self, clf, feature_names, model_name):
        try:
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
            elif hasattr(clf, 'coef_'):
                coef = np.abs(clf.coef_)
                importances = coef.mean(axis=0) if coef.ndim > 1 else coef.ravel()
            else:
                QMessageBox.critical(self.parent, "Ошибка", f"Нет важности: {model_name}")
                return

            df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            df_imp = df_imp.sort_values('Importance', ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_imp, x='Importance', y='Feature')
            plt.title(f"Важность признаков — {model_name}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.critical(self.parent, "Ошибка", f"Не удалось построить график: {e}")
