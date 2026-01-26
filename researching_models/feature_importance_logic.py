# feature_importance_logic.py
from PySide6.QtWidgets import QMessageBox
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from .check_models_loading_screen import LoadingScreen


class FeatureImportanceCalculator:
    def __init__(self, parent, X_train, y_train, labels_and_lines, task_type, top_features_callback):
        self.parent = parent
        self.X_train = X_train
        self.y_train = y_train
        self.labels_and_lines = labels_and_lines
        self.task_type = task_type
        self.top_features_callback = top_features_callback

    def calculate(self, selected_models):
        splash = LoadingScreen()
        splash.show()

        X_scaled = StandardScaler().fit_transform(self.X_train)
        feature_names = self.X_train.columns.tolist()

        for model_name in selected_models:
            try:
                params = self.labels_and_lines.get(model_name, {})
                clf = self._create_model(model_name, params)
                clf.fit(X_scaled, self.y_train)
                importances = self._get_importances(clf)

                # ТОП-5 признаков
                idx = np.argsort(importances)[::-1]
                top_5 = [feature_names[i] for i in idx[:5]]
                self.top_features_callback(model_name, top_5)

                # Построение графика
                self._plot_with_plt(importances, feature_names, model_name)
            except Exception as e:
                QMessageBox.critical(self.parent, "Ошибка", f"Ошибка в {model_name}:\n{e}")

        splash.close()

    def _create_model(self, name, params):
        random_state = self.safe_int(params, 'Random State', 42)
        n_estimators = self.safe_int(params, 'Кол-во деревьев', 100)

        if 'Random Forest Classification' in name:
            max_depth = self.safe_int_or_none(params, 'Max Depth', None)
            min_samples_split = self.safe_int(params, 'Min Samples Split', 2)
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        elif 'Gradient Boosting Classification' in name:
            max_depth = self.safe_int_or_none(params, 'Max Depth', 3)
            learning_rate = self.safe_float(params, 'Learning Rate', 0.1)
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        elif 'Logistic Regression Classification' in name:
            C = self.safe_float(params, 'C', 1.0)
            max_iter = self.safe_int(params, 'Max Iterations', 100)
            penalty = params.get('Penalty', None)
            penalty = penalty.text().strip() if penalty else 'l2'
            penalty = penalty if penalty in ['l1', 'l2', 'none'] else 'l2'
            solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
            return LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, random_state=random_state)
        elif 'Random Forest Regression' in name:
            max_depth = self.safe_int_or_none(params, 'Max Depth', None)
            min_samples_split = self.safe_int(params, 'Min Samples Split', 2)
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        elif 'Gradient Boosting Regression' in name:
            max_depth = self.safe_int_or_none(params, 'Max Depth', 3)
            learning_rate = self.safe_float(params, 'Learning Rate', 0.1)
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            raise ValueError(f"Неизвестная модель: {name}")

    def _get_importances(self, clf):
        if hasattr(clf, 'feature_importances_'):
            return clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            coef = np.abs(clf.coef_)
            return coef.mean(axis=0) if coef.ndim > 1 else coef.ravel()
        else:
            raise AttributeError("Модель не поддерживает важность признаков")

    def _plot_with_plt(self, importances, feature_names, model_name):
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values('Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_imp, x='Importance', y='Feature')
        plt.title(f"Важность признаков — {model_name}")
        plt.tight_layout()
        plt.show()

    def safe_int(self, params, key, default):
        try:
            val = params[key].text().strip()
            return int(val) if val else default
        except:
            return default

    def safe_float(self, params, key, default):
        try:
            val = params[key].text().strip()
            return float(val) if val else default
        except:
            return default

    def safe_int_or_none(self, params, key, default):
        try:
            val = params[key].text().strip()
            if not val or val.lower() in ('none', 'null'):
                return None
            return int(val)
        except:
            return default
