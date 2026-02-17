from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
import numpy as np
from PySide6.QtWidgets import QComboBox, QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget, QGridLayout, QCheckBox, QSpacerItem, QSizePolicy, QHBoxLayout, QPushButton
import os
import psutil

def create_model(model_name, params):
    """Создаёт модель по имени и параметрам"""
    random_state = safe_int(params, 'Random State', 42)
    n_estimators = safe_int(params, 'Кол-во деревьев', 100)

    if 'Random Forest Classification' in model_name:
        max_depth = safe_int_or_none(params, 'Max Depth', None)
        min_samples_split = safe_int(params, 'Min Samples Split', 2)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
    elif 'Gradient Boosting Classification' in model_name:
        max_depth = safe_int_or_none(params, 'Max Depth', 3)
        learning_rate = safe_float(params, 'Learning Rate', 0.1)
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    elif 'Logistic Regression Classification' in model_name:
        C = safe_float(params, 'C', 1.0)
        max_iter = safe_int(params, 'Max Iterations', 100)
        penalty = params.get('Penalty', None)
        penalty = penalty.currentText() if isinstance(penalty, QComboBox) else penalty.text().strip() if penalty else 'l2'
        penalty = penalty if penalty in ['l1', 'l2', 'none'] else 'l2'
        solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
        return LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, random_state=random_state)
    elif 'Random Forest Regression' in model_name:
        max_depth = safe_int_or_none(params, 'Max Depth', None)
        min_samples_split = safe_int(params, 'Min Samples Split', 2)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
    elif 'Gradient Boosting Regression' in model_name:
        max_depth = safe_int_or_none(params, 'Max Depth', 3)
        learning_rate = safe_float(params, 'Learning Rate', 0.1)
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    elif 'Linear Regression' in model_name:
        fit_intercept = params.get('Fit Intercept', None)
        fit_intercept = fit_intercept.currentText() == 'True' if isinstance(fit_intercept, QComboBox) else True
        normalize = params.get('Normalize', None)
        normalize = normalize.currentText() == 'True' if isinstance(normalize, QComboBox) else False
        return LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")


def get_importances(clf):
    """Получает важность признаков из модели"""
    if hasattr(clf, 'feature_importances_'):
        return clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        coef = np.abs(clf.coef_)
        return coef.mean(axis=0) if coef.ndim > 1 else coef.ravel()
    else:
        raise AttributeError("Модель не поддерживает важность признаков")


def safe_int(params, key, default):
    """Безопасное извлечение целого числа из параметров"""
    try:
        val = params[key]
        if isinstance(val, QComboBox):
            val = val.currentText()
        else:
            val = val.text().strip()
        return int(val) if val else default
    except:
        return default


def safe_float(params, key, default):
    """Безопасное извлечение вещественного числа"""
    try:
        val = params[key].text().strip()
        return float(val) if val else default
    except:
        return default


def safe_int_or_none(params, key, default):
    """Безопасное извлечение целого или None"""
    try:
        val = params[key].text().strip()
        if not val or val.lower() in ('none', 'null'):
            return None
        return int(val)
    except:
        return default


def train_model(model_name, params, X_train, y_train, n_jobs=1):
    """Обучает модель и возвращает её"""
    try:
        X_scaled = StandardScaler().fit_transform(X_train)
        clf = create_model(model_name, params)
        with parallel_backend('loky', n_jobs=n_jobs):
            clf.fit(X_scaled, y_train)
        return {
            'success': True,
            'model': clf,
            'importances': get_importances(clf) if hasattr(clf, 'feature_importances_') or hasattr(clf, 'coef_') else None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def analyze_shap(explainer_type, model, X_train, sample_size="1000", model_task="auto"):
    """Анализ SHAP-значений"""
    try:
        import shap
        print(f"X_train type: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'unknown'}, dtype: {X_train.dtypes if hasattr(X_train, 'dtypes') else 'unknown'}")
        X_scaled = StandardScaler().fit_transform(X_train)
        print(f"X_scaled contains NaN: {np.isnan(X_scaled).any()}")
        
        if sample_size == "все":
            X_sample = X_scaled
        else:
            sample_size = int(sample_size)
            # Проверяем, что размер выборки не превышает размер данных
            actual_sample_size = min(sample_size, len(X_scaled))
            # При необходимости разрешаем замены
            replace = actual_sample_size > len(X_scaled)
            idx = np.random.choice(len(X_scaled), actual_sample_size, replace=replace)
            X_sample = X_scaled[idx]

        # Выбор объяснителя
        # Принудительная проверка X_sample перед использованием
        if X_sample.size == 0:
            raise ValueError("X_sample пустой после выборки")
        
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(-1, 1)

        if X_sample.ndim != 2:
            raise ValueError(f"X_sample должен быть 2D массивом, но имеет форму {X_sample.shape}")

        if np.isnan(X_sample).any():
            X_sample = np.nan_to_num(X_sample, nan=0.0)
            print("Внимание: X_sample содержал NaN и был заменён на нули.")

        # Выбор объяснителя
        if explainer_type == "TreeExplainer" and hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
        elif explainer_type == "LinearExplainer" and hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_sample)
        elif explainer_type == "KernelExplainer":
            explainer = shap.KernelExplainer(model.predict, X_sample)
        else:
            explainer = shap.Explainer(model)

        # Повторная проверка после всех преобразований
        if X_sample.ndim != 2:
            raise ValueError(f"X_sample должен быть 2D массивом, но имеет форму {X_sample.shape}")

        if np.isnan(X_sample).any():
            X_sample = np.nan_to_num(X_sample, nan=0.0)
            print("Внимание: X_sample содержал NaN и был заменён на нули.")

        shap_values = explainer(X_sample)
        
        return {
            'success': True,
            'shap_values': shap_values,
            'explainer': explainer,
            'X_sample': X_sample
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def kill_child_processes():
    """Forcibly terminates all child processes (e.g., from joblib)."""
    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        if not children:
            return
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        gone, alive = psutil.wait_procs(children, timeout=3)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"Error terminating processes: {e}")