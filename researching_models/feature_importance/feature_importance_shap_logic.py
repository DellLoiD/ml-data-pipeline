from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
import numpy as np
from PySide6.QtWidgets import QComboBox
import os
import psutil
import logging

logger = logging.getLogger(__name__)

def create_model(model_name, params, task_type="classification"):
    """Создаёт модель по имени и параметрам"""
    random_state = safe_int(params, 'Random State')
    n_estimators = safe_int(params, 'Кол-во деревьев')

    if 'Random Forest' in model_name and task_type == "classification":
        print(f"[DEBUG] Условие 'Random Forest' и 'classification' выполнено для model_name='{model_name}'")
        max_depth = safe_int_or_none(params, 'Max Depth')
        min_samples_split = safe_int(params, 'Min Samples Split')
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
    elif 'Gradient Boosting' in model_name and task_type == "classification":
        max_depth = safe_int_or_none(params, 'Max Depth')
        learning_rate = safe_float(params, 'Learning Rate')
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    elif 'Logistic Regression' in model_name and task_type == "classification":
        C = safe_float(params, 'C')
        max_iter = safe_int(params, 'Max Iterations')
        penalty = params.get('Penalty', None)
        penalty = penalty.currentText() if isinstance(penalty, QComboBox) else penalty.text().strip()
        penalty = penalty if penalty in ['l1', 'l2', 'none'] else 'l2'
        solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
        return LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, random_state=random_state)
    elif 'Random Forest' in model_name and task_type == "regression":
        max_depth = safe_int_or_none(params, 'Max Depth')
        min_samples_split = safe_int(params, 'Min Samples Split')
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
    elif 'Gradient Boosting' in model_name and task_type == "regression":
        max_depth = safe_int_or_none(params, 'Max Depth')
        learning_rate = safe_float(params, 'Learning Rate')
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    elif 'Linear Regression' in model_name and task_type == "regression":
        fit_intercept = params.get('Fit Intercept', None)
        fit_intercept = fit_intercept.currentText() == 'True' if isinstance(fit_intercept, QComboBox) else True
        normalize = params.get('Normalize', None)
        normalize = normalize.currentText() == 'True' if isinstance(normalize, QComboBox) else True
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


def safe_int(params, key):
    """Безопасное извлечение целого числа из параметров"""
    try:
        val = params[key]
        if isinstance(val, QComboBox):
            val = val.currentText()
        else:
            val = val.text().strip()
        return int(val) if val else None
    except:
        return None


def safe_float(params, key):
    """Безопасное извлечение вещественного числа"""
    try:
        val = params[key].text().strip()
        return float(val) if val else None
    except:
        return None


def safe_int_or_none(params, key):
    """Безопасное извлечение целого или None"""
    try:
        val = params[key].text().strip()
        if not val or val.lower() in ('none', 'null'):
            return None
        return int(val)
    except:
        return None


def train_model(model_name, params, X_train, y_train, n_jobs):
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
    logger.info("Начало выполнения analyze_shap в feature_importance_shap_logic.py")
    logger.info(f"Тип explainer_type: {type(explainer_type)}, значение: {explainer_type}")
    logger.info(f"Тип model: {type(model)}")
    logger.info(f"Тип X_train: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'unknown'}")
    logger.info(f"Тип sample_size: {type(sample_size)}, значение: {sample_size}")
    logger.info(f"Тип model_task: {type(model_task)}, значение: {model_task}")
    
    try:
        import shap
        logger.info("SHAP импортирован успешно")
        
        print(f"X_train type: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'unknown'}, dtype: {X_train.dtypes if hasattr(X_train, 'dtypes') else 'unknown'}")
        if X_train.isnull().values.any():
            logger.warning("X_train содержит NaN значения, заполняем нулями")
            X_train = X_train.fillna(0)
        X_scaled = StandardScaler().fit_transform(X_train)
        logger.info(f"X_scaled преобразован, форма: {X_scaled.shape}, содержит NaN: {np.isnan(X_scaled).any()}")
        
        if sample_size == "все":
            X_sample = X_scaled
            logger.info("Выбрана вся обучающая выборка для X_sample")
        else:
            try:
                sample_size = int(sample_size)
            except ValueError:
                logger.error(f"Невозможно преобразовать sample_size '{sample_size}' в целое число. Используем значение по умолчанию 100.")
                sample_size = 100
            
            # Проверяем, что размер выборки не превышает размер данных
            actual_sample_size = min(sample_size, len(X_scaled))
            # При необходимости разрешаем замены
            replace = actual_sample_size > len(X_scaled)
            logger.info(f"Выборка: запрашиваемый размер {sample_size}, фактический размер {actual_sample_size}, replace={replace}")
            
            idx = np.random.choice(len(X_scaled), actual_sample_size, replace=replace)
            X_sample = X_scaled[idx]
            logger.info(f"X_sample сформирован, форма: {X_sample.shape}")

        # Принудительная проверка X_sample перед использованием
        if X_sample.size == 0:
            error_msg = "X_sample пустой после выборки"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(-1, 1)
            logger.info(f"X_sample был 1D, преобразован в 2D: {X_sample.shape}")

        if X_sample.ndim != 2:
            error_msg = f"X_sample должен быть 2D массивом, но имеет форму {X_sample.shape}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

        if np.isnan(X_sample).any():
            X_sample = np.nan_to_num(X_sample, nan=0.0)
            logger.warning("X_sample содержал NaN и был заменён на нули.")

        # Выбор объяснителя
        logger.info(f"Инициализация SHAP объяснителя: {explainer_type}")
        
        if explainer_type == "TreeExplainer" and hasattr(model, 'estimators_'):
            logger.info("Используется TreeExplainer")
            explainer = shap.TreeExplainer(model)
        elif explainer_type == "LinearExplainer" and hasattr(model, 'coef_'):
            logger.info("Используется LinearExplainer")
            explainer = shap.LinearExplainer(model, X_sample)
        elif explainer_type == "KernelExplainer":
            logger.info("Используется KernelExplainer")
            explainer = shap.KernelExplainer(model.predict, X_sample)
        else:
            logger.info("Используется универсальный Explainer")
            explainer = shap.Explainer(model.predict, X_sample)

        # Повторная проверка после всех преобразований
        if X_sample.ndim != 2:
            error_msg = f"X_sample должен быть 2D массивом, но имеет форму {X_sample.shape}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

        if np.isnan(X_sample).any():
            X_sample = np.nan_to_num(X_sample, nan=0.0)
            logger.warning("X_sample содержал NaN и был заменён на нули.")

        logger.info("Начало вычисления SHAP значений...")
        if not isinstance(explainer, shap.explainers._tree.TreeExplainer):
            shap_values = explainer(X_sample)
        else:
            # Для TreeExplainer обработка может отличаться
            shap_values_list = []
            for i in range(X_sample.shape[0]):
                try:
                    sv = explainer.shap_values(X_sample[i:i+1])
                    if isinstance(sv, list):
                        sv = np.array(sv)
                    shap_values_list.append(sv)
                except Exception as e_inner:
                    logger.error(f"Ошибка при вычислении SHAP значений для образца {i}: {e_inner}")
                    continue
            if shap_values_list:
                # Агрегация всех значений
                if isinstance(shap_values_list[0], np.ndarray) and shap_values_list[0].ndim > 1:
                    shap_values = np.concatenate(shap_values_list, axis=0)
                else:
                    shap_values = np.array(shap_values_list)
                logger.info(f"SHAP значения вычислены для {len(shap_values_list)} образцов, форма: {shap_values.shape}")
            else:
                error_msg = "Не удалось вычислить SHAP значения для любого образца"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg
                }
        
        # Агрегация значений для категориальных признаков, если нужно
        # В данном случае, так как признаки уже закодированы, агрегация не требуется
        # Но если нужно объединить, например, dummy-переменные, можно добавить логику здесь
        
        logger.info("Анализ SHAP успешно завершен. Возвращаем результат.")
        return {
            'success': True,
            'shap_values': shap_values,
            'explainer': explainer,
            'X_sample': X_sample
        }
    except Exception as e:
        logger.error(f"Исключение в analyze_shap: {type(e).__name__}: {e}", exc_info=True)
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