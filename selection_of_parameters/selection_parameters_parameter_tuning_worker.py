import logging
from PySide6.QtCore import Signal, QThread 
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from .selection_of_parameters_logic import (get_hyperparameters, get_random_search_params)

logger = logging.getLogger(__name__)
    
class ParameterTuningWorker(QThread):
    """Поток для оптимизации параметров, предотвращающей замораживание GUI-интерфейса"""
    # Сигнал обновления прогресса (текущие итерация / общее число итераций)
    progress_updated = Signal(float, int, int)
    # Сигнал завершения настройки параметров (лучшая модель, лучшие гиперпараметры, точность)
    tuning_completed = Signal(object, dict, float)
    # Сигнал возникновения ошибки
    error_occurred = Signal(str)
        
    def __init__(self, parent=None, dataset_path=None, target_variable=None):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.target_variable = target_variable 
    
    def run(self):
        logger.info("ParameterTuningWorker.run метод запущен")
        try:# Загрузка датасета
            logger.info(f"Загружаем датасет из файла: {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Датасет загружен успешно. Размер: {df.shape}")                                
            feature_cols = list(df.columns.drop(self.target_variable))
            X = df[feature_cols]
            y = df[self.target_variable]  
            # Получение параметров для поиска
            search_params = get_random_search_params()
            hyperparams = get_hyperparameters()    
            # Проверяем и задаем минимально допустимое значение n_iter
            n_iter_value = search_params.get('n_iter')           
            # Разделение данных на обучение и тестирование
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=search_params.get('test_size'), random_state=search_params.get('random_state'))             
            # Создаем классификатор
            estimator = RandomForestClassifier(random_state=search_params.get('random_state'))
            # Настройка случайного поиска параметров
            random_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=hyperparams,
                n_iter=n_iter_value,
                cv=search_params.get('cv'),
                scoring=search_params.get('scoring', ['accuracy', 'f1_macro']),
                refit=search_params.get('refit', 'accuracy'),
                random_state=search_params.get('random_state'),
                verbose=search_params.get('verbose'), 
                n_jobs=search_params.get('n_jobs')
            )
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            accuracy = best_model.score(X_test, y_test)
            # Отправляем сигнал о завершении
            self.tuning_completed.emit(best_model, best_params, accuracy)            
        except Exception as e:
            logger.error(f"Ошибка в ParameterTuningWorker.run: {str(e)}")
            logger.error(f"Тип исключения: {type(e).__name__}")
            import traceback
            logger.error(f"След исключения: {traceback.format_exc()}")
            self.error_occurred.emit(str(e))