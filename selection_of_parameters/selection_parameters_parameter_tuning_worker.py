import logging
from PySide6.QtCore import QThread, Signal
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from .selection_of_parameters_logic import (get_hyperparameters, get_random_search_params)

logger = logging.getLogger(__name__)

class ParameterTuningWorker(QThread):
    """Поток для оптимизации параметров, предотвращающей замораживание GUI-интерфейса"""
    progress_updated = Signal(int, int)       # Сигнал обновления прогресса
    tuning_completed = Signal(object, dict, float)  # Сигнал завершения подбора параметров
    error_occurred = Signal(str)               # Сигнал возникновения ошибки
    
    def __init__(self, dataset_path):
        """
        Конструктор класса, инициализирует рабочий объект с путём к набору данных
        
        :param dataset_path: путь к файлу CSV с набором данных
        """
        logger.info(f"ParameterTuningWorker.__init__ вызван с параметром dataset_path: {dataset_path}")
        super().__init__()
        self.dataset_path = dataset_path
        logger.info("ParameterTuningWorker успешно инициализирован")
    
    def run(self):
        """
        Основной метод выполнения рабочего потока
        """
        logger.info("ParameterTuningWorker.run метод запущен")
        try:
            # Загрузка датасета
            logger.info(f"Загружаем датасет из файла: {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Датасет загружен успешно. Размер: {df.shape}")
            
            # Подготовка признаков и целевого столбца
            target_col = 'Diabetes_012'
            if target_col not in df.columns:
                possible_targets = [col for col in df.columns if 'diabetes' in col.lower()]
                if possible_targets:
                    target_col = possible_targets[0]
                else:
                    raise ValueError("Невозможно найти целевой столбец 'Diabetes_012' или аналогичный")
            
            feature_cols = list(df.columns.drop(target_col))
            X = df[feature_cols]
            y = df[target_col]
            
            # Разделение данных на обучение и тестирование
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Получение параметров для поиска
            search_params = get_random_search_params()
            hyperparams = get_hyperparameters()
            
            # Создание классификатора случайного леса
            estimator = RandomForestClassifier(random_state=42)
            
            # Определение метрик оценки
            scoring = search_params.get('scoring', ['accuracy', 'f1_macro'])
            if isinstance(scoring, str):
                try:
                    scoring = eval(scoring) if scoring.startswith('[') else [scoring]
                except:
                    scoring = ['accuracy', 'f1_macro']
            
            # Парсим параметр рефитинга
            refit = search_params.get('refit', 'accuracy')
            if isinstance(refit, str) and refit.startswith("'"):
                refit = eval(refit)
            
            # Настройка случайного поиска параметров
            random_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=hyperparams,
                n_iter=search_params.get('n_iter', 100),
                cv=search_params.get('cv', 6),
                scoring=scoring,
                refit=refit,
                random_state=search_params.get('random_state', 42),
                verbose=0,
                n_jobs=1
            )
            
            # Выполнение процедуры подбора лучших параметров
            random_search.fit(X_train, y_train)
            
            # Получение лучшей модели и оценка её точности
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            accuracy = best_model.score(X_test, y_test)
            
            # Отправляем сигнал о завершении подбора параметров
            self.tuning_completed.emit(best_model, best_params, accuracy)
            
        except Exception as e:
            logger.error(f"Ошибка в ParameterTuningWorker.run: {str(e)}")
            logger.error(f"Тип исключения: {type(e).__name__}")
            import traceback
            logger.error(f"След исключения: {traceback.format_exc()}")
            self.error_occurred.emit(str(e))