from PySide6.QtWidgets import *
from PySide6.QtCore import *
import logging


 # Запуск оптимизации параметров задерживается, чтобы сначала появилось окно
def delayed_start():
    simulate_tuning()
    
QTimer.singleShot(1000, delayed_start)  # Задержка в 1 секунду

def simulate_tuning(self):        
        # Имитация процесса подбора параметров для тестирования.                
        # Моделируем обновление прогресса
        total_iterations = 10  # Уменьшено для теста
        for i in range(total_iterations + 1):
            QTimer.singleShot(i * 200, lambda iter=i: self.update_progress(iter, total_iterations))        
        # Моделируем завершение через 3 секунды
        def simulate_completion():
            fake_best_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'criterion': 'gini'
            }
            fake_accuracy = 0.9289
            self.on_tuning_completed(None, fake_best_params, fake_accuracy)        
        QTimer.singleShot(3000, simulate_completion)