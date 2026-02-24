from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import psutil
import pandas as pd
import numpy as np

from .feature_importance_help_dialog import HelpDialog, PLOT_HELP_TEXT

class ShapUiLogic:
    """Класс, отвечающий за логику управления данными и состоянием в UI SHAP."""

    def __init__(self):
        self.df = None
        self.X_train = None
        self.y_train = None
        self.target_col = None
        self.categorical_columns = []
        self.label_encoders = {}
        self.trained_models = {}
        self.feature_importances = {}
        self.task_type = "classification"
        self.process = psutil.Process(os.getpid())
        self.shap_values = None

    def set_trained_model(self, model, model_name):
        """Устанавливает предварительно обученную модель извне."""
        if model is not None and model_name:
            self.trained_models = {model_name: model}
            return True
        return False

    def set_data(self, df, target_col):
        """Устанавливает данные для анализа извне. Вызывает подготовку данных."""
        if df is None or target_col is None or target_col not in df.columns:
            return False
        
        self.df = df.copy()
        self.target_col = target_col
        
        # Подготовка данных
        self._prepare_data()
        
        return True

    def _prepare_data(self):
        """Подготавливает X_train и y_train из self.df и self.target_col."""
        if self.df is None or self.target_col is None:
            self.X_train = None
            self.y_train = None
            return
        
        # Разделяем на признаки и целевую переменную
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Обработка категориальных признаков
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X_encoded = X.copy()
        
        self.label_encoders = {}
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Преобразуем в строку, чтобы избежать проблем с типами
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        # Сохраняем обработанные данные
        self.X_train = X_encoded
        self.y_train = y
        
        # Определяем тип задачи
        if y.dtype.kind in ['i', 'u'] and len(y.unique()) < 20:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

    def update_button_states(self):
        """Обновляет состояние кнопок на основе текущего состояния."""
        model_trained = len(self.trained_models) > 0
        shap_values_exist = self.shap_values is not None
        # Возвращаем состояние, так как UI обновляется отдельно
        return {
            'analyze_btn_enabled': model_trained,
            'delete_columns_btn_enabled': shap_values_exist
        }

    def show_plot_help(self, parent=None):
        text = PLOT_HELP_TEXT
        HelpDialog("Справка по графикам", text, parent or self).exec_()