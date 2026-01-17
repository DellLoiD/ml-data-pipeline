# preprocessing/data_balancing/dataset_trim/dataset_trim_window_logic.py
import pandas as pd
import numpy as np

class DatasetTrimLogic:
    def __init__(self):
        self.df_original = None
        self.df_trimmed = None
        self.target_col = None

    def set_data(self, df, target_col):
        self.df_original = df.copy()
        self.target_col = target_col
        self.df_trimmed = None

    def trim_equal(self, n_per_class):
        """Оставить по N записей в каждом классе"""
        if self.df_original is None:
            raise ValueError("Данные не загружены")
        dfs = []
        for cls in self.df_original[self.target_col].unique():
            subset = self.df_original[self.df_original[self.target_col] == cls]
            if len(subset) > n_per_class:
                subset = subset.sample(n=n_per_class, random_state=42)
            dfs.append(subset)
        self.df_trimmed = pd.concat(dfs, ignore_index=True)
        return self.df_trimmed

    def trim_proportional(self, max_total):
        """Сохранить пропорции, уменьшить до max_total записей"""
        if self.df_original is None:
            raise ValueError("Данные не загружены")
        total = len(self.df_original)
        ratio = max_total / total
        dfs = []
        for cls in self.df_original[self.target_col].unique():
            subset = self.df_original[self.df_original[self.target_col] == cls]
            n = max(1, int(len(subset) * ratio))
            if len(subset) > n:
                subset = subset.sample(n=n, random_state=42)
            dfs.append(subset)
        self.df_trimmed = pd.concat(dfs, ignore_index=True)
        return self.df_trimmed

    def trim_majority_only(self, max_majority):
        """Обрезать только мажоритарный класс"""
        if self.df_original is None:
            raise ValueError("Данные не загружены")
        class_counts = self.df_original[self.target_col].value_counts()
        majority_class = class_counts.index[0]
        dfs = []
        for cls in self.df_original[self.target_col].unique():
            subset = self.df_original[self.df_original[self.target_col] == cls]
            if cls == majority_class and len(subset) > max_majority:
                subset = subset.sample(n=max_majority, random_state=42)
            dfs.append(subset)
        self.df_trimmed = pd.concat(dfs, ignore_index=True)
        return self.df_trimmed
