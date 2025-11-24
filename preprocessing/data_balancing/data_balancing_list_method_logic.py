# data_balancing_list_method.py
#Undersampling, ClusterCentroidsSampling, NearMiss, Oversampling, ADASYN, BorderlineSMOTE, гибридные и ансамблевые подходы).
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import numpy as np

#Метод 1: Оригинальный SMOTE (уже реализован)
def balance_classes_smote(X_train, y_train, round_labels=False, **kwargs):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")
    
    # Используем kwargs для передачи дополнительных параметров
    smote = SMOTE(**kwargs)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Округляем метки классов, если требуется округление
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)
    
    return X_resampled, y_resampled
# Метод 2: Редукция большинства (Random Under-Sampling)
def balance_classes_random_undersampling(X_train, y_train, round_labels=False):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)    
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)    
    return X_resampled, y_resampled

# Метод 3: Cluster Centroids Sampling
def balance_classes_cluster_centroids(X_train, y_train, round_labels=False):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    cc = ClusterCentroids(random_state=42)
    X_resampled, y_resampled = cc.fit_resample(X_train, y_train)    
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)    
    return X_resampled, y_resampled

# Метод 4: NearMiss Algorithms
def balance_classes_nearmiss(X_train, y_train, round_labels=False):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    near_miss = NearMiss(version=1, random_state=42)
    X_resampled, y_resampled = near_miss.fit_resample(X_train, y_train)    
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)    
    return X_resampled, y_resampled

# Метод 5: Перегрузка миноритарного класса (Random Over-Sampling)
def balance_classes_random_oversampling(X_train, y_train, round_labels=False):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)    
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)    
    return X_resampled, y_resampled

# Метод 6: Adaptive Synthetic Sampling (ADASYN)
def balance_classes_adasyn(X_train, y_train, round_labels=False):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)    
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)    
    return X_resampled, y_resampled

# Метод 7: BorderlineSMOTE
def balance_classes_borderlinesmote(X_train, y_train, round_labels=False):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    borderline_smote = BorderlineSMOTE(random_state=42)
    X_resampled, y_resampled = borderline_smote.fit_resample(X_train, y_train)    
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)    
    return X_resampled, y_resampled

# Метод 8: Гибридные методы (SMOTE-TOMEK и SMOTE-ENN)
# a) SMOTE-TOMEK (Удаление выбросов после SMOTE)
def balance_classes_hybrid_smotetomek(X_train, y_train, round_labels=False):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    smt_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt_tomek.fit_resample(X_train, y_train)    
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)    
    return X_resampled, y_resampled

# b) SMOTE-ENN (Удаление шума после SMOTE)
def balance_classes_hybrid_smoteenn(X_train, y_train, round_labels=False):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    smt_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smt_enn.fit_resample(X_train, y_train)    
    if round_labels:
        y_resampled = np.round(y_resampled).astype(int)
        X_resampled = np.round(X_resampled).astype(int)    
    return X_resampled, y_resampled

#Метод 9: Ансамблевый подход (Bagging Classifier + Балансировка)
#Реализуем ансамбль классификаторов вместе с предварительной обработкой классов.
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def balance_classes_bagging_classifier(X_train, y_train, base_estimator=None, n_estimators=10, random_state=42):
    if X_train is None or y_train is None:
        raise ValueError("Переданы пустые данные для балансировки.")    
    bagging_model = BaggingClassifier(base_estimator=base_estimator or DecisionTreeClassifier(), 
                                      n_estimators=n_estimators, 
                                      random_state=random_state)
    bagging_model.fit(X_train, y_train)    
    return X_train, y_train