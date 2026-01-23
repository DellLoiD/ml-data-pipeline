# load_params_and_train_final_model.py
# Загрузка параметров из JSON → обучение → сохранение модели

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QGroupBox, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import json
import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error
)

import pandas as pd


class FinalTrainingWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = None
        self.df_train = None
        self.df_test = None
        self.df_train_path = None  # ✅ Сохраняем путь
        self.df_test_path = None
        self.trained_model = None
        self.target_variable = None
        self.task_type = None
        self.model_type = None
        self.primary_metric_name = None
        self.primary_metric_value = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Финальное обучение модели")
        self.setGeometry(300, 300, 900, 700)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        main_layout = QVBoxLayout()

        title = QLabel("🚀 Финальное обучение модели")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # === КНОПКИ УПРАВЛЕНИЯ ===
        btn_layout = QHBoxLayout()

        self.load_params_btn = QPushButton("📂 Загрузить параметры (.json)")
        self.load_params_btn.clicked.connect(self.load_params)
        btn_layout.addWidget(self.load_params_btn)

        self.load_data_btn = QPushButton("📊 Загрузить train/test")
        self.load_data_btn.clicked.connect(self.load_train_test_data)
        self.load_data_btn.setEnabled(False)
        btn_layout.addWidget(self.load_data_btn)

        main_layout.addLayout(btn_layout)

        # === ОТОБРАЖЕНИЕ ПАРАМЕТРОВ ===
        params_group = QGroupBox("📋 Загруженные параметры")
        params_layout = QVBoxLayout()
        self.params_display = QLabel("Параметры не загружены.")
        self.params_display.setWordWrap(True)
        self.params_display.setStyleSheet("font-family: Courier; font-size: 12px;")
        params_layout.addWidget(self.params_display)
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # === ДАННЫЕ ===
        data_group = QGroupBox("💾 Загруженные данные")
        data_layout = QVBoxLayout()
        self.data_info = QLabel("Данные не загружены.")
        self.data_info.setWordWrap(True)
        data_layout.addWidget(self.data_info)
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)

        # === РЕЗУЛЬТАТ ОБУЧЕНИЯ ===
        self.result_group = QGroupBox("📈 Результаты оценки на test")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("Обучение не запущено.")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        self.result_group.setLayout(result_layout)
        self.result_group.setVisible(False)
        main_layout.addWidget(self.result_group)

        # === КНОПКИ: ЗАПУСК И СОХРАНЕНИЕ ===
        action_layout = QHBoxLayout()

        self.train_btn = QPushButton("▶️ Обучить на train-данных")
        self.train_btn.clicked.connect(self.train_final_model)
        self.train_btn.setEnabled(False)
        action_layout.addWidget(self.train_btn)

        self.save_model_btn = QPushButton("💾 Сохранить финальную модель")
        self.save_model_btn.clicked.connect(self.save_final_model)
        self.save_model_btn.setEnabled(False)
        action_layout.addWidget(self.save_model_btn)

        main_layout.addLayout(action_layout)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def load_params(self):
        # ✅ Открываем в папке model_params
        initial_dir = "model_params"
        if not os.path.exists(initial_dir):
            os.makedirs(initial_dir, exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл параметров", initial_dir, "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.params = data.get("best_params")
            self.target_variable = data.get("target_variable")
            self.task_type = data.get("task_type", "classification")
            self.model_type = data.get("model_type")
            self.primary_metric_name = data.get("primary_metric", {}).get("name", "unknown")
            self.primary_metric_value = data.get("primary_metric", {}).get("value", 0.0)

            if not self.params or not self.target_variable:
                raise ValueError("Нет данных 'best_params' или 'target_variable'")

            # Отображаем
            self.display_params(data)
            self.load_data_btn.setEnabled(True)
            self.update()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить параметры:\n{e}")

    def display_params(self, data):
        text = f"<b>Модель:</b> {data.get('model_type', 'Unknown')}<br>"
        text += f"<b>Целевая переменная:</b> {data.get('target_variable', 'Unknown')}<br>"
        text += f"<b>Тип задачи:</b> {'Классификация' if data.get('task_type') == 'classification' else 'Регрессия'}<br>"
        primary = data.get('primary_metric', {})
        text += f"<b>Ключевая метрика:</b> {primary.get('name', 'Unknown')} = {primary.get('value', 0):.4f}<br><br>"
        text += "<b>Гиперпараметры:</b><br>"
        for k, v in data.get("best_params", {}).items():
            text += f"• <b>{k}:</b> {v}<br>"
        self.params_display.setText(text)

    def load_train_test_data(self):
        # ✅ Открываем в папке dataset
        initial_dir = "dataset"
        if not os.path.exists(initial_dir):
            os.makedirs(initial_dir, exist_ok=True)

        train_path, _ = QFileDialog.getOpenFileName(self, "Выберите train-файл", initial_dir, "CSV Files (*.csv)")
        if not train_path:
            return
        test_path, _ = QFileDialog.getOpenFileName(self, "Выберите test-файл", initial_dir, "CSV Files (*.csv)")
        if not test_path:
            return

        try:
            # ✅ Игнорируем строки с #
            df_train = pd.read_csv(train_path, comment='#')
            df_test = pd.read_csv(test_path, comment='#')

            if df_train.empty:
                raise ValueError("Train-файл пуст после пропуска комментариев")
            if df_test.empty:
                raise ValueError("Test-файл пуст после пропуска комментариев")

            if self.target_variable not in df_train.columns:
                raise ValueError(f"Целевая переменная '{self.target_variable}' не найдена в train")
            if self.target_variable not in df_test.columns:
                raise ValueError(f"Целевая переменная '{self.target_variable}' не найдена в test")

            # ✅ Сохраняем датафреймы и пути
            self.df_train = df_train
            self.df_test = df_test
            self.df_train_path = train_path  # ✅ Запоминаем путь
            self.df_test_path = test_path

            info = (
                f"Train: {df_train.shape[0]} строк × {df_train.shape[1]} признаков<br>"
                f"Test: {df_test.shape[0]} строк × {df_test.shape[1]} признаков<br>"
                f"Общие колонки: {len(set(df_train.columns) & set(df_test.columns))}"
            )
            self.data_info.setText(info)
            self.train_btn.setEnabled(True)
            self.update()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные:\n{e}")

    def train_final_model(self):
        if self.df_train is None or self.df_train.empty or self.params is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите параметры и данные!")
            return

        try:
            X_train = self.df_train.drop(columns=[self.target_variable])
            y_train = self.df_train[self.target_variable].copy()

            X_train = X_train.select_dtypes(include=['number'])
            if X_train.empty:
                raise ValueError("Нет числовых признаков после очистки")

            # Кодируем y_train
            if self.task_type == "classification" and y_train.dtype == "object":
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                self.label_encoder = le

            # Создаём и обучаем модель
            model_class = self.get_model_class()
            model = model_class(**self.params)
            model.fit(X_train, y_train)

            # Подготовка test
            if self.df_test is None or self.df_test.empty:
                raise ValueError("Test-датасет не загружен или пуст")

            X_test = self.df_test.drop(columns=[self.target_variable]).select_dtypes(include=['number'])
            y_test = self.df_test[self.target_variable].copy()

            if X_test.empty:
                raise ValueError("Нет числовых признаков в test после очистки")

            if self.task_type == "classification" and hasattr(self, 'label_encoder'):
                y_test = self.label_encoder.transform(y_test)

            y_pred = model.predict(X_test)

            # ✅ Вычисляем все метрики
            if self.task_type == "classification":
                n_classes = len(np.unique(y_train))
                avg = 'weighted' if n_classes > 2 else 'binary'

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
                rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

                try:
                    if hasattr(model, 'predict_proba'):
                        probas = model.predict_proba(X_test)
                        if probas.shape[1] == 2:
                            auc = roc_auc_score(y_test, probas[:, 1])
                        else:
                            auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
                    else:
                        auc = "Недоступно"
                except:
                    auc = "Ошибка"

                result_text = (
                    f"<b>Точность (Accuracy):</b> {acc:.4f}<br>"
                    f"<b>Precision:</b> {prec:.4f}<br>"
                    f"<b>Recall:</b> {rec:.4f}<br>"
                    f"<b>F1-Score:</b> {f1:.4f}<br>"
                    f"<b>ROC-AUC:</b> {auc if isinstance(auc, str) else f'{auc:.4f}'}"
                )
            else:
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                result_text = (
                    f"<b>R² Score:</b> {r2:.4f}<br>"
                    f"<b>Mean Squared Error (MSE):</b> {mse:.4f}<br>"
                    f"<b>Mean Absolute Error (MAE):</b> {mae:.4f}"
                )

            self.trained_model = model
            self.result_label.setText(result_text)
            self.result_group.setVisible(True)
            self.save_model_btn.setEnabled(True)

            QMessageBox.information(self, "Успех", "Модель обучена на train и оценена на test.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обучения:\n{e}")

    def get_model_class(self):
        mapping = {
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "LogisticRegression": LogisticRegression,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
        }
        cls = mapping.get(self.model_type)
        if not cls:
            raise ValueError(f"Неизвестная модель: {self.model_type}")
        return cls

    def save_final_model(self):
        if not self.trained_model:
            QMessageBox.warning(self, "Ошибка", "Нет обученной модели для сохранения!")
            return

        # ✅ Создаём имя файла: модель_датасет_метрика_значение.pkl
        try:
            trained_models_dir = "trained_models"
            os.makedirs(trained_models_dir, exist_ok=True)

            # Имя модели
            model_name = self.model_type

            # ✅ Имя датасета — из сохранённого пути
            if not self.df_train_path:
                dataset_name = "unknown"
            else:
                dataset_name = os.path.splitext(os.path.basename(self.df_train_path))[0]

            # Метрика
            metric_name = self.primary_metric_name if self.primary_metric_name else 'score'
            metric_value = f"{self.primary_metric_value:.4f}".replace('.', '_')

            # Имя файла
            filename = f"{model_name}_{dataset_name}_{metric_name}_{metric_value}.pkl"
            file_path = os.path.join(trained_models_dir, filename)

            # Сохраняем
            joblib.dump(self.trained_model, file_path)

            QMessageBox.information(self, "Успех", f"Финальная модель сохранена:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить модель:\n{e}")

    def closeEvent(self, event):
        if self.trained_model and not self.isVisible():
            reply = QMessageBox.question(
                self, "Закрытие",
                "Вы уверены, что хотите закрыть? Модель будет потеряна.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
