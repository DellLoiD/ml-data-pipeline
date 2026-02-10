from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, 
    QGroupBox, QRadioButton, QLineEdit,  QDialog, QFrame, QComboBox, QScrollArea
)
from PySide6.QtWidgets import QScrollArea
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from researching_models.check_models_loading_screen import LoadingScreen
import gc
import matplotlib.pyplot as plt

from researching_models.cross_validation.cross_validation_random_search_logic import RandomSearchAnalyzer
import logging

logger = logging.getLogger(__name__)
import numpy as np


class HelpDialog(QDialog):
    """Справка по метрикам и параметрам"""
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.setModal(True)
        self.resize(300, 300)
        layout = QVBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
        self.setLayout(layout)


class CrossValidationRandomSearchTab(QWidget):
    def __init__(self, analyzer=None, main_window=None):
        super().__init__()
        self.analyzer = analyzer or RandomSearchAnalyzer()
        self.main_window = main_window
        self.results_layout = main_window.results_layout if main_window else QVBoxLayout()
        self.curve_params = {}
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # === ВЫБОР МОДЕЛИ И RANDOM SEARCH ===
        models_group = QGroupBox("🔍 Random Search: Подбор гиперпараметров")
        models_layout = QVBoxLayout()

        # === СТРОКА: Модель + Итерации + n_estimators + max_depth + n_jobs_rs + learning rate ===
        model_params_layout = QHBoxLayout()

        # Модель
        model_group = QGroupBox("Модель")
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "Gradient Boosting"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        model_params_layout.addWidget(model_group)

        # Итерации
        trials_group = QGroupBox("Итерации")
        trials_layout = QHBoxLayout()
        self.n_trials_le = QLineEdit("50")
        trials_layout.addWidget(self.n_trials_le)
        trials_group.setLayout(trials_layout)
        model_params_layout.addWidget(trials_group)

        # n_estimators
        n_est_group = QGroupBox("n_estimators")
        n_est_group_layout = QHBoxLayout()
        self.n_est_le = QLineEdit("50-200")
        n_est_group_layout.addWidget(self.n_est_le)
        n_est_group.setLayout(n_est_group_layout)
        model_params_layout.addWidget(n_est_group)

        # max_depth
        max_depth_group = QGroupBox("max_depth")
        max_depth_group_layout = QHBoxLayout()
        self.max_depth_le = QLineEdit("2-5")
        max_depth_group_layout.addWidget(self.max_depth_le)
        max_depth_group.setLayout(max_depth_group_layout)
        model_params_layout.addWidget(max_depth_group)

        # n_jobs_rs
        n_jobs_rs_group = QGroupBox("n_jobs_rs")
        n_jobs_rs_layout = QHBoxLayout()
        self.n_jobs_rs_le = QLineEdit("1")
        n_jobs_rs_layout.addWidget(self.n_jobs_rs_le)
        n_jobs_rs_group.setLayout(n_jobs_rs_layout)
        model_params_layout.addWidget(n_jobs_rs_group)

        # learning rate
        lr_group = QGroupBox("learning rate")
        lr_layout = QHBoxLayout()
        self.learning_rate_le = QLineEdit("0.01-0.3")
        self.learning_rate_le.setFixedWidth(80)
        lr_layout.addWidget(self.learning_rate_le)
        lr_group.setLayout(lr_layout)
        model_params_layout.addWidget(lr_group)
        
        models_layout.addLayout(model_params_layout)

        models_group.setLayout(models_layout)
        main_layout.addWidget(models_group)

        # === МЕТРИКА И ЦЕЛЬ ===
        scoring_opt_layout = QHBoxLayout()
        scoring_group = QGroupBox("Метрика")
        scoring_layout = QHBoxLayout()
        self.accuracy_radio = QRadioButton("accuracy")
        self.f1_radio = QRadioButton("f1")
        self.precision_radio = QRadioButton("precision")
        self.recall_radio = QRadioButton("recall")
        self.roc_auc_radio = QRadioButton("roc_auc")
        self.r2_radio = QRadioButton("r2")
        self.neg_mse_radio = QRadioButton("neg_mean_squared_error")
        self.neg_mae_radio = QRadioButton("neg_mean_absolute_error")
        scoring_layout.addWidget(self.accuracy_radio)
        scoring_layout.addWidget(self.f1_radio)
        scoring_layout.addWidget(self.precision_radio)
        scoring_layout.addWidget(self.recall_radio)
        scoring_layout.addWidget(self.roc_auc_radio)
        scoring_layout.addWidget(self.r2_radio)
        scoring_layout.addWidget(self.neg_mse_radio)
        scoring_layout.addWidget(self.neg_mae_radio)
        scoring_group.setLayout(scoring_layout)
        scoring_opt_layout.addWidget(scoring_group)

        direction_group = QGroupBox("Цель")
        direction_layout = QHBoxLayout()
        self.maximize_radio = QRadioButton("maximize")
        self.minimize_radio = QRadioButton("minimize")
        self.maximize_radio.setChecked(True)
        direction_layout.addWidget(self.maximize_radio)
        direction_layout.addWidget(self.minimize_radio)
        direction_group.setLayout(direction_layout)
        scoring_opt_layout.addWidget(direction_group)
        main_layout.addLayout(scoring_opt_layout)

        # === ПАРАМЕТРЫ КРИВОЙ ОБУЧЕНИЯ ===
        curve_group = QGroupBox("⚙️ Параметры кривой обучения")
        curve_layout = QHBoxLayout()
        params = [("CV", "5"), ("n_jobs", "1"), ("Число точек", "10"), ("Random State", "42")]
        for label_text, default_value in params:
            group_box = QGroupBox(label_text)
            le = QLineEdit(default_value)
            le.setFixedWidth(60)
            layout = QHBoxLayout()
            layout.addWidget(le)
            group_box.setLayout(layout)
            self.curve_params[label_text] = le
            curve_layout.addWidget(group_box)
        curve_group.setLayout(curve_layout)
        main_layout.addWidget(curve_group)

        # === КНОПКА ЗАПУСКА ===
        self.analyze_btn = QPushButton("🚀 Запустить анализ (Random Search)")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analyze_btn.setEnabled(True)
        main_layout.addWidget(self.analyze_btn)

        self.setLayout(main_layout)

        self.update_scoring_options()
        self.update_models()
        self.on_model_changed()

    def update_scoring_options(self):
        # Активируем только нужные метрики в зависимости от задачи
        is_classification = self.analyzer.task_type == "classification"
        for radio in [self.accuracy_radio, self.f1_radio, self.precision_radio, self.recall_radio, self.roc_auc_radio]:
            radio.setVisible(is_classification)
        for radio in [self.r2_radio, self.neg_mse_radio, self.neg_mae_radio]:
            radio.setVisible(not is_classification)

    def update_models(self):
        """Обновляет список моделей в зависимости от задачи"""
        self.model_combo.clear()
        if self.analyzer.task_type == "classification":
            models = ["Random Forest", "Gradient Boosting"]
        else:  # regression
            models = ["Random Forest", "Gradient Boosting"]  
        self.model_combo.addItems(models)
        self.on_model_changed() 

    def on_model_changed(self):
        is_gb = self.model_combo.currentText() == "Gradient Boosting"
        self.learning_rate_le.setEnabled(is_gb)
        
    def on_analyze(self):
        try:
            # Сбор параметров
            n_trials = int(self.n_trials_le.text())
            direction = "maximize" if self.maximize_radio.isChecked() else "minimize"
            
            # Определяем выбранную метрику
            scoring = None
            classification_metrics = {
                self.accuracy_radio: "accuracy",
                self.f1_radio: "f1",
                self.precision_radio: "precision",
                self.recall_radio: "recall",
                self.roc_auc_radio: "roc_auc"
            }
            regression_metrics = {
                self.r2_radio: "r2",
                self.neg_mse_radio: "neg_mean_squared_error",
                self.neg_mae_radio: "neg_mean_absolute_error"
            }
            for radio, metric in (classification_metrics if self.analyzer.task_type == "classification" else regression_metrics).items():
                if radio.isChecked():
                    scoring = metric
                    break
            if not scoring:
                QMessageBox.warning(self, "Ошибка", "Не выбрана метрика!")
                return
            
            # Показ заглушки
            self.loading_screen = LoadingScreen()
            self.loading_screen.show()
            
            cv = int(self.curve_params['CV'].text())
            n_points = int(self.curve_params['Число точек'].text())
            rs = int(self.curve_params['Random State'].text())
            
            # n_jobs для cross_val_score (Random Search)
            try:
                n_jobs_cv = int(self.n_jobs_rs_le.text())
            except:
                n_jobs_cv = 1

            # n_jobs для кривой обучения
            try:
                n_jobs_lc = int(self.curve_params['n_jobs'].text())
            except:
                n_jobs_lc = 1

            # Диапазоны
            n_est = self.parse_range(self.n_est_le.text(), int)
            max_depth = self.parse_range(self.max_depth_le.text(), int)
            lr = self.parse_range(self.learning_rate_le.text(), float)

            # Вызов логики из нового модуля
            result = self.analyzer.run_random_search(
                n_trials=n_trials,
                model_name=self.model_combo.currentText(),
                scoring=scoring,
                direction=direction,
                n_est_range=n_est,
                max_depth_range=max_depth,
                lr_range=lr,
                cv=cv,
                n_jobs_cv=n_jobs_cv,
                n_jobs_lc=n_jobs_lc,
                n_points=n_points,
                random_state=rs
            )
            
            if result is None:
                QMessageBox.warning(self, "Random Search", "Не найдено решений.")
                return
            
            # Сохраняем результаты для построения графика
            self.current_cv_scores = result.get('cv_scores')
            self.current_model_name = result['model_name']
            self.current_scoring = scoring
            
            # Закрываем заглушку ПЕРЕД отображением результатов
            if self.loading_screen:
                self.loading_screen.close()
            
            # Отображение результатов
            logger.info("Перед вызовом display_result в RandomSearchTab")
            
            try:
                # Удаляем model_name из result чтобы избежать дублирования
                result_data = result.copy()
                if 'model_name' in result_data:
                    del result_data['model_name']
                if 'scoring' in result_data:
                    del result_data['scoring']
                
                # Удаляем метрики кросс-валидации из result_data, так как они будут добавлены отдельно
                if 'cv_scores' in result_data:
                    del result_data['cv_scores']
                
                self.display_result(
                    model_name=result['model_name'],
                    scoring=scoring,
                    **result_data
                )
                logger.info("Метод display_result успешно вызван в RandomSearchTab")
            except Exception as e:
                logger.error(f"Ошибка при вызове display_result в RandomSearchTab: {e}")
                raise

            # Финальная очистка памяти
            gc.collect()
            
        finally:
            # Всегда закрываем заглушку
            if hasattr(self, 'loading_screen') and self.loading_screen:
                self.loading_screen.close()
                logger.info("Заглушка закрыта в блоке finally")
 

    def parse_range(self, text, dtype):
        text = text.strip()
        if not text or 'none' in text.lower():
            return (3, 10)  # default
        if '-' in text:
            a, b = map(dtype, text.split('-'))
            return (a, b)
        return (dtype(text), dtype(text))
    
    def display_result(self, model_name, mean_score, std_score, scoring, best_params, cv_scores=None):
        # Создаём центральный виджет с результатами, если его ещё нет
        if not hasattr(self, 'central_results_group'):
            self.central_results_group = QGroupBox("📊 Результаты анализа")
            self.central_results_layout = QHBoxLayout()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFixedHeight(250)
            scroll_content = QWidget()
            scroll_content.setLayout(self.central_results_layout)
            scroll.setWidget(scroll_content)
            self.central_results_group.setLayout(QVBoxLayout())
            self.central_results_group.layout().addWidget(scroll)
            
            # Логируем создание
            logger.info(f"Создан central_results_layout в RandomSearchTab с id: {id(self.central_results_layout)}")

            # Добавляем в основной макет вкладки
            self.layout().insertWidget(3, self.central_results_group)  # после вкладок

        model_group = QGroupBox(f" {model_name} (Random Search)")
        model_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #aaa;
                border-radius: 6px;
                padding: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Финальные метрики
        def add_metric(layout, label_text, value, help_text):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label_text}: {value:.4f}"))
            btn = QPushButton("?")
            btn.setFixedSize(20, 20)
            btn.clicked.connect(lambda: HelpDialog(label_text, help_text, self).exec_())
            row.addWidget(btn)
            layout.addLayout(row)

        add_metric(layout, "Средняя метрика", mean_score, f"Среднее значение метрики: {mean_score:.4f} ± {std_score:.4f}")
        add_metric(layout, "Стандартное отклонение", std_score, "Стандартное отклонение метрики по фолдам")

        # Параметры
        if best_params:
            param_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in best_params.items()])
            params_label = QLabel(f"<small><u>Лучшие параметры:</u><br>{param_text}</small>")
            params_label.setWordWrap(True)
            params_label.setStyleSheet("font-size: 12px; color: #777;")
            layout.addWidget(params_label)

        # Кнопка для графика кросс-валидации
        if cv_scores is not None and len(cv_scores) > 0:
            plot_btn = QPushButton("📊 Показать график кросс-валидации")
            plot_btn.clicked.connect(self.plot_cv_scores)
            layout.addWidget(plot_btn)
            logger.info(f"Кнопка 'Показать график кросс-валидации' добавлена для модели {model_name}")
        else:
            logger.warning(f"Кнопка 'Показать график кросс-валидации' НЕ добавлена для модели {model_name} - отсутствуют оценки кросс-валидации")

        model_group.setLayout(layout)
        self.central_results_layout.addWidget(model_group)

        # Ограничиваем количество результатов (оставляем последние 3)
        while self.central_results_layout.count() > 3:
            item = self.central_results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if self.main_window:
            self.main_window.update_memory_usage()

    def plot_cv_scores(self):
        """Построение столбчатой диаграммы оценок кросс-валидации"""
        if not hasattr(self, 'current_cv_scores') or self.current_cv_scores is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для построения графика.")
            return

        plt.figure(figsize=(10, 6))
        folds = range(1, len(self.current_cv_scores) + 1)
        plt.bar(folds, self.current_cv_scores, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Фолд')
        
        metric_names = {
            "r2": "R²",
            "neg_mean_squared_error": "RMSE",
            "neg_mean_absolute_error": "MAE",
            "accuracy": "Accuracy",
            "f1": "F1 Score",
            "precision": "Precision",
            "recall": "Recall",
            "roc_auc": "ROC AUC"
        }
        ylabel = metric_names.get(self.current_scoring, self.current_scoring.replace("neg_", "").replace("_", " ").title())
        plt.ylabel(ylabel)
        plt.title(f"Оценки кросс-валидации — {self.current_model_name}")
        plt.xticks(folds)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def update_memory_usage(self):
        if self.main_window:
            self.main_window.update_memory_usage()