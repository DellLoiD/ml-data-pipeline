# researching_models/learning_curve_optuna_ui.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QInputDialog,
    QGroupBox, QButtonGroup, QRadioButton, QLineEdit, QScrollArea, QDialog, QFrame, QComboBox, QFormLayout
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import os
import pandas as pd
import matplotlib.pyplot as plt
import gc
import psutil
from .learning_curve_optuna_logic import ModelAnalyzer


class HelpDialog(QDialog):
    """Справка по метрикам и параметрам"""
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.setModal(True)
        self.resize(400, 300)
        layout = QVBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
        self.setLayout(layout)


class LearningCurveUI(QWidget):
    def __init__(self):
        super().__init__()
        self.analyzer = ModelAnalyzer()
        self.results_layout = None
        self.curve_params = {}
        self.process = psutil.Process(os.getpid())
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Кривые обучения + Optuna")
        main_layout = QVBoxLayout()

        title_label = QLabel("Подбор гиперпараметров и кривые обучения")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)

        # === СТРОКА: Задача + Кнопка загрузки ===
        task_load_layout = QHBoxLayout()

        # Группа для типа задачи (чтобы не растягивалась)
        task_widget = QWidget()
        task_layout = QHBoxLayout(task_widget)
        task_layout.addWidget(QLabel("Задача:"))
        self.classification_radio = QRadioButton("Классификация")
        self.regression_radio = QRadioButton("Регрессия")
        self.classification_radio.setChecked(True)
        self.task_group = QButtonGroup()
        self.task_group.addButton(self.classification_radio, 1)
        self.task_group.addButton(self.regression_radio, 2)
        self.task_group.buttonClicked.connect(self.on_task_selected)
        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.regression_radio)
        task_layout.addStretch()  # внутри widget'а задачи

        # Кнопка загрузки
        self.load_btn = QPushButton("📁 Загрузить датасет")
        self.load_btn.clicked.connect(self.on_load_dataset)

        # Добавляем в основную строку: сначала группу задач, потом кнопку
        task_load_layout.addWidget(task_widget)  # фиксированная ширина
        task_load_layout.addWidget(self.load_btn)  # занимает оставшееся место
        task_load_layout.setStretch(0, 0)  # task_widget — не растягивается
        task_load_layout.setStretch(1, 1)  # load_btn — растягивается

        main_layout.addLayout(task_load_layout)
        # Целевая переменная и метка памяти в одной строке
        target_memory_layout = QHBoxLayout()
        self.target_label = QLabel("Целевая переменная: не выбрана")
        self.target_label.setStyleSheet("font-weight: bold;")
        self.memory_label = QLabel("📊 Память: ? МБ")
        self.memory_label.setStyleSheet("color: #555; font-size: 11px;")
        target_memory_layout.addWidget(self.target_label)
        target_memory_layout.addWidget(self.memory_label)
        target_memory_layout.addStretch()  # чтобы метка памяти не растягивалась
        main_layout.addLayout(target_memory_layout)

        # === ВЫБОР МОДЕЛИ И OPTUNA ===
        models_group = QGroupBox("🤖 Optuna: Подбор гиперпараметров")
        models_layout = QVBoxLayout()

        # === СТРОКА: Модель + Кнопки режима + Optuna + CV + learning_rate ===
        model_jobs_layout = QHBoxLayout()

        # Модель
        model_jobs_layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "Gradient Boosting"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_jobs_layout.addWidget(self.model_combo)

        # Кнопки режимов
        self.day_btn = QPushButton("🌞")
        self.night_btn = QPushButton("🌙")
        self.day_btn.setFixedSize(40, 25)
        self.night_btn.setFixedSize(40, 25)
        self.day_btn.clicked.connect(self.set_day_mode)
        self.night_btn.clicked.connect(self.set_night_mode)
        model_jobs_layout.addWidget(self.day_btn)
        model_jobs_layout.addWidget(self.night_btn)

        # Optuna n_jobs
        optuna_job_group = QWidget()
        optuna_job_layout = QHBoxLayout(optuna_job_group)
        optuna_job_layout.setContentsMargins(0, 0, 0, 0)
        optuna_job_layout.addWidget(QLabel("Optuna_n_jobs: "))
        self.optuna_n_jobs_le = QLineEdit("1")
        self.optuna_n_jobs_le.setFixedWidth(60)
        optuna_job_layout.addWidget(self.optuna_n_jobs_le)
        help_optuna_btn = QPushButton("?")
        help_optuna_btn.setFixedSize(20, 20)
        help_optuna_text = "<b>Optuna n_jobs</b><br><br>Основной кран параллелизма.<br>Число trialов, запускаемых параллельно."
        help_optuna_btn.clicked.connect(lambda: HelpDialog("Optuna n_jobs", help_optuna_text, self).exec_())
        optuna_job_layout.addWidget(help_optuna_btn)
        model_jobs_layout.addWidget(optuna_job_group)

        # CV n_jobs
        cv_job_group = QWidget()
        cv_job_layout = QHBoxLayout(cv_job_group)
        cv_job_layout.setContentsMargins(0, 0, 0, 0)
        cv_job_layout.addWidget(QLabel("CV_n_jobs: "))
        self.cv_n_jobs_le = QLineEdit("1")
        self.cv_n_jobs_le.setFixedWidth(60)
        cv_job_layout.addWidget(self.cv_n_jobs_le)
        help_cv_btn = QPushButton("?")
        help_cv_btn.setFixedSize(20, 20)
        help_cv_text = "<b>CV n_jobs</b><br><br>Дополнительное ускорение (макс. = 2).<br>Число процессов внутри одного trial (в cross_val_score)."
        help_cv_btn.clicked.connect(lambda: HelpDialog("CV n_jobs", help_cv_text, self).exec_())
        cv_job_layout.addWidget(help_cv_btn)
        model_jobs_layout.addWidget(cv_job_group)

        # learning_rate в той же строке
        model_jobs_layout.addWidget(QLabel("lr:"))
        self.learning_rate_le = QLineEdit("0.01-0.3")
        self.learning_rate_le.setFixedWidth(80)
        model_jobs_layout.addWidget(self.learning_rate_le)
        model_jobs_layout.addStretch()  # Растягиваем для выравнивания
        models_layout.addLayout(model_jobs_layout)

        # === Число итераций и таймаут в одной строке ===
        trials_timeout_layout = QHBoxLayout()
        trials_group = QGroupBox("Число итераций")
        trials_layout = QHBoxLayout()
        self.optuna_trials = QLineEdit("50")
        trials_layout.addWidget(self.optuna_trials)
        trials_group.setLayout(trials_layout)

        timeout_group = QGroupBox("Таймаут (сек)")
        timeout_layout = QHBoxLayout()
        self.optuna_timeout = QLineEdit("600")
        timeout_layout.addWidget(self.optuna_timeout)
        timeout_group.setLayout(timeout_layout)

        trials_timeout_layout.addWidget(trials_group)
        trials_timeout_layout.addWidget(timeout_group)
        models_layout.addLayout(trials_timeout_layout)

        # === Метрика и цель оптимизации в одной строке ===
        scoring_opt_layout = QHBoxLayout()

        scoring_group = QGroupBox("Метрика")
        scoring_layout = QVBoxLayout()
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

        direction_group = QGroupBox("Цель оптимизации")
        direction_layout = QVBoxLayout()
        self.maximize_radio = QRadioButton("maximize")
        self.minimize_radio = QRadioButton("minimize")
        self.maximize_radio.setChecked(True)
        direction_layout.addWidget(self.maximize_radio)
        direction_layout.addWidget(self.minimize_radio)
        direction_group.setLayout(direction_layout)
        scoring_opt_layout.addWidget(direction_group)

        models_layout.addLayout(scoring_opt_layout)

        # === Параметры модели: n_estimators и max_depth в одной строке ===
        params_layout = QHBoxLayout()

        n_est_group = QGroupBox("n_estimators")
        n_est_group_layout = QHBoxLayout()
        self.n_est_le = QLineEdit("50-200")
        n_est_group_layout.addWidget(self.n_est_le)
        n_est_group.setLayout(n_est_group_layout)
        params_layout.addWidget(n_est_group)

        max_depth_group = QGroupBox("max_depth")
        max_depth_group_layout = QHBoxLayout()
        self.max_depth_le = QLineEdit("2-5")
        max_depth_group_layout.addWidget(self.max_depth_le)
        max_depth_group.setLayout(max_depth_group_layout)
        params_layout.addWidget(max_depth_group)

        models_layout.addLayout(params_layout)


        models_group.setLayout(models_layout)
        main_layout.addWidget(models_group)        

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
        self.analyze_btn = QPushButton("🚀 Запустить анализ")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analyze_btn.setEnabled(False)
        main_layout.addWidget(self.analyze_btn)
        # === РЕЗУЛЬТАТЫ ===
        results_group = QGroupBox("📊 Результаты")
        results_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.results_layout = QVBoxLayout()
        scroll_content = QWidget()
        scroll_content.setLayout(self.results_layout)
        scroll.setWidget(scroll_content)
        scroll.setFixedHeight(250)
        results_layout.addWidget(scroll)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        self.setLayout(main_layout)
        self.resize(1000, 800)
        self.show()
        self.update_scoring_options()
        self.update_memory_usage()
        self.update_models() 
        self.curve_params['n_jobs'].textChanged.connect(self.on_n_jobs_changed)
        self.optuna_n_jobs_le.textChanged.connect(self.on_n_jobs_changed)
        self.on_n_jobs_changed()
        
    def set_day_mode(self):
        """Устанавливает дневной режим"""
        self.optuna_n_jobs_le.setText("1")
        self.cv_n_jobs_le.setText("1")

    def set_night_mode(self):
        """Устанавливает ночной режим"""
        self.optuna_n_jobs_le.setText("6")
        self.cv_n_jobs_le.setText("1")

    def on_n_jobs_changed(self):
        """Показывает предупреждение при потенциально опасной комбинации n_jobs"""
        try:
            cv_n_jobs = int(self.cv_n_jobs_le.text())
            optuna_n_jobs = int(self.optuna_n_jobs_le.text())

            total_processes = abs(cv_n_jobs * optuna_n_jobs)

            if cv_n_jobs > 1 and optuna_n_jobs > 1:
                warn_text = (
                    f"⚠️ Высокий риск перегрузки!<br><br>"
                    f"Optuna запустит {optuna_n_jobs} trial'ов,<br>"
                    f"каждый из которых будет использовать {cv_n_jobs} процессов.<br><br>"
                    f"<b>Общее число процессов:</b> до {total_processes}<br><br>"
                    "Рекомендуется:<br>"
                    "• Оставить <b>Optuna n_jobs > 1</b> и <b>CV n_jobs = 1</b><br>"
                    "• Или наоборот."
                )
                self.cv_n_jobs_le.setStyleSheet("background-color: #fff3cd; border: 1px solid #ffeaa7;")
                self.optuna_n_jobs_le.setStyleSheet("background-color: #fff3cd; border: 1px solid #ffeaa7;")
            else:
                self.cv_n_jobs_le.setStyleSheet("")
                self.optuna_n_jobs_le.setStyleSheet("")

        except Exception as e:
            pass 

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

    def on_task_selected(self):
        self.analyzer.task_type = "classification" if self.classification_radio.isChecked() else "regression"
        self.update_scoring_options()
        self.update_models()

    def on_model_changed(self):
        is_gb = self.model_combo.currentText() == "Gradient Boosting"
        self.learning_rate_le.setVisible(is_gb)
        
    def on_load_dataset(self):
            reply = QMessageBox.question(
                self, "Режим загрузки",
                "Разделить датасет на train/test?\n"
                "• Да — загрузить два файла\n"
                "• Нет — один файл, разделю автоматически",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.load_separate_datasets()
            else:
                self.load_single_dataset()

    def load_single_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV", "./dataset/", "CSV (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path, comment='#')
            target, ok = QInputDialog.getItem(self, "Целевая", "Выберите:", df.columns, 0, False)
            if not ok: return
            self.analyzer.load_from_dataframe(df, target, self.analyzer.task_type)
            self.target_label.setText(f"Целевая: {target}")
            self.analyze_btn.setEnabled(True)
            self.load_btn.setText(f"📁 {os.path.basename(path)}")

            self.update_memory_usage()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не загружен:\n{e}")

    def load_separate_datasets(self):
        train_path, _ = QFileDialog.getOpenFileName(self, "Train", "./dataset/", "CSV (*.csv)")
        if not train_path: return
        test_path, _ = QFileDialog.getOpenFileName(self, "Test", "./dataset/", "CSV (*.csv)")
        if not test_path: return
        try:
            df_train = pd.read_csv(train_path, comment='#')
            df_test = pd.read_csv(test_path, comment='#')

            common_cols = set(df_train.columns) & set(df_test.columns)
            if not common_cols:
                QMessageBox.critical(self, "Ошибка", "Нет общих колонок!")
                return

            possible_targets = [col for col in common_cols if df_train[col].nunique() < 0.9 * len(df_train)]
            if not possible_targets:
                possible_targets = list(common_cols)

            target, ok = QInputDialog.getItem(self, "Целевая", "Выберите:", sorted(possible_targets), 0, False)
            if not ok or not target:
                return

            # Загружаем через analyzer
            self.analyzer.load_separate_datasets(train_path, test_path, target, self.analyzer.task_type)

            self.target_label.setText(f"Целевая: {target}")
            self.analyze_btn.setEnabled(True)

            train_name = os.path.basename(train_path)
            test_name = os.path.basename(test_path)
            self.load_btn.setText(f"📁 train: {train_name}\n   test: {test_name}")

            self.update_memory_usage()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки:\n{e}")


    def on_analyze(self):
        try:
            # Сбор параметров
            n_trials = int(self.optuna_trials.text())
            timeout = int(self.optuna_timeout.text())
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

            cv = int(self.curve_params['CV'].text())
            n_jobs_cv = int(self.curve_params['n_jobs'].text())
            n_points = int(self.curve_params['Число точек'].text())
            rs = int(self.curve_params['Random State'].text())

            # Диапазоны
            n_est = self.parse_range(self.n_est_le.text(), int)
            max_depth = self.parse_range(self.max_depth_le.text(), int)
            lr = self.parse_range(self.learning_rate_le.text(), float)

            # Запуск Optuna
            optuna_n_jobs = int(self.optuna_n_jobs_le.text())
            cv_n_jobs = int(self.cv_n_jobs_le.text())  # Используем новое поле

            study = self.analyzer.run_optuna_study(
                model_name=self.model_combo.currentText(),
                n_trials=n_trials, timeout=timeout, direction=direction, scoring=scoring,
                n_est_range=n_est, max_depth_range=max_depth, learning_rate_range=lr,
                cv=cv, n_jobs_cv=cv_n_jobs, random_state=rs,
                optuna_n_jobs=optuna_n_jobs
            )

            # Принудительная очистка памяти после Optuna
            gc.collect()

            if not study.best_trial:
                QMessageBox.warning(self, "Optuna", "Не найдено решений.")
                return
            # Создание лучшей модели
            best_params = study.best_params

            if self.model_combo.currentText() == "Random Forest":
                model_cls = RandomForestRegressor if self.analyzer.task_type == "regression" else RandomForestClassifier
            elif self.model_combo.currentText() == "Gradient Boosting":
                model_cls = GradientBoostingRegressor if self.analyzer.task_type == "regression" else GradientBoostingClassifier
            else:
                raise ValueError("Модель не поддерживается")

            best_model = model_cls(**best_params, random_state=rs)

            # Кривая обучения
            lc_result = self.analyzer.compute_learning_curve(best_model, scoring=scoring, cv=cv, n_points=n_points, n_jobs_cv=n_jobs_cv, random_state=rs)

            # Отображение
            self.display_result(
                model_name=f"{self.model_combo.currentText()} ({self.analyzer.task_type})",
                **lc_result,
                scoring=scoring,
                best_params=best_params
            )

            # Финальная очистка памяти после всего анализа
            gc.collect()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка анализа:\n{e}")

    def parse_range(self, text, dtype):
        text = text.strip()
        if not text or 'none' in text.lower():
            return (3, 10)  # default
        if '-' in text:
            a, b = map(dtype, text.split('-'))
            return (a, b)
        return (dtype(text), dtype(text))
    
    def display_result(self, model_name, final_val, gap, final_test, train_sizes, train_mean, val_mean, scoring, best_params):
        model_group = QGroupBox(f" {model_name} (Optuna)")
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

        add_metric(layout, "Val Final", final_val, "Средняя метрика на валидации при полном обучении")
        add_metric(layout, "Gap", gap, "Разница между обучением и валидацией. >0.1 — признак переобучения")
        add_metric(layout, "Test", final_test, "Оценка на независимом тестовом наборе")

        # Параметры
        param_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in best_params.items()])
        params_label = QLabel(f"<small><u>Лучшие параметры:</u><br>{param_text}</small>")
        params_label.setWordWrap(True)
        params_label.setStyleSheet("font-size: 12px; color: #777;")
        layout.addWidget(params_label)

        # Кнопка графика
        plot_btn = QPushButton("📈 Показать график")
        plot_btn.clicked.connect(lambda: self.plot_curve(train_sizes, train_mean, val_mean, model_name, scoring))
        layout.addWidget(plot_btn)

        model_group.setLayout(layout)
        self.results_layout.addWidget(model_group)

        # Ограничиваем количество результатов (оставляем последние 3)
        while self.results_layout.count() > 3:
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.update_memory_usage()


    def plot_curve(self, train_sizes, train_mean, val_mean, model_name, scoring):
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Обучение')
        plt.plot(train_sizes, val_mean, 'o-', label='Валидация')
        plt.xlabel('Размер выборки')
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
        ylabel = metric_names.get(scoring, scoring.replace("neg_", "").replace("_", " ").title())
        plt.ylabel(ylabel)
        plt.title(f"Кривая обучения — {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def update_memory_usage(self):
        try:
            mem_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"📊 Память: {mem_mb:.1f} МБ")
        except:
            self.memory_label.setText("📊 Память: ошибка")

    def closeEvent(self, event):
        plt.close('all')
        self.analyzer = None
        gc.collect()
        super().closeEvent(event)
