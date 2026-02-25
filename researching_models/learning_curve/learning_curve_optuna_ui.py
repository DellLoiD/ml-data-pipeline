from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QApplication,
    QGroupBox, QButtonGroup, QRadioButton, QLineEdit, QScrollArea, QDialog, QFrame, QComboBox, QFormLayout
)
from PySide6.QtWidgets import QScrollArea
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import gc

from researching_models.learning_curve.learning_curve_optuna_logic import OptunaAnalyzer, logger
from researching_models.check_models_loading_screen import LoadingScreen
from researching_models.learning_curve.learning_curve_worker import LearningCurveWorker

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

class LearningCurveOptunaTab(QWidget):
    def __init__(self, analyzer=None, main_window=None):
        super().__init__()
        self.analyzer = analyzer or OptunaAnalyzer()
        self.main_window = main_window
        self.results_layout = main_window.results_layout if main_window else QVBoxLayout()
        self.curve_params = {}
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # === ВЫБОР МОДЕЛИ И OPTUNA ===
        models_group = QGroupBox("🤖 Optuna: Подбор гиперпараметров")
        models_layout = QVBoxLayout()

        # === СТРОКА: Модель + Кнопки режима + Optuna + CV + learning_rate + Число итераций + Таймаут ===
        model_jobs_layout = QHBoxLayout()

        # Модель
        model_group = QGroupBox("Модель")
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "Gradient Boosting"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        model_jobs_layout.addWidget(model_group)

        # Кнопки режимов
        mode_group = QGroupBox("Режим")
        mode_layout = QHBoxLayout()
        self.day_btn = QPushButton("🌞")
        self.night_btn = QPushButton("🌙")
        self.day_btn.setFixedSize(40, 25)
        self.night_btn.setFixedSize(40, 25)
        self.day_btn.clicked.connect(self.set_day_mode)
        self.night_btn.clicked.connect(self.set_night_mode)
        mode_layout.addWidget(self.day_btn)
        mode_layout.addWidget(self.night_btn)
        mode_group.setLayout(mode_layout)
        model_jobs_layout.addWidget(mode_group)

        # Optuna n_jobs
        optuna_job_group = QGroupBox("Optuna_n_jobs")
        optuna_job_layout = QHBoxLayout()
        self.optuna_n_jobs_le = QLineEdit("1")
        self.optuna_n_jobs_le.setFixedWidth(60)
        optuna_job_layout.addWidget(self.optuna_n_jobs_le)
        help_optuna_btn = QPushButton("?")
        help_optuna_btn.setFixedSize(20, 20)
        help_optuna_text = "<b>Optuna n_jobs</b><br><br>Основной кран параллелизма.<br>Число trialов, запускаемых параллельно."
        help_optuna_btn.clicked.connect(lambda: HelpDialog("Optuna n_jobs", help_optuna_text, self).exec_())
        optuna_job_layout.addWidget(help_optuna_btn)
        optuna_job_group.setLayout(optuna_job_layout)
        model_jobs_layout.addWidget(optuna_job_group)

        # CV n_jobs
        cv_job_group = QGroupBox("CV_n_jobs")
        cv_job_layout = QHBoxLayout()
        self.cv_n_jobs_le = QLineEdit("1")
        self.cv_n_jobs_le.setFixedWidth(30)
        cv_job_layout.addWidget(self.cv_n_jobs_le)
        help_cv_btn = QPushButton("?")
        help_cv_btn.setFixedSize(20, 20)
        help_cv_text = "<b>CV n_jobs</b><br><br>Дополнительное ускорение (макс. = 2).<br>Число процессов внутри одного trial (в cross_val_score)."
        help_cv_btn.clicked.connect(lambda: HelpDialog("CV n_jobs", help_cv_text, self).exec_())
        cv_job_layout.addWidget(help_cv_btn)
        cv_job_group.setLayout(cv_job_layout)
        model_jobs_layout.addWidget(cv_job_group)

        # learning_rate в той же строке
        lr_group = QGroupBox("learning rate")
        lr_layout = QHBoxLayout()
        self.learning_rate_le = QLineEdit("0.01-0.3")
        self.learning_rate_le.setFixedWidth(80)
        lr_layout.addWidget(self.learning_rate_le)
        lr_group.setLayout(lr_layout)
        model_jobs_layout.addWidget(lr_group)

        # Число итераций
        trials_group = QGroupBox("Число итераций")
        trials_layout = QHBoxLayout()
        self.optuna_trials = QLineEdit("20")
        trials_layout.addWidget(self.optuna_trials)
        trials_group.setLayout(trials_layout)
        model_jobs_layout.addWidget(trials_group)

        # Таймаут
        timeout_group = QGroupBox("Таймаут (сек)")
        timeout_layout = QHBoxLayout()
        self.optuna_timeout = QLineEdit("600")
        timeout_layout.addWidget(self.optuna_timeout)
        timeout_group.setLayout(timeout_layout)
        model_jobs_layout.addWidget(timeout_group)

        model_jobs_layout.addStretch() 
        models_layout.addLayout(model_jobs_layout)

        # === Вторая строка: Метрика, Цель, n_estimators, max_depth ===
        second_row_layout = QHBoxLayout()

        # Метрика
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
        second_row_layout.addWidget(scoring_group)

        # Цель оптимизации
        direction_group = QGroupBox("Цель оптимизации")
        direction_layout = QHBoxLayout()
        self.maximize_radio = QRadioButton("maximize")
        self.minimize_radio = QRadioButton("minimize")
        self.maximize_radio.setChecked(True)
        direction_layout.addWidget(self.maximize_radio)
        direction_layout.addWidget(self.minimize_radio)
        direction_group.setLayout(direction_layout)
        second_row_layout.addWidget(direction_group)

        # n_estimators
        n_est_group = QGroupBox("n_estimators")
        n_est_group_layout = QHBoxLayout()
        self.n_est_le = QLineEdit("50-200")
        n_est_group_layout.addWidget(self.n_est_le)
        n_est_group.setLayout(n_est_group_layout)
        second_row_layout.addWidget(n_est_group)

        # max_depth
        max_depth_group = QGroupBox("max_depth")
        max_depth_group_layout = QHBoxLayout()
        self.max_depth_le = QLineEdit("2-5")
        max_depth_group_layout.addWidget(self.max_depth_le)
        max_depth_group.setLayout(max_depth_group_layout)
        second_row_layout.addWidget(max_depth_group)

        models_layout.addLayout(second_row_layout)
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
        self.analyze_btn = QPushButton("🚀 Запустить анализ (Optuna)")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analyze_btn.setEnabled(True)
        main_layout.addWidget(self.analyze_btn)

        self.setLayout(main_layout)

        self.update_scoring_options()

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

    def on_model_selected_from_main(self, model_name):
        """Вызывается из основного окна при смене модели"""
        self.model_combo.setCurrentText(model_name)
        self.on_model_changed()

    def on_task_type_changed(self, task_type):
        """Вызывается из основного окна при смене типа задачи"""
        self.analyzer.task_type = task_type
        self.update_scoring_options()
        self.update_models()

    def on_model_changed(self):
        # Включает/отключает поле learning_rate в зависимости от выбранной модели.
        # Поле активно только для Gradient Boosting.
        is_gb = self.model_combo.currentText() == "Gradient Boosting"
        self.learning_rate_le.setEnabled(is_gb)
        
    def on_analyze(self):
        try:
            # Проверка загрузки датасета
            if self.analyzer.X_train is None:
                QMessageBox.warning(self, "Ошибка", "Датасет не загружен. Пожалуйста, загрузите датасет перед запуском анализа.")
                return
            
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
                QMessageBox.warning(self, "Ошибка", "Не выбрана метрика! Пожалуйста, выберите целевую метрику.")
                return
            
            # Показ заглушки
            self.loading_screen = LoadingScreen()
            self.loading_screen.show()
            QApplication.processEvents()  

            cv = int(self.curve_params['CV'].text())
            n_jobs_cv = int(self.curve_params['n_jobs'].text())
            n_points = int(self.curve_params['Число точек'].text())
            rs = int(self.curve_params['Random State'].text())

            # Диапазоны
            n_est = self.parse_range(self.n_est_le.text(), int)
            max_depth = self.parse_range(self.max_depth_le.text(), int)
            lr = self.parse_range(self.learning_rate_le.text(), float)

            # Запуск в потоке
            self.worker = LearningCurveWorker(
                analyzer=self.analyzer,
                model_name=self.model_combo.currentText(),
                n_trials=n_trials,
                timeout=timeout,
                direction=direction,
                scoring=scoring,
                n_est_range=n_est,
                max_depth_range=max_depth,
                learning_rate_range=lr,
                cv=cv,
                n_jobs_cv=n_jobs_cv,
                random_state=rs,
                optuna_n_jobs=int(self.optuna_n_jobs_le.text())
            )
            self.worker.result_ready.connect(self.on_worker_result_ready)
            self.worker.error_occurred.connect(self.on_worker_error)
            self.worker.start()

        finally:
            # Оставляем заглушку открытой — закроется из потока
            pass

    def on_worker_result_ready(self, result):
        """Вызывается из потока при успешном завершении"""
        try:
            best_model = result['best_model']
            lc_result = result['lc_result']
            best_params = result['best_params']
            scoring = result['scoring']
            model_name = result['model_name']

            logger.info("Отправка результатов в интерфейс...")
            logger.info(f"Данные для display_result: model_name={model_name}, final_val={lc_result['final_val']:.4f}, final_test={lc_result['final_test']:.4f}, gap={lc_result['gap']:.4f}, scoring={scoring}, best_params={best_params}")
            self.display_result(
                model_name=f"{model_name} ({self.analyzer.task_type})",
                **lc_result,
                scoring=scoring,
                best_params=best_params
            )
            logger.info("Результаты успешно переданы в интерфейс.")
        except Exception as e:
            logger.error(f"Ошибка при вызове display_result: {e}")
            self.on_worker_error(str(e))
        finally:
            self._cleanup_loading_screen()

    def on_worker_error(self, error_msg):
        """Вызывается из потока при ошибке"""
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка в фоновом потоке:\n{error_msg}")
        logger.error(f"Ошибка в фоновом потоке: {error_msg}")
        self._cleanup_loading_screen()

    def _cleanup_loading_screen(self):
        """Закрывает заглушку"""
        if hasattr(self, 'loading_screen') and self.loading_screen:
            self.loading_screen.close()
            self.loading_screen = None
            logger.info("Заглушка закрыта")

    def parse_range(self, text, dtype):
        text = text.strip()
        if not text or 'none' in text.lower():
            return (3, 10)  # default
        if '-' in text:
            a, b = map(dtype, text.split('-'))
            return (a, b)
        return (dtype(text), dtype(text))
    
    def display_result(self, model_name, final_val, gap, final_test, train_sizes, train_mean, val_mean, scoring, best_params):
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
            logger.info(f"Создан central_results_layout в OptunaTab с id: {id(self.central_results_layout)}")

            # Добавляем в основной макет вкладки
            self.layout().insertWidget(3, self.central_results_group)  # после вкладок

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
        self.central_results_layout.addWidget(model_group)

        # Ограничиваем количество результатов (оставляем последние 3)
        while self.central_results_layout.count() > 3:
            item = self.central_results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if self.main_window:
            self.main_window.update_memory_usage()

    def plot_curve(self, train_sizes, train_mean, val_mean, model_name, scoring):
        import matplotlib.pyplot as plt
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
        if self.main_window:
            self.main_window.update_memory_usage()