from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QApplication,
    QGroupBox, QButtonGroup, QRadioButton, QLineEdit, QScrollArea, QDialog, QFrame, QComboBox, QFormLayout
)
from PySide6.QtWidgets import QScrollArea
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import gc

from researching_models.cross_validation.cross_validation_optuna_logic import OptunaAnalyzer, logger
from researching_models.check_models_loading_screen import LoadingScreen
import matplotlib.pyplot as plt

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

class CrossValidationOptunaTab(QWidget):
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

        # === ПАРАМЕТРЫ КРОСС-ВАЛИДАЦИИ ===
        cv_group = QGroupBox("⚙️ Параметры кросс-валидации")
        cv_layout = QHBoxLayout()
        params = [("CV", "5"), ("n_jobs", "1"), ("Random State", "42")]
        for label_text, default_value in params:
            group_box = QGroupBox(label_text)
            le = QLineEdit(default_value)
            le.setFixedWidth(60)
            layout = QHBoxLayout()
            layout.addWidget(le)
            group_box.setLayout(layout)
            self.curve_params[label_text] = le
            cv_layout.addWidget(group_box)
        cv_group.setLayout(cv_layout)
        main_layout.addWidget(cv_group)

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

    def update_models(self):
        # Функция оставлена пустой, так как в текущей реализации она больше не требуется.
        # Ранее она использовалась для обновления списка моделей при смене типа задачи, но теперь это не нужно.
        pass

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
            rs = int(self.curve_params['Random State'].text())

            # Диапазоны
            n_est = self.parse_range(self.n_est_le.text(), int)
            max_depth = self.parse_range(self.max_depth_le.text(), int)
            lr = self.parse_range(self.learning_rate_le.text(), float)

            # Запуск Optuna
            optuna_n_jobs = int(self.optuna_n_jobs_le.text())
            cv_n_jobs = int(self.cv_n_jobs_le.text()) 

            logger.info("Запуск анализа Optuna и кросс-валидации...")
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
                logger.warning("Оптимизация Optuna не нашла подходящих решений.")
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

            # Кросс-валидация
            cv_result = self.analyzer.compute_cross_validation_scores(best_model, scoring=scoring, cv=cv, n_jobs_cv=n_jobs_cv, random_state=rs)

            # Сохраняем данные для графика
            self.current_cv_scores = cv_result.get('scores')
            self.current_model_name = f"{self.model_combo.currentText()} ({self.analyzer.task_type})"
            self.current_scoring = scoring

            # Закрываем заглушку перед обновлением интерфейса
            if self.loading_screen:
                self.loading_screen.close()
                logger.info("Заглушка закрыта перед отображением результатов")
            
            # Принудительное обновление интерфейса
            QApplication.processEvents()

            # Отображение результатов
            logger.info("Отправка результатов в интерфейс...")
            logger.info(f"Данные для display_result: model_name={self.model_combo.currentText()}, mean_score={cv_result['mean_score']:.4f}, std_score={cv_result['std_score']:.4f}, scoring={scoring}, best_params={best_params}")
            try:
                self.display_result(
                    model_name=f"{self.model_combo.currentText()} ({self.analyzer.task_type})",
                    **cv_result,
                    scoring=scoring,
                    best_params=best_params
                )
                logger.info("Результаты успешно переданы в интерфейс.")
            except Exception as e:
                logger.error(f"Ошибка при вызове display_result: {e}")
                raise

            # Финальная очистка памяти после всего анализа
            gc.collect()
            logger.info("Анализ Optuna и кросс-валидации завершен. Результаты отображены.")

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
    
    def display_result(self, model_name, mean_score, std_score, scoring, best_params, scores=None):
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

        add_metric(layout, "Средняя метрика", mean_score, f"Среднее значение метрики: {mean_score:.4f} ± {std_score:.4f}")
        add_metric(layout, "Стандартное отклонение", std_score, "Стандартное отклонение метрики по фолдам")

        # Параметры
        param_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in best_params.items()])
        params_label = QLabel(f"<small><u>Лучшие параметры:</u><br>{param_text}</small>")
        params_label.setWordWrap(True)
        params_label.setStyleSheet("font-size: 12px; color: #777;")
        layout.addWidget(params_label)

        # Кнопка для графика кросс-валидации
        if scores is not None:
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