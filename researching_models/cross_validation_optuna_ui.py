# cross_validation_ui.py — кросс-валидация + Optuna-подбор

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QInputDialog,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton, QLineEdit, QScrollArea, QDialog, QFrame, QComboBox,
    QTabWidget, QFormLayout
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import psutil
from joblib import parallel_backend
import optuna  




class HelpDialog(QDialog):
    """Справка по метрикам и параметрам"""
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.setModal(True)
        self.resize(400, 300)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
        self.setLayout(layout)


class CrossValidationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.target_col = None
        self.task_type = "classification"
        self.results_layout = None
        self.process = psutil.Process(os.getpid())
        self.optuna_results = {}
        self.manual_checkboxes = []
        self.manual_lines = {}
        self.optuna_checkboxes = []        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Кросс-валидация и Optuna")
        main_layout = QVBoxLayout()

        title_label = QLabel("Анализ моделей с подбором гиперпараметров")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title_label)

        task_layout = QHBoxLayout()
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
        task_layout.addStretch()
        main_layout.addLayout(task_layout)

        self.load_btn = QPushButton("📁 Загрузить датасет")
        self.load_btn.clicked.connect(self.on_load_dataset)
        main_layout.addWidget(self.load_btn)

        self.target_label = QLabel("Целевая переменная: не выбрана")
        self.target_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.target_label)

        self.memory_label = QLabel("📊 Память: ? МБ")
        self.memory_label.setStyleSheet("color: #555; font-size: 11px;")
        main_layout.addWidget(self.memory_label)

        tabs = QTabWidget()
        manual_tab = self.create_manual_tab()
        optuna_tab = self.create_optuna_tab()
        tabs.addTab(manual_tab, "Ручные параметры")
        tabs.addTab(optuna_tab, "Автонастройка (Optuna)")
        main_layout.addWidget(tabs)

        btn_layout = QHBoxLayout()
        self.analyze_manual_btn = QPushButton("✅ Оценить модели (ручные)")
        self.analyze_manual_btn.clicked.connect(self.on_analyze_manual)
        btn_layout.addWidget(self.analyze_manual_btn)

        self.analyze_optuna_btn = QPushButton("🤖 Автоподбор и оценка (Optuna)")
        self.analyze_optuna_btn.clicked.connect(self.on_analyze_optuna)
        btn_layout.addWidget(self.analyze_optuna_btn)

        main_layout.addLayout(btn_layout)

        results_group = QGroupBox("📊 Результаты")
        results_layout = QVBoxLayout()
        help_label = QLabel("Результаты CV и финальный тест.\nПрокрутите вправо.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("font-size: 11px; color: #555;")
        results_layout.addWidget(help_label)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        results_layout.addWidget(line)
        self.results_layout = QHBoxLayout()
        self.results_layout.setSpacing(15)
        scroll_content = QWidget()
        scroll_content.setLayout(self.results_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        scroll.setFixedHeight(250)
        results_layout.addWidget(scroll)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        self.setLayout(main_layout)
        self.resize(1100, 900)
        self.show()
        self.update_memory_usage()
        
    def update_memory_usage(self):
        """Обновляет метку с текущим использованием ОЗУ"""
        try:
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024  # в МБ
            self.memory_label.setText(f"📊 Память: {mem_mb:.1f} МБ")
        except Exception as e:
            self.memory_label.setText("📊 Память: ошибка")
            
    def kill_child_processes(self):
        """Принудительно завершает дочерние процессы (например, от joblib или Optuna)"""
        try:
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)

            if not children:
                return

            # Пытаемся корректно завершить
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Ждём 3 секунды
            gone, alive = psutil.wait_procs(children, timeout=3)

            # Принудительно убиваем оставшиеся
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"[DEBUG] Ошибка при завершении процессов: {e}")



    def create_manual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        models_group = QGroupBox("Модели для анализа (ручные)")
        models_layout = QVBoxLayout()
        self.manual_classification_box = QGroupBox("Классификация")
        self.manual_classification_layout = QVBoxLayout()
        self.manual_classification_box.setLayout(self.manual_classification_layout)
        models_layout.addWidget(self.manual_classification_box)
        self.manual_regression_box = QGroupBox("Регрессия")
        self.manual_regression_layout = QVBoxLayout()
        self.manual_regression_box.setLayout(self.manual_regression_layout)
        models_layout.addWidget(self.manual_regression_box)
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

        cv_group = QGroupBox("⚙️ Параметры кросс-валидации")
        cv_layout = QHBoxLayout()
        params = [("CV", "5"), ("n_jobs", "-1"), ("Random State", "42")]
        self.cv_params = {}
        for label_text, default_value in params:
            group_box = QGroupBox(label_text)
            group_layout = QHBoxLayout()
            le = QLineEdit(default_value)
            le.setFixedWidth(60)
            btn = QPushButton("?")
            btn.setFixedSize(20, 20)
            btn.clicked.connect(lambda ch, t=label_text: self.show_cv_help(t))
            group_layout.addWidget(le)
            group_layout.addWidget(btn)
            group_box.setLayout(group_layout)
            self.cv_params[label_text] = le
            cv_layout.addWidget(group_box)
        cv_group.setLayout(cv_layout)
        layout.addWidget(cv_group)

        self.create_manual_models()
        tab.setLayout(layout)
        return tab

    def create_optuna_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        optuna_group = QGroupBox("🤖 Optuna: Параметры оптимизации")
        optuna_layout = QFormLayout()
        self.trials_le = QLineEdit("50")
        self.timeout_le = QLineEdit("600")
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["maximize", "minimize"])
        self.scoring_combo = QComboBox()
        self.cv_folds_le = QLineEdit("5")
        self.n_jobs_le = QLineEdit("-1")
        optuna_layout.addRow("Число итераций (trials):", self.trials_le)
        optuna_layout.addRow("Таймаут (сек):", self.timeout_le)
        optuna_layout.addRow("Цель:", self.direction_combo)
        optuna_layout.addRow("Метрика:", self.scoring_combo)
        optuna_layout.addRow("Фолды CV:", self.cv_folds_le)
        optuna_layout.addRow("n_jobs:", self.n_jobs_le)
        optuna_group.setLayout(optuna_layout)
        layout.addWidget(optuna_group)

        models_group = QGroupBox("Модели для автонастройки")
        models_layout = QVBoxLayout()
        self.optuna_clf_box = QGroupBox("Классификация")
        self.optuna_clf_layout = QVBoxLayout()
        self.optuna_clf_box.setLayout(self.optuna_clf_layout)
        models_layout.addWidget(self.optuna_clf_box)
        self.optuna_reg_box = QGroupBox("Регрессия")
        self.optuna_reg_layout = QVBoxLayout()
        self.optuna_reg_box.setLayout(self.optuna_reg_layout)
        models_layout.addWidget(self.optuna_reg_box)
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

        self.create_optuna_model_inputs()
        tab.setLayout(layout)
        return tab

    def show_cv_help(self, param):
        texts = {
            "CV": "Количество фолдов. 5 — стандарт.",
            "n_jobs": "-1 = все ядра, 1 = один поток",
            "Random State": "Фиксирует разбиение на фолды"
        }
        HelpDialog("Параметры CV", texts.get(param, ""), self).exec_()

    def create_manual_models(self):
        clf_models = {
            'Random Forest Classification': ['Test Size', 'Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Random State'],
            'Gradient Boosting Classification': ['Test Size', 'Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Random State'],
            'Logistic Regression Classification': ['Test Size', 'C', 'Max Iterations', 'Penalty', 'Random State']
        }
        reg_models = {
            'Random Forest Regression': ['Test Size', 'Кол-во деревьев', 'Max Depth', 'Min Samples Split', 'Random State'],
            'Gradient Boosting Regression': ['Test Size', 'Кол-во деревьев', 'Learning Rate', 'Max Depth', 'Random State']
        }
        defaults = {
            'Test Size': '0.2', 'Кол-во деревьев': '100', 'Max Depth': 'None', 'Min Samples Split': '2',
            'Random State': '42', 'Learning Rate': '0.1', 'C': '1.0', 'Max Iterations': '100', 'Penalty': 'l2'
        }

        for name, params in clf_models.items():
            self._add_model_to_layout(name, params, defaults, self.manual_classification_layout, "manual_")

        for name, params in reg_models.items():
            self._add_model_to_layout(name, params, defaults, self.manual_regression_layout, "manual_")

        self.manual_classification_box.setVisible(self.task_type == "classification")
        self.manual_regression_box.setVisible(self.task_type == "regression")

    def create_optuna_model_inputs(self):
        self.update_scoring_options()
        rf_group = QGroupBox("🌲 Random Forest")
        rf_layout = QFormLayout()
        rf_layout.addRow("n_estimators:", QLineEdit("50-300"))
        rf_layout.addRow("max_depth:", QLineEdit("3-10,None"))
        rf_layout.addRow("min_samples_split:", QLineEdit("2-20"))
        rf_layout.addRow("min_samples_leaf:", QLineEdit("1-10"))
        rf_group.setLayout(rf_layout)
        self.optuna_clf_layout.addWidget(rf_group)
        self.optuna_reg_layout.addWidget(rf_group)

        gb_group = QGroupBox("⚡ Gradient Boosting")
        gb_layout = QFormLayout()
        gb_layout.addRow("n_estimators:", QLineEdit("50-300"))
        gb_layout.addRow("learning_rate:", QLineEdit("0.01-0.3"))
        gb_layout.addRow("max_depth:", QLineEdit("3-10"))
        gb_layout.addRow("subsample:", QLineEdit("0.8-1.0"))
        gb_group.setLayout(gb_layout)
        self.optuna_clf_layout.addWidget(gb_group)
        self.optuna_reg_layout.addWidget(gb_group)

        lr_group = QGroupBox("📏 Logistic Regression")
        lr_layout = QFormLayout()
        lr_layout.addRow("C:", QLineEdit("0.1-10.0"))
        lr_layout.addRow("max_iter:", QLineEdit("100-1000"))
        penalty_layout = QHBoxLayout()
        self.lr_l1_cb = QCheckBox("l1"); self.lr_l1_cb.setChecked(True)
        self.lr_l2_cb = QCheckBox("l2"); self.lr_l2_cb.setChecked(True)
        self.lr_en_cb = QCheckBox("elasticnet")
        penalty_layout.addWidget(self.lr_l1_cb)
        penalty_layout.addWidget(self.lr_l2_cb)
        penalty_layout.addWidget(self.lr_en_cb)
        lr_layout.addRow("penalty:", penalty_layout)
        lr_group.setLayout(lr_layout)
        self.optuna_clf_layout.addWidget(lr_group)

        self.optuna_clf_box.setVisible(self.task_type == "classification")
        self.optuna_reg_box.setVisible(self.task_type == "regression")

    def _add_model_to_layout(self, model_name, params, defaults, layout, prefix):
        hbox = QHBoxLayout()
        cb = QCheckBox(model_name)
        self.__dict__[f"{prefix}checkboxes"].append(cb)
        hbox.addWidget(cb)
        lines = {}
        for param in params:
            lbl = QLabel(param)
            le = QLineEdit(defaults.get(param, "0"))
            le.setFixedWidth(80)
            btn = QPushButton("?")
            btn.setFixedSize(20, 20)
            btn.clicked.connect(lambda ch, p=param: self.show_param_help(p))
            hbox.addWidget(lbl)
            hbox.addWidget(le)
            hbox.addWidget(btn)
            lines[param] = le
        self.__dict__[f"{prefix}lines"][model_name] = lines
        hbox.addStretch()
        layout.addLayout(hbox)

    def show_param_help(self, param):
        texts = {
            'Test Size': "Доля данных для теста",
            'Кол-во деревьев': "Число деревьев в ансамбле",
            'Max Depth': "Макс. глубина дерева",
            'Min Samples Split': "Мин. объектов для разбиения",
            'Learning Rate': "Скорость обучения",
            'C': "Сила регуляризации",
            'Max Iterations': "Макс. итераций обучения",
            'Penalty': "Тип регуляризации",
            'Random State': "Фиксация случайности"
        }
        HelpDialog("Параметр", texts.get(param, param), self).exec_()

    def on_task_selected(self):
        self.task_type = "classification" if self.classification_radio.isChecked() else "regression"
        self.manual_classification_box.setVisible(self.task_type == "classification")
        self.manual_regression_box.setVisible(self.task_type == "regression")
        self.optuna_clf_box.setVisible(self.task_type == "classification")
        self.optuna_reg_box.setVisible(self.task_type == "regression")
        self.update_scoring_options()

    def update_scoring_options(self):
        self.scoring_combo.clear()
        metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"] if self.task_type == "classification" \
            else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
        self.scoring_combo.addItems(metrics)

    def on_load_dataset(self):
        reply = QMessageBox.question(self, "Режим", "Загрузить train/test отдельно?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.load_separate_datasets()
        else:
            self.load_single_dataset()

    def load_single_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV", "./dataset/", "CSV (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path, comment='#')
            self.df = df
            self.select_target_variable()
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
            target, ok = QInputDialog.getItem(self, "Целевая", "Выберите:", sorted(common_cols), 0, False)
            if not ok or not target: return
            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]
            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test
            self.target_col = target
            self.target_label.setText(f"Целевая: {target}")
            self.analyze_manual_btn.setEnabled(True)
            self.analyze_optuna_btn.setEnabled(True)
            self.update_memory_usage()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка:\n{e}")

    def select_target_variable(self):
        if self.df is None: return
        target, ok = QInputDialog.getItem(self, "Целевая", "Выберите:", sorted(self.df.columns), 0, False)
        if not ok or not target: return
        df_local = self.df.copy()
        if self.task_type == "classification" and df_local[target].dtype == 'object':
            df_local[target] = LabelEncoder().fit_transform(df_local[target])
        X = df_local.drop(columns=[target]).select_dtypes(include=['number'])
        y = df_local[target]
        if X.empty:
            QMessageBox.critical(self, "Ошибка", "Нет числовых признаков.")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.target_col = target
        self.target_label.setText(f"Целевая: {target}")
        self.analyze_manual_btn.setEnabled(True)
        self.analyze_optuna_btn.setEnabled(True)
        self.update_memory_usage()

    def parse_range(self, text, dtype=float):
        text = text.strip()
        if not text or 'none' in text.lower(): return None
        if '-' in text:
            a, b = text.split('-')
            return (dtype(a), dtype(b))
        if ',' in text:
            return [x.strip() for x in text.split(',')]
        return dtype(text)

    def check_data_ready(self):
        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "Ошибка", "Загрузите данные.")
            return False
        if not self.target_col:
            QMessageBox.warning(self, "Ошибка", "Не выбрана целевая переменная.")
            return False
        return True

    def get_selected_models(self, mode):
        checkboxes = self.manual_checkboxes if mode == "manual" else self.optuna_checkboxes
        return [cb.text() for cb in checkboxes if cb.isChecked()]

    def get_cv_params(self):
        cv = self.safe_int(None, self.cv_params['CV'].text(), 5)
        n_jobs = self.safe_int(None, self.cv_params['n_jobs'].text(), -1)
        rs = self.safe_int(None, self.cv_params['Random State'].text(), 42)
        return cv, n_jobs, rs

    def safe_int(self, container, val, default):
        try:
            return int(val.strip()) if val.strip() else default
        except:
            return default

    def safe_float(self, val, default):
        try:
            return float(val.strip()) if val.strip() else default
        except:
            return default

    def on_analyze_manual(self):
        self.run_manual_analysis()

    def on_analyze_optuna(self):
        self.run_optuna_analysis()

    def run_manual_analysis(self):
        self.kill_child_processes()
        if not self.check_data_ready(): return
        selected = self.get_selected_models("manual")
        if not selected: return
        cv, n_jobs, rs = self.get_cv_params()
        for model_name in selected:
            self.evaluate_model_manual(model_name, cv, n_jobs, rs)

    def evaluate_model_manual(self, model_name, cv, n_jobs, rs):
        try:
            params = self.manual_lines[model_name]
            model = self._create_model(model_name, params)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
            scoring = 'accuracy' if 'Classification' in model_name else 'r2'
            with parallel_backend('loky', n_jobs=n_jobs):
                scores = cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
            cv_mean, cv_std = np.mean(scores), np.std(scores)
            model.fit(X_train_scaled, self.y_train)
            final_score = model.score(X_test_scaled, self.y_test)
            self.display_results(model_name, cv_mean, cv_std, final_score, scores, scoring, params)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"{model_name}:\n{e}")

    def run_optuna_analysis(self):
        self.kill_child_processes()
        if not self.check_data_ready(): return
        selected = self.get_selected_models("optuna")
        if not selected: return
        n_trials = self.safe_int(None, self.trials_le.text(), 50)
        timeout = self.safe_int(None, self.timeout_le.text(), 600) or None
        direction = self.direction_combo.currentText()
        scoring = self.scoring_combo.currentText()
        cv = self.safe_int(None, self.cv_folds_le.text(), 5)
        n_jobs_cv = self.safe_int(None, self.n_jobs_le.text(), -1)
        for model_name in selected:
            self.optimize_with_optuna(model_name, n_trials, timeout, direction, scoring, cv, n_jobs_cv)

    def optimize_with_optuna(self, model_name, n_trials, timeout, direction, scoring, cv, n_jobs):
        try:
            def objective(trial):
                if "Random Forest" in model_name:
                    n_est = trial.suggest_int('n_estimators', *self.parse_range(self.rf_n_est.text(), int))
                    max_depth_val = self.parse_range(self.rf_max_depth.text(), int)
                    max_depth = trial.suggest_categorical('max_depth', [3, 5, 7, 10, None]) if 'None' in self.rf_max_depth.text() else \
                                 trial.suggest_int('max_depth', *max_depth_val)
                    min_split = trial.suggest_int('min_samples_split', *self.parse_range(self.rf_min_split.text(), int))
                    min_leaf = trial.suggest_int('min_samples_leaf', *self.parse_range(self.rf_min_leaf.text(), int))
                    model = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, min_samples_split=min_split,
                                                   min_samples_leaf=min_leaf, random_state=42) if 'Class' in model_name else \
                            RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, min_samples_split=min_split,
                                                  min_samples_leaf=min_leaf, random_state=42)

                elif "Gradient Boosting" in model_name:
                    n_est = trial.suggest_int('n_estimators', *self.parse_range(self.gb_n_est.text(), int))
                    lr = trial.suggest_float('learning_rate', *self.parse_range(self.gb_lr.text(), float), log=True)
                    max_depth = trial.suggest_int('max_depth', *self.parse_range(self.gb_max_depth.text(), int))
                    subsample = trial.suggest_float('subsample', *self.parse_range(self.gb_subsample.text(), float))
                    model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                                                      subsample=subsample, random_state=42) if 'Class' in model_name else \
                            GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                                                      subsample=subsample, random_state=42)

                elif "Logistic Regression" in model_name:
                    C = trial.suggest_float('C', *self.parse_range(self.lr_C.text(), float))
                    max_iter = trial.suggest_int('max_iter', *self.parse_range(self.lr_max_iter.text(), int))
                    penalties = [p for p, cb in zip(['l1','l2','elasticnet'], [self.lr_l1_cb, self.lr_l2_cb, self.lr_en_cb]) if cb.isChecked()]
                    penalty = trial.suggest_categorical('penalty', penalties) if penalties else 'l2'
                    solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
                    model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, random_state=42)

                else:
                    return 0.0

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(self.X_train)
                scores = cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
                return np.mean(scores)

            study = optuna.create_study(direction="maximize" if direction == "maximize" else "minimize")
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            best_model = self._create_model_from_trial(model_name, study.best_params)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
            scores = cross_val_score(best_model, X_train_scaled, self.y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
            cv_mean, cv_std = np.mean(scores), np.std(scores)
            best_model.fit(X_train_scaled, self.y_train)
            final_score = best_model.score(X_test_scaled, self.y_test)
            self.display_results(model_name, cv_mean, cv_std, final_score, scores, scoring, study.best_params, is_optuna=True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Optuna {model_name}:\n{e}")

    def create_optuna_model_inputs(self):
        self.update_scoring_options()

        # Random Forest
        rf_group = QGroupBox("🌲 Random Forest")
        rf_layout = QFormLayout()
        self.rf_n_est = QLineEdit("50-300")
        self.rf_max_depth = QLineEdit("3-10,None")
        self.rf_min_split = QLineEdit("2-20")
        self.rf_min_leaf = QLineEdit("1-10")
        rf_layout.addRow("n_estimators:", self.rf_n_est)
        rf_layout.addRow("max_depth:", self.rf_max_depth)
        rf_layout.addRow("min_samples_split:", self.rf_min_split)
        rf_layout.addRow("min_samples_leaf:", self.rf_min_leaf)
        rf_group.setLayout(rf_layout)
        self.optuna_clf_layout.addWidget(rf_group)
        self.optuna_reg_layout.addWidget(rf_group)

        # Gradient Boosting
        gb_group = QGroupBox("⚡ Gradient Boosting")
        gb_layout = QFormLayout()
        self.gb_n_est = QLineEdit("50-300")
        self.gb_lr = QLineEdit("0.01-0.3")
        self.gb_max_depth = QLineEdit("3-10")
        self.gb_subsample = QLineEdit("0.8-1.0")
        gb_layout.addRow("n_estimators:", self.gb_n_est)
        gb_layout.addRow("learning_rate:", self.gb_lr)
        gb_layout.addRow("max_depth:", self.gb_max_depth)
        gb_layout.addRow("subsample:", self.gb_subsample)
        gb_group.setLayout(gb_layout)
        self.optuna_clf_layout.addWidget(gb_group)
        self.optuna_reg_layout.addWidget(gb_group)

        # Logistic Regression
        lr_group = QGroupBox("📏 Logistic Regression")
        lr_layout = QFormLayout()
        self.lr_C = QLineEdit("0.1-10.0")
        self.lr_max_iter = QLineEdit("100-1000")
        lr_layout.addRow("C:", self.lr_C)
        lr_layout.addRow("max_iter:", self.lr_max_iter)
        penalty_layout = QHBoxLayout()
        self.lr_l1_cb = QCheckBox("l1"); self.lr_l1_cb.setChecked(True)
        self.lr_l2_cb = QCheckBox("l2"); self.lr_l2_cb.setChecked(True)
        self.lr_en_cb = QCheckBox("elasticnet")
        penalty_layout.addWidget(self.lr_l1_cb)
        penalty_layout.addWidget(self.lr_l2_cb)
        penalty_layout.addWidget(self.lr_en_cb)
        lr_layout.addRow("penalty:", penalty_layout)
        lr_group.setLayout(lr_layout)
        self.optuna_clf_layout.addWidget(lr_group)

        # Обновление видимости
        self.optuna_clf_box.setVisible(self.task_type == "classification")
        self.optuna_reg_box.setVisible(self.task_type == "regression")
        
    def _create_model_from_trial(self, name, params):
        """Создаёт модель с лучшими параметрами из Optuna"""
        if "Random Forest Classification" in name:
            return RandomForestClassifier(**params, random_state=42)
        elif "Random Forest Regression" in name:
            return RandomForestRegressor(**params, random_state=42)
        elif "Gradient Boosting Classification" in name:
            return GradientBoostingClassifier(**params, random_state=42)
        elif "Gradient Boosting Regression" in name:
            return GradientBoostingRegressor(**params, random_state=42)
        elif "Logistic Regression Classification" in name:
            solver = 'liblinear' if params.get('penalty') in ['l1', 'l2'] else 'saga'
            return LogisticRegression(**params, solver=solver, random_state=42)
        else:
            raise ValueError(f"Неизвестная модель: {name}")
        
    def display_results(self, model_name, cv_mean, cv_std, final_score, scores, scoring, params, is_optuna=False):
        """Отображает результаты CV и теста"""
        model_group = QGroupBox(f" {model_name} ")
        model_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #aaa;
                border-radius: 6px;
                margin: 0;
                padding: 10px;
                min-width: 260px;
            }
        """)
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)

        # Результаты
        row1 = QHBoxLayout()
        lbl1 = QLabel(f"CV среднее: {cv_mean:.4f}")
        btn1 = QPushButton("?")
        btn1.setFixedSize(20, 20)
        btn1.clicked.connect(lambda: HelpDialog(
            "Кросс-валидация", f"Средняя метрика по {self.safe_int(None, self.cv_folds_le.text(), 5)} фолдам.", self).exec_())
        row1.addWidget(lbl1)
        row1.addWidget(btn1)
        model_layout.addLayout(row1)

        row2 = QHBoxLayout()
        lbl2 = QLabel(f"CV std: ±{cv_std:.4f}")
        btn2 = QPushButton("?")
        btn2.setFixedSize(20, 20)
        btn2.clicked.connect(lambda: HelpDialog(
            "Стандартное отклонение", "Разброс по фолдам. Малый std — стабильная модель.", self).exec_())
        row2.addWidget(lbl2)
        row2.addWidget(btn2)
        model_layout.addLayout(row2)

        row3 = QHBoxLayout()
        lbl3 = QLabel(f"Final Test: {final_score:.4f}")
        btn3 = QPushButton("?")
        btn3.setFixedSize(20, 20)
        btn3.clicked.connect(lambda: HelpDialog(
            "Финальный тест", "Оценка на независимом test датасете.", self).exec_())
        row3.addWidget(lbl3)
        row3.addWidget(btn3)
        model_layout.addLayout(row3)

        # Параметры
        param_text = "<br>".join([f"{k}: {v}" for k, v in params.items()])
        mode = "Optuna" if is_optuna else "Ручные"
        params_label = QLabel(f"<small><b>{mode}:</b><br>{param_text}</small>")
        params_label.setWordWrap(True)
        params_label.setStyleSheet("font-size: 12px; color: #777;")
        model_layout.addWidget(params_label)

        # Кнопка графика
        plot_btn = QPushButton("📊 График CV")
        plot_btn.clicked.connect(
            lambda: self.plot_cv_scores(scores, model_name, scoring, 42)
        )
        model_layout.addWidget(plot_btn)

        model_group.setLayout(model_layout)
        self.results_layout.addWidget(model_group)

        # Ограничение: максимум 3 блока
        while self.results_layout.count() > 3:
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.update_memory_usage()
        
    def plot_cv_scores(self, scores, model_name, scoring, random_state):
        self.kill_child_processes()
        folds = np.arange(1, len(scores) + 1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        plt.figure(figsize=(8, 5))
        plt.bar(folds, scores, color='skyblue', edgecolor='black', alpha=0.7, label='Оценка фолда')
        plt.axhline(mean_score, color='red', linestyle='--', label=f'Среднее: {mean_score:.4f}')
        plt.fill_between(folds, mean_score - std_score, mean_score + std_score,
                         color='orange', alpha=0.2, label=f'±std ({std_score:.4f})')

        for i, score in enumerate(scores):
            plt.text(i + 1, score + 0.005 * (max(scores) - min(scores)), f"{score:.3f}",
                     ha='center', fontsize=9, color='darkblue')

        plt.xlabel('Фолд')
        plt.ylabel(scoring.replace("neg_", "").replace("_", " ").title())
        plt.title(f"Кросс-валидация — {model_name}")
        plt.xticks(folds)
        plt.legend(loc='best')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        self.update_memory_usage()
        
    def closeEvent(self, event):
        plt.close('all')
        self.df = self.X_train = self.y_train = self.X_test = self.y_test = None
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        gc.collect()
        super().closeEvent(event)





