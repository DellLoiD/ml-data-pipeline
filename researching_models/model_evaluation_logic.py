# model_evaluation_logic.py
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox, QLabel, QHBoxLayout, QPushButton, QWidget, QVBoxLayout
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
from .check_models_loading_screen import LoadingScreen


class EvaluationThread(QThread):
    finished_signal = Signal(list, str)
    error_signal = Signal(str)

    def __init__(self, parent, models_config, X_train, X_test, y_train, y_test, task_type):
        super().__init__(parent)
        self.models_config = models_config
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task_type = task_type

    def run(self):
        try:
            results = []
            total_time = 0.0
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)

            for name, clf in self.models_config:
                start_time = datetime.now()
                clf.fit(X_train_scaled, self.y_train)
                y_pred = clf.predict(X_test_scaled)

                if self.task_type == "classification":
                    n_classes = len(np.unique(self.y_train))
                    avg = 'weighted' if n_classes > 2 else 'binary'
                    acc = accuracy_score(self.y_test, y_pred)
                    prec = precision_score(self.y_test, y_pred, average=avg, zero_division=0)
                    rec = recall_score(self.y_test, y_pred, average=avg, zero_division=0)
                    f1 = f1_score(self.y_test, y_pred, average=avg, zero_division=0)
                    try:
                        if hasattr(clf, "predict_proba"):
                            probas = clf.predict_proba(X_test_scaled)
                            auc = roc_auc_score(self.y_test, probas, multi_class='ovr', average='weighted') if probas.shape[1] > 2 else roc_auc_score(self.y_test, probas[:, 1])
                        else:
                            auc = "Недоступно"
                    except:
                        auc = "Ошибка"
                    metrics = {
                        "Точность": f"{acc:.4f}",
                        "Precision": f"{prec:.4f}",
                        "Recall": f"{rec:.4f}",
                        "F1-Score": f"{f1:.4f}",
                        "ROC-AUC": f"{auc:.4f}" if isinstance(auc, float) else auc
                    }
                else:
                    r2 = r2_score(self.y_test, y_pred)
                    mse = mean_squared_error(self.y_test, y_pred)
                    mae = mean_absolute_error(self.y_test, y_pred)
                    metrics = {
                        "R²": f"{r2:.4f}",
                        "MSE": f"{mse:.4f}",
                        "MAE": f"{mae:.4f}"
                    }

                elapsed = (datetime.now() - start_time).total_seconds()
                total_time += elapsed
                results.append((name, metrics))

            self.finished_signal.emit(results, f"{total_time:.4f}")

        except Exception as e:
            self.error_signal.emit(str(e))


class ModelEvaluator:
    def __init__(self, parent, checkboxes, labels_and_lines, results_layout, task_type="classification"):
        self.parent = parent
        self.checkboxes = checkboxes
        self.labels_and_lines = labels_and_lines
        self.results_layout = results_layout
        self.task_type = task_type
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_col = None
        self.df = None
        self.thread = None
        self.splash = None

    def update_dataframe(self, df, target_col):
        """Сохраняем датасет и целевую переменную"""
        if df is None or target_col not in df.columns:
            return
        self.df = df.copy()
        self.target_col = target_col  # ✅ Сохраняем
        self.X_train = self.X_test = self.y_train = self.y_test = None
        print(f"[DEBUG] Датасет и целевая переменная сохранены: {target_col}")  # 🔹 Отладка

    def set_split_data(self, X_train, X_test, y_train, y_test, target_col):
        """При загрузке train/test отдельно"""
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.target_col = target_col
        self.df = None
        print(f"[DEBUG] Загружены X_train, X_test и целевая переменная: {target_col}")  # 🔹

    def evaluate_models(self):
        X_train, X_test, y_train, y_test = None, None, None, None

        # Сценарий 1: уже есть train/test (два файла)
        if self.X_train is not None and self.y_train is not None:
            X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
            print(f"[DEBUG] Используем уже разделённые данные")  # 🔹
        else:
            # Сценарий 2: один датасет
            if self.df is None:
                QMessageBox.critical(self.parent, "Ошибка", "Датасет не загружен!")
                print("[ERROR] self.df is None")  # 🔹
                return

            if not self.target_col:
                QMessageBox.critical(self.parent, "Ошибка", "Не выбрана целевая переменная!")
                print(f"[ERROR] self.target_col = {self.target_col}")  # 🔹
                return

            if self.target_col not in self.df.columns:
                QMessageBox.critical(self.parent, "Ошибка", f"Целевая переменная '{self.target_col}' не найдена в датасете!")
                print(f"[ERROR] Столбец '{self.target_col}' отсутствует в данных")  # 🔹
                return

            print(f"[DEBUG] Найдена целевая переменная: {self.target_col}")  # 🔹

            df_local = self.df.copy()

            # 🔹 Кодируем ТОЛЬКО для классификации
            if self.task_type == "classification":
                if df_local[self.target_col].dtype == 'object' or df_local[self.target_col].nunique() < 10:
                    le = LabelEncoder()
                    df_local[self.target_col] = le.fit_transform(df_local[self.target_col])

            # Выбираем признаки
            X = df_local.drop(columns=[self.target_col]).select_dtypes(include=['number'])
            y = df_local[self.target_col]

            if X.empty:
                QMessageBox.critical(self.parent, "Ошибка", "Нет числовых признаков для обучения.")
                return

            # Парсим параметры
            try:
                test_size = float(self.get_param_value("Test Size", "0.2"))
                if not (0 < test_size < 1):
                    test_size = 0.2
            except:
                test_size = 0.2

            try:
                random_state = int(self.get_param_value("Random State", "42"))
            except:
                random_state = 42

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            print(f"[DEBUG] Данные разделены: {len(X_train)} train, {len(X_test)} test")  # 🔹

        # Сборка моделей
        models_config = []
        for cb in self.checkboxes:
            if not cb.isChecked():
                continue
            name = cb.text()
            params = self.labels_and_lines.get(name, {})
            try:
                if 'Random Forest Classification' in name:
                    n_estimators = self.safe_int(params, 'Кол-во деревьев', 100)
                    max_depth = self.safe_int_or_none(params, 'Max Depth', None)
                    min_samples_split = self.safe_int(params, 'Min Samples Split', 2)
                    random_state = self.safe_int(params, 'Random State', 42)
                    clf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    )
                elif 'Gradient Boosting Classification' in name:
                    n_estimators = self.safe_int(params, 'Кол-во деревьев', 100)
                    learning_rate = self.safe_float(params, 'Learning Rate', 0.1)
                    max_depth = self.safe_int_or_none(params, 'Max Depth', 3)
                    random_state = self.safe_int(params, 'Random State', 42)
                    clf = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                elif 'Logistic Regression Classification' in name:
                    C = self.safe_float(params, 'C', 1.0)
                    max_iter = self.safe_int(params, 'Max Iterations', 100)
                    penalty = params['Penalty'].text().strip() if 'Penalty' in params else 'l2'
                    if penalty not in ['l1', 'l2', 'elasticnet', 'none']:
                        penalty = 'l2'
                    solver = 'saga' if penalty == 'elasticnet' else 'liblinear'
                    clf = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, random_state=42)
                elif 'Random Forest Regression' in name:
                    n_estimators = self.safe_int(params, 'Кол-во деревьев', 100)
                    max_depth = self.safe_int_or_none(params, 'Max Depth', None)
                    min_samples_split = self.safe_int(params, 'Min Samples Split', 2)
                    random_state = self.safe_int(params, 'Random State', 42)
                    clf = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    )
                elif 'Gradient Boosting Regression' in name:
                    n_estimators = self.safe_int(params, 'Кол-во деревьев', 100)
                    learning_rate = self.safe_float(params, 'Learning Rate', 0.1)
                    max_depth = self.safe_int_or_none(params, 'Max Depth', 3)
                    random_state = self.safe_int(params, 'Random State', 42)
                    clf = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                else:
                    continue
                models_config.append((name, clf))
            except Exception as e:
                print(f"[ERROR] Ошибка создания модели {name}: {e}")  # 🔹
                continue

        if not models_config:
            QMessageBox.warning(self.parent, "Предупреждение", "Ни одна модель не была корректно настроена.")
            return

        self.splash = LoadingScreen()
        self.splash.show()
        self.thread = EvaluationThread(self.parent, models_config, X_train, X_test, y_train, y_test, self.task_type)
        self.thread.finished_signal.connect(self.on_evaluation_finished)
        self.thread.error_signal.connect(self.on_evaluation_error)
        self.thread.start()

    def get_param_value(self, param_name, default="0"):
        for lines in self.labels_and_lines.values():
            if param_name in lines:
                val = lines[param_name].text().strip()
                if val:
                    return val
        return default

    def safe_int(self, params, key, default):
        try:
            val = params[key].text().strip()
            return int(val) if val else default
        except:
            return default

    def safe_float(self, params, key, default):
        try:
            val = params[key].text().strip()
            return float(val) if val else default
        except:
            return default

    def safe_int_or_none(self, params, key, default):
        try:
            val = params[key].text().strip()
            if not val or val.lower() in ('none', 'null', ''):
                return None
            return int(val)
        except:
            return default

    def on_evaluation_finished(self, results, time):
        if self.splash:
            self.splash.close()
        print(f"[DEBUG] Оценка завершена: {len(results)} моделей, время: {time} сек")  # 🔹
        self.add_result_column(results, time)

    def add_result_column(self, results, time):
        col_widget = QWidget()
        col_layout = QVBoxLayout()
        col_layout.setSpacing(6)

        for name, metrics in results:
            title = QLabel(f"<b>{name}</b>")
            title.setWordWrap(True)
            col_layout.addWidget(title)

            for metric_name, value in metrics.items():
                row = QHBoxLayout()
                label = QLabel(f"{metric_name} = {value}")
                btn = QPushButton("?")
                btn.setFixedSize(24, 24)
                btn.clicked.connect(lambda _, m=metric_name: self.show_help(m))
                row.addWidget(label)
                row.addWidget(btn)
                row.addStretch()
                col_layout.addLayout(row)

        time_label = QLabel(f"<b>Время: {time} сек</b>")
        time_label.setWordWrap(True)
        col_layout.addWidget(time_label)
        col_layout.addStretch()

        col_widget.setLayout(col_layout)
        self.results_layout.addWidget(col_widget)

    def show_help(self, metric_name):
        descriptions = {
            "Точность": "Доля правильных предсказаний среди всех предсказаний.",
            "Precision": "Точность положительного класса: TP / (TP + FP).",
            "Recall": "Полнота: TP / (TP + FN).",
            "F1-Score": "Гармоническое среднее Precision и Recall.",
            "ROC-AUC": "Площадь под ROC-кривой. Чем ближе к 1 — тем лучше.",
            "R²": "Коэффициент детерминации. Показывает, насколько модель объясняет дисперсию.",
            "MSE": "Средний квадрат ошибки. Чувствителен к выбросам.",
            "MAE": "Средняя абсолютная ошибка. Более устойчива к выбросам."
        }
        QMessageBox.information(self.parent, f"Справка: {metric_name}", descriptions.get(metric_name, "Нет описания."))

    def on_evaluation_error(self, error_msg):
        if self.splash:
            self.splash.close()
        QMessageBox.critical(self.parent, "Ошибка", f"Произошла ошибка:\n{error_msg}")
