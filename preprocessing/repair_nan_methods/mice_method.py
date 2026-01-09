# preprocessing/repair_nan_methods/mice_method.py

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QSpinBox, QCheckBox,
    QPushButton, QHBoxLayout, QGroupBox, QFormLayout, QMessageBox,
    QProgressBar, QTextEdit
)
from PySide6.QtCore import QThread, Signal, QObject, Qt
from PySide6.QtGui import QFont

# === Сигналы для потока ===
class MiceWorkerSignals(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(pd.DataFrame, str)
    error = Signal(str)

# === Рабочий поток для MICE ===
class MiceImputationWorker(QThread):
    def __init__(self, df: pd.DataFrame, column: str, settings: dict, parent=None):
        super().__init__(parent)
        self.df = df.copy()
        self.column = column
        self.settings = settings
        self.signals = MiceWorkerSignals()
        self._is_cancelled = False

    def run(self):
        try:
            numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
            if self.column not in numeric_df.columns:
                raise ValueError(f"Колонка '{self.column}' не является числовой.")

            missing_count = numeric_df[self.column].isnull().sum()
            if missing_count == 0:
                self.signals.finished.emit(self.df, f"Нет пропусков в '{self.column}'.")
                return

            # --- Инициализация модели ---
            estimator_name = self.settings['estimator']
            estimators = {
                'Bayesian Ridge': BayesianRidge(),
                'Random Forest': RandomForestRegressor(
                    n_estimators=10, random_state=42, max_depth=10
                ),
            }
            estimator = estimators[estimator_name]

            # --- Настройки ---
            max_iter = self.settings['max_iter']
            impute_strategy = self.settings['initial_strategy']
            clip_min = self.settings['clip_min']
            clip_max = self.settings['clip_max']
            round_decimals = self.settings['round_decimals']

            # --- Подготовка ---
            before_values = self.df[self.column].copy()

            # --- Запуск MICE ---
            imputer = IterativeImputer(
                estimator=estimator,
                max_iter=max_iter,
                initial_strategy=impute_strategy,
                random_state=42,
                skip_complete=True,
                sample_posterior=False
            )

            X_numeric = numeric_df.values.copy()
            X_imputed = X_numeric.copy()

            try:
                for i in range(max_iter):
                    if self._is_cancelled:
                        raise KeyboardInterrupt("Прервано пользователем")

                    self.signals.progress.emit(i + 1)
                    self.signals.status.emit(f"Итерация {i + 1} из {max_iter}...")

                    X_imputed = imputer.fit_transform(X_imputed)

            except KeyboardInterrupt:
                self.signals.finished.emit(None, "❌ Восстановление прервано пользователем.")
                return

            df_numeric_restored = pd.DataFrame(X_imputed, columns=numeric_df.columns, index=self.df.index)
            self.df[self.column] = df_numeric_restored[self.column]

            # --- Постобработка ---
            # Ограничение диапазона
            if clip_min is not None:
                self.df[self.column] = self.df[self.column].clip(lower=clip_min)
            if clip_max is not None:
                self.df[self.column] = self.df[self.column].clip(lower=None, upper=clip_max)

            # Округление
            if round_decimals >= 0:
                self.df[self.column] = self.df[self.column].round(round_decimals)
            elif round_decimals == -1:  # До целого
                self.df[self.column] = self.df[self.column].round().astype('Int64')

            # --- Отчёт ---
            after_values = self.df[self.column]
            filled_mask = before_values.isnull()
            filled_values = after_values[filled_mask]
            filled_sample = filled_values.head(5).tolist()
            filled_str = ", ".join([f"{x:.2f}" if isinstance(x, float) else str(x) for x in filled_sample])

            stats = {
                'min': self.df[self.column].min(),
                'max': self.df[self.column].max(),
                'mean': self.df[self.column].mean(),
                'median': self.df[self.column].median()
            }

            report = f"""
📊 **Отчёт о восстановлении пропусков — MICE**

• Колонка: **{self.column}**
• Восстановлено значений: **{len(filled_values)}**
• Метод: **{estimator_name} → MICE (max_iter={max_iter})**
• Начальная стратегия: **{impute_strategy}**
• Ограничения: clip={clip_min}..{clip_max if clip_max else '∞'}
• Округление: до {('целого' if round_decimals == -1 else f'{round_decimals} знаков')}

🔧 **Примеры вставленных значений**: {filled_str}

📈 **Статистика после восстановления**:
   Среднее: {stats['mean']:.2f}
   Медиана: {stats['median']:.2f}
   Мин/Макс: {stats['min']:.2f} / {stats['max']:.2f}

✅ Восстановление завершено.
"""
            self.signals.finished.emit(self.df, report)

        except Exception as e:
            self.signals.error.emit(str(e))


# === Окно настроек MICE ===
class MiceSettingsDialog(QDialog):
    def __init__(self, df: pd.DataFrame, column: str, parent=None):
        super().__init__(parent)
        self.df = df
        self.column = column
        self.settings = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Настройки MICE")
        self.resize(500, 600)

        layout = QVBoxLayout()

        title = QLabel("Настройки метода MICE")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # === Группа: Основные параметры ===
        params_group = QGroupBox("Основные параметры")
        form = QFormLayout()

        # Estimator
        self.combo_estimator = QComboBox()
        self.combo_estimator.addItems(["Bayesian Ridge", "Random Forest"])
        self.combo_estimator.setCurrentText("Random Forest")
        form.addRow("Модель восстановления:", self.combo_estimator)
        self.add_help(form, "Модель, которая предсказывает пропуски. Random Forest устойчив к выбросам и не даёт абсурдных значений.")

        # Max Iter
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 50)
        self.spin_iter.setValue(10)
        form.addRow("Макс. итераций:", self.spin_iter)
        self.add_help(form, "Сколько раз алгоритм пройдёт по всем колонкам. Чем больше — тем точнее, но дольше.")

        # Initial Strategy
        self.combo_init = QComboBox()
        self.combo_init.addItems(["mean", "median", "most_frequent"])
        self.combo_init.setCurrentText("median")
        form.addRow("Начальное заполнение:", self.combo_init)
        self.add_help(form, "Как заполнить пропуски перед первой итерацией. 'Медиана' — устойчива к выбросам.")

        params_group.setLayout(form)
        layout.addWidget(params_group)

        # === Группа: Постобработка ===
        post_group = QGroupBox("Постобработка")
        post_form = QFormLayout()

        self.spin_clip_min = QSpinBox()
        self.spin_clip_min.setRange(-1000000, 1000000)
        self.spin_clip_min.setValue(0)
        self.check_clip_min = QCheckBox("Ограничить минимум")
        self.check_clip_min.setChecked(True)
        post_form.addRow(self.check_clip_min, self.spin_clip_min)
        self.add_help(post_form, "Не допускать значений ниже указанного (например, площадь ≥ 0).")

        self.spin_clip_max = QSpinBox()
        self.spin_clip_max.setRange(-1000000, 1000000)
        self.spin_clip_max.setValue(1000)
        self.check_clip_max = QCheckBox("Ограничить максимум")
        self.check_clip_max.setChecked(False)
        post_form.addRow(self.check_clip_max, self.spin_clip_max)
        self.add_help(post_form, "Не допускать слишком больших значений (например, площадь < 1000 м²).")

        self.combo_round = QComboBox()
        self.combo_round.addItems(["Без округления", "0 знаков", "1 знак", "2 знака", "3 знака"])
        self.combo_round.setCurrentText("1 знак")
        post_form.addRow("Округление:", self.combo_round)
        self.add_help(post_form, "Сколько знаков после запятой оставить. Полезно для интерпретации.")

        post_group.setLayout(post_form)
        layout.addWidget(post_group)

        # === Кнопки ===
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_run = QPushButton("🚀 Запустить MICE")
        self.btn_run.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_run)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def add_help(self, layout: QFormLayout, text: str):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 11px; color: #555;")
        layout.addRow(label)

    def get_settings(self):
        return {
            'estimator': self.combo_estimator.currentText(),
            'max_iter': self.spin_iter.value(),
            'initial_strategy': self.combo_init.currentText(),
            'clip_min': self.spin_clip_min.value() if self.check_clip_min.isChecked() else None,
            'clip_max': self.spin_clip_max.value() if self.check_clip_max.isChecked() else None,
            'round_decimals': {
                "Без округления": -2,
                "0 знаков": -1,
                "1 знак": 1,
                "2 знака": 2,
                "3 знака": 3
            }[self.combo_round.currentText()]
        }


# === Основная функция: запуск с настройками ===
def impute_mice(df: pd.DataFrame, column: str, parent=None) -> tuple[pd.DataFrame, str]:
    """
    Запускает MICE с настройками.
    Показывает окно настроек → прогресс → отчёт.
    Восстановление сохраняется даже после закрытия отчёта.
    """
    # Показываем окно настроек
    settings_dialog = MiceSettingsDialog(df, column, parent)
    if settings_dialog.exec() != QDialog.Accepted:
        return df, "MICE отменён пользователем."

    settings = settings_dialog.get_settings()

    # Результат будет сохранён здесь
    result_df = None
    final_message = ""

    # Показываем прогресс
    progress_dialog = QDialog(parent)
    progress_dialog.setWindowTitle("MICE — Восстановление пропусков")
    progress_dialog.resize(400, 180)
    progress_dialog.setModal(True)

    layout = QVBoxLayout()

    label = QLabel(f"Восстановление пропусков в:\n<b>{column}</b>")
    label.setWordWrap(True)
    layout.addWidget(label)

    progress = QProgressBar()
    progress.setRange(0, settings['max_iter'])
    layout.addWidget(progress)

    status = QLabel("Инициализация...")
    layout.addWidget(status)

    btn_layout = QHBoxLayout()
    cancel_btn = QPushButton("Прервать")
    btn_layout.addStretch()
    btn_layout.addWidget(cancel_btn)
    layout.addLayout(btn_layout)

    progress_dialog.setLayout(layout)
    progress_dialog.show()

    # Запускаем в потоке
    worker = MiceImputationWorker(df, column, settings, parent=parent)

    def on_progress(value):
        progress.setValue(value)

    def on_status(text):
        status.setText(text)

    def on_finished(res_df, message):
        nonlocal result_df, final_message
        result_df = res_df
        final_message = message
        progress_dialog.accept()

        # Показываем отчёт — НЕ модально, или модально, но не блокируя возврат
        report_dialog = QDialog(parent)
        report_dialog.setWindowTitle("📊 Отчёт о восстановлении")
        report_dialog.setAttribute(Qt.WA_DeleteOnClose)  # Удалится при закрытии
        report_layout = QVBoxLayout()
        report_text = QTextEdit()
        report_text.setMarkdown(message)
        report_text.setReadOnly(True)
        report_layout.addWidget(report_text)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(report_dialog.accept)
        report_layout.addWidget(close_btn)
        report_dialog.setLayout(report_layout)
        report_dialog.resize(600, 400)
        report_dialog.show()  # ⚠️ Не exec(), а show() — не блокирует поток
        # Или можно оставить exec(), но результат всё равно будет возвращён

    def on_error(error_msg):
        nonlocal final_message
        final_message = f"Ошибка: {error_msg}"
        QMessageBox.critical(parent, "Ошибка", final_message)
        progress_dialog.reject()

    cancel_btn.clicked.connect(lambda: setattr(worker, '_is_cancelled', True))

    worker.signals.progress.connect(on_progress)
    worker.signals.status.connect(on_status)
    worker.signals.finished.connect(on_finished)
    worker.signals.error.connect(on_error)

    worker.start()
    progress_dialog.exec()  # Ждём завершения прогресса

    # 🔽 Возвращаем результат, даже если отчёт уже закрыт
    if result_df is not None:
        return result_df, "MICE (успешно восстановлено)"
    else:
        return df, final_message or "MICE: восстановление не выполнено"

