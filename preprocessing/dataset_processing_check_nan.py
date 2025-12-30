# preprocessing/dataset_processing_check_nan.py
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QComboBox, QGroupBox,
    QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import os

# 📝 Справки по методам
IMPUTATION_HELP = {
    "mean": (
        "Среднее значение (Mean Imputation)\n\n"
        "Замена пропущенного значения средним значением остальных наблюдений признака.\n\n"
        "x_miss = Σx_i / n\n\n"
        "Где n — количество ненулевых значений признака.\n\n"
        "✔ Подходит для нормально распределённых данных\n"
        "✖ Чувствителен к выбросам"
    ),
    "median": (
        "Медианное значение (Median Imputation)\n\n"
        "Аналогично среднему, но используется медиана.\n\n"
        "✔ Устойчив к выбросам\n"
        "✖ Может сместить распределение"
    ),
    "mode": (
        "Модальное значение (Mode Imputation)\n\n"
        "Заменяет пропущенные значения наиболее частым значением в колонке.\n\n"
        "✔ Подходит для категориальных признаков\n"
        "✖ Может исказить баланс классов"
    ),
    "interpolate": (
        "Интерполяция\n\n"
        "Восстанавливает пропущенные значения на основе соседних значений.\n\n"
        "Чаще используется для временных рядов.\n\n"
        "Поддерживает: линейную, квадратичную, сплайн-интерполяцию.\n\n"
        "✔ Сохраняет тенденции\n"
        "✖ Не подходит для категорий"
    ),
    "knn": (
        "KNN-Imputer (K-Nearest Neighbors)\n\n"
        "Находит похожие строки и заполняет пропуски на основе значений ближайших соседей.\n\n"
        "✔ Учитывает связи между признаками\n"
        "✖ Требует нормализации и много памяти"
    ),
    "mice": (
        "MICE (Multiple Imputation by Chained Equations)\n\n"
        "Итеративный метод множественного восстановления, учитывающий неопределённость.\n\n"
        "Каждый пропуск восстанавливается несколько раз, затем усредняется.\n\n"
        "✔ Очень точный и статистически обоснованный\n"
        "✖ Медленный, сложен в настройке"
    ),
    "hot_deck": (
        "Hot Deck Imputation\n\n"
        "Находит похожие объекты (по другим признакам) и копирует значение из них.\n\n"
        "✔ Сохраняет реалистичные значения\n"
        "✖ Трудно масштабируется"
    ),
    "em": (
        "Multivariate Imputation using Expectation Maximization (EM)\n\n"
        "Байесовский подход: оценивает совместное распределение признаков и восстанавливает пропуски.\n\n"
        "✔ Учитывает корреляции\n"
        "✖ Сложный, требует нормальности распределения"
    )
}


class HelpDialog(QDialog):
    """Диалог со справкой по методу"""
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 300)

        layout = QVBoxLayout(self)

        text_edit = QLabel(text)
        text_edit.setWordWrap(True)
        layout.addWidget(text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class MissingValuesDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.df = None
        self.df_original = None  # Сохраняем оригинал
        self.selected_file_path = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # === Заголовок ===
        title = QLabel("Проверка и обработка пропусков")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # === Кнопка выбора датасета ===
        self.btn_select_dataset = QPushButton('📁 Выбрать датасет')
        self.btn_select_dataset.clicked.connect(self.select_raw_dataset)
        main_layout.addWidget(self.btn_select_dataset)

        # === Общее количество строк и колонок ===
        self.label_total_rows = QLabel("Всего строк: —")
        self.label_total_rows.setStyleSheet("font-weight: bold; color: #0066cc;")
        main_layout.addWidget(self.label_total_rows)

        self.label_total_cols = QLabel("Всего колонок: —")
        self.label_total_cols.setStyleSheet("font-weight: bold; color: #0066cc;")
        main_layout.addWidget(self.label_total_cols)

        # === Кнопка показа пропусков ===
        self.btn_show_missing = QPushButton('🔍 Показать пропуски')
        self.btn_show_missing.clicked.connect(self.show_missing_values)
        main_layout.addWidget(self.btn_show_missing)

        # === Отображение списка пропусков ===
        self.label_missing_info = QLabel("Пропуски не показаны. Нажмите 'Показать пропуски'.")
        self.label_missing_info.setWordWrap(True)
        self.label_missing_info.setStyleSheet("font-family: 'Courier'; font-size: 12px; background: #f5f5f5; padding: 10px; border-radius: 5px;")
        main_layout.addWidget(self.label_missing_info)

        # === Список колонок с пропусками ===
        self.combo_missing_cols = QComboBox()
        self.combo_missing_cols.setEnabled(False)
        self.combo_missing_cols.setPlaceholderText("Колонки с пропусками")
        main_layout.addWidget(QLabel("Выберите колонку для обработки:"))
        main_layout.addWidget(self.combo_missing_cols)

        # === Группа действий с пропусками ===
        actions_group = QGroupBox("Действия")
        actions_layout = QVBoxLayout()

        # Кнопка удаления пропусков в выбранной колонке
        self.btn_drop_col_na = QPushButton("🗑️ Удалить строки с NaN в колонке")
        self.btn_drop_col_na.clicked.connect(self.drop_na_in_column)
        self.btn_drop_col_na.setEnabled(False)
        actions_layout.addWidget(self.btn_drop_col_na)

        # === Методы восстановления ===
        impute_label = QLabel("Выберите метод восстановления:")
        impute_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        actions_layout.addWidget(impute_label)

        # Простые методы
        self.add_imputation_button(actions_layout, "Среднее", "mean")
        self.add_imputation_button(actions_layout, "Медиана", "median")
        self.add_imputation_button(actions_layout, "Мода", "mode")
        self.add_imputation_button(actions_layout, "Интерполяция", "interpolate")
        self.add_imputation_button(actions_layout, "KNN-Imputer", "knn")
        self.add_imputation_button(actions_layout, "MICE", "mice")

        actions_group.setLayout(actions_layout)
        main_layout.addWidget(actions_group)

        # === Кнопки: Сохранить / Закрыть ===
        buttons_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 Сохранить датасет")
        self.btn_save.clicked.connect(self.save_dataset)
        self.btn_save.setEnabled(False)
        buttons_layout.addWidget(self.btn_save)

        close_button = QPushButton("❌ Закрыть")
        close_button.clicked.connect(self.close)
        buttons_layout.addWidget(close_button)

        main_layout.addLayout(buttons_layout)

        # === Настройки ===
        self.setLayout(main_layout)
        self.setWindowTitle('Обработка пропусков')
        self.resize(600, 700)

    def add_imputation_button(self, layout, label, method_key):
        """Добавляет кнопку метода + кнопку '?'"""
        row_layout = QHBoxLayout()
        btn = QPushButton(label)
        btn.clicked.connect(lambda: self.impute_column(method_key))
        row_layout.addWidget(btn)

        help_btn = QPushButton("?")
        help_btn.setFixedSize(24, 24)
        help_btn.clicked.connect(lambda: self.show_help(method_key))
        row_layout.addWidget(help_btn)

        row_layout.addStretch()
        layout.addLayout(row_layout)

    def show_help(self, method_key):
        """Показывает справку по методу"""
        if method_key in IMPUTATION_HELP:
            title = method_key.replace("_", " ").title()
            dialog = HelpDialog(title, IMPUTATION_HELP[method_key], self)
            dialog.exec()

    def select_raw_dataset(self):
        """Выбор датасета"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Выбрать датасет', './dataset', 'CSV Files (*.csv)'
        )
        if not filename:
            return

        try:
            self.df = pd.read_csv(filename)
            self.df_original = self.df.copy()
            basename = os.path.basename(filename)
            self.btn_select_dataset.setText(f'✅ {basename}')
            self.selected_file_path = filename

            total_rows = len(self.df)
            total_cols = len(self.df.columns)

            self.label_total_rows.setText(f"Всего строк: {total_rows}")
            self.label_total_cols.setText(f"Всего колонок: {total_cols}")

            self.combo_missing_cols.clear()
            self.combo_missing_cols.setEnabled(False)
            self.btn_drop_col_na.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.label_missing_info.setText("Пропуски не показаны. Нажмите 'Показать пропуски'.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить датасет:\n{e}")

    def show_missing_values(self):
        """Показывает пропуски в формате: колонка (тип) — количество"""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите датасет!")
            return

        missing_data = self.df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]

        if missing_cols.empty:
            self.label_missing_info.setText("✅ В датасете нет пропусков.")
            self.combo_missing_cols.clear()
            self.combo_missing_cols.addItem("Нет колонок с пропусками")
            self.combo_missing_cols.setEnabled(False)
            self.btn_drop_col_na.setEnabled(False)
            return

        # Обновляем комбобокс
        self.combo_missing_cols.clear()
        self.combo_missing_cols.addItems(missing_cols.index.tolist())
        self.combo_missing_cols.setEnabled(True)
        self.btn_drop_col_na.setEnabled(True)
        self.btn_save.setEnabled(True)

        # Формируем текст: колонка (тип) — количество
        result_text = "<b>Пропуски найдены в:</b><br>"
        for col, count in missing_cols.items():
            dtype = str(self.df[col].dtype)
            result_text += f'{col} <span style="color:gray;">({dtype})</span> — <span style="color:red;">{count}</span><br>'
        
        self.label_missing_info.setTextFormat(Qt.RichText)
        self.label_missing_info.setText(result_text)

        # Обновляем строки/колонки (на всякий случай)
        self.label_total_rows.setText(f"Всего строк: {len(self.df)}")
        self.label_total_cols.setText(f"Всего колонок: {len(self.df.columns)}")

    def drop_na_in_column(self):
        """Удаляет строки с NaN в выбранной колонке"""
        col = self.combo_missing_cols.currentText()
        if not col or col not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", "Выберите корректную колонку!")
            return

        before = len(self.df)
        self.df = self.df.dropna(subset=[col])
        after = len(self.df)
        deleted = before - after

        QMessageBox.information(
            self, "Готово",
            f"Удалено {deleted} строк с NaN в колонке '{col}'.\n"
            f"Теперь в датасете {after} строк."
        )
        self.show_missing_values()  # Обновляем список

    def impute_column(self, method):
        """Восстановление пропусков в выбранной колонке"""
        col = self.combo_missing_cols.currentText()
        if not col or col not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", "Выберите колонку!")
            return

        series = self.df[col]

        try:
            if method == "mean":
                if series.dtype not in ['int64', 'float64']:
                    raise ValueError("Среднее применимо только к числовым колонкам")
                value = series.mean()
                self.df[col] = series.fillna(value)
                self.log_action(f"Заполнено средним: {value:.4f}")

            elif method == "median":
                if series.dtype not in ['int64', 'float64']:
                    raise ValueError("Медиана применима только к числовым колонкам")
                value = series.median()
                self.df[col] = series.fillna(value)
                self.log_action(f"Заполнено медианой: {value:.4f}")

            elif method == "mode":
                value = series.mode()
                if value.empty:
                    value = series.dropna().iloc[0] if not series.dropna().empty else "Unknown"
                else:
                    value = value[0]
                self.df[col] = series.fillna(value)
                self.log_action(f"Заполнено модой: {value}")

            elif method == "interpolate":
                if series.dtype not in ['int64', 'float64']:
                    raise ValueError("Интерполяция доступна только для числовых колонок")
                self.df[col] = series.interpolate(method='linear', limit_direction='both')
                self.log_action("Интерполяция (линейная)")

            elif method == "knn":
                self.show_not_implemented("KNN-Imputer требует нормализации и установки kneighbors. Доступно в расширенной версии.")
            elif method == "mice":
                self.show_not_implemented("MICE — сложный метод. Реализация в разработке.")
            elif method == "hot_deck":
                self.show_not_implemented("Hot Deck — в разработке.")
            elif method == "em":
                self.show_not_implemented("EM — требует предположений о распределении. В разработке.")

            QMessageBox.information(self, "Успех", f"Пропуски в '{col}' восстановлены методом: {method}")
            self.show_missing_values()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось восстановить:\n{e}")

    def log_action(self, message):
        print(f"[Imputation] {message}")

    def show_not_implemented(self, msg):
        QMessageBox.information(self, "В разработке", msg)

    def save_dataset(self):
        """Сохраняет обновлённый датасет"""
        if self.df is None or self.selected_file_path is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения!")
            return

        file_name = os.path.splitext(os.path.basename(self.selected_file_path))[0]
        suggested_name = f"dataset/{file_name}_cleaned.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить датасет", suggested_name, "CSV Files (*.csv)"
        )
        if not save_path:
            return

        try:
            self.df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Сохранено", f"Датасет сохранён:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")
