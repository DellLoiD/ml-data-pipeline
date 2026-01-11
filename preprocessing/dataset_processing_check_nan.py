# preprocessing/dataset_processing_check_nan.py
import pandas as pd
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QComboBox, QGroupBox, QDialog, QDialogButtonBox, QGridLayout
)
from PySide6.QtCore import Qt
from preprocessing.repair_nan_methods.mice_method import impute_mice
# Импорт логики восстановления
from .dataset_processing_check_nan_logic import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_interpolate,
    impute_knn,
    impute_hot_deck,
    impute_em
)
# Импорт нового трекера
from utils.meta_tracker import MetaTracker

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
        self.resize(400, 300)

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
        self.df_original = None
        self.selected_file_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # === Кнопка выбора датасета ===
        self.btn_select_dataset = QPushButton('📁 Выбрать датасет')
        self.btn_select_dataset.clicked.connect(self.select_raw_dataset)
        main_layout.addWidget(self.btn_select_dataset)

        # === Всего строк и колонок — в одну строку ===
        stats_layout = QHBoxLayout()
        self.label_total_rows = QLabel("Всего строк: —")
        self.label_total_rows.setStyleSheet("font-weight: bold; color: #0066cc;")
        self.label_total_cols = QLabel("Всего колонок: —")
        self.label_total_cols.setStyleSheet("font-weight: bold; color: #0066cc;")
        stats_layout.addWidget(self.label_total_rows)
        stats_layout.addWidget(self.label_total_cols)
        stats_layout.addStretch()
        main_layout.addLayout(stats_layout)

        # === Отображение пропусков ===
        self.label_missing_info = QLabel("Пропуски не показаны.")
        self.label_missing_info.setWordWrap(True)
        self.label_missing_info.setStyleSheet("font-family: 'Courier'; font-size: 12px; background: #f5f5f5; padding: 10px; border-radius: 5px;")
        main_layout.addWidget(self.label_missing_info)

        # === Группа действий ===
        actions_group = QGroupBox("Действия")
        actions_layout = QVBoxLayout()

        # Кнопка показа пропусков
        self.btn_show_missing = QPushButton('🔍 Показать пропуски')
        self.btn_show_missing.clicked.connect(self.show_missing_values)
        actions_layout.addWidget(self.btn_show_missing)
        self.btn_show_missing.hide()

        # === Список колонок с пропусками ===
        self.combo_missing_cols = QComboBox()
        self.combo_missing_cols.setEnabled(False)
        self.combo_missing_cols.setPlaceholderText("Колонки с пропусками")
        self.combo_missing_cols.currentTextChanged.connect(self.on_column_selected)
        actions_layout.addWidget(QLabel("Выберите колонку для обработки:"))
        actions_layout.addWidget(self.combo_missing_cols)

        # Примеры значений
        self.label_example_values = QLabel("Примеры значений: —")
        self.label_example_values.setWordWrap(True)
        self.label_example_values.setStyleSheet("font-style: italic; color: #555;")
        actions_layout.addWidget(self.label_example_values)

        # Удаление строк с NaN
        self.btn_drop_col_na = QPushButton("🗑️ Удалить строки с NaN в колонке")
        self.btn_drop_col_na.clicked.connect(self.drop_na_in_column)
        self.btn_drop_col_na.setEnabled(False)
        actions_layout.addWidget(self.btn_drop_col_na)

        # === Методы восстановления — сетка 2×4 ===
        impute_label = QLabel("Методы восстановления:")
        impute_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        actions_layout.addWidget(impute_label)

        grid_layout = QGridLayout()
        methods = [
            ("Среднее", "mean"),
            ("Медиана", "median"),
            ("Мода", "mode"),
            ("Интерполяция", "interpolate"),
            ("KNN-Imputer", "knn"),
            ("MICE", "mice"),
            ("Hot Deck", "hot_deck"),
            ("EM", "em"),
        ]
        for i, (label, key) in enumerate(methods):
            row = i // 2
            col = (i % 2) * 2
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, k=key: self.impute_column(k))
            grid_layout.addWidget(btn, row, col)

            help_btn = QPushButton("?")
            help_btn.setFixedSize(24, 24)
            help_btn.clicked.connect(lambda _, k=key: self.show_help(k))
            grid_layout.addWidget(help_btn, row, col + 1)

        actions_layout.addLayout(grid_layout)
        actions_group.setLayout(actions_layout)
        main_layout.addWidget(actions_group)

        # === Кнопка сохранения ===
        self.btn_save = QPushButton("💾 Сохранить датасет")
        self.btn_save.clicked.connect(self.save_dataset)
        self.btn_save.setEnabled(False)
        main_layout.addWidget(self.btn_save)

        self.setLayout(main_layout)
        self.setWindowTitle('Обработка пропусков')
        self.resize(600, 750)

    def show_help(self, method_key):
        """Показывает справку по методу"""
        if method_key in IMPUTATION_HELP:
            title = method_key.replace("_", " ").title()
            dialog = HelpDialog(title, IMPUTATION_HELP[method_key], self)
            dialog.exec()

    def select_raw_dataset(self):
        """Выбор датасета с загрузкой меты"""
        filename, _ = self.get_open_filename()
        if not filename:
            return

        try:
            # Загружаем мету
            self.meta_tracker.load_from_file(filename)

            # Читаем данные
            self.df = pd.read_csv(filename, comment='#', skipinitialspace=True)
            self.df_original = self.df.copy()
            basename = os.path.basename(filename)
            self.btn_select_dataset.setText(f'✅ {basename}')
            self.selected_file_path = filename

            # Обновляем интерфейс
            total_rows = len(self.df)
            total_cols = len(self.df.columns)
            self.label_total_rows.setText(f"Всего строк: {total_rows}")
            self.label_total_cols.setText(f"Всего колонок: {total_cols}")

            self.show_missing_values()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить датасет:\n{e}")

    def get_open_filename(self):
        return QFileDialog.getOpenFileName(
            self, 'Выбрать датасет', './dataset', 'CSV Files (*.csv)'
        )

    def show_missing_values(self):
        """Показывает пропуски"""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите датасет!")
            return

        missing_data = self.df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]

        self.combo_missing_cols.clear()
        if missing_cols.empty:
            self.label_missing_info.setText("✅ В датасете нет пропусков.")
            self.combo_missing_cols.addItem("Нет колонок с пропусками")
            self.combo_missing_cols.setEnabled(False)
            self.btn_drop_col_na.setEnabled(False)
            self.label_example_values.setText("Примеры значений: —")
        else:
            self.combo_missing_cols.addItems(missing_cols.index.tolist())
            self.combo_missing_cols.setEnabled(True)
            self.btn_drop_col_na.setEnabled(True)
            self.btn_save.setEnabled(True)

            result_text = "<b>Пропуски найдены в:</b><br>"
            for col, count in missing_cols.items():
                dtype = str(self.df[col].dtype)
                result_text += f'{col} <span style="color:gray;">({dtype})</span> — <span style="color:red;">{count}</span><br>'
            self.label_missing_info.setTextFormat(Qt.RichText)
            self.label_missing_info.setText(result_text)

            first_col = missing_cols.index[0]
            self.show_example_values(first_col)

    def on_column_selected(self, column):
        """При выборе колонки обновляем примеры значений"""
        if column and column != "Нет колонок с пропусками" and column in self.df.columns:
            self.show_example_values(column)

    def show_example_values(self, column):
        """Показывает до 3 уникальных непустых значений"""
        non_null = self.df[column].dropna().unique()
        examples = non_null[:3]
        if len(examples) == 0:
            self.label_example_values.setText("Примеры значений: (все значения — пропуски)")
            return

        example_strs = [str(x)[:30] for x in examples]
        joined = " • ".join(example_strs)
        self.label_example_values.setText(f"Примеры значений: {joined}")

    def drop_na_in_column(self):
        """Удаление строк с NaN"""
        col = self.combo_missing_cols.currentText()
        if not col or col not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", "Выберите корректную колонку!")
            return

        before = len(self.df)
        self.df = self.df.dropna(subset=[col])
        after = len(self.df)
        deleted = before - after

        # Добавляем в историю
        self.meta_tracker.add_change(f"удалены строки с NaN в '{col}'")
        self.btn_save.setEnabled(True)

        QMessageBox.information(self, "Готово", f"Удалено {deleted} строк. Осталось: {after}.")
        self.show_missing_values()

    def impute_column(self, method):
        """Восстановление пропусков"""
        col = self.combo_missing_cols.currentText()
        if not col or col not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", "Выберите корректную колонку!")
            return

        method_map = {
            "mean": impute_mean,
            "median": impute_median,
            "mode": impute_mode,
            "interpolate": impute_interpolate,
            "knn": impute_knn,
            "mice": impute_mice,
            "hot_deck": impute_hot_deck,
            "em": impute_em,
        }

        if method not in method_map:
            QMessageBox.critical(self, "Ошибка", f"Метод '{method}' не реализован.")
            return

        try:
            old_missing = self.df[col].isnull().sum()
            self.df, description = method_map[method](self.df.copy(), col, parent=self)
            new_missing = self.df[col].isnull().sum()

            method_name = {
                "mean": "среднего",
                "median": "медианы",
                "mode": "моды",
                "interpolate": "интерполяции",
                "knn": "KNN",
                "mice": "MICE",
                "hot_deck": "Hot Deck",
                "em": "EM"
            }.get(method, method)

            filled = old_missing - new_missing
            self.meta_tracker.add_change(f"пропуски в '{col}' заполнены методом {method_name} ({filled})")
            self.btn_save.setEnabled(True)

            QMessageBox.information(self, "Успех", f"Пропуски восстановлены:\n{description}")
            self.show_missing_values()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось восстановить:\n{e}")

    def save_dataset(self):
        """Сохранение с использованием MetaTracker"""
        if self.df is None or self.selected_file_path is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения!")
            return

        # Определяем путь: base_name_vN.csv
        base_name = os.path.splitext(os.path.basename(self.selected_file_path))[0]
        base_name = base_name.split("_v")[0] if "_v" in base_name else base_name
        save_path = os.path.join("dataset", f"{base_name}_v{self.meta_tracker.version}.csv")

        try:
            # Сохраняем через MetaTracker
            success = self.meta_tracker.save_to_file(save_path, self.df)
            if success:
                self.selected_file_path = save_path
                self.btn_save.setEnabled(False)

                # Увеличиваем версию для следующего сохранения
                self.meta_tracker.version += 1

                QMessageBox.information(
                    self, "Сохранено",
                    f"Датасет сохранён:\n{os.path.basename(save_path)}\n\n"
                    f"Версия: v{self.meta_tracker.version - 1}"
                )
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")
