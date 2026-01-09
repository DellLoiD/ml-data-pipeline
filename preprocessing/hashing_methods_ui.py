# preprocessing/hashing_methods_ui.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QComboBox, QGroupBox, QSpinBox,
    QDialog, QScrollArea, QTextEdit, QFrame, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import os
import pandas as pd
import hashlib
import random


class HashingMethodsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.selected_column = None
        self.unique_count = 0
        self._meta_line = "# META:"  # Для хранения строки метаданных
        self._has_changes = False  # Контроль кнопки "Сохранить"
        self._last_loaded_path = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Хеширование строковых признаков")
        self.resize(500, 500)

        layout = QVBoxLayout()

        # === Кнопка загрузки датасета ===
        self.load_btn = QPushButton("📂 Выбрать датасет (CSV)")
        self.load_btn.clicked.connect(self.load_dataset)
        layout.addWidget(self.load_btn)

        # === Выбор колонки ===
        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel("Выберите строковую колонку:"))
        self.column_combo = QComboBox()
        self.column_combo.setEnabled(False)
        self.column_combo.currentTextChanged.connect(self.on_column_selected)
        col_layout.addWidget(self.column_combo)
        layout.addLayout(col_layout)

        # === Информация о количестве уникальных значений ===
        self.info_label = QLabel("Количество уникальных значений: —")
        self.info_label.setStyleSheet("font-weight: bold; margin: 10px 0;")
        layout.addWidget(self.info_label)

        # === Отображение примеров значений ===
        self.sample_label = QLabel("Примеры значений: —")
        self.sample_label.setStyleSheet("color: gray; font-size: 12px; font-style: italic;")
        self.sample_label.setWordWrap(True)
        layout.addWidget(self.sample_label)

        # === Разделитель ===
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # === Группа методов хеширования ===
        hash_group = QGroupBox("Методы хеширования")
        hash_layout = QVBoxLayout()

        # Список методов
        self.methods = [
            {
                "name": "1. Простое хеширование (hash())",
                "desc": "Использует встроенную функцию hash() для преобразования строки в целое число. "
                        "Быстро, но результат зависит от сессии Python (не стабилен между запусками). "
                        "Подходит для временных преобразований.",
                "min_size": 1,
                "stable": False
            },
            {
                "name": "2. Feature Hashing («хэш-трик»)",
                "desc": "Отображает большое количество признаков в фиксированное пространство (n_features) "
                        "с помощью одной или нескольких хеш-функций. Уменьшает размерность, но возможны коллизии. "
                        "Используется в моделях с большими категориальными признаками (например, текст, IP).",
                "min_size": 2,
                "stable": True
            },
            {
                "name": "3. One-Hot + Хеширование",
                "desc": "Создаёт one-hot вектор для каждой категории, затем применяет хеширование, "
                        "чтобы сжать его в меньшее пространство. Позволяет уменьшить размерность "
                        "при сохранении информации о разреженности.",
                "min_size": 2,
                "stable": True
            },
            {
                "name": "4. Embedding + Хеширование",
                "desc": "Каждая строка сначала преобразуется в эмбеддинг (например, через усреднение букв или "
                        "предобученную модель), затем применяется хеширование. Полезно, когда важна семантика строк.",
                "min_size": 1,
                "stable": True
            },
            {
                "name": "5. Universal Hash Functions",
                "desc": "Использует случайно выбранную хеш-функцию из семейства, чтобы минимизировать коллизии. "
                        "Подходит для строгих требований к равномерности распределения хешей.",
                "min_size": 2,
                "stable": True
            },
            {
                "name": "6. Count Min Sketch",
                "desc": "Оценивает частоту элементов с помощью нескольких хеш-функций и двумерной таблицы. "
                        "Позволяет работать с потоками данных и экономить память. Результат — приближённый.",
                "min_size": 2,
                "stable": True
            }
        ]

        # Добавляем кнопки и кнопки помощи
        for method in self.methods:
            row = QHBoxLayout()

            btn = QPushButton(method["name"])
            btn.clicked.connect(lambda _, m=method: self.run_hashing_method(m))
            row.addWidget(btn, 4)

            help_btn = QPushButton("?")
            help_btn.setFixedSize(25, 25)
            help_btn.clicked.connect(lambda _, d=method["desc"]: self.show_help(d))
            row.addWidget(help_btn)

            hash_layout.addLayout(row)

        hash_group.setLayout(hash_layout)
        layout.addWidget(hash_group)

        # === Кнопка сохранения ===
        self.save_btn = QPushButton("💾 Сохранить изменённый датасет")
        self.save_btn.clicked.connect(self.save_dataset)
        self.save_btn.setEnabled(False)  # Только после изменений
        layout.addWidget(self.save_btn)

        # === Кнопка закрытия УДАЛЕНА ===
        # Больше не добавляем "Закрыть"

        self.setLayout(layout)
        self.reset_ui()

    def reset_ui(self):
        """Сброс всех полей"""
        self.df = None
        self.selected_column = None
        self.unique_count = 0
        self._has_changes = False
        self.column_combo.clear()
        self.column_combo.setEnabled(False)
        self.info_label.setText("Количество уникальных значений: —")
        self.sample_label.setText("Примеры значений: —")
        self.save_btn.setEnabled(False)

    def load_dataset(self):
        """Загрузка датасета с обработкой #META"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV файл", "./dataset", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            # Читаем #META строку
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            if first_line.startswith("# META:"):
                self._meta_line = first_line
            else:
                self._meta_line = "# META:"

            # Читаем данные, игнорируя комментарии
            self.df = pd.read_csv(file_path, comment='#', dtype=str).fillna("")
            self._last_loaded_path = file_path

            # Определяем строковые колонки (не числовой тип)
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            string_cols = []
            for col in categorical_cols:
                sample = self.df[col].dropna().astype(str).head(100)
                if not pd.to_numeric(sample, errors='coerce').notna().all():
                    string_cols.append(col)

            if not string_cols:
                QMessageBox.warning(self, "Нет данных", "В датасете нет строковых колонок для хеширования.")
                return

            self.column_combo.clear()
            self.column_combo.addItems(string_cols)
            self.column_combo.setEnabled(True)
            self.on_column_selected(string_cols[0])

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")
            self.reset_ui()

    def on_column_selected(self, column):
        """Обновление информации при выборе колонки"""
        if not column or self.df is None or column not in self.df.columns:
            return

        self.selected_column = column
        unique_vals = self.df[column].dropna().unique()
        self.unique_count = len(unique_vals)
        self.info_label.setText(f"🔢 Уникальных значений: <b>{self.unique_count}</b>")

        # --- ПОКАЗАТЬ 3 СЛУЧАЙНЫХ УНИКАЛЬНЫХ ЗНАЧЕНИЯ ---
        if len(unique_vals) == 0:
            self.sample_label.setText("Примеры значений: —")
        else:
            # Выбираем до 3 случайных значений
            sample_values = pd.Series(unique_vals).sample(n=min(3, len(unique_vals)), random_state=None).tolist()
            # Форматируем как строки в кавычках
            formatted = ", ".join(f"'{str(v)}'" for v in sample_values)
            self.sample_label.setText(f"Примеры значений: {formatted}")

    def show_help(self, description):
        """Справка по методу"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Справка по методу")
        dialog.resize(600, 300)

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(description)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec()

    def run_hashing_method(self, method):
        """Применение выбранного метода хеширования"""
        if not self.selected_column:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите колонку!")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Параметры: {method['name']}")
        layout = QVBoxLayout()

        default_size = max(method["min_size"], self.unique_count * 2)

        layout.addWidget(QLabel(f"Метод: <b>{method['name']}</b>"))
        layout.addWidget(QLabel("Размер хеш-таблицы (n):"))

        size_input = QSpinBox()
        size_input.setRange(method["min_size"], 10_000_000)
        size_input.setValue(default_size)
        layout.addWidget(size_input)

        hint = QLabel(f"Рекомендуется: ≥ {method['min_size']}")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        cancel_btn = QPushButton("Отмена")
        ok_btn = QPushButton("Запустить")
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

        cancel_btn.clicked.connect(dialog.reject)
        ok_btn.clicked.connect(dialog.accept)

        dialog.setLayout(layout)

        if dialog.exec() != QDialog.Accepted:
            return

        n = size_input.value()
        if n < method["min_size"]:
            QMessageBox.warning(self, "Ошибка", f"Размер таблицы должен быть не менее {method['min_size']}.")
            return

        # === Применяем метод ===
        try:
            new_col_name = f"{self.selected_column}_hashed"

            if method["name"].startswith("1."):
                self.df[new_col_name] = self.df[self.selected_column].apply(
                    lambda x: self.simple_hash(x) % n
                )
                method_desc = "простое хеширование (hash)"
            elif method["name"].startswith("2."):
                self.df[new_col_name] = self.df[self.selected_column].apply(
                    lambda x: self.feature_hash(x, n)
                )
                method_desc = "Feature Hashing"
            elif method["name"].startswith("3."):
                value_to_idx = {val: i for i, val in enumerate(self.df[self.selected_column].unique())}
                self.df[new_col_name] = self.df[self.selected_column].map(value_to_idx).apply(
                    lambda x: self.feature_hash(str(x), n)
                )
                method_desc = "One-Hot + Хеширование"
            elif method["name"].startswith("4."):
                def simple_embedding(s):
                    return sum(ord(c) for c in s) % (2**31)
                self.df[new_col_name] = self.df[self.selected_column].apply(
                    lambda x: (simple_embedding(x) + hash(x)) % n
                )
                method_desc = "Embedding + Хеширование"
            elif method["name"].startswith("5."):
                a, b = random.randint(1, 100), random.randint(0, 100)
                self.df[new_col_name] = self.df[self.selected_column].apply(
                    lambda x: self.universal_hash(x, n, a, b)
                )
                method_desc = "Universal Hash Functions"
            elif method["name"].startswith("6."):
                counts = self.count_min_sketch(self.df[self.selected_column].tolist(), n)
                self.df[new_col_name] = self.df[self.selected_column].map(
                    lambda x: counts.get(x, 0)
                )
                method_desc = "Count Min Sketch"
            else:
                method_desc = "неизвестный метод"

            # Логируем операцию
            self._meta_line += f", хеширована колонка '{self.selected_column}' методом {method_desc} (n={n})"
            self._has_changes = True
            self.save_btn.setEnabled(True)

            QMessageBox.information(
                self, "Успех",
                f"✅ Хеширование завершено!\n"
                f"Новая колонка: '{new_col_name}'\n"
                f"Размер таблицы: {n}\n"
                f"Метод: {method['name']}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить хеширование:\n{e}")

    def save_dataset(self):
        """Сохранение датасета с обновлением #META и версионированием"""
        if self.df is None or not self._has_changes:
            return

        # Извлекаем имя и версию
        original_name = "dataset.csv"
        if self._last_loaded_path:
            original_name = os.path.basename(self._last_loaded_path)

        base_name = os.path.splitext(original_name)[0]
        if "_v" in base_name:
            try:
                version = int(base_name.split("_v")[1]) + 1
                base_name = base_name.split("_v")[0]
            except:
                version = 1
        else:
            version = 1

        save_path = os.path.join("dataset", f"{base_name}_v{version}.csv")

        # Спрашиваем, удалить ли оригинальную колонку
        dialog = QDialog(self)
        dialog.setWindowTitle("Удалить оригинальную колонку?")
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Удалить исходную колонку '{self.selected_column}' после хеширования?"))
        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        if dialog.exec() == QDialog.Accepted:
            df_to_save = self.df.drop(columns=[self.selected_column])
            action_log = f", удалена колонка '{self.selected_column}'"
        else:
            df_to_save = self.df.copy()
            action_log = f", колонка '{self.selected_column}' сохранена"

        # Обновляем мету
        self._meta_line += action_log

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(self._meta_line + "\n")
                df_to_save.to_csv(f, index=False)

            QMessageBox.information(
                self, "Сохранено",
                f"✅ Датасет сохранён:\n{save_path}\n\n"
                f"Версия: v{version}"
            )
            self.save_btn.setEnabled(False)
            self._has_changes = False
            self._last_loaded_path = save_path

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")

    # === Реализации методов хеширования ===

    def simple_hash(self, value: str) -> int:
        return hash(value) % (2**31)

    def feature_hash(self, value: str, n: int) -> int:
        return int(hashlib.md5(value.encode()).hexdigest(), 16) % n

    def universal_hash(self, value: str, n: int, a: int, b: int, p=2147483647) -> int:
        x = int(hashlib.sha256(value.encode()).hexdigest()[:15], 16)
        return ((a * x + b) % p) % n

    def count_min_sketch(self, items: list, n: int, d: int = 3) -> dict:
        tables = [[0] * n for _ in range(d)]
        hashes = [lambda x, i=i: int(hashlib.sha256(f"{i}{x}".encode()).hexdigest(), 16) % n for i in range(d)]
        counts = {}

        for item in items:
            min_count = min(tables[i][hashes[i](item)] for i in range(d))
            for i in range(d):
                tables[i][hashes[i](item)] += 1
            counts[item] = min_count + 1  # простая аппроксимация

        return counts
