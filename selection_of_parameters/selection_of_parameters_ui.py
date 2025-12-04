from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QComboBox
)
from PySide6.QtGui import QIcon
import ast
from selection_of_parameters.selection_of_parameters_logic import save_random_grid, get_random_grid


class HyperParameterOptimizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.model_type = "RandomForest"
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        self.setWindowTitle("Настройка гиперпараметров")
        self.setWindowIcon(QIcon.fromTheme("configure"))

        # === Выбор модели ===
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["RandomForest", "GradientBoosting", "LogisticRegression"])
        self.model_combo.setCurrentText(self.model_type)
        self.model_combo.currentTextChanged.connect(self.load_model_params)
        model_layout.addWidget(self.model_combo)
        main_layout.addLayout(model_layout)

        # === Поля параметров ===
        self.param_fields = {}
        self.fields_layout = QVBoxLayout()
        main_layout.addLayout(self.fields_layout)

        # === Кнопка сохранения ===
        save_button = QPushButton("Сохранить параметры")
        save_button.clicked.connect(self.on_save_clicked)
        save_button.setStyleSheet("font-size: 12px; padding: 10px;")
        main_layout.addWidget(save_button)

        self.setLayout(main_layout)
        self.resize(600, 500)

        # Загружаем параметры из random_grid
        self.load_model_params(self.model_type)

    def clear_layout(self, layout):
        """Безопасная очистка вложенной компоновки."""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    self.clear_layout(child_layout)

    def load_model_params(self, model_type):
        """Загружает и отображает параметры для выбранной модели."""
        self.model_type = model_type
        self.clear_layout(self.fields_layout)
        self.param_fields.clear()

        full_grid = get_random_grid()
        model_params = full_grid.get(model_type, {})
        param_info = self.get_param_info()

        for param_name, info in param_info.items():
            # Фильтр по модели
            if self.skip_param_for_model(model_type, param_name):
                continue

            row = QHBoxLayout()

            # === Специальная логика: n_estimators ===
            if param_name == "n_estimators":
                value = model_params.get("n_estimators", info["default"])
                min_val, max_val, step_val = self.extract_range_from_value(value)

                row.addWidget(QLabel("n_estimators:"))

                min_edit = QLineEdit(str(min_val))
                max_edit = QLineEdit(str(max_val))
                step_edit = QLineEdit(str(step_val))

                min_edit.setPlaceholderText("мин")
                max_edit.setPlaceholderText("макс")
                step_edit.setPlaceholderText("шаг")
                min_edit.setFixedWidth(60)
                max_edit.setFixedWidth(60)
                step_edit.setFixedWidth(60)

                row.addWidget(QLabel("мин:"))
                row.addWidget(min_edit)
                row.addWidget(QLabel("макс:"))
                row.addWidget(max_edit)
                row.addWidget(QLabel("шаг:"))
                row.addWidget(step_edit)

                self.param_fields["n_estimators"] = ("range", min_edit, max_edit, step_edit)
            else:
                # === Остальные параметры ===
                row.addWidget(QLabel(f"{param_name}:"))

                current_value = model_params.get(param_name, info["default"])
                widget = self.create_widget_for_param(info, current_value)
                self.param_fields[param_name] = ("single", widget)
                row.addWidget(widget)

            # === Кнопка помощи ===
            help_btn = QPushButton()
            help_btn.setFixedSize(24, 24)
            help_btn.setIcon(QIcon.fromTheme("dialog-question"))
            help_btn.clicked.connect(lambda _, tip=info["tooltip"]: self.show_help(tip))
            row.addWidget(help_btn)

            self.fields_layout.addLayout(row)

    def skip_param_for_model(self, model_type, param_name):
        """Проверяет, должен ли параметр отображаться для модели."""
        mapping = {
            "RandomForest": [
                'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                'bootstrap', 'criterion', 'class_weight', 'max_features', 'ccp_alpha', 'verbose'
            ],
            "GradientBoosting": [
                'n_estimators', 'max_depth', 'learning_rate', 'subsample',
                'criterion', 'min_samples_split', 'min_samples_leaf', 'ccp_alpha', 'verbose'
            ],
            "LogisticRegression": [
                'penalty', 'C', 'solver', 'verbose'
            ]
        }
        return param_name not in mapping.get(model_type, [])

    def extract_range_from_value(self, value):
        """Извлекает min, max, step из range, list или int."""
        if isinstance(value, range):
            return value.start, value.stop, value.step
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            sorted_vals = sorted([int(x) for x in value])
            step = sorted_vals[1] - sorted_vals[0] if len(sorted_vals) > 1 else 1
            return sorted_vals[0], sorted_vals[-1], step
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            v = int(value[0])
            return v, v, 1
        else:
            try:
                v = int(value)
                return v, v, 1
            except:
                return 10, 100, 10  # значение по умолчанию

    def create_widget_for_param(self, info, value):
        """Создаёт виджет в зависимости от типа параметра."""
        if info["type"] == "bool":
            combo = QComboBox()
            combo.addItems(["True", "False"])
            combo.setCurrentText(str(bool(value)).title())
            return combo

        elif info["type"] == "choice":
            combo = QComboBox()
            options_str = [str(x) for x in info["options"]]
            combo.addItems(options_str)
            current_str = str(value)
            if current_str in options_str:
                combo.setCurrentText(current_str)
            return combo

        elif info["type"] in ["int", "float"]:
            if isinstance(value, (list, tuple, range)):
                text_value = str(list(value))  # Принудительно в строку
            else:
                try:
                    text_value = str(int(value)) if info["type"] == "int" else f"{float(value):.3f}"
                except:
                    text_value = str(info["default"])
            edit = QLineEdit(text_value)
            placeholder = "целое или [10,20]" if info["type"] == "int" else "число или [0.1,0.2]"
            edit.setPlaceholderText(placeholder)
            return edit

        else:
            return QLineEdit(str(value))

    def get_param_info(self):
        """Информация для UI: подсказки, типы, значения по умолчанию."""
        return {
            'n_estimators': {
                "default": [10, 50, 100],
                "type": "range_int",
                "tooltip": "Задаёт диапазон количества деревьев.\n"
                           "Выбираются значения от Мин до Макс с шагом Шаг.\n"
                           "Например: Мин=10, Макс=50, Шаг=10 → [10,20,30,40,50]"
            },
            'max_depth': {
                "default": None,
                "type": "choice",
                "options": [None, 3, 5, 7, 10],
                "tooltip": "Максимальная глубина дерева. None — без ограничений."
            },
            'min_samples_split': {
                "default": 2,
                "type": "int",
                "tooltip": "Минимальное число образцов в узле для разделения."
            },
            'min_samples_leaf': {
                "default": 1,
                "type": "int",
                "tooltip": "Минимальное число образцов в листе."
            },
            'bootstrap': {
                "default": True,
                "type": "bool",
                "tooltip": "Использовать bootstrap-выборку для деревьев?"
            },
            'criterion': {
                "default": "gini",
                "type": "choice",
                "options": ["gini", "entropy"],
                "tooltip": "Метрика для разделения."
            },
            'class_weight': {
                "default": "balanced",
                "type": "choice",
                "options": ["balanced", None],
                "tooltip": "Учёт дисбаланса классов."
            },
            'max_features': {
                "default": "sqrt",
                "type": "choice",
                "options": ["sqrt", "log2", None],
                "tooltip": "Число признаков, проверяемых при разделении."
            },
            'ccp_alpha': {
                "default": 0.0,
                "type": "float",
                "tooltip": "Регуляризация — обрезка дерева."
            },
            'verbose': {
                "default": 0,
                "type": "int",
                "options": [0, 1, 2],
                "tooltip": "Уровень подробного вывода."
            },
            'learning_rate': {
                "default": 0.1,
                "type": "float",
                "tooltip": "Скорость обучения для GradientBoosting."
            },
            'subsample': {
                "default": 0.8,
                "type": "float",
                "tooltip": "Доля данных для каждого дерева."
            },
            'penalty': {
                "default": "l2",
                "type": "choice",
                "options": ["l1", "l2"],
                "tooltip": "Тип регуляризации."
            },
            'C': {
                "default": 1.0,
                "type": "float",
                "tooltip": "Инверсия силы регуляризации."
            },
            'solver': {
                "default": "liblinear",
                "type": "choice",
                "options": ["liblinear", "saga"],
                "tooltip": "Метод оптимизации."
            },
        }

    def show_help(self, message):
        """Отображает всплывающую подсказку."""
        QMessageBox.information(self, "Справка", message)

    def update_model_grid(self):
        """Собирает значения из полей и формирует корректную сетку параметров."""
        full_grid = get_random_grid()
        model_params = {}
        param_info = self.get_param_info()

        for param, field_data in self.param_fields.items():
            value = None
            info = param_info.get(param)

            try:
                if param == "n_estimators":
                    # Генерируем список: range(min, max, step)
                    _, min_w, max_w, step_w = field_data
                    start = int(min_w.text().strip())
                    stop = int(max_w.text().strip())
                    step = int(step_w.text().strip())

                    if step <= 0:
                        step = 1
                    if start > stop:
                        start, stop = stop, start

                    value = list(range(start, stop + 1, step))
                else:
                    # Обычные параметры
                    field_type, widget = field_data
                    if info["type"] == "bool":
                        value = widget.currentText() == "True"
                    elif info["type"] == "choice":
                        text = widget.currentText()
                        value = None if text == "None" else text
                    elif info["type"] == "int":
                        text = widget.text().strip()
                        value = ast.literal_eval(text) if text.startswith(("[", "(")) else int(text)
                    elif info["type"] == "float":
                        text = widget.text().strip()
                        value = ast.literal_eval(text) if text.startswith(("[", "(")) else float(text)
                    else:
                        text = widget.text().strip()
                        value = ast.literal_eval(text) if text else info["default"]

                # ✅ ГАРАНТИРУЕМ: значение — список
                if not isinstance(value, (list, tuple, range)) and not hasattr(value, 'rvs'):
                    value = [value]

            except Exception as e:
                print(f"Ошибка при обработке {param}: {e}")
                value = info["default"]
                if not isinstance(value, (list, tuple)) and not hasattr(value, 'rvs'):
                    value = [value]

            model_params[param] = value

        full_grid[self.model_type] = model_params
        return full_grid

    def on_save_clicked(self):
        """Сохраняет параметры в random_grid."""
        try:
            updated_grid = self.update_model_grid()
            save_random_grid(updated_grid)
            QMessageBox.information(self, "Успех", f"Параметры модели '{self.model_type}' успешно сохранены!")
            self.load_model_params(self.model_type)  # Перезагрузка для актуального отображения
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить параметры:\n{str(e)}")
