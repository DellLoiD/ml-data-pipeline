from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QComboBox, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
import ast
from selection_of_parameters.selection_of_parameters_logic import save_random_grid, get_random_grid


class HyperParameterOptimizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.model_type = "RandomForestClassifier"
        self.task_type = "classification"  # по умолчанию
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        self.setWindowTitle("Настройка гиперпараметров")
        self.setWindowIcon(QIcon.fromTheme("configure"))

        # === Тип задачи: Классификация / Регрессия ===
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Тип задачи:"))

        self.classification_radio = QRadioButton("1. Классификация")
        self.regression_radio = QRadioButton("2. Регрессия")
        self.classification_radio.setChecked(True)

        self.task_group = QButtonGroup()
        self.task_group.addButton(self.classification_radio, 1)
        self.task_group.addButton(self.regression_radio, 2)

        self.classification_radio.toggled.connect(self.on_task_changed)
        self.regression_radio.toggled.connect(self.on_task_changed)

        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.regression_radio)
        task_layout.addStretch()
        main_layout.addLayout(task_layout)

        # === Выбор модели ===
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.load_model_params)
        model_layout.addWidget(self.model_combo)
        main_layout.addLayout(model_layout)

        # --- ВАЖНО: сначала создаём fields_layout ---
        self.param_fields = {}
        self.fields_layout = QVBoxLayout()  # ✅ Сначала создаём
        main_layout.addLayout(self.fields_layout)

        # === Кнопка сохранения ===
        save_button = QPushButton("Сохранить параметры")
        save_button.clicked.connect(self.on_save_clicked)
        save_button.setStyleSheet("font-size: 12px; padding: 10px;")
        main_layout.addWidget(save_button)

        self.setLayout(main_layout)
        self.resize(600, 500)

        # --- А ТЕПЕРЬ вызываем load_model_params ---
        self.update_model_list()
        self.model_combo.setCurrentText(self.model_type)
        self.load_model_params(self.model_type)  # ✅ После создания fields_layout


    def get_task_type(self):
        """Возвращает текущий тип задачи"""
        return "classification" if self.classification_radio.isChecked() else "regression"

    @Slot()
    def on_task_changed(self):
        """Обновление списка моделей при смене типа задачи"""
        current_model = self.model_combo.currentText()
        self.update_model_list()
        # Сохраняем базовое имя (без Classifier/Regressor)
        base_model = current_model.replace("Classifier", "").replace("Regressor", "")
        # Пытаемся восстановить модель для новой задачи
        suggested_model = base_model + ("Classifier" if self.get_task_type() == "classification" else "Regressor")
        if suggested_model in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(suggested_model)
        else:
            self.model_combo.setCurrentIndex(0)
        self.load_model_params(self.model_combo.currentText())

    def update_model_list(self):
        """Обновляет список моделей в зависимости от задачи"""
        self.model_combo.clear()
        task = self.get_task_type()
        if task == "classification":
            models = ["RandomForestClassifier", "GradientBoostingClassifier", "LinearClassifier"]
        else:
            models = ["RandomForestRegressor", "GradientBoostingRegressor"]
        self.model_combo.addItems(models)

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
        """Загружает и отображает параметры для выбранной модели и задачи."""
        if not model_type:
            return
        self.model_type = model_type
        self.clear_layout(self.fields_layout)
        self.param_fields.clear()

        full_grid = get_random_grid()
        model_params = full_grid.get(model_type, {})

        # Получаем параметры, отфильтрованные по модели и задаче
        param_info = self.get_filtered_param_info(model_type)

        for param_name, info in param_info.items():
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

    def get_filtered_param_info(self, model_type):
        """Возвращает параметры, отфильтрованные по модели и типу задачи."""
        task = self.get_task_type()
        all_params = self.get_param_info()

        # Маппинг моделей к группам
        if "RandomForestClassifier" in model_type:
            base = "rf"
        elif "GradientBoostingClassifier" in model_type:
            base = "gb"
        elif model_type == "LogisticRegression":
            base = "lr"
        elif model_type == "LinearRegression":
            base = "linreg"
        else:
            base = ""

        # Фильтруем параметры
        result = {}
        for param, info in all_params.items():
            include = False
            t = info.get("task", "both")  # по умолчанию — обе задачи
            g = info.get("group", "")

            if t == "both" or t == task:
                if g == "":
                    include = True
                elif g == base:
                    include = True

            if include:
                result[param] = info

        return result

    def get_param_info(self):
        """Полная информация о параметрах с привязкой к задаче и группе."""
        return {
            'n_estimators': {
                "default": [10, 50, 100],
                "type": "range_int",
                "tooltip": "Количество деревьев в ансамбле.",
                "task": "both",
                "group": ""
            },
            'max_depth': {
                "default": None,
                "type": "choice",
                "options": [None, 3, 5, 7, 10],
                "tooltip": "Максимальная глубина дерева. None — без ограничений.",
                "task": "both",
                "group": ""
            },
            'min_samples_split': {
                "default": 2,
                "type": "int",
                "tooltip": "Минимальное число образцов в узле для разделения.",
                "task": "both",
                "group": ""
            },
            'min_samples_leaf': {
                "default": 1,
                "type": "int",
                "tooltip": "Минимальное число образцов в листе.",
                "task": "both",
                "group": ""
            },
            'bootstrap': {
                "default": True,
                "type": "bool",
                "tooltip": "Использовать bootstrap-выборку?",
                "task": "both",
                "group": "rf"
            },
            'criterion': {
                "default": "gini",
                "type": "choice",
                "options": ["gini", "entropy"],
                "tooltip": "Критерий разделения для классификации.",
                "task": "classification",
                "group": "rf"
            },
            'criterion_reg': {
                "name": "criterion",
                "default": "squared_error",
                "type": "choice",
                "options": ["squared_error", "absolute_error", "friedman_mse"],
                "tooltip": "Критерий разделения для регрессии.",
                "task": "regression",
                "group": "rf"
            },
            'class_weight': {
                "default": "balanced",
                "type": "choice",
                "options": ["balanced", None],
                "tooltip": "Учёт дисбаланса классов.",
                "task": "classification",
                "group": "rf"
            },
            'ccp_alpha': {
                "default": 0.0,
                "type": "float",
                "tooltip": "Регуляризация — обрезка дерева.",
                "task": "both",
                "group": ""
            },
            'verbose': {
                "default": 0,
                "type": "int",
                "options": [0, 1, 2],
                "tooltip": "Уровень подробного вывода.",
                "task": "both",
                "group": ""
            },
            'learning_rate': {
                "default": 0.1,
                "type": "float",
                "tooltip": "Скорость обучения для GradientBoosting.",
                "task": "both",
                "group": "gb"
            },
            'subsample': {
                "default": 0.8,
                "type": "float",
                "tooltip": "Доля данных для каждого дерева.",
                "task": "both",
                "group": "gb"
            },
            'penalty': {
                "default": "l2",
                "type": "choice",
                "options": ["l1", "l2"],
                "tooltip": "Тип регуляризации.",
                "task": "classification",
                "group": "lr"
            },
            'C': {
                "default": 1.0,
                "type": "float",
                "tooltip": "Инверсия силы регуляризации.",
                "task": "classification",
                "group": "lr"
            },
            'solver': {
                "default": "liblinear",
                "type": "choice",
                "options": ["liblinear", "saga"],
                "tooltip": "Метод оптимизации.",
                "task": "classification",
                "group": "lr"
            },
        }

    def extract_range_from_value(self, value):
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
                return 10, 100, 10

    def create_widget_for_param(self, info, value):
        if info["type"] == "bool":
            combo = QComboBox()
            combo.addItems(["True", "False"])
            combo.setCurrentText(str(bool(value)).title())
            return combo

        elif info["type"] == "choice":
            combo = QComboBox()
            # Преобразуем все значения в строки, но сохраняем оригинальные типы
            options_str = [str(x) for x in info["options"]]
            combo.addItems(options_str)
            current_str = str(value)
            if current_str in options_str:
                combo.setCurrentText(current_str)
            return combo

        elif info["type"] in ["int", "float"]:
            if isinstance(value, (list, tuple, range)):
                text_value = str(list(value))
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

    def show_help(self, message):
        QMessageBox.information(self, "Справка", message)

    def update_model_grid(self):
        """Собирает значения из полей и формирует корректную сетку параметров."""
        full_grid = get_random_grid()
        model_params = {}
        param_info = self.get_filtered_param_info(self.model_type)

        for param, field_data in self.param_fields.items():
            info = param_info.get(param)
            if info is None:
                continue

            try:
                if param == "n_estimators":
                    _, min_w, max_w, step_w = field_data
                    start = int(min_w.text().strip())
                    stop = int(max_w.text().strip())
                    step = int(step_w.text().strip())
                    if step <= 0: step = 1
                    if start > stop: start, stop = stop, start
                    value = list(range(start, stop + 1, step))

                else:
                    field_type, widget = field_data
                    if info["type"] == "bool":
                        value = widget.currentText() == "True"

                    elif info["type"] == "choice":
                        text = widget.currentText().strip()
                        # ✅ Преобразуем в оригинальный тип из options
                        options = info["options"]
                        for opt in options:
                            if str(opt) == text:
                                value = opt  # ✅ Сохраняем оригинальный тип (int, None и т.д.)
                                break
                        else:
                            value = info["default"]

                    elif info["type"] == "int":
                        text = widget.text().strip()
                        if text.startswith(("[", "(")):
                            value = ast.literal_eval(text)
                            value = [int(x) for x in value]
                        else:
                            value = int(text)

                    elif info["type"] == "float":
                        text = widget.text().strip()
                        if text.startswith(("[", "(")):
                            value = ast.literal_eval(text)
                            value = [float(x) for x in value]
                        else:
                            value = float(text)

                    else:
                        text = widget.text().strip()
                        value = ast.literal_eval(text) if text else info["default"]

                    # Если значение — список, но не range
                    if not isinstance(value, (list, tuple, range)) and not hasattr(value, 'rvs'):
                        value = [value]

            except Exception as e:
                print(f"Ошибка при обработке {param}: {e}")
                value = info["default"]
                if not isinstance(value, (list, tuple)):
                    value = [value]

            # Переименование псевдонимов
            if param in ["criterion_reg", "criterion_gb"]:
                model_params["criterion"] = value
            else:
                model_params[param] = value

        full_grid[self.model_type] = model_params
        return full_grid

    def on_save_clicked(self):
        try:
            updated_grid = self.update_model_grid()
            save_random_grid(updated_grid)
            QMessageBox.information(self, "Успех", f"Параметры модели '{self.model_type}' сохранены!")
            self.load_model_params(self.model_type)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить параметры:\n{str(e)}")
