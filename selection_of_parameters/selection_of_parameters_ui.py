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
        self.model_type = "RandomForest"  # по умолчанию
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        self.setWindowTitle("Настройка гиперпараметров")

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
        main_layout.addWidget(save_button)

        self.setLayout(main_layout)

        # Загружаем параметры для начальной модели
        self.load_model_params(self.model_type)
        
    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    # Если это вложенный макет или спейсер — рекурсивно очищаем
                    child_layout = item.layout()
                    if child_layout is not None:
                        self.clear_layout(child_layout)

    def load_model_params(self, model_type):
        self.model_type = model_type
        # Очищаем макет с помощью безопасной функции
        self.clear_layout(self.fields_layout)
        self.param_fields.clear()

        # Получаем параметры ТОЛЬКО для выбранной модели
        full_grid = get_random_grid()
        model_params = full_grid.get(model_type, {})

        # Определяем, какие параметры доступны для каждой модели
        param_info = self.get_param_info()

        for param_name, info in param_info.items():
            if param_name not in model_params and model_type != "RandomForest":
                continue  # например, max_features нет у LogisticRegression

            # Группа для параметра
            row = QHBoxLayout()
            label = QLabel(f"{param_name}:")
            row.addWidget(label)

            # Текущее значение
            current_value = model_params.get(param_name, info["default"])
            edit = QLineEdit(str(current_value))
            row.addWidget(edit)
            self.param_fields[param_name] = edit

            # Кнопка справки
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setIcon(QIcon.fromTheme("dialog-question"))
            btn.clicked.connect(lambda _, tip=info["tooltip"]: self.show_help(tip))
            row.addWidget(btn)

            self.fields_layout.addLayout(row)

    def get_param_info(self):
        return {
            'n_estimators': {
                "default": [100],
                "tooltip": "Количество деревьев в лесу."
            },
            'max_depth': {
                "default": [None, 10, 20],
                "tooltip": "Макс. глубина дерева. None — неограничена."
            },
            'min_samples_split': {
                "default": [2, 5],
                "tooltip": "Мин. кол-во объектов для разделения узла."
            },
            'min_samples_leaf': {
                "default": [1, 2],
                "tooltip": "Мин. кол-во объектов в листе."
            },
            'bootstrap': {
                "default": [True, False],
                "tooltip": "Использовать bootstrapping?"
            },
            'criterion': {
                "default": ['gini', 'entropy'],
                "tooltip": "Функция качества разделения."
            },
            'class_weight': {
                "default": ['balanced', None],
                "tooltip": "Веса классов: balanced — автоматически."
            },
            'max_features': {
                "default": [None, 'sqrt', 'log2'],
                "tooltip": "Число признаков для выбора лучшего разделения."
            },
            'ccp_alpha': {
                "default": [0.0],
                "tooltip": "Коэффициент регуляризации (обрезка дерева)."
            },
            'verbose': {
                "default": [0],
                "tooltip": "Уровень подробности вывода (0 — нет)."
            },
            'learning_rate': {
                "default": [0.1],
                "tooltip": "Скорость обучения для Gradient Boosting."
            },
            'subsample': {
                "default": [1.0],
                "tooltip": "Доля данных, используемых для построения дерева."
            },
            'penalty': {
                "default": ['l2'],
                "tooltip": "Тип регуляризации: l1 (Lasso), l2 (Ridge)."
            },
            'C': {
                "default": [1.0],
                "tooltip": "Обратная сила регуляризации. Меньше — сильнее штраф."
            },
            'solver': {
                "default": ['liblinear'],
                "tooltip": "Алгоритм оптимизации."
            },
        }

    def show_help(self, message):
        QMessageBox.information(self, "Справка", message)

    def update_model_grid(self):
        full_grid = get_random_grid()  # текущий полный словарь
        model_params = {}

        for param, edit in self.param_fields.items():
            raw = edit.text().strip()
            try:
                value = ast.literal_eval(raw)
            except Exception:
                try:
                    value = eval(raw)  # fallback для range, loguniform — ОПАСНО, но локально допустимо
                except:
                    value = raw
            model_params[param] = value

        # Обновляем только нужную модель
        full_grid[self.model_type] = model_params
        return full_grid

    def on_save_clicked(self):
        try:
            updated_grid = self.update_model_grid()
            save_random_grid(updated_grid)  # сохраняем весь словарь
            QMessageBox.information(self, "Успех", f"Параметры модели {self.model_type} сохранены!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить параметры:\n{str(e)}")
