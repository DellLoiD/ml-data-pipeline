from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QFrame
)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyle
import ast
import sys

# Импорты логики
from .selection_of_parameters_logic import (
    get_random_search_params,
    save_random_search_params
)


class RandomSearchConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.text_fields = {}
        self.param_info = self.get_param_info()  # Кэшируем
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        self.setWindowTitle("Конфигурация RandomizedSearchCV")
        self.setWindowIcon(QIcon.fromTheme("configure"))

        # === Описание ===
        desc = QLabel("Настройки поиска гиперпараметров (RandomizedSearchCV)")
        desc.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(desc)

        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        # === Поля параметров ===
        self.fields_layout = QVBoxLayout()
        self.load_search_params()
        main_layout.addLayout(self.fields_layout)

        # === Кнопка сохранения ===
        save_button = QPushButton("Сохранить параметры")
        save_button.clicked.connect(self.on_save_clicked)
        save_button.setStyleSheet("font-size: 12px; padding: 10px;")
        main_layout.addWidget(save_button)

        # Устанавливаем макет
        self.setLayout(main_layout)
        self.resize(600, 400)

    def clear_layout(self, layout):
        """Безопасная очистка QVBoxLayout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    child_layout = item.layout()
                    if child_layout is not None:
                        self.clear_layout(child_layout)

    def get_param_info(self):
        """Описание параметров RandomizedSearchCV с подсказками и значениями по умолчанию"""
        return {
            'n_iter': {
                "default": 100,
                "tooltip": "Количество итераций случайного поиска.\n"
                           "Больше — точнее, но дольше."
            },
            'cv': {
                "default": 5,
                "tooltip": "Количество фолдов кросс-валидации.\n"
                           "Обычно 3–10. Чем больше — надёжнее оценка, но медленнее."
            },
            'scoring': {
                "default": ['accuracy'],
                "tooltip": "Метрики для оценки моделей.\n"
                           "Можно указать несколько: ['accuracy', 'f1_macro']"
            },
            'refit': {
                "default": 'accuracy',
                "tooltip": "Метрика, по которой выбирается лучшая модель.\n"
                           "Должна быть одной из 'scoring'."
            },
            'test_size': {
                "default": 0.2,
                "tooltip": "Доля данных, выделенных на тестовую выборку.\n"
                           "Обычно: 0.2 (20%)"
            },
            'random_state': {
                "default": 42,
                "tooltip": "Seed для воспроизводимости результатов."
            },
            'verbose': {
                "default": 1,
                "tooltip": "Уровень детализации вывода:\n"
                           "0 — тихо, 1 — информация по итерациям, 2 — подробно."
            },
            'n_jobs': {
                "default": -1,
                "tooltip": "Количество ядер CPU.\n"
                           "-1 = все доступные ядра."
            }
        }

    def load_search_params(self):
        """Загружает текущие параметры и создаёт поля ввода"""
        self.clear_layout(self.fields_layout)
        self.text_fields.clear()

        current_params = get_random_search_params()

        for param_name, info in self.param_info.items():
            current_value = current_params.get(param_name, info["default"])

            row = QHBoxLayout()
            label = QLabel(f"{param_name}:")
            row.addWidget(label)

            edit = QLineEdit(self.format_value(current_value))
            edit.setToolTip(str(current_value))
            row.addWidget(edit)
            self.text_fields[param_name] = edit

            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setIcon(QIcon.fromTheme("dialog-question", self.style().standardIcon(QStyle.SP_MessageBoxQuestion)))
            btn.clicked.connect(lambda _, tip=info["tooltip"]: self.show_help(tip))
            row.addWidget(btn)

            self.fields_layout.addLayout(row)

    def format_value(self, value):
        """Форматирует значение для отображения в QLineEdit"""
        if isinstance(value, (list, tuple)):
            return str(value)
        elif isinstance(value, str):
            return f"'{value}'"
        else:
            return str(value)

    def show_help(self, message):
        """Показывает подсказку в виде QMessageBox"""
        QMessageBox.information(self, "Справка: Параметр RandomizedSearchCV", message)

    def parse_value(self, param_name, text):
        """Безопасно парсит значение из строки"""
        text = text.strip()
        default_value = self.param_info[param_name]["default"]

        try:
            if isinstance(default_value, int):
                return int(text)
            elif isinstance(default_value, float):
                return float(text)
            elif isinstance(default_value, list):
                # Безопасный разбор списка
                if text.startswith('[') and text.endswith(']'):
                    return ast.literal_eval(text)
                else:
                    # Попробуем как строку
                    return text
            elif isinstance(default_value, str):
                if text.startswith("'") and text.endswith("'"):
                    return text[1:-1]
                return text
            else:
                return ast.literal_eval(text)
        except Exception:
            return default_value  # Возвращаем значение по умолчанию при ошибке

    def on_save_clicked(self):
        """Обработчик кнопки 'Сохранить'"""
        try:
            updated_params = {}
            for param_name in self.param_info:
                text = self.text_fields[param_name].text()
                value = self.parse_value(param_name, text)
                updated_params[param_name] = value

            # Сохраняем
            save_random_search_params(updated_params)
            QMessageBox.information(self, "Успех", "Параметры RandomizedSearchCV успешно сохранены!")

            # Перезагружаем, чтобы убедиться
            self.load_search_params()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить параметры:\n{str(e)}")