from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QMessageBox, QFrame, QInputDialog
)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyle
import ast

# Импорты логики
from .selection_of_parameters_logic import (
    get_random_search_params,
    save_random_search_params
)


class RandomSearchConfigGUI(QWidget):
    def __init__(self):
        super().__init__()

        # === 🔥 ШАГ 1: Показываем диалог выбора задачи при старте ===
        task, ok = QInputDialog.getItem(
            self, "Тип задачи", "Выберите тип задачи:",
            ["Классификация", "Регрессия"],
            current=0,
            editable=False
        )
        if not ok:
            # Если отмена — выбираем по умолчанию
            self.task_type = "classification"
            QMessageBox.warning(self, "Внимание", "Тип задачи не выбран. Будет использована классификация.")
        else:
            self.task_type = "classification" if task == "Классификация" else "regression"

        # === 🔥 ШАГ 2: Устанавливаем refit и scoring по умолчанию для выбранной задачи ===
        self._set_default_scoring_and_refit()

        # === Инициализация UI ===
        self.text_fields = {}
        self.param_info = self.get_param_info()  # Теперь зависит от self.task_type
        self.initUI()

    def _set_default_scoring_and_refit(self):
        """Устанавливает scoring и refit в зависимости от задачи"""
        params = get_random_search_params()

        if self.task_type == "classification":
            default_scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}
            default_refit = "roc_auc"
        else:
            default_scoring = {
                "r2": "r2",
                "neg_mean_squared_error": "neg_mean_squared_error",
                "neg_mean_absolute_error": "neg_mean_absolute_error"
            }
            default_refit = "r2"

        # Обновляем только если изменилось
        updated = False
        if params.get("scoring") != default_scoring:
            params["scoring"] = default_scoring
            updated = True
        if params.get("refit") != default_refit:
            params["refit"] = default_refit
            updated = True

        if updated:
            save_random_search_params(params)

    def initUI(self):
        main_layout = QVBoxLayout()
        self.setWindowTitle("Конфигурация RandomizedSearchCV")
        self.setWindowIcon(QIcon.fromTheme("configure"))

        # === Заголовок ===
        title = QLabel("Настройки поиска гиперпараметров (RandomizedSearchCV)")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # === Метка типа задачи ===
        self.task_label = QLabel(f"Текущая задача: <b>{'Классификация' if self.task_type == 'classification' else 'Регрессия'}</b>")
        self.task_label.setStyleSheet("color: #1E90FF; font-weight: bold;")
        main_layout.addWidget(self.task_label)

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
        """Возвращает параметры с учётом типа задачи"""
        if self.task_type == "classification":
            default_scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}
            default_refit = "roc_auc"
            scoring_tooltip = "Для классификации: accuracy, f1_macro, roc_auc, precision_macro, recall_macro"
        else:
            default_scoring = {
                "r2": "r2",
                "neg_mean_squared_error": "neg_mean_squared_error",
                "neg_mean_absolute_error": "neg_mean_absolute_error"
            }
            default_refit = "r2"
            scoring_tooltip = "Для регрессии: r2, neg_mean_squared_error, neg_mean_absolute_error, explained_variance"

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
                "default": default_scoring,
                "tooltip": "Словарь метрик для оценки.\n"
                           "Формат: {'название': 'метрика'}\n"
                           "Пример: {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}\n\n"
                           + scoring_tooltip
            },
            'refit': {
                "default": default_refit,
                "tooltip": "Ключ из 'scoring', по которому выбирается лучшая модель.\n"
                           "Пример: 'accuracy', 'f1_macro', 'r2' и т.д."
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
        """Загружает и создаёт поля ввода. Обновляет refit динамически."""
        self.clear_layout(self.fields_layout)
        self.text_fields.clear()

        current_params = get_random_search_params()

        for param_name, info in self.param_info.items():
            current_value = current_params.get(param_name, info["default"])

            row = QHBoxLayout()
            label = QLabel(f"{param_name}:")
            row.addWidget(label)

            # === Особое поле для 'refit' ===
            if param_name == "refit":
                # Используем scoring из текущих настроек (уже может быть обновлён)
                scoring_dict = current_params.get("scoring", info["default"])
                if isinstance(scoring_dict, dict):
                    available_metrics = list(scoring_dict.keys())
                else:
                    # fallback на метрики по типу задачи
                    available_metrics = ["r2"] if self.task_type == "regression" else ["accuracy"]

                combo = QComboBox()
                combo.addItems(available_metrics)
                # Устанавливаем текущее значение, если допустимо
                if current_value in available_metrics:
                    combo.setCurrentText(current_value)
                else:
                    combo.setCurrentText(available_metrics[0])  # fallback
                row.addWidget(combo)
                self.text_fields[param_name] = combo

            else:
                # Для других параметров — QLineEdit
                edit = QLineEdit(self.format_value(current_value))
                edit.setToolTip(str(current_value))
                row.addWidget(edit)
                self.text_fields[param_name] = edit

            # Кнопка помощи
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setIcon(QIcon.fromTheme("dialog-question", self.style().standardIcon(QStyle.SP_MessageBoxQuestion)))
            btn.clicked.connect(lambda _, tip=info["tooltip"]: self.show_help(tip))
            row.addWidget(btn)

            self.fields_layout.addLayout(row)

    def format_value(self, value):
        """Форматирует значение для отображения в QLineEdit"""
        if isinstance(value, dict):
            items = [f"'{k}': '{v}'" for k, v in value.items()]
            return "{" + ", ".join(items) + "}"
        elif isinstance(value, (list, tuple)):
            return str(value)
        elif isinstance(value, str):
            return f"'{value}'"
        else:
            return str(value)

    def show_help(self, message):
        """Показывает подсказку"""
        QMessageBox.information(self, "Справка: Параметр RandomizedSearchCV", message)

    def parse_value(self, param_name, widget):
        """Безопасно парсит значение из виджета"""
        if isinstance(widget, QComboBox):
            return widget.currentText()

        text = widget.text().strip() if isinstance(widget, QLineEdit) else str(widget)
        default_value = self.param_info[param_name]["default"]

        try:
            if isinstance(default_value, int):
                return int(text)
            elif isinstance(default_value, float):
                return float(text)
            elif isinstance(default_value, list):
                if text.startswith('[') and text.endswith(']'):
                    return ast.literal_eval(text)
                else:
                    return default_value
            elif isinstance(default_value, dict):
                if text.startswith("{") and text.endswith("}"):
                    return ast.literal_eval(text)
                else:
                    return default_value
            elif isinstance(default_value, str):
                if text.startswith("'") and text.endswith("'"):
                    return text[1:-1]
                return text
            else:
                return ast.literal_eval(text)
        except Exception as e:
            print(f"Ошибка парсинга {param_name}: {text} -> {e}")
            return default_value

    def on_save_clicked(self):
        """Сохранение параметров"""
        try:
            updated_params = {}
            for param_name in self.param_info:
                widget = self.text_fields[param_name]
                value = self.parse_value(param_name, widget)
                updated_params[param_name] = value

            # Сохраняем
            save_random_search_params(updated_params)
            QMessageBox.information(self, "Успех", "Параметры RandomizedSearchCV успешно сохранены!")

            # Перезагружаем для актуального отображения
            self.load_search_params()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить параметры:\n{str(e)}")
