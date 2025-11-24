from ast import literal_eval
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QSpinBox,
    QGroupBox,
    QDialogButtonBox,
    QLabel
)
from PySide6.QtCore import Qt


class SMOTEDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры SMOTE")

        main_layout = QVBoxLayout()

        # Случайное состояние (random_state)
        random_state_label = QLabel("Случайное состояние:")
        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 999999)
        self.random_state_spin.setValue(42)  # Значение по умолчанию
        hbox_random_state = QHBoxLayout()
        hbox_random_state.addWidget(random_state_label)
        hbox_random_state.addWidget(self.random_state_spin)
        main_layout.addLayout(hbox_random_state)

        # Группа выбора стратегии выборки
        strategy_group_box = QGroupBox("Стратегия выборки")
        group_box_layout = QVBoxLayout()

        # Список предопределенных вариантов
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['all', 'auto', 'minority', 'not majority', 'not minority'])
        group_box_layout.addWidget(self.strategy_combo)

        # Чекбокс для включения ручного ввода
        self.custom_input_checkbox = QCheckBox("Использовать собственный ввод:")
        self.custom_input_checkbox.stateChanged.connect(self.toggle_custom_input)
        group_box_layout.addWidget(self.custom_input_checkbox)

        # Устанавливаем стандартное значение в поле ввода
        self.custom_input_field = QLineEdit()
        self.custom_input_field.setText("{0: 500, 1: 500}")
        self.custom_input_field.setEnabled(False)
        group_box_layout.addWidget(self.custom_input_field)

        strategy_group_box.setLayout(group_box_layout)
        main_layout.addWidget(strategy_group_box)

        # Добавляем кнопки Apply и Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_and_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def toggle_custom_input(self, state):
        """Активация/деактивация возможности ввести собственное значение."""
        is_checked = bool(state)
        self.strategy_combo.setDisabled(is_checked)
        self.custom_input_field.setEnabled(is_checked)

    def apply_and_accept(self):
        """Обработка нажатия кнопки "Apply" и получение параметров."""
        params = self.get_parameters()
        print(f"Параметры метода SMOTE:\n{params}")
        self.accept()

    def get_parameters(self):
        """Получение текущих параметров из GUI."""
        selected_value = ""
        if self.custom_input_checkbox.isChecked():
            selected_value = self.custom_input_field.text().strip()
        else:
            selected_value = self.strategy_combo.currentText()

        return {
            "random_state": self.random_state_spin.value(),
            "sampling_strategy": selected_value
        }


def show_smote_parameter_dialog(parent=None):
    dialog = SMOTEDialog(parent)
    result = dialog.exec()
    if result == QDialog.DialogCode.Accepted:
        # Получаем введённые параметры из диалогового окна
        random_state_value = dialog.random_state_spin.value()
        sampling_strategy_value = dialog.get_parameters()['sampling_strategy']
        
        try:
            # Преобразуем строку в словарь (если пользователь ввёл строку вида "{...}")
            sampling_strategy_value = literal_eval(sampling_strategy_value)
        except ValueError:
            pass  # Оставляем исходное значение, если это строка (например, 'auto')
        
        return {
            'random_state': random_state_value,
            'sampling_strategy': sampling_strategy_value
        }
    else:
        return None