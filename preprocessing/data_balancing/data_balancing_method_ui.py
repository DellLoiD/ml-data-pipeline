# data_balancing_method_ui.py
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QApplication, QDialog
)
from preprocessing.data_balancing.data_balancing_list_method_ui import BalancingMethodsWindow
from preprocessing.data_balancing.dataset_trim.dataset_trim_window_ui import DatasetTrimWindow
from preprocessing.data_balancing.data_balancing_operaiting_classes import FeatureSelector
from preprocessing.data_balancing.align_columns_ui import AlignColumnsApp  # ← Новый импорт

# === Глобальные ссылки на окна (чтобы не открывалось несколько раз) ===
balancing_window_instance = None
trim_window_instance = None
feature_selector_instance = None
align_columns_instance = None  # ← Новая глобальная переменная


class DataBalancingApp(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Балансировка датасета")
        layout = QVBoxLayout(self)

        # === Кнопка: Выбор метода балансировки ===
        balance_button = QPushButton("Выбрать метод балансировки")
        balance_button.clicked.connect(self._open_balancing_window)
        layout.addWidget(balance_button)

        # === Кнопка: Обрезать датасет ===
        trim_dataset_button = QPushButton("Обрезать датасет")
        trim_dataset_button.clicked.connect(self._open_trim_window)
        layout.addWidget(trim_dataset_button)

        # === Кнопка: Удалить колонку ===
        operaiting_classes_button = QPushButton("Удалить колонку")
        operaiting_classes_button.clicked.connect(self._open_operaiting_classes)
        layout.addWidget(operaiting_classes_button)

        # === КНОПКА: Выровнять порядок колонок ===
        align_columns_button = QPushButton("🔧 Выровнять порядок колонок")
        align_columns_button.clicked.connect(self._open_align_columns_window)
        layout.addWidget(align_columns_button)

        # === Настройки окна ===
        self.setLayout(layout)
        self.resize(400, 300)

    def _open_balancing_window(self):
        global balancing_window_instance
        if balancing_window_instance is None or not balancing_window_instance.isVisible():
            balancing_window_instance = BalancingMethodsWindow()
            balancing_window_instance.show()

    def _open_trim_window(self):
        global trim_window_instance
        if trim_window_instance is None or not trim_window_instance.isVisible():
            trim_window_instance = DatasetTrimWindow()
            trim_window_instance.show()

    def _open_operaiting_classes(self):
        global feature_selector_instance
        if feature_selector_instance is None or not feature_selector_instance.isVisible():
            feature_selector_instance = FeatureSelector()
            feature_selector_instance.show()

    def _open_align_columns_window(self):
        global align_columns_instance
        if align_columns_instance is None or not align_columns_instance.isVisible():
            align_columns_instance = AlignColumnsApp()
            align_columns_instance.show()
