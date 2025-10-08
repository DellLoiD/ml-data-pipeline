import os, sys, subprocess
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QLineEdit, QHBoxLayout
from PySide6.QtCore import Qt
import pandas as pd

# Списки возможных скриптов для запуска
scripts_to_run = [
    'experiments/diabetes_prediction_forest.py',
    'experiments/diabetes_prediction_logistic_regression.py',    
    'experiments/diabetes_prediction_gradient_boosting.py',
]

class ScriptRunnerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def runSelectedScripts(self):
        # Проверка выбраны ли скрипты
        selected_scripts = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        if not selected_scripts:
            QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы один скрипт!")
            return

    def initUI(self):
        # Основной вертикальный макет
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Чекбоксы для выбора скриптов
        self.checkboxes = []
        for script_name in scripts_to_run:
            checkbox = QCheckBox(script_name)
            main_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        # Кнопка запуска
        run_button = QPushButton("Запустить выбранные скрипты")
        run_button.clicked.connect(self.runSelectedScripts)
        main_layout.addWidget(run_button)

        # Виджет для выбора файла (изначально скрыт)
        file_label = QLabel('Файл:')
        self.file_input = QLineEdit()
        browse_button = QPushButton('Обзор...')
        browse_button.clicked.connect(self.browseDataset)
        file_hlayout = QHBoxLayout()
        file_hlayout.addWidget(file_label)
        file_hlayout.addWidget(self.file_input)
        file_hlayout.addWidget(browse_button)
        main_layout.addLayout(file_hlayout)

        # Комбо-бокс для выбора целевой переменной (изначально пуст)
        target_label = QLabel('Цель:')
        self.target_combo = QComboBox()
        target_hlayout = QHBoxLayout()
        target_hlayout.addWidget(target_label)
        target_hlayout.addWidget(self.target_combo)
        main_layout.addLayout(target_hlayout)

        # Показываем / скрываем элементы, если нужен выбор датасета
        self.dataset_selection_visible(False)

        # Обработчик события проверки чекбокса
        for cb in self.checkboxes:
            cb.stateChanged.connect(self.handleCheckboxStateChange)

        # Настраиваем окно
        self.setWindowTitle("Выбор и выполнение скриптов")
        self.resize(400, 300)

    def handleCheckboxStateChange(self, state):
        """
        Когда меняется состояние одного из чекбоксов,
        проверяется необходимость включения блока выбора датасета
        """
        any_checked = any(cb.isChecked() for cb in self.checkboxes)
        self.dataset_selection_visible(any_checked)

    def dataset_selection_visible(self, visible):
        """Показывать или скрывать блок выбора датасета"""
        self.file_input.setVisible(visible)
        self.target_combo.setVisible(visible)

    def browseDataset(self):
        """
        Открывает диалог выбора файла и загружает его признаки
        """
        filename, _ = QFileDialog.getOpenFileName(
            None,           # Родительского элемента нет
            "Выбрать датасет",  # Название диалога
            dir="./dataset/",  # Начальная директория
            filter="CSV files (*.csv)"  # Тип фильтрации файлов
        )
        if filename:
            try:
                df = pd.read_csv(filename)
                columns = list(df.columns)
                self.target_combo.clear()
                self.target_combo.addItems(columns)
                self.file_input.setText(filename)
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", str(e))

def runSelectedScripts(self):
    selected_scripts = []
    for i, checkbox in enumerate(self.checkboxes):
        if checkbox.isChecked():
            selected_scripts.append(scripts_to_run[i])

    if not selected_scripts:
        QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы один скрипт!")
        return

    # Получаем путь к файлу и выбранную цель
    filename = self.file_input.text()
    target_column = self.target_combo.currentText()

    if not filename or not target_column:
        QMessageBox.warning(self, "Ошибка", "Укажите файл и выберите целевую колонку.")
        return

    # Передача пути к файлу и имени столбца цели в скрипты
    arguments = ['--file', filename, '--target', target_column]

    # Запуск выбранных скриптов
    py_path = sys.executable
    success = True

    for script in selected_scripts:
        print(f'\nЗапускаем скрипт: {script}\n')
        cmd = [py_path, script] + arguments
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            error_message = f"Ошибка при выполнении скрипта '{script}'.\n\nOutput:\n{result.stdout}\nError:\n{result.stderr}"
            QMessageBox.critical(self, "Ошибка", error_message)
            success = False
            break
        else:
            print(f"\nСкрипт '{script}' успешно выполнен!\n")

    if success:
        QMessageBox.information(self, "Готово", "Все выбранные скрипты выполнены успешно.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ScriptRunnerApp()
    window.show()
    sys.exit(app.exec())


