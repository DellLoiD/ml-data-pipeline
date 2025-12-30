# preprocessing/data_balancing/data_balancing_list_method_ui.py

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QCheckBox,
    QMessageBox, QInputDialog, QGroupBox, QTableWidget, QTableWidgetItem,
    QLabel, QApplication
)
import preprocessing.data_balancing.data_balancing_list_method_logic as balancing_methods
from preprocessing.data_balancing.setting_window_of_methods_balansing.smote_dialog_window_ui import show_smote_parameter_dialog


class BalancingMethodsWindow(QDialog):
    balancing_finished_signal = Signal(str)

    def __init__(self, X_train=None, y_train=None):
        super(BalancingMethodsWindow, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_resampled = None
        self.y_resampled = None
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # === Верхняя панель кнопок ===
        top_buttons_layout = QHBoxLayout()
        load_button = QPushButton("Загрузить датасет")
        load_button.clicked.connect(self.load_dataset)
        top_buttons_layout.addWidget(load_button)

        save_button = QPushButton("Сохранить датасет")
        save_button.clicked.connect(self.save_dataset)
        top_buttons_layout.addWidget(save_button)
        main_layout.addLayout(top_buttons_layout)

        # === Чекбокс округления меток ===
        self.round_labels_checkbox = QCheckBox("Округлять метки?")
        main_layout.addWidget(self.round_labels_checkbox)

        # === Заголовок для распределения классов ===
        class_counts_label = QLabel("Распределение количества записей по классам:")
        class_counts_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(class_counts_label)

        # === Таблица "До балансировки" ===
        before_layout = QVBoxLayout()
        before_layout.addWidget(QLabel("До балансировки:"))
        self.before_table = self.create_class_table()
        before_layout.addWidget(self.before_table)
        main_layout.addLayout(before_layout)

        # === Таблица "После балансировки" ===
        after_layout = QVBoxLayout()
        after_layout.addWidget(QLabel("После балансировки:"))
        self.after_table = self.create_class_table()
        after_layout.addWidget(self.after_table)
        main_layout.addLayout(after_layout)

        # === Блок кнопок методов балансировки ===
        methods_layout = QVBoxLayout()

        self.method_to_function_map = {
            'Original SMOTE': balancing_methods.balance_classes_smote,
            'Random Undersampling': balancing_methods.balance_classes_random_undersampling,
            'Cluster Centroids Sampling': balancing_methods.balance_classes_cluster_centroids,
            'NearMiss Algorithms': balancing_methods.balance_classes_nearmiss,
            'Random Oversampling': balancing_methods.balance_classes_random_oversampling,
            'ADASYN': balancing_methods.balance_classes_adasyn,
            'Borderline-SMOTE': balancing_methods.balance_classes_borderlinesmote,
            'SMOTE-TOMEK Hybrid Method': balancing_methods.balance_classes_hybrid_smotetomek,
            'SMOTE-ENN Hybrid Method': balancing_methods.balance_classes_hybrid_smoteenn,
            'Bagging Classifier Ensemble Approach': balancing_methods.balance_classes_bagging_classifier
        }

        self.methods = {
            'Original SMOTE': 'Создает синтетические объекты между существующими объектами.',
            'Random Undersampling': 'Уменьшение доминирующего класса путем случайного удаления объектов.',
            'Cluster Centroids Sampling': 'Выбор центров кластеров мажоритарного класса для уменьшения объема выборки.',
            'NearMiss Algorithms': 'Отбор ближайших соседей среди разных классов для оптимизации расстояния между ними.',
            'Random Oversampling': 'Дублирование миноритарных объектов для повышения их числа.',
            'ADASYN': 'Генерация синтетических объектов пропорционально степени редкости области пространства признаков.',
            'Borderline-SMOTE': 'Создание новых объектов рядом с границей классов для лучшего разделения.',
            'SMOTE-TOMEK Hybrid Method': 'Комбинация SMOTE и TOMEK links для очистки шума и коррекции смещения классов.',
            'SMOTE-ENN Hybrid Method': 'Применение SMOTE вместе с edited nearest neighbor для очищения набора данных.',
            'Bagging Classifier Ensemble Approach': 'Совместное использование ансамбля классификаторов для минимизации влияния несбалансированности.'
        }

        # === Группы методов ===
        self.create_method_group(methods_layout, "Методы, увеличивающие количество образцов",
                                 ['Original SMOTE', 'Random Oversampling', 'ADASYN', 'Borderline-SMOTE'])

        self.create_method_group(methods_layout, "Методы, уменьшающие количество образцов",
                                 ['Random Undersampling', 'Cluster Centroids Sampling', 'NearMiss Algorithms'])

        self.create_method_group(methods_layout, "Гибридные методы",
                                 ['SMOTE-TOMEK Hybrid Method', 'SMOTE-ENN Hybrid Method'])

        self.create_method_group(methods_layout, "Дополнительные методы",
                                 ['Bagging Classifier Ensemble Approach'])

        main_layout.addLayout(methods_layout)
        self.setLayout(main_layout)

        # === Настройки окна ===
        self.setWindowTitle("Методы балансировки классов")
        self.resize(400, 900)
        self.setMinimumSize(600, 500)

    def create_class_table(self):
        """Создаёт таблицу для отображения распределения классов"""
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Класс", "Количество"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)  # Только для чтения
        table.setMaximumHeight(200)  # Ограничиваем высоту
        table.verticalHeader().setVisible(False)  # Убираем нумерацию строк
        return table

    def create_method_group(self, layout, title, method_names):
        """Создаёт группу методов с кнопками и справкой"""
        group = QGroupBox(title)
        group_layout = QVBoxLayout()

        for method_name in method_names:
            hbox = QHBoxLayout()

            button = QPushButton(method_name)
            button.setFixedSize(300, 30)
            button.clicked.connect(lambda checked=False, m=method_name: self.applyMethod(m))
            hbox.addWidget(button)

            help_button = QPushButton('Справка')
            help_button.setFixedSize(80, 30)
            help_button.clicked.connect(lambda checked=False, m=method_name: self.show_help(m))
            hbox.addWidget(help_button)

            group_layout.addLayout(hbox)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def show_help(self, method_name):
        """Показывает описание метода"""
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Справка по методу")
        msg_box.setText(f"{method_name}\n\nОписание:\n{self.methods.get(method_name, 'Нет описания')}")
        msg_box.exec_()

    def load_dataset(self):
        """Загружает датасет и позволяет выбрать целевую переменную."""
        file_dialog = QFileDialog()
        file_dialog.setDirectory('./dataset')
        file_name, _ = file_dialog.getOpenFileName(self, "Выбор датасета", "", "CSV Files (*.csv)")

        if file_name:
            df = pd.read_csv(file_name)
            self.dataset_filename = file_name

            numeric_columns = df.select_dtypes(include=['number', 'bool']).columns.tolist()

            item, ok = QInputDialog.getItem(
                self,
                "Выбор целевой переменной (number или bool)",
                "Выберите целевую переменную:",
                numeric_columns,
                editable=False
            )

            if ok and item:
                target_col = item
                feature_cols = list(set(numeric_columns) - {target_col})
                self.feature_cols = feature_cols
                self.target_col = target_col

                self.X = df[feature_cols].values
                self.y = df[target_col].values

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=0.2, random_state=42
                )

                self.update_class_distribution_labels()
            else:
                QMessageBox.warning(self, "Предупреждение", "Необходимо выбрать целевую переменную!")

    def update_class_distribution_labels(self):
        """Обновляет таблицы распределения классов (ограничено 20 строками)"""
        self.update_table(self.before_table, self.y_train, "До балансировки")

    def update_table(self, table, y_data, title_prefix=""):
        """Заполняет таблицу данными о классах (до 20 строк + 'и ещё...')"""
        class_counts = pd.Series(y_data).value_counts().sort_values(ascending=False)
        total_classes = len(class_counts)

        # Очищаем таблицу
        table.setRowCount(0)

        if total_classes == 0:
            return

        # Показываем максимум 20 строк
        top_classes = class_counts.iloc[:20]
        hidden_count = total_classes - 20

        for cls, count in top_classes.items():
            row_position = table.rowCount()
            table.insertRow(row_position)
            table.setItem(row_position, 0, QTableWidgetItem(str(cls)))
            table.setItem(row_position, 1, QTableWidgetItem(str(count)))

        # Если есть скрытые классы — добавляем строку с подсказкой
        if hidden_count > 0:
            row_position = table.rowCount()
            table.insertRow(row_position)
            table.setItem(row_position, 0, QTableWidgetItem("... и ещё"))
            table.setItem(row_position, 1, QTableWidgetItem(str(hidden_count)))

    def applyMethod(self, method_name):
        method_func = self.method_to_function_map.get(method_name)
        if method_func is None:
            QMessageBox.critical(self, "Ошибка", f"Метод '{method_name}' не найден!")
            return

        balanced_X_train = None
        balanced_y_train = None

        try:
            round_labels = self.round_labels_checkbox.isChecked()

            if method_name == 'Original SMOTE':
                parameters = show_smote_parameter_dialog(self)
                if not parameters:
                    return
                balanced_X_train, balanced_y_train = method_func(
                    self.X_train, self.y_train, round_labels=round_labels, **parameters
                )
            else:
                balanced_X_train, balanced_y_train = method_func(
                    self.X_train, self.y_train, round_labels=round_labels
                )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить метод:\n{str(e)}")
            return

        if balanced_X_train is not None and balanced_y_train is not None:
            self.X_resampled = balanced_X_train
            self.y_resampled = balanced_y_train

            # Обновляем таблицу "после"
            self.update_table(self.after_table, self.y_resampled, "После балансировки")

            QMessageBox.information(self, "Успех", f"Метод '{method_name}' успешно применён.")
        else:
            QMessageBox.warning(self, "Предупреждение", "Не удалось сбалансировать данные.")

    def save_dataset(self):
        if self.X_resampled is None or self.y_resampled is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните балансировку.")
            return

        try:
            resampled_df = pd.DataFrame(data=self.X_resampled, columns=self.feature_cols)
            resampled_df[self.target_col] = self.y_resampled

            original_basename = os.path.splitext(os.path.basename(self.dataset_filename))[0]
            target_variable = self.target_col
            most_common_class = pd.Series(self.y_resampled).value_counts().index[0]
            count_most_common_class = pd.Series(self.y_resampled).value_counts()[most_common_class]

            new_filename = f"{original_basename}-balanced-{target_variable}-size{count_most_common_class}.csv"
            output_path = os.path.join("./dataset", new_filename)

            resampled_df.to_csv(output_path, index=False)
            QMessageBox.information(self, "Успех", f"Датасет успешно сохранён в {output_path}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при сохранении файла: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BalancingMethodsWindow()
    window.show()
    sys.exit(app.exec())
