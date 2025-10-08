import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QCheckBox, QMessageBox,  QInputDialog, )
import preprocessing.data_balancing.data_balancing_list_method_logic as balancing_methods  

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
        # Верхняя панель кнопок
        top_buttons_layout = QHBoxLayout()
        load_button = QPushButton("Загрузить датасет")
        load_button.clicked.connect(self.load_dataset)
        top_buttons_layout.addWidget(load_button)
        save_button = QPushButton("Сохранить датасет")
        save_button.clicked.connect(self.save_dataset)
        top_buttons_layout.addWidget(save_button)
        main_layout.addLayout(top_buttons_layout)        
        # Чекбокс округления меток
        self.round_labels_checkbox = QCheckBox("Округлять метки?")
        main_layout.addWidget(self.round_labels_checkbox)               
        # Лейбл количества записей по классам
        self.class_counts_label = QLabel("Количество записей каждого класса:")
        main_layout.addWidget(self.class_counts_label)
        self.before_label = QLabel("Распределение классов до балансировки:")
        main_layout.addWidget(self.before_label)        
        self.after_label = QLabel("Распределение классов после балансировки:")
        main_layout.addWidget(self.after_label)        
        self.setLayout(main_layout)
        self.show()
        # Блок кнопок методов балансировки
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
            'Original SMOTE': 'Классический SMOTE создает синтетические объекты между существующими объектами.',
            'Random Undersampling': 'Редукция мажоритарного класса путем случайного удаления объектов.',
            'Cluster Centroids Sampling': 'Выбор центров кластеров мажоритарного класса для уменьшения объема выборки.',
            'NearMiss Algorithms': 'Отбор ближайших соседей среди разных классов для оптимизации расстояния между ними.',
            'Random Oversampling': 'Дублирование миноритарных объектов для повышения их числа.',
            'ADASYN': 'Генерация синтетических объектов пропорционально степени редкости области пространства признаков.',
            'Borderline-SMOTE': 'Создание новых объектов рядом с границей классов для лучшего разделения.',
            'SMOTE-TOMEK Hybrid Method': 'Комбинация SMOTE и TOMEK links для очистки шума и коррекции смещения классов.',
            'SMOTE-ENN Hybrid Method': 'Применение SMOTE вместе с edited nearest neighbor для очищения набора данных.',
            'Bagging Classifier Ensemble Approach': 'Совместное использование ансамбля классификаторов для минимизации влияния несбалансированности.'
        }
        # Создание элементов интерфейса для каждого метода
        for method_name in self.methods.keys():
            hbox = QHBoxLayout()
            
            # Кнопка выбора метода
            button_method = QPushButton(method_name)
            button_method.setFixedSize(300, 30)
            button_method.clicked.connect(lambda checked=False, method=method_name: self.applyMethod(method))
            hbox.addWidget(button_method)
            
            # Справочная кнопка для описания метода
            help_button = QPushButton('Справка')
            help_button.setFixedSize(80, 30)
            help_button.clicked.connect(lambda checked=False, method=method_name: self.show_help(method))
            hbox.addWidget(help_button)
            
            methods_layout.addLayout(hbox)
        
        main_layout.addLayout(methods_layout)
        self.setLayout(main_layout)
        self.show()
        
    def show_help(self, method_name):
        """Показываем всплывающее окно с описанием выбранного метода"""
        msg_box = QMessageBox()
        msg_box.setText(f"{method_name}\n\nОписание:\n{self.methods.get(method_name)}")
        msg_box.exec_()
        
    def load_dataset(self):
        """Загружает датасет и позволяет выбрать целевую переменную."""
        file_dialog = QFileDialog()
        file_dialog.setDirectory('./dataset')
        file_name, _ = file_dialog.getOpenFileName(self, "Выбор датасета", "", "CSV Files (*.csv)")
        
        if file_name:
            df = pd.read_csv(file_name)
            self.dataset_filename = file_name
            
            # Выбираем только численные столбцы
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            # Диалог выбора целевой переменной
            item, ok = QInputDialog.getItem(
                self, 
                "Выбор целевой переменной",
                "Выберите целевую переменную:", 
                numeric_columns,
                editable=False
            )
            
            if ok and item:
                target_col = item
                feature_cols = list(set(numeric_columns) - {target_col})
                self.feature_cols = feature_cols
                self.target_col = target_col
                
                # Разделение признаков и цели
                self.X = df[self.feature_cols].values
                self.y = df[target_col].values
                
                # Создание тренировочной и тестовой выборки
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                
                # Теперь вызываем метод для обновления лейбла класса
                self.update_class_distribution_labels()
            else:
                QMessageBox.warning(self, "Предупреждение", "Необходимо выбрать целевую переменную!")
                
    def update_class_distribution_labels(self):
        """Обновляет текстовые метки количества записей в каждом классе."""
        # Обновляем лейбл с распределением классов ДО балансировки
        class_dist_before = pd.Series(self.y_train).value_counts().sort_values(ascending=True)
        text_before = f"Записей по классам:\n{class_dist_before.to_string()}"
        self.before_label.setText(text_before)
        
        # Текущий код очищает лейбл "после балансировки", пока сам процесс балансировки не реализован
        self.after_label.clear()

    def applyMethod(self, method_name):
        method_func = self.method_to_function_map.get(method_name)
        if method_func is None:
            print(f"МЕТОД '{method_name}' НЕ НАЙДЕН!")
            return
        round_labels = self.round_labels_checkbox.isChecked()
        balanced_X_train, balanced_y_train = method_func(self.X_train, self.y_train, round_labels=round_labels)
        self.X_resampled = balanced_X_train
        self.y_resampled = balanced_y_train
        class_counts = dict(zip(*np.unique(balanced_y_train, return_counts=True)))
        counts_str = ', '.join([f"{cls}: {count}" for cls, count in sorted(class_counts.items())])
        self.after_label.setText(f"Распределение классов после балансировки:\n{counts_str}")

    def save_dataset(self):
        if self.X_resampled is None or self.y_resampled is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните балансировку или обрезку.")
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