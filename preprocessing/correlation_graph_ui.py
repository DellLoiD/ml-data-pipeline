import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox
)
import numpy as np


class CorrelationGraphUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('График корреляции')
        self.initUI()
    
    def initUI(self):
        # Элементы управления
        btn_select_dataset = QPushButton('Выбрать датасет')
        btn_select_target_variable = QPushButton('Выбрать целевую переменную')
        btn_build_correlation_graph = QPushButton('Построить график корреляции')
        btn_remove_target_variable = QPushButton('Удалить признак')
        btn_save_processed_data = QPushButton('Сохранить датасет')
        self.label_dataset_status = QLabel('')
        self.combo_box_columns = QComboBox()
        self.class_distribution_label = QLabel('')
        self.info_label = QLabel('''
Коэффициент корреляции близок к нулю: признак практически не зависит от другого параметра.<br/>
Высокий коэффициент (+0.8 и выше): два признака сильно взаимосвязаны, возможно, один из них избыточен.<br/>
Отрицательные коэффициенты (-0.8 и ниже): признаки движутся противоположно друг другу.
''')
        
        # Обработчики кнопок
        btn_select_dataset.clicked.connect(self.selectDataset)
        btn_select_target_variable.clicked.connect(self.selectTargetVariable)
        btn_build_correlation_graph.clicked.connect(self.buildCorrelationGraph)
        btn_remove_target_variable.clicked.connect(self.removeTargetVariable)
        btn_save_processed_data.clicked.connect(self.saveProcessedData)
        
        # Макеты элементов
        h_layout_buttons = QHBoxLayout()
        h_layout_buttons.addWidget(btn_select_dataset)
        h_layout_buttons.addWidget(btn_select_target_variable)
        h_layout_buttons.addWidget(btn_build_correlation_graph)
        h_layout_buttons.addWidget(btn_remove_target_variable)
        
        v_layout_main = QVBoxLayout()
        v_layout_main.addLayout(h_layout_buttons)
        v_layout_main.addWidget(self.label_dataset_status)
        v_layout_main.addWidget(self.combo_box_columns)
        v_layout_main.addWidget(self.class_distribution_label)
        v_layout_main.addWidget(self.info_label)
        v_layout_main.addWidget(btn_save_processed_data)
        
        self.setLayout(v_layout_main)
    
    def selectDataset(self):
        # Получаем абсолютный путь к текущему файлу
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        start_directory = os.path.join(project_root, 'dataset')
        
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            'Открыть файл',
            start_directory,
            'CSV (*.csv);;Excel (*.xls *.xlsx)'
        )
        
        if file_name:
            try:
                self.df = pd.read_csv(file_name) if '.csv' in file_name else pd.read_excel(file_name)
                self.file_name = file_name
                self.label_dataset_status.setText(f'Загружено {len(self.df)} строк.')
                self.combo_box_columns.clear()
                self.combo_box_columns.addItems(list(self.df.columns))
            except Exception as e:
                print(e)
                self.label_dataset_status.setText('Ошибка загрузки файла.')
                
    def selectTargetVariable(self):
        target_column = self.combo_box_columns.currentText()
        if target_column:
            self.target_variable = target_column
            class_counts = self.df[self.target_variable].value_counts().sort_values(ascending=False)
            
            top_classes = class_counts.head(15)
            
            distribution_text = f'\n{"-" * 30}\nРаспределение записей по категориям:\n'
            distribution_text += "\n".join([f"{cls}: {cnt}" for cls, cnt in top_classes.items()])
            
            total_classes_count = len(class_counts)
            if total_classes_count > 15:
                remaining_classes_count = total_classes_count - 15
                distribution_text += f"\n... и ещё {remaining_classes_count} категорий."
                
            self.class_distribution_label.setText(distribution_text)
            self.label_dataset_status.setText(f'Цель выбрана: {target_column}')
    
    def removeTargetVariable(self):
        if hasattr(self, 'target_variable'):
            self.df.drop(columns=self.target_variable, inplace=True)
            del self.target_variable
            self.combo_box_columns.clear()
            self.combo_box_columns.addItems(list(self.df.columns))
            self.class_distribution_label.clear()
            self.label_dataset_status.setText("Признак удалён.")
        else:
            self.label_dataset_status.setText("Ничего не выбрано для удаления.")
    
    def buildCorrelationGraph(self):
        if not hasattr(self, 'df'):
            self.label_dataset_status.setText('Сначала выберите датасет!')
            return

        # ✅ 1. Оставляем только числовые столбцы
        numeric_df = self.df.select_dtypes(include=['number', 'Int64'])
        
        if numeric_df.empty:
            QMessageBox.warning(self, "Внимание", "Нет числовых столбцов для построения корреляции.")
            return

        # ✅ 2. Сообщаем, какие столбцы использованы, а какие проигнорированы
        all_cols = set(self.df.columns)
        num_cols = set(numeric_df.columns)
        cat_cols = all_cols - num_cols
        
        if cat_cols:
            ignored_list = ', '.join(sorted(cat_cols))
            QMessageBox.information(
                self, "Информация",
                f"Корреляция строится только по числовым столбцам.\n"
                f"Игнорируются: {ignored_list}"
            )

        # Считаем корреляцию
        corr_matrix = numeric_df.corr()

        # Строим график
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title('Матрица корреляций', fontsize=16)
        plt.tight_layout()
        plt.show()

    def saveProcessedData(self):
        if not hasattr(self, 'df'):
            self.label_dataset_status.setText('Нет загруженного датасета для сохранения.')
            return
        
        project_dir = os.getcwd()
        output_folder = os.path.join(project_dir, 'dataset')
        os.makedirs(output_folder, exist_ok=True)
        
        input_filename = os.path.basename(self.file_name)
        output_path = os.path.join(output_folder, input_filename)
        
        try:
            self.df.to_csv(output_path, index=False)
            self.label_dataset_status.setText(f'Датасет успешно сохранён в {output_path}')
        except Exception as e:
            print(e)
            self.label_dataset_status.setText('Ошибка при сохранении файла.')


if __name__ == '__main__':
    app = QApplication([])
    window = CorrelationGraphUI()
    window.show()
    app.exec()
