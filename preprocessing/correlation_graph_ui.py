# preprocessing/correlation_graph_ui.py
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
from utils.meta_tracker import MetaTracker  # Импорт трекера


class CorrelationGraphUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('График корреляции')
        self.df = None
        self.file_name = None
        self.removed_column = None
        self._last_loaded_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)  # Управление историей
        self.initUI()

    def initUI(self):
        # Элементы управления
        btn_select_dataset = QPushButton('Выбрать датасет')
        btn_select_target_variable = QPushButton('Выбрать целевую переменную')
        btn_build_correlation_graph = QPushButton('Построить график корреляции')
        btn_remove_target_variable = QPushButton('Удалить признак')
        self.btn_save_processed_data = QPushButton('Сохранить датасет')
        self.btn_save_processed_data.setEnabled(False)  # Кнопка активируется после изменений

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
        self.btn_save_processed_data.clicked.connect(self.saveProcessedData)

        # Макеты
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
        v_layout_main.addWidget(self.btn_save_processed_data)

        self.setLayout(v_layout_main)

    def selectDataset(self):
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
                # Загружаем мета-информацию
                self.meta_tracker.load_from_file(file_name)

                # Загружаем данные
                if file_name.endswith('.csv'):
                    self.df = pd.read_csv(file_name, comment='#', skipinitialspace=True)
                else:
                    self.df = pd.read_excel(file_name)

                self.file_name = file_name
                self._last_loaded_path = file_name
                self.label_dataset_status.setText(f'Загружено {len(self.df)} строк.')
                self.combo_box_columns.clear()
                self.combo_box_columns.addItems(list(self.df.columns))

                # Добавляем в историю
                self.meta_tracker.add_change("загружен датасет для анализа корреляции")
            except Exception as e:
                self.label_dataset_status.setText('Ошибка загрузки файла.')
                print(e)

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
            removed_col = self.target_variable
            self.df.drop(columns=self.target_variable, inplace=True)
            del self.target_variable
            self.combo_box_columns.clear()
            self.combo_box_columns.addItems(list(self.df.columns))
            self.class_distribution_label.clear()
            self.label_dataset_status.setText(f"Признак '{removed_col}' удалён.")

            # Добавляем в историю
            self.meta_tracker.add_change(f"удалён признак '{removed_col}'")

            # Активируем кнопку сохранения
            self.btn_save_processed_data.setEnabled(True)

            self.removed_column = removed_col
        else:
            self.label_dataset_status.setText("Ничего не выбрано для удаления.")

    def buildCorrelationGraph(self):
        if not hasattr(self, 'df'):
            self.label_dataset_status.setText('Сначала выберите датасет!')
            return

        numeric_df = self.df.select_dtypes(include=['number', 'Int64'])
        if numeric_df.empty:
            QMessageBox.warning(self, "Внимание", "Нет числовых столбцов для построения корреляции.")
            return

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

        plt.figure(figsize=(12, 9))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title('Матрица корреляций', fontsize=16)
        plt.tight_layout()
        plt.show()

    def saveProcessedData(self):
        if not hasattr(self, 'df'):
            self.label_dataset_status.setText('Нет загруженного датасета для сохранения.')
            return

        try:
            # Определяем базовое имя
            base_name = "correlation_processed"
            if self._last_loaded_path:
                name = os.path.splitext(os.path.basename(self._last_loaded_path))[0]
                base_name = name.split("_v")[0]

            save_path = os.path.join("dataset", f"{base_name}_v{self.meta_tracker.version}.csv")

            # Сохраняем через MetaTracker
            success = self.meta_tracker.save_to_file(save_path, self.df)
            if success:
                self._last_loaded_path = save_path
                self.meta_tracker.version += 1
                self.label_dataset_status.setText(f'✅ Датасет сохранён: {os.path.basename(save_path)}')
                self.btn_save_processed_data.setEnabled(False)
            else:
                self.label_dataset_status.setText('Ошибка при сохранении.')

        except Exception as e:
            self.label_dataset_status.setText('Ошибка при сохранении файла.')
            print(e)


if __name__ == '__main__':
    app = QApplication([])
    window = CorrelationGraphUI()
    window.show()
    app.exec()
