# preprocessing/correlation_graph_ui.py
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QLabel,
    QComboBox, QVBoxLayout, QHBoxLayout, QMessageBox, QSplitter,
    QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from utils.meta_tracker import MetaTracker  # Импорт трекера
import gc


class CorrelationGraphUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('График корреляции')
        self.df = None
        self.file_name = None
        self.removed_column = None
        self._last_loaded_path = None
        self.meta_tracker = MetaTracker(max_line_length=150)
        self.graph_window = None  # Для отслеживания окна графика
        self.canvas = None
        self.initUI()

    def initUI(self):
        # Элементы управления
        btn_select_dataset = QPushButton('Выбрать датасет')
        btn_select_target_variable = QPushButton('Выбрать целевую переменную')
        btn_build_correlation_graph = QPushButton('Построить график корреляции')
        btn_remove_target_variable = QPushButton('Удалить признак')
        self.btn_save_processed_data = QPushButton('Сохранить датасет')
        self.btn_save_processed_data.setEnabled(False)

        self.label_dataset_status = QLabel('')
        self.combo_box_columns = QComboBox()
        self.class_distribution_label = QLabel('')

        # Обработчики кнопок
        btn_select_dataset.clicked.connect(self.selectDataset)
        btn_select_target_variable.clicked.connect(self.selectTargetVariable)
        btn_build_correlation_graph.clicked.connect(self.buildCorrelationGraph)
        btn_remove_target_variable.clicked.connect(self.removeTargetVariable)
        self.btn_save_processed_data.clicked.connect(self.saveProcessedData)

        # Макет кнопок
        h_layout_buttons = QHBoxLayout()
        h_layout_buttons.addWidget(btn_select_dataset)
        h_layout_buttons.addWidget(btn_select_target_variable)
        h_layout_buttons.addWidget(btn_build_correlation_graph)
        h_layout_buttons.addWidget(btn_remove_target_variable)

        # Основной макет
        v_layout_main = QVBoxLayout()
        v_layout_main.addLayout(h_layout_buttons)
        v_layout_main.addWidget(self.label_dataset_status)
        v_layout_main.addWidget(self.combo_box_columns)
        v_layout_main.addWidget(self.class_distribution_label)
        v_layout_main.addWidget(self.btn_save_processed_data)

        self.setLayout(v_layout_main)
        self.resize(900, 700)

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
                self.meta_tracker.load_from_file(file_name)

                if file_name.endswith('.csv'):
                    self.df = pd.read_csv(file_name, comment='#', skipinitialspace=True)
                else:
                    self.df = pd.read_excel(file_name)

                self.file_name = file_name
                self._last_loaded_path = file_name
                self.label_dataset_status.setText(f'Загружено {len(self.df)} строк.')
                self.combo_box_columns.clear()
                self.combo_box_columns.addItems(list(self.df.columns))

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
            self.meta_tracker.add_change(f"удалён признак '{removed_col}'")
            self.btn_save_processed_data.setEnabled(True)
            self.removed_column = removed_col
        else:
            self.label_dataset_status.setText("Ничего не выбрано для удаления.")

    def load_param_descriptions(self, file_path):
        """Загружает описания параметров из .txt файла"""
        descriptions = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    key, desc = line.split(':', 1)
                    descriptions[key.strip()] = desc.strip()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать файл описаний:\n{e}")
        return descriptions

    def buildCorrelationGraph(self):
        if not hasattr(self, 'df'):
            self.label_dataset_status.setText('Сначала выберите датасет!')
            return

        numeric_df = self.df.select_dtypes(include=['number', 'Int64'])
        if numeric_df.empty:
            QMessageBox.warning(self, "Внимание", "Нет числовых столбцов для построения корреляции.")
            return

        # Выбор файла описаний
        desc_file, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с описаниями параметров", "./", "Text Files (*.txt)"
        )
        descriptions = self.load_param_descriptions(desc_file) if desc_file else None

        num_cols = numeric_df.columns.tolist()

        # Уничтожаем предыдущее окно, если оно есть
        if self.graph_window:
            self.graph_window.close()
            self.graph_window = None

        # Создаём новое окно
        graph_window = QWidget()
        graph_window.setWindowTitle("Матрица корреляций с описанием параметров")
        graph_window.resize(1100, 700)

        # Разделитель
        splitter = QSplitter(Qt.Horizontal)

        # Создаём график
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
        ax.set_title('Матрица корреляций', fontsize=16)
        plt.tight_layout()

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        canvas = FigureCanvasQTAgg(fig)

        # Правая панель — описание
        desc_widget = QWidget()
        desc_layout = QVBoxLayout()
        desc_label_title = QLabel("<b>Описание параметров:</b>")
        desc_label_title.setFont(QFont("Arial", 12, QFont.Bold))
        desc_layout.addWidget(desc_label_title)

        if descriptions:
            for col in num_cols:
                if col in descriptions:
                    label = QLabel(f"<b>{col}:</b> {descriptions[col]}")
                    label.setWordWrap(True)
                    label.setTextFormat(Qt.RichText)
                    desc_layout.addWidget(label)
                else:
                    label = QLabel(f"{col}: — описание отсутствует")
                    label.setStyleSheet("color: gray;")
                    desc_layout.addWidget(label)
        else:
            no_desc = QLabel("Описание не добавлено")
            no_desc.setStyleSheet("color: red; font-style: italic;")
            desc_layout.addWidget(no_desc)

        # Подсказка по интерпретации
        info_label = QLabel('''
            <b>Интерпретация корреляции:</b><br><br>
            • <b>Близко к 0</b>: признаки практически не связаны.<br>
            • <b>+0.8 и выше</b>: сильная прямая зависимость, возможно, один из признаков избыточен.<br>
            • <b>-0.8 и ниже</b>: сильная обратная зависимость.
        ''')
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background-color: #f0f8ff; padding: 10px; border-radius: 5px;")
        desc_layout.addWidget(info_label)

        desc_layout.addStretch()
        desc_widget.setLayout(desc_layout)

        # Добавляем на splitter
        splitter.addWidget(canvas)
        splitter.addWidget(desc_widget)
        splitter.setSizes([700, 400])

        # Макет окна
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        graph_window.setLayout(main_layout)

        # Сохраняем ссылки
        self.graph_window = graph_window
        self.canvas = canvas

        # Привязываем закрытие окна к очистке
        self.graph_window.setAttribute(Qt.WA_DeleteOnClose)
        self.graph_window.destroyed.connect(self.on_graph_window_closed)

        graph_window.show()

    def on_graph_window_closed(self):
        """Вызывается при уничтожении окна графика"""
        self.canvas = None
        plt.close('all')  # Закрываем все фигуры matplotlib
        gc.collect()
        
    def closeEvent(self, event):
        """Очистка при закрытии основного окна"""
        # Закрываем окно графика, если оно существует и ещё не удалено
        if self.graph_window is not None:
            try:
                # Проверим, можно ли ещё вызывать методы
                if not self.isAncestorOf(self.graph_window): 
                    pass
                else:
                    self.graph_window.close()
            except RuntimeError:
                # Объект уже удалён — игнорируем
                pass
            finally:
                self.graph_window = None 

        # Явно закрываем все matplotlib-окна
        plt.close('all')

        # Очищаем данные
        self.df = None
        self.canvas = None

        # Сборка мусора
        import gc
        gc.collect()

        super().closeEvent(event)


    def saveProcessedData(self):
        if not hasattr(self, 'df'):
            self.label_dataset_status.setText('Нет загруженного датасета для сохранения.')
            return

        try:
            base_name = "correlation_processed"
            if self._last_loaded_path:
                name = os.path.splitext(os.path.basename(self._last_loaded_path))[0]
                base_name = name.split("_v")[0]

            save_path = os.path.join("dataset", f"{base_name}_v{self.meta_tracker.version}.csv")

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
