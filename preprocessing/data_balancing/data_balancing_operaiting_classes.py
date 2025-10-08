import os
import pandas as pd
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

class FeatureSelector(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Выбор признаков датасета")
        layout = QVBoxLayout(self)
        
        # Виджет для загрузки файла
        load_data_btn = QPushButton("Выбрать датасет")
        load_data_btn.clicked.connect(self.load_dataset)
        layout.addWidget(load_data_btn)
        
        # Комбобокс для выбора признака
        self.feature_combo_box = QComboBox()
        self.feature_combo_box.currentIndexChanged.connect(self.update_feature_info)
        layout.addWidget(self.feature_combo_box)
        
        # Информационная панель
        self.feature_info_label = QLabel("", alignment=Qt.AlignLeft)
        layout.addWidget(self.feature_info_label)
        
        # Кнопка для удаления признака
        delete_feature_btn = QPushButton("Удалить признак")
        delete_feature_btn.clicked.connect(self.delete_selected_feature)
        layout.addWidget(delete_feature_btn)
        
        # Кнопка для сохранения датасета
        save_data_btn = QPushButton("Сохранить датасет")
        save_data_btn.clicked.connect(self.save_dataset)
        layout.addWidget(save_data_btn)
        
        # Переменные для хранения данных
        self.df = None
        self.dataset_filename = ""
        self.selected_feature = None
    
    def load_dataset(self):
        file_dialog = QFileDialog()
        file_dialog.setDirectory('./dataset')
        file_name, _ = file_dialog.getOpenFileName(self, "Выбор датасета", "", "CSV Files (*.csv)")
        
        if file_name:
            self.dataset_filename = file_name
            self.df = pd.read_csv(file_name)
            
            # Определяем признаки (features)
            features = self.df.columns.tolist()
            self.feature_combo_box.clear()
            self.feature_combo_box.addItems(features)
            
            # Очищаем информацию о признаке
            self.feature_info_label.clear()
            self.selected_feature = None
    
    def update_feature_info(self):
        selected_feature = self.feature_combo_box.currentText()
        if selected_feature:
            # Группируем по выбранному признаку и считаем количество записей
            class_counts = self.df.groupby(selected_feature).size()
            
            # Ограничиваем вывод первыми 10 и последними 2 классами
            total_classes = len(class_counts)
            first_10 = class_counts.head(10)
            last_2 = class_counts.tail(2)
            
            # Объединяем обе группы и создаем финальное сообщение
            # Исключаем дублирование с помощью .drop_duplicates()
            combined_info = pd.concat([first_10, last_2], ignore_index=False).drop_duplicates()
            
            # Формируем сообщение с результатом группировки
            info = []
            separator_needed = False  # Установим True, если нужен разделитель
            
            # Проходим по всем классам в комбинированной коллекции
            for class_val, count in combined_info.items():
                info.append(f"Класс {class_val}: {count} экземпляров")
            
            # Если классов больше 10, ставим разделитель
            if total_classes > 10:
                skipped_classes = total_classes - 12  # Минус первые 10 и последние 2
                info.insert(len(first_10), f"... (ещё {skipped_classes} классов)")
            
            final_message = "\n".join(info)
            self.feature_info_label.setText(final_message)
            self.selected_feature = selected_feature
    
    def delete_selected_feature(self):
        if self.selected_feature is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите признак.")
            return
        
        # Удаляем выбранный признак из DataFrame
        del self.df[self.selected_feature]
        
        # Обновляем список признаков
        remaining_features = self.df.columns[:-1].tolist()
        self.feature_combo_box.clear()
        self.feature_combo_box.addItems(remaining_features)
        
        # Очищаем информацию о признаке
        self.feature_info_label.clear()
        self.selected_feature = None
        
        QMessageBox.information(self, "Информация", "Признак удалён из датасета.")
    
    def save_dataset(self):
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите датасет.")
            return
        
        # Генерируем новое имя файла
        basename = os.path.splitext(os.path.basename(self.dataset_filename))[0]
        new_filename = f"{basename}_cleaned.csv"
        output_path = os.path.join("./dataset", new_filename)
        
        # Сохраняем файл
        self.df.to_csv(output_path, index=False)
        QMessageBox.information(self, "Успех", f"Датасет успешно сохранён в {output_path}")

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = FeatureSelector()
    window.show()
    sys.exit(app.exec())