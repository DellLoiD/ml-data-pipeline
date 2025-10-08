import pandas as pd
from PySide6.QtWidgets import (QFileDialog, QMessageBox, QInputDialog)
from sklearn.model_selection import train_test_split
from preprocessing.data_balancing.data_balancing_list_method_ui import BalancingMethodsWindow
import os

def load_dataset(self):
    file_dialog = QFileDialog()
    file_dialog.setDirectory('./dataset')
    file_name, _ = file_dialog.getOpenFileName(self, "Выбор датасета", "", "CSV Files (*.csv)")
        
    if file_name:
        df = pd.read_csv(file_name)            
        # Запоминаем полное имя файла
        self.dataset_filename = file_name            
        # Числовые колонки считаем признаками
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()            
        # Диалог выбора целевой переменной
        item, ok = QInputDialog.getItem(self, "Выбор целевой переменной", "Выберите целевую переменную:", numeric_columns, editable=False)
        if ok and item:
            target_col = item
            feature_cols = list(set(numeric_columns) - set([item]))  # Остальные колонки становятся признаками
            self.feature_cols = feature_cols                
            # Присваиваем атрибуту target_col значение выбранной переменной
            self.target_col = target_col                
            self.X = df[self.feature_cols].values
            self.y = df[target_col].values                
            # Тренировочная/тестовая выборка
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            # Информация до балансировки
            self.before_label.setText(f"До балансировки:\n{pd.Series(self.y_train).value_counts()}")
            self.after_label.clear()
        else:
            QMessageBox.warning(self, "Предупреждение", "Необходимо выбрать целевую переменную!")      
def update_class_distribution(self, distribution_text):
    """Метод обновления текста метки после завершения балансировки"""
    self.after_label.setText(distribution_text)

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