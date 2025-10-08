import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from PySide6.QtWidgets import QMessageBox, QInputDialog, QFileDialog
import numpy as np
import os

class DatasetTrimLogic:
    def __init__(self, parent=None):
        
        self.dataset_filename = ''
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_resampled = None
        self.y_resampled = None
        self.feature_cols = []
        self.target_col = ''
        self.parent_widget = parent


    def load_dataset(self, ui):        
        file_dialog = QFileDialog()
        file_dialog.setDirectory('./dataset')
        file_name, _ = file_dialog.getOpenFileName(ui, "Выбор датасета", "", "CSV Files (*.csv)")
        if file_name:
            df = pd.read_csv(file_name)
            self.dataset_filename = file_name
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            item, ok = QInputDialog.getItem(ui, "Выбор целевой переменной", "Выберите целевую переменную:", numeric_columns, editable=False)
            if ok and item:
                target_col = item
                feature_cols = list(set(numeric_columns) - set([item]))
                self.feature_cols = feature_cols
                self.target_col = target_col
                self.X = df[self.feature_cols].values
                self.y = df[target_col].values
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                before_stats = pd.Series(self.y_train).value_counts().to_string()
                ui.before_label.setText(f"До балансировки:\n{before_stats}")
                ui.after_label.clear()
                ui.file_name_label.setText(f"Датасет загружен: {file_name.split('/')[-1]}")
            else:
                QMessageBox.warning(ui, "Предупреждение", "Необходимо выбрать целевую переменную!")
                
    def trim_dataset(self, target_samples):
        if self.X_train is None or self.y_train is None:
            raise Exception("Сначала загрузите датасет.")

        current_samples = pd.Series(self.y_train).value_counts()
        unique_classes = len(current_samples)
        
        for i in range(unique_classes):
            class_value = current_samples.index[i]
            if target_samples > current_samples.loc[class_value]:
                raise Exception(f"({target_samples}) превышает количество {class_value}. значения: {current_samples}")

        sampling_strategy = {value: target_samples for value in current_samples.index}
        
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_trimmed, y_trimmed = sampler.fit_resample(self.X_train, self.y_train)
        
        y_trimmed = np.round(y_trimmed).astype(int)
        X_trimmed = np.round(X_trimmed).astype(int)

        after_stats = pd.Series(y_trimmed).value_counts().to_string()
    
        # Обновляем атрибуты класса новым набором данных
        self.X_resampled = X_trimmed
        self.y_resampled = y_trimmed
        
        return X_trimmed, y_trimmed
    
    def save_trimmed_dataset(self, target_samples):
        if not hasattr(self, 'X_resampled'):
            raise Exception('Обрезанный датасет ещё не создан.')
        
        try:
            # Проверяем наличие обязательных атрибутов
            required_attrs = ['feature_cols', 'target_col', 'dataset_filename']
            missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
            if missing_attrs:
                raise AttributeError(f"Атрибуты {missing_attrs} отсутствуют.")
            
            # Печать текущих значений атрибутов
            print(f"Значения атрибутов перед сохранением датасета:")
            print(f"Feature cols: {self.feature_cols}")
            print(f"Target col: {self.target_col}")
            print(f"Dataset filename: {self.dataset_filename}")
            
            # Продолжаем стандартную процедуру
            filename_base = os.path.basename(self.dataset_filename).split('.')[0]
            new_filename = f"{filename_base}_trimmed_{target_samples}.csv"
            
            # Формируем датафрейм с сохранением оригинальной структуры
            data_dict = {}
            for idx, column in enumerate(self.feature_cols):
                data_dict[column] = self.X_resampled[:, idx]
            data_dict[self.target_col] = self.y_resampled
            
            trimmed_df = pd.DataFrame(data_dict)
            
            directory = './dataset'
            full_path = os.path.join(directory, new_filename)
            
            trimmed_df.to_csv(full_path, index=False)
            
            print(f'Датасет успешно сохранён: {full_path}')
        except AttributeError as ae:
            print(f'Ошибка: {ae}')
        except FileNotFoundError:
            print("Каталог для сохранения датасетов не найден.")
        except IOError:
            print("Ошибка записи файла.")
        except Exception as e:
            print(f'Ошибка при сохранении: {e}')
           
