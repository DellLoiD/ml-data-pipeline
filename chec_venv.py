import os

file_path = './preprocessing/data_balancing/data_balancing_method_ui.py'
if os.path.exists(file_path):
    print(f"Файл {file_path} найден.")
else:
    print(f"Файл {file_path} отсутствует.")