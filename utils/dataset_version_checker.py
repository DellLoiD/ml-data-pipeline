# utils/dataset_version_checker.py
import re
import os
from PySide6.QtWidgets import QMessageBox


def extract_version(filename):
    """
    Извлекает номер версии из имени файла по шаблону: ..._vN.csv
    Например:
        "diabetes_train_v3.csv" → 3
        "data_v5.csv" → 5
        "data.csv" → None

    Параметры:
        filename (str): Имя файла или полный путь

    Возвращает:
        int: Номер версии, если найден
        None: Если версия не найдена
    """
    if not filename:
        return None
    # Ищем шаблон: _v<число>.csv в конце строки
    match = re.search(r'_v(\d+)\.csv$', str(filename))
    return int(match.group(1)) if match else None


def check_train_test_versions(train_path, test_path, parent=None):
    """
    Проверяет, что train и test имеют одинаковую версию (vN).
    Также проверяет совпадение базового имени (без _train/_test).

    Параметры:
        train_path (str): Путь к train-файлу
        test_path (str): Путь к test-файлу
        parent (QWidget): Родительский виджет для отображения QMessageBox

    Возвращает:
        tuple: (is_valid: bool, version: int or None)
               Если is_valid == False, дальнейшее обучение следует заблокировать
               Если пользователь игнорирует предупреждение — вернёт (True, version)
    """
    train_version = extract_version(train_path)
    test_version = extract_version(test_path)

    # --- Проверка: найдены ли версии ---
    if train_version is None and test_version is None:
        _show_warning(
            parent,
            "Предупреждение",
            "Не удалось найти номер версии в именах файлов:\n"
            f"train: {os.path.basename(train_path) if train_path else 'отсутствует'}\n"
            f"test:  {os.path.basename(test_path) if test_path else 'отсутствует'}\n\n"
            "Возможно, файлы не были обработаны через систему балансировки/обрезки.\n"
            "Продолжение возможно, но рискуете использовать несогласованные данные."
        )
        return True, None  # Разрешаем продолжить, но с осторожностью

    if train_version is None:
        _show_error(
            parent,
            "Ошибка версии",
            f"Файл train не содержит номер версии:\n{os.path.basename(train_path)}\n\n"
            "Ожидается формат: имя_vN.csv\n"
            "Загрузите файл, сохранённый через систему обработки."
        )
        return False, None

    if test_version is None:
        _show_error(
            parent,
            "Ошибка версии",
            f"Файл test не содержит номер версии:\n{os.path.basename(test_path)}\n\n"
            "Ожидается формат: имя_vN.csv\n"
            "Загрузите файл, сохранённый через систему обработки."
        )
        return False, None

    # --- Проверка: совпадают ли версии ---
    if train_version != test_version:
        msg = (
            f"❗ Предупреждение: версии train и test не совпадают!\n\n"
            f"• train: v{train_version}\n"
            f"• test:  v{test_version}\n\n"
            "Это может привести к:\n"
            "• Обучению на одном наборе данных\n"
            "• Тестированию на другом\n"
            "• Неверной оценке точности модели\n\n"
            "Рекомендуется использовать пару с одинаковой версией."
        )
        reply = QMessageBox.question(
            parent,
            "Несовпадение версий",
            msg,
            QMessageBox.Ignore | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        if reply == QMessageBox.Cancel:
            return False, None
        # Иначе — пользователь проигнорировал, продолжаем
        return True, train_version

    # --- Дополнительно: проверим, что имена согласованы ---
    train_base = _get_base_name(train_path)
    test_base = _get_base_name(test_path)
    if train_base != test_base:
        _show_warning(
            parent,
            "Предупреждение",
            f"Базовые имена файлов не совпадают:\n"
            f"• train: {train_base}\n"
            f"• test:  {test_base}\n\n"
            "Данные могут быть из разных экспериментов.\n"
            "Продолжить можно, но с осторожностью."
        )
        # Не блокируем, но предупреждаем
    else:
        # Все хорошо
        pass

    return True, train_version


def _get_base_name(filepath):
    """
    Получает базовое имя без суффиксов _train, _test и версии
    Пример:
        diabetes_train_v3.csv → diabetes
        housing_test_v5.csv → housing
    """
    if not filepath:
        return ""
    name = os.path.basename(filepath)
    name = os.path.splitext(name)[0]  # Убираем .csv
    name = re.sub(r'_v\d+$', '', name)  # Убираем _v3
    name = re.sub(r'_train$', '', name)
    name = re.sub(r'_test$', '', name)
    return name


def _show_warning(parent, title, text):
    """Выводит предупреждение"""
    QMessageBox.warning(parent, title, text)


def _show_error(parent, title, text):
    """Выводит ошибку"""
    QMessageBox.critical(parent, title, text)


# === Пример использования (для тестирования) ===
if __name__ == "__main__":
    # Тестирование
    print(extract_version("data_train_v3.csv"))  # → 3
    print(extract_version("data_v5.csv"))        # → 5
    print(extract_version("data.csv"))           # → None
    print(_get_base_name("patient_data_train_v2.csv"))  # → patient_data

    # Проверка пары
    valid, version = check_train_test_versions(
        "data/diabetes_train_v3.csv",
        "data/diabetes_test_v3.csv"
    )
    print("Valid:", valid, "Version:", version)  # → True, 3
