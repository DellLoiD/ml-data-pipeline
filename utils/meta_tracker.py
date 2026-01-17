# utils/meta_tracker.py

import os
import re
from typing import Optional


class MetaTracker:
    """
    Универсальный класс для управления # META: строками.
    Поддерживает:
      - Накопление изменений
      - Версионирование (v1, v2, ...)
      - Разбивку на строки по max_line_length без потери данных
    """

    def __init__(self, max_line_length=150):
        self.max_line_length = max_line_length
        self.changes = []  # Новые изменения (ещё не версионированные)
        self.version = 1   # Следующая версия при сохранении
        self.original_path = None
        self._loaded_from_version = 0

    def load_from_file(self, filepath: str):
        """Загружает существующую историю из файла и определяет следующую версию"""
        self.changes = []
        self.original_path = filepath
        self._loaded_from_version = 0

        raw_lines = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("# META:"):
                        raw_lines.append(stripped)
                    elif stripped and not stripped.startswith("#"):
                        break  # Первые данные — останавливаемся
        except Exception:
            pass  # Игнорируем ошибки — начинаем с чистого листа

        # Собираем все # META: строки в одну
        full_text = " ".join(raw_lines).replace("# META:", "", 1).strip()
        if full_text.startswith(","):
            full_text = full_text[1:].strip()

        # Разделяем на части по запятым
        parts = [p.strip() for p in full_text.split(",") if p.strip()]

        # Извлекаем существующие версии и определяем номер последней
        for part in parts:
            part = part.strip()
            if part.startswith("v") and re.match(r"v\d+ ", part):
                match = re.search(r"v(\d+)", part)
                if match:
                    ver = int(match.group(1))
                    self._loaded_from_version = max(self._loaded_from_version, ver)

        # Следующая версия — последняя + 1
        self.version = self._loaded_from_version + 1

        # "Сырые" изменения (без версий) — не нужно сохранять, они уже в файле
        # Все новые изменения будут добавлены через add_change()
        self.changes = []

    def add_change(self, change_description: str):
        """Добавляет новое изменение (будет сохранено как часть текущей версии)"""
        cleaned = change_description.strip()
        if cleaned and cleaned not in self.changes:
            self.changes.append(cleaned)

    def get_version_label(self) -> str:
        """Возвращает текущую версию: v1, v2..."""
        return f"v{self.version}"

    def get_meta_lines(self) -> list:
        """
        Генерирует список строк # META:,
        разбивая по max_line_length, но НЕ ОБРЕЗАЯ НИКАКИХ ДАННЫХ.
        Каждая строка начинается с # META:
        """
        # Формируем новую запись: vN действие1, действие2...
        version_tag = f"v{self.version}"
        if self.changes:
            current_summary = ", ".join(self.changes)
            new_entry = f"{version_tag} {current_summary}"
        else:
            new_entry = f"{version_tag} без изменений"

        # Извлекаем все существующие версионированные записи из исходного файла
        history = []
        raw_lines = []
        if self.original_path and os.path.exists(self.original_path):
            try:
                with open(self.original_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped.startswith("# META:"):
                            raw_lines.append(stripped)
                        elif stripped and not stripped.startswith("#"):
                            break
            except Exception:
                pass

        # Объединяем и парсим только версионированные записи
        full_meta = " ".join(raw_lines).replace("# META:", "", 1).strip()
        if full_meta.startswith(","):
            full_meta = full_meta[1:].strip()

        parts = [p.strip() for p in full_meta.split(",") if p.strip()]
        for part in parts:
            if part.startswith("v") and re.match(r"v\d+ ", part):
                history.append(part)

        # Добавляем новую запись
        history.append(new_entry)

        # Формируем строки длиной не более max_line_length
        meta_lines = []
        current_line = "# META:"

        for entry in history:
            # Формат: ", v3 изменение" или " v3 изменение"
            separator = ", " if current_line != "# META:" else " "

            # Пробуем добавить запись
            if len(current_line) + len(separator) + len(entry) <= self.max_line_length:
                current_line += separator + entry
            else:
                # Текущая строка переполнена — сохраняем и начинаем новую
                if current_line != "# META:":
                    meta_lines.append(current_line)
                # Начинаем новую строку только с этой записи
                if len("# META: " + entry) <= self.max_line_length:
                    current_line = "# META: " + entry
                else:
                    # Даже одна запись слишком длинная — разбить нельзя, но мы НЕ ОБРЕЗАЕМ!
                    # Продолжаем с переполнением (редкий случай, но безопаснее)
                    current_line = "# META: " + entry

        # Добавляем последнюю строку
        if current_line != "# META:":
            meta_lines.append(current_line)

        return meta_lines

    def save_to_file(self, file_path, df, preserve_version=False):
        try:
            # Добавляем мета-информацию в начало
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in self.get_meta_lines():
                    f.write(line + '\n')

            # Добавляем данные
            df.to_csv(file_path, mode='a', index=False, encoding='utf-8')

            if not preserve_version:
                self.version += 1
            return True
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
            return False


    def __str__(self):
        return "\n".join(self.get_meta_lines())

    def get_change_description(self, version: str) -> str:
        """
        Возвращает описание изменений для указанной версии
        (полезно для UI, например, при клике на v1, v2...)
        """
        raw_lines = []
        if self.original_path and os.path.exists(self.original_path):
            try:
                with open(self.original_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped.startswith("# META:"):
                            raw_lines.append(stripped)
                        elif stripped and not stripped.startswith("#"):
                            break
            except Exception:
                pass

        full_meta = " ".join(raw_lines).replace("# META:", "", 1).strip()
        if full_meta.startswith(","):
            full_meta = full_meta[1:].strip()

        parts = [p.strip() for p in full_meta.split(",") if p.strip()]
        for part in parts:
            if part.startswith(f"{version} "):
                return part[len(version)+1:].strip()

        return "Информация недоступна"
