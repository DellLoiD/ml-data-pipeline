# utils/meta_tracker.py

import os
import re
from typing import List, Optional


class MetaTracker:
    """
    Универсальный класс для управления # META: строками.
    Поддерживает:
      - Накопление изменений
      - Версионирование (v1, v2, ...)
      - Одна строка — одна версия: # META: v1: ..., # META: v2: ...
      - Полная обратная совместимость со старыми форматами
      - Не обрезает длинные строки — сохраняет всё
    """

    def __init__(self, max_line_length: Optional[int] = 150):
        """
        Добавлен max_line_length для обратной совместимости.
        Параметр игнорируется, но не вызывает ошибку.
        """
        self.changes: List[str] = []
        self.version: int = 1
        self.original_path: Optional[str] = None
        self._loaded_from_version: int = 0

    def load_from_file(self, filepath: str) -> None:
        """
        Загружает историю из файла. Поддерживает старый и новый форматы:
          - # META: v1 ...
          - # META: v2: ...
        """
        self.changes = []
        self.original_path = filepath
        self._loaded_from_version = 0

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("# META:"):
                        # Ищем vN — независимо от наличия ":"
                        match = re.search(r"v(\d+)", stripped)
                        if match:
                            ver = int(match.group(1))
                            self._loaded_from_version = max(self._loaded_from_version, ver)
                    elif stripped and not stripped.startswith("#"):
                        break  # Дошли до данных
        except Exception as e:
            print(f"Не удалось прочитать файл метаданных {filepath}: {e}")

        self.version = self._loaded_from_version + 1

    def add_change(self, change_description: str) -> None:
        """Добавляет изменение для текущей версии"""
        cleaned = change_description.strip()
        if cleaned and cleaned not in self.changes:
            self.changes.append(cleaned)

    def get_version_label(self) -> str:
        """Возвращает метку текущей версии"""
        return f"v{self.version}"

    def get_meta_lines(self) -> List[str]:
        """
        Генерирует по одной строке на версию.
        Поддерживает загрузку из старого формата, но сохраняет в новом:
          # META: v1: изменение 1, изменение 2
        """
        lines = []

        # === 1. Загружаем старые # META: строки (все версии) ===
        if self.original_path and os.path.exists(self.original_path):
            try:
                with open(self.original_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped.startswith("# META:"):
                            # Сохраняем только строки, содержащие версию
                            if re.search(r"v\d+", stripped):
                                # Приводим к новому формату, если нужно
                                new_line = self._normalize_meta_line(stripped)
                                if new_line:
                                    lines.append(new_line)
                        elif stripped and not stripped.startswith("#"):
                            break
            except Exception as e:
                print(f"Ошибка чтения оригинального файла: {e}")

        # === 2. Добавляем новую версию ===
        version_tag = f"v{self.version}"
        if self.changes:
            changes_str = ", ".join(self.changes)
            new_line = f"# META: {version_tag}: {changes_str}"
        else:
            new_line = f"# META: {version_tag}: без изменений"

        lines.append(new_line)
        return lines

    def _normalize_meta_line(self, line: str) -> str:
        """
        Приводит старую строку к новому формату.
        Помогает поддерживать единообразие.
        """
        # Убираем # META:
        content = line.replace("# META:", "", 1).strip()

        # Ищем vN
        match = re.search(r"v\d+", content)
        if not match:
            return ""

        ver = match.group()
        # Убираем саму метку версии из контента
        rest = content[len(ver):].strip()
        # Убираем лишние разделители в начале
        if rest.startswith(":"):
            rest = rest[1:].strip()
        if rest.startswith(",") or rest.startswith("-"):
            rest = rest[1:].strip()

        # Формируем единый формат
        description = rest.strip()
        if not description:
            description = "без изменений"

        return f"# META: {ver}: {description}"

    def save_to_file(self, file_path: str, df, preserve_version: bool = False) -> bool:
        """
        Сохраняет метаданные и данные.
        Полностью совместим с предыдущими версиями.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                meta_lines = self.get_meta_lines()
                for line in meta_lines:
                    f.write(line + '\n')  # Не режем, не меняем

            # Добавляем данные
            df.to_csv(file_path, mode='a', index=False, encoding='utf-8')

            if not preserve_version:
                self.version += 1

            return True

        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
            return False

    def __str__(self) -> str:
        return "\n".join(self.get_meta_lines())

    def get_change_description(self, version: str) -> str:
        """
        Возвращает описание для версии (v1, v2...).
        Поддерживает оба формата: с ":" и без.
        """
        # Проверяем оригинальный файл
        if self.original_path and os.path.exists(self.original_path):
            try:
                with open(self.original_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped.startswith("# META:"):
                            continue
                        if re.search(rf"{re.escape(version)}\s*:", stripped) or \
                           (re.search(rf"{re.escape(version)}\b", stripped) and not re.search(rf"{re.escape(version)}\d", stripped)):
                            # Удаляем # META: и версию
                            start = stripped.find(version)
                            if start != -1:
                                desc_part = stripped[start + len(version):].strip()
                                # Убираем : или пробел
                                if desc_part.startswith(":"):
                                    desc_part = desc_part[1:].strip()
                                return desc_part or "без изменений"
            except Exception as e:
                print(f"Ошибка при чтении описания версии: {e}")

        # Проверяем текущую (ещё не сохранённую) версию
        if version == f"v{self.version}" and self.changes:
            return ", ".join(self.changes)

        return "Информация недоступна"
