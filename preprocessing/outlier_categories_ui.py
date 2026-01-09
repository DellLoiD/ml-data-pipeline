# preprocessing/outlier_categories_ui.py
import os
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QLineEdit, QTextEdit, QMessageBox, QGroupBox, QInputDialog
)


class OutlierCategoriesApp(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.dataset_file_name = ""
        self.original_file_path = ""
        self._version = 1   
        self._meta_data = {} 
        self._pending_changes = [] 
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.load_btn = QPushButton("📁 Загрузить датасет")
        self.load_btn.clicked.connect(self.load_dataset)
        layout.addWidget(self.load_btn)
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Столбец для анализа:"))
        self.category_combo = QComboBox()
        self.category_combo.setEnabled(False)
        self.category_combo.setPlaceholderText("Выберите столбец")
        category_layout.addWidget(self.category_combo)
        layout.addLayout(category_layout)
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Фильтр по значению (от):"))
        self.min_val_input = QLineEdit()
        self.min_val_input.setPlaceholderText("мин, напр. 1800")
        self.min_val_input.setFixedWidth(100)
        self.min_val_input.setEnabled(False)
        range_layout.addWidget(self.min_val_input)

        range_layout.addWidget(QLabel("до:"))
        self.max_val_input = QLineEdit()
        self.max_val_input.setPlaceholderText("макс, напр. 1950")
        self.max_val_input.setFixedWidth(100)
        self.max_val_input.setEnabled(False)
        range_layout.addWidget(self.max_val_input)
        layout.addLayout(range_layout)
        n_layout = QHBoxLayout()
        n_layout.addWidget(QLabel("Макс. кол-во записей (N):"))
        self.n_input = QLineEdit("5")
        self.n_input.setPlaceholderText("Напр.: 5")
        self.n_input.setFixedWidth(100)
        n_layout.addWidget(self.n_input)
        layout.addLayout(n_layout)
        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("🔍 Найти редкие значения")
        self.analyze_btn.clicked.connect(self.analyze_rare_classes)
        self.analyze_btn.setEnabled(False)
        btn_layout.addWidget(self.analyze_btn)

        self.merge_btn = QPushButton("🔗 Объединить в класс...")
        self.merge_btn.clicked.connect(self.merge_interval_values)
        self.merge_btn.setEnabled(False)
        btn_layout.addWidget(self.merge_btn)
        layout.addLayout(btn_layout)
        results_group = QGroupBox("Редкие значения (количество ≤ N)")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("После анализа здесь появятся результаты...")
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        self.save_btn = QPushButton("💾 Сохранить датасет")
        self.save_btn.clicked.connect(self.save_dataset)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)
        self.resize(650, 700)
        self.setWindowTitle("Анализ редких классов")
        self.show()

    def load_dataset(self):
        """Загрузка датасета + парсинг истории # META:"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV файл", "./dataset/", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                meta_lines = []
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("# META:"):
                        meta_lines.append(stripped)
                    elif stripped and not stripped.startswith("#"):
                        break

            self._meta_data = {}
            for line in meta_lines:
                line = line.replace("# META:", "").strip()
                parts = line.split("|")
                for part in parts:
                    part = part.strip()
                    if part.startswith("v"):
                        version_part = part.split(maxsplit=1)
                        if len(version_part) == 2:
                            ver = version_part[0]
                            changes = version_part[1]
                            self._meta_data[ver] = changes

            # Загружаем данные
            self.df = pd.read_csv(file_path, comment='#', skipinitialspace=True)
            self.dataset_file_name = os.path.basename(file_path)
            self.original_file_path = file_path
            name, ext = os.path.splitext(self.dataset_file_name)
            if "_v" in name:
                try:
                    self._version = int(name.split("_v")[1]) + 1
                except:
                    self._version = 1
            else:
                self._version = 1

            self.load_btn.setText(f"✅ {self.dataset_file_name}")
            self.category_combo.clear()
            all_columns = self.df.columns.tolist()
            if not all_columns:
                QMessageBox.warning(self, "Ошибка", "Файл пустой — нет столбцов.")
                return

            self.category_combo.addItems(all_columns)
            self.category_combo.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.merge_btn.setEnabled(True)
            self.save_btn.setEnabled(False)

            self.category_combo.currentTextChanged.connect(self.on_column_changed)

            QMessageBox.information(self, "Успех", f"Датасет '{self.dataset_file_name}' загружен!\n"
                                                  f"Доступно столбцов: {len(all_columns)}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def on_column_changed(self, column_name):
        """Включает/выключает поля интервала в зависимости от типа столбца"""
        if not column_name or column_name not in self.df.columns:
            return

        is_numeric = pd.api.types.is_numeric_dtype(self.df[column_name])
        self.min_val_input.setEnabled(is_numeric)
        self.max_val_input.setEnabled(is_numeric)

        if not is_numeric:
            self.min_val_input.clear()
            self.max_val_input.clear()

    def analyze_rare_classes(self):
        """Поиск редких значений с фильтрацией по интервалу"""
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет!")
            return

        column_name = self.category_combo.currentText()
        if not column_name or column_name not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", "Выберите корректный столбец!")
            return
        is_numeric = pd.api.types.is_numeric_dtype(self.df[column_name])
        min_val, max_val = None, None
        use_range = False

        if is_numeric:
            min_text = self.min_val_input.text().strip()
            max_text = self.max_val_input.text().strip()
            if min_text or max_text:
                try:
                    min_val = float(min_text) if min_text else None
                    max_val = float(max_text) if max_text else None
                    use_range = True
                except ValueError:
                    QMessageBox.warning(self, "Ошибка", "Введите корректные числа в поля 'от' и 'до'.")
                    return
        if use_range and is_numeric:
            mask = True
            if min_val is not None:
                mask &= (self.df[column_name] >= min_val)
            if max_val is not None:
                mask &= (self.df[column_name] <= max_val)
            filtered_series = self.df[column_name][mask]
        else:
            filtered_series = self.df[column_name]
        try:
            n = int(self.n_input.text().strip())
            if n < 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректное положительное число в поле N.")
            return

        value_counts = filtered_series.value_counts(dropna=False).sort_index()
        rare_values = value_counts[value_counts <= n]
        if rare_values.empty:
            result_text = f"✅ Нет значений с количеством записей ≤ {n}."
            if use_range:
                result_text += f"<br><i>(в диапазоне от {min_val} до {max_val})</i>"
        else:
            count_rare = len(rare_values)
            result_text = f"🔍 Найдено <b>{count_rare}</b> редких значений "
            if use_range:
                result_text += f"(в диапазоне от {min_val} до {max_val})"
            result_text += f" (≤ {n} записей):\n\n"
            result_text += "<pre>Значение → Количество</pre>\n"
            result_text += "<pre>" + "-" * 40 + "</pre>\n"
            for value, count in rare_values.items():
                val_str = "(пусто)" if pd.isna(value) else str(value)
                val_str = val_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                result_text += f"<pre>{val_str:<25} → {count:>8}</pre>\n"

        self.results_text.setHtml(result_text)
        self.results_text.append("")
        total_filtered = len(filtered_series)
        total_unique = len(value_counts)
        summary = (
            f"<hr>"
            f"<b>📊 Сводка по '{column_name}':</b><br>"
            f"• Всего записей (после фильтра): {total_filtered}<br>"
            f"• Уникальных значений: {total_unique}<br>"
            f"• Мин. частота: {value_counts.min() if len(value_counts) > 0 else 0}<br>"
            f"• Макс. частота: {value_counts.max() if len(value_counts) > 0 else 0}"
        )
        self.results_text.append(summary)

    def merge_interval_values(self):
        """Объединяет значения в указанном интервале в одно значение"""
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите датасет!")
            return

        column_name = self.category_combo.currentText()
        if not column_name or column_name not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", "Выберите корректный столбец!")
            return

        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            QMessageBox.warning(self, "Ошибка", f"Столбец '{column_name}' должен быть числовым для объединения.")
            return
        min_text = self.min_val_input.text().strip()
        max_text = self.max_val_input.text().strip()

        if not min_text or not max_text:
            QMessageBox.warning(self, "Ошибка", "Введите оба значения: 'от' и 'до'.")
            return

        try:
            min_val = float(min_text)
            max_val = float(max_text)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректные числа в поля 'от' и 'до'.")
            return

        if min_val > max_val:
            QMessageBox.warning(self, "Ошибка", "Значение 'от' не может быть больше 'до'.")
            return
        target_val, ok = QInputDialog.getDouble(
            self,
            "Объединение значений",
            f"В какое значение объединить все записи\nв диапазоне [{min_val}, {max_val}]?",
            decimals=0 if self.df[column_name].dtype == 'int64' else 2,
            value=min_val
        )
        if not ok:
            return
        if target_val < -1e10 or target_val > 1e10:
            QMessageBox.warning(self, "Ошибка", "Значение вне допустимого диапазона.")
            return
        mask = (self.df[column_name] >= min_val) & (self.df[column_name] <= max_val)
        count = mask.sum()
        if count == 0:
            QMessageBox.information(self, "Нет данных", "Нет записей в указанном диапазоне.")
            return

        self.df.loc[mask, column_name] = target_val
        change_text = f"объединены значения в '{column_name}' от {min_val} до {max_val} в {target_val}"
        self._pending_changes.append(change_text)
        self.save_btn.setEnabled(True)
        QMessageBox.information(
            self, "Успешно", f"✅ {count} записей в столбце '{column_name}'\n"
                             f"в диапазоне [{min_val}, {max_val}]\n"
                             f"объединены в значение: <b>{target_val}</b>"
        )
        self.analyze_rare_classes()

    def save_dataset(self):
        """Сохранение датасета с обновлением истории # META:"""
        if self.df is None or self.original_file_path is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения!")
            return
        file_name = os.path.splitext(os.path.basename(self.original_file_path))[0]
        base_name = file_name.split("_v")[0] if "_v" in file_name else file_name
        save_path = os.path.join("dataset", f"{base_name}_v{self._version}.csv")

        try:
            current_version = f"v{self._version}"
            if self._pending_changes:
                self._meta_data[current_version] = ", ".join(self._pending_changes)
            else:
                if current_version not in self._meta_data:
                    self._meta_data[current_version] = "без изменений"
            parts = []
            for ver in sorted(self._meta_data.keys(), key=lambda x: int(x[1:])):
                changes = self._meta_data[ver]
                parts.append(f"{ver} {changes}")
            full_content = "|".join(parts)
            full_line = f"# META: {full_content}"
            max_len = 150
            meta_lines = []
            if len(full_line) <= max_len:
                meta_lines.append(full_line)
            else:
                words = full_content.split("|")
                current = "# META:"
                for word in words:
                    test = current + ("|" if current != "# META:" else " ") + word
                    if len(test) <= max_len:
                        if current == "# META:":
                            current = f"# META: {word}"
                        else:
                            current += "|" + word
                    else:
                        if current != "# META:":
                            meta_lines.append(current)
                        current = f"# META: {word}"
                if current != "# META:":
                    meta_lines.append(current)
            with open(save_path, "w", encoding="utf-8") as f:
                for line in meta_lines:
                    f.write(line + "\n")
                self.df.to_csv(f, index=False)
            self._pending_changes.clear()
            self.original_file_path = save_path
            self._version += 1
            self.save_btn.setEnabled(False)
            QMessageBox.information(
                self, "Сохранено", f"✅ Датасет сохранён:\n{os.path.basename(save_path)}\n\n"
                                   f"Теперь версия: v{self._version - 1}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")
