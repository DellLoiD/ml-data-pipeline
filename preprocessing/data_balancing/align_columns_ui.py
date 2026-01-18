# preprocessing/data_balancing/align_columns_ui.py
import os
from PySide6.QtCore import Qt
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QTextEdit, QGroupBox, QInputDialog, QListWidget,
    QDialog, QVBoxLayout as QLayout, QDialogButtonBox, QListWidgetItem
)
from PySide6.QtGui import QFont, QColor

# Импорт нового трекера
from utils.meta_tracker import MetaTracker


class ColumnTypeMismatchDialog(QDialog):
    """Диалог для выбора колонок с несовпадающими типами"""
    def __init__(self, mismatches, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выберите колонки для выравнивания типов")
        self.resize(500, 400)

        layout = QLayout()

        info_label = QLabel("Колонки с разными типами данных:")
        info_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(info_label)

        self.list_widget = QListWidget()
        for col, ref_type, target_type in mismatches:
            item = QListWidgetItem(f"{col} | Реф: {ref_type} → Цель: {target_type}")
            item.setData(1, col)  # Храним имя колонки
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_selected_columns(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(1))
        return selected


class AlignColumnsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.reference_df = None
        self.target_df = None
        self.reference_file_name = ""
        self.target_file_name = ""
        self._last_loaded_path = None  # Для сохранения
        self.meta_tracker = MetaTracker(max_line_length=150)  # Управление историей
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # === Заголовок ===
        title = QLabel("Выравнивание порядка колонок и типов данных")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # === Описание ===
        desc = QLabel("Выберите референсный датасет (образец) и целевой, который нужно изменить.")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # === Кнопка выбора референсного датасета ===
        self.ref_btn = QPushButton("📎 Выбрать референсный датасет (образец)")
        self.ref_btn.clicked.connect(self.load_reference_dataset)
        layout.addWidget(self.ref_btn)

        # === Кнопка выбора целевого датасета ===
        self.target_btn = QPushButton("🎯 Выбрать целевой датасет (для выравнивания)")
        self.target_btn.clicked.connect(self.load_target_dataset)
        layout.addWidget(self.target_btn)

        # === Кнопка выравнивания типов ===
        self.align_types_btn = QPushButton("🔧 Сделать типы данных всех колонок идентичными")
        self.align_types_btn.clicked.connect(self.align_column_types)
        self.align_types_btn.setEnabled(False)
        layout.addWidget(self.align_types_btn)

        # === Кнопка запуска выравнивания колонок ===
        self.align_btn = QPushButton("🔄 Выровнять порядок колонок")
        self.align_btn.clicked.connect(self.align_columns)
        self.align_btn.setEnabled(False)
        layout.addWidget(self.align_btn)

        # === Область результатов ===
        results_group = QGroupBox("Результат")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Здесь появится отчёт о выравнивании...")
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # === Настройки окна ===
        self.setLayout(layout)
        self.resize(750, 600)
        self.setWindowTitle("Выравнивание колонок и типов данных")

    def load_reference_dataset(self):
        """Загрузка референсного датасета"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите референсный CSV", "./dataset/", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            self.meta_tracker.load_from_file(file_path)
            self.reference_df = pd.read_csv(file_path, comment='#')
            self.reference_file_name = os.path.basename(file_path)
            self.ref_btn.setText(f"✅ {self.reference_file_name}")

            self.results_text.setText(f"📌 Референсный датасет загружен:\n"
                                      f"• Файл: {self.reference_file_name}\n"
                                      f"• Колонки: {len(self.reference_df.columns)}\n"
                                      f"• Строки: {len(self.reference_df)}")

            self.meta_tracker.add_change("загружен референсный датасет для выравнивания")

            self.check_ready()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить референсный датасет:\n{e}")

    def load_target_dataset(self):
        """Загрузка целевого датасета"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите целевой CSV", "./dataset/", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            self.target_df = pd.read_csv(file_path, comment='#')
            self.target_file_name = os.path.basename(file_path)
            self._last_loaded_path = file_path

            self.target_btn.setText(f"✅ {self.target_file_name}")

            self.results_text.append(f"\n🎯 Целевой датасет загружен:\n"
                                     f"• Файл: {self.target_file_name}\n"
                                     f"• Колонки: {len(self.target_df.columns)}\n"
                                     f"• Строки: {len(self.target_df)}")

            self.meta_tracker.add_change("загружен целевой датасет для выравнивания")

            self.check_ready()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить целевой датасет:\n{e}")

    def check_ready(self):
        """Проверка готовности кнопок"""
        ready = self.reference_df is not None and self.target_df is not None
        self.align_btn.setEnabled(ready)
        self.align_types_btn.setEnabled(ready)

    def get_type_mismatches(self):
        """Возвращает список колонок с разными типами (col, ref_type, target_type)"""
        if self.reference_df is None or self.target_df is None:
            return []

        mismatches = []
        ref_cols = set(self.reference_df.columns)
        target_cols = set(self.target_df.columns)
        common_cols = ref_cols & target_cols

        for col in common_cols:
            ref_dtype = str(self.reference_df[col].dtype)
            target_dtype = str(self.target_df[col].dtype)
            if ref_dtype != target_dtype:
                mismatches.append((col, ref_dtype, target_dtype))

        return mismatches

    def align_column_types(self):
        """Выравнивание типов данных выбранных колонок"""
        # ✅ ИСПРАВЛЕНО: проверяем, что оба датасета загружены
        if self.reference_df is None or self.target_df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите оба датасета!")
            return

        mismatches = self.get_type_mismatches()
        if not mismatches:
            QMessageBox.information(self, "Готово", "Нет колонок с разными типами данных.")
            return

        # Показываем диалог с выбором
        dialog = ColumnTypeMismatchDialog(mismatches, self)
        if dialog.exec() != QDialog.Accepted:
            return

        selected_cols = dialog.get_selected_columns()
        if not selected_cols:
            QMessageBox.information(self, "Отмена", "Не выбрано ни одной колонки.")
            return

        changes = []
        errors = []

        for col in selected_cols:
            ref_dtype = self.reference_df[col].dtype
            target_dtype = self.target_df[col].dtype

            if ref_dtype == target_dtype:
                continue

            try:
                # Особые правила для числовых типов
                if pd.api.types.is_integer_dtype(ref_dtype):
                    # Приводим к int
                    self.target_df[col] = pd.to_numeric(self.target_df[col], errors='coerce').astype('Int64')
                elif pd.api.types.is_float_dtype(ref_dtype):
                    # Приводим к float
                    self.target_df[col] = pd.to_numeric(self.target_df[col], errors='coerce').astype('float64')
                elif pd.api.types.is_bool_dtype(ref_dtype):
                    # Приводим к bool
                    self.target_df[col] = self.target_df[col].astype(bool)
                elif pd.api.types.is_datetime64_any_dtype(ref_dtype):
                    # Приводим к datetime
                    self.target_df[col] = pd.to_datetime(self.target_df[col], errors='coerce')
                else:
                    # Приводим к строке, если не получается
                    self.target_df[col] = self.target_df[col].astype(str)

                changes.append(f"• {col}: {target_dtype} → {ref_dtype}")

            except Exception as e:
                errors.append(f"{col}: {str(e)}")

        # Отчёт
        result_text = "<b>🔧 Типы данных выровнены:</b><br>"
        if changes:
            result_text += "<br>".join(changes)
            self.meta_tracker.add_change(f"выровнены типы для колонок: {', '.join(selected_cols)}")
        else:
            result_text += "Ничего не изменено."

        if errors:
            result_text += f"<br><br><b>❌ Ошибки:</b><br>" + "<br>".join([f"• {e}" for e in errors])

        self.results_text.setHtml(result_text)

        # Показываем сообщение об успехе и предлагаем сохранить
        if changes:
            reply = QMessageBox.question(
                self, "Сохранить",
                "Типы данных выровнены. Сохранить обновлённый целевой датасет?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.save_aligned_dataset()
        else:
            QMessageBox.information(self, "Готово", "Изменений не было — сохранение не требуется.")



    def align_columns(self):
        """Выравнивает порядок колонок целевого датасета по референсному"""
        if self.reference_df is None or self.target_df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите оба датасета!")
            return

        ref_cols = self.reference_df.columns.tolist()
        target_cols = self.target_df.columns.tolist()

        missing_in_target = [col for col in ref_cols if col not in target_cols]
        extra_in_target = [col for col in target_cols if col not in ref_cols]

        if missing_in_target:
            QMessageBox.critical(
                self, "Ошибка",
                f"В целевом датасете отсутствуют колонки:\n" + ", ".join(missing_in_target) +
                "\n\nВыравнивание невозможно."
            )
            return

        if extra_in_target:
            reply = QMessageBox.question(
                self, "Лишние колонки",
                f"В целевом датасете есть лишние колонки:\n" + ", ".join(extra_in_target) +
                "\n\nОставить их или удалить при выравнивании?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.target_df = self.target_df[ref_cols]
                dropped_count = len(extra_in_target)
                self.meta_tracker.add_change(f"удалены лишние колонки: {', '.join(extra_in_target)}")
            else:
                ordered_cols = [col for col in ref_cols if col in target_cols] + \
                               [col for col in target_cols if col not in ref_cols]
                self.target_df = self.target_df[ordered_cols]
                dropped_count = 0
                self.meta_tracker.add_change(f"лишние колонки сохранены, но перемещены в конец")
        else:
            self.target_df = self.target_df[ref_cols]
            dropped_count = 0
            self.meta_tracker.add_change("выровнен порядок колонок по референсному датасету")

        result_text = f"""
        <b>✅ Выравнивание выполнено!</b><br><br>
        • Референсный датасет: <b>{self.reference_file_name}</b><br>
        • Целевой датасет: <b>{self.target_file_name}</b><br>
        • Колонки приведены к порядку референсного<br>
        • Количество колонок: {len(ref_cols)}<br>
        """
        if dropped_count:
            result_text += f"• Удалено лишних колонок: <b>{dropped_count}</b><br>"

        result_text += "<br><b>Первые 5 колонок после выравнивания:</b><br>"
        result_text += "<pre>" + " → ".join(ref_cols[:5]) + ("..." if len(ref_cols) > 5 else "") + "</pre>"

        self.results_text.setHtml(result_text)
        self.ask_save_aligned_dataset()

    def ask_save_aligned_dataset(self):
        """Спрашивает, сохранить ли выровненный датасет"""
        reply = QMessageBox.question(
            self, "Сохранить",
            "Выравнивание завершено. Сохранить отредактированный датасет?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.save_aligned_dataset()

    def save_aligned_dataset(self):
        """Сохраняет выровненный датасет с использованием MetaTracker"""
        if self.target_df is None:
            return

        base_name = "aligned_dataset"
        if self._last_loaded_path:
            name = os.path.splitext(os.path.basename(self._last_loaded_path))[0]
            base_name = name.split("_v")[0]

        save_path = os.path.join("dataset", f"{base_name}_v{self.meta_tracker.version}.csv")

        try:
            success = self.meta_tracker.save_to_file(save_path, self.target_df)
            if success:
                self._last_loaded_path = save_path
                self.meta_tracker.version += 1
                QMessageBox.information(
                    self, "Сохранено",
                    f"✅ Датасет сохранён:\n{os.path.basename(save_path)}\n\n"
                    f"Версия: v{self.meta_tracker.version - 1}"
                )
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")
