# preprocessing/data_balancing/align_columns_ui.py
import os
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QTextEdit, QGroupBox
)
from PySide6.QtGui import QFont


class AlignColumnsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.reference_df = None
        self.target_df = None
        self.reference_file_name = ""
        self.target_file_name = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # === Заголовок ===
        title = QLabel("Выравнивание порядка колонок в датасетах")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # === Описание ===
        desc = QLabel("Выберите референсный датасет (образец порядка колонок) и целевой датасет, который нужно изменить.")
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

        # === Кнопка запуска ===
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
        self.resize(700, 500)
        self.setWindowTitle("Выравнивание колонок датасетов")
        self.show()

    def load_reference_dataset(self):
        """Загрузка референсного датасета (образец порядка)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите референсный CSV", "./dataset/", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            self.reference_df = pd.read_csv(file_path)
            self.reference_file_name = os.path.basename(file_path)
            self.ref_btn.setText(f"✅ {self.reference_file_name}")

            self.results_text.setText(f"📌 Референсный датасет загружен:\n"
                                      f"• Файл: {self.reference_file_name}\n"
                                      f"• Колонки: {len(self.reference_df.columns)}\n"
                                      f"• Строки: {len(self.reference_df)}")

            self.check_alignment_ready()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить референсный датасет:\n{e}")

    def load_target_dataset(self):
        """Загрузка целевого датасета (который нужно выровнять)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите целевой CSV", "./dataset/", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            self.target_df = pd.read_csv(file_path)
            self.target_file_name = os.path.basename(file_path)
            self.target_btn.setText(f"✅ {self.target_file_name}")

            self.results_text.append(f"\n🎯 Целевой датасет загружен:\n"
                                     f"• Файл: {self.target_file_name}\n"
                                     f"• Колонки: {len(self.target_df.columns)}\n"
                                     f"• Строки: {len(self.target_df)}")

            self.check_alignment_ready()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить целевой датасет:\n{e}")

    def check_alignment_ready(self):
        """Проверяет, можно ли запустить выравнивание"""
        ready = self.reference_df is not None and self.target_df is not None
        self.align_btn.setEnabled(ready)

    def align_columns(self):
        """Выравнивает порядок колонок целевого датасета по референсному"""
        if self.reference_df is None or self.target_df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите оба датасета!")
            return

        ref_cols = self.reference_df.columns.tolist()
        target_cols = self.target_df.columns.tolist()

        # Проверка, все ли колонки из референса есть в целевом
        missing_in_target = [col for col in ref_cols if col not in target_cols]
        extra_in_target = [col for col in target_cols if col not in ref_cols]

        if missing_in_target:
            QMessageBox.critical(
                self, "Ошибка",
                f"В целевом датасете отсутствуют колонки:\n" + ", ".join(missing_in_target) +
                "\n\nВыравнивание невозможно."
            )
            return

        # Логируем предупреждение, если есть лишние колонки
        if extra_in_target:
            reply = QMessageBox.question(
                self, "Лишние колонки",
                f"В целевом датасете есть лишние колонки:\n" + ", ".join(extra_in_target) +
                "\n\nОставить их или удалить при выравнивании?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                # Удаляем лишние
                self.target_df = self.target_df[ref_cols]
                dropped_count = len(extra_in_target)
            else:
                # Оставляем, но в нужном порядке + оставшиеся
                ordered_cols = [col for col in ref_cols if col in target_cols] + \
                               [col for col in target_cols if col not in ref_cols]
                self.target_df = self.target_df[ordered_cols]
                dropped_count = 0
        else:
            # Просто выравниваем порядок
            self.target_df = self.target_df[ref_cols]
            dropped_count = 0

        # Отчёт
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

        # Спрашиваем о сохранении
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
        """Сохраняет выровненный датасет"""
        if self.target_df is None:
            return

        default_name = f"aligned_{self.target_file_name}"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить выровненный датасет",
            f"./dataset/{default_name}",
            "CSV Files (*.csv)"
        )
        if not save_path:
            return

        try:
            self.target_df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Сохранено", f"✅ Датасет сохранён:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")
