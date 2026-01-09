# load_dataset.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
import os
import pandas as pd


class LoadDatasetWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_folder = "dataset"
        self.setup_ui()
        self.make_dataset_dir()

    def setup_ui(self):
        self.setWindowTitle("Загрузка датасета")
        self.resize(600, 400)

        layout = QVBoxLayout()

        # Заголовок
        title = QLabel("Загрузка датасета")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Кнопка загрузки
        self.load_btn = QPushButton("📂 Выбрать CSV-файл")
        self.load_btn.clicked.connect(self.load_dataset)
        self.load_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.load_btn)

        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # Информация о датасете
        self.info_label = QLabel("Датасет не загружен.")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-family: Courier; font-size: 12px;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.info_label)
        scroll.setMaximumHeight(200)
        layout.addWidget(QLabel("<b>Информация о датасете:</b>"))
        layout.addWidget(scroll)

        self.setLayout(layout)

    def make_dataset_dir(self):
        """Создаёт папку dataset, если её нет"""
        os.makedirs(self.dataset_folder, exist_ok=True)

    def load_dataset(self):
        """Открывает диалог выбора файла и загружает как {name}_v0.csv + # META: v0"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV-файл", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return  # Пользователь отменил выбор

        # Исходный путь и имя
        file_path = os.path.abspath(file_path)
        original_filename = os.path.basename(file_path)
        name, ext = os.path.splitext(original_filename)

        # ✅ Правильное имя: {name}_v0.csv
        new_filename = f"{name}_v0{ext}"
        dest_path = os.path.join(self.dataset_folder, new_filename)
        dest_path = os.path.abspath(dest_path)

        try:
            # Проверяем, не совпадает ли путь
            if file_path == dest_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                # Если уже есть # META: v0 — ничего не делаем
                if first_line == "# META: v0":
                    QMessageBox.information(self, "Готово", "Файл уже загружен в нужном формате.")
                    return

            # Перезапись?
            if os.path.exists(dest_path) and file_path != dest_path:
                reply = QMessageBox.question(
                    self,
                    "Файл существует",
                    f"Файл '{new_filename}' уже существует. Перезаписать?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # Читаем датасет
            df = pd.read_csv(file_path, skipinitialspace=True)

            # ✅ Записываем с одной строкой: # META: v0
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write("# META: v0\n")  # 🔥 Только это — как вы и хотели
                df.to_csv(f, index=False, encoding="utf-8", lineterminator="\n")

            # Собираем информацию
            rows, cols = df.shape
            dtypes = df.dtypes.value_counts()
            object_cols = df.select_dtypes(include=['object']).columns.tolist()

            info = f"✅ Файл успешно загружён:\n  {new_filename}\n\n"
            info += f"📊 Размер: {rows} строк × {cols} столбцов\n\n"
            info += f"🔢 Типы данных:\n"
            for dtype, count in dtypes.items():
                info += f"  • {dtype}: {count} столбец(ов)\n"

            if object_cols:
                info += f"\n⚠️  Столбцы с текстом (object): {len(object_cols)}\n"
                info += "   Рекомендуется обработать:\n"
                for col in object_cols[:10]:
                    info += f"   - {col}\n"
                if len(object_cols) > 10:
                    info += f"   ... и ещё {len(object_cols) - 10}\n"
            else:
                info += "\n✅ Нет текстовых столбцов — можно продолжать."

            self.info_label.setText(info)
            QMessageBox.information(self, "Успех", f"Датасет сохранён как:\n{dest_path}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить или сохранить датасет:\n{str(e)}")
            self.info_label.setText("❌ Ошибка при загрузке датасета.")
