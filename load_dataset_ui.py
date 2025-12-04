# load_dataset.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
import os
import shutil
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
        """Открывает диалог выбора файла и загружает его"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV-файл", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return  # Отмена

        try:
            # Копируем файл в папку dataset
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.dataset_folder, filename)

            shutil.copy(file_path, dest_path)

            # Читаем датасет
            df = pd.read_csv(dest_path)

            # Собираем информацию
            rows, cols = df.shape
            dtypes = df.dtypes.value_counts()
            object_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Формируем текст
            info = f"✅ Файл успешно загружён:\n  {filename}\n\n"
            info += f"📊 Размер: {rows} строк × {cols} столбцов\n\n"
            info += f"🔢 Типы данных:\n"
            for dtype, count in dtypes.items():
                info += f"  • {dtype}: {count} столбец(ов)\n"

            if object_cols:
                info += f"\n⚠️  Столбцы с текстом (object): {len(object_cols)}\n"
                info += "   Рекомендуется обработать:\n"
                for col in object_cols[:10]:  # первые 10
                    info += f"   - {col}\n"
                if len(object_cols) > 10:
                    info += f"   ... и ещё {len(object_cols) - 10}\n"
            else:
                info += "\n✅ Нет текстовых столбцов — можно продолжать."

            self.info_label.setText(info)

            # Показываем сообщение
            QMessageBox.information(self, "Успех", f"Датасет сохранён в:\n{dest_path}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить датасет:\n{str(e)}")
            self.info_label.setText("❌ Ошибка при загрузке датасета.")
