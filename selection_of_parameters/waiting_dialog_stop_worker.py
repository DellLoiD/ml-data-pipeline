from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer

class WaitingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Прерывание обучения")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)  # Убираем крестик
        self.resize(300, 100)

        layout = QVBoxLayout()

        self.label = QLabel("Идёт процесс прерывания обучения...\nПодождите")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Индикатор анимации
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.setLayout(layout)
