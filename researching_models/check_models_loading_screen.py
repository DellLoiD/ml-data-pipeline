from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

class LoadingScreen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setWindowModality(Qt.ApplicationModal)  # Блокировка основного окна
        self.setWindowFlags(Qt.FramelessWindowHint)  # Без рамки
        self.setStyleSheet("background-color: rgba(255, 255, 255, 180);")
        # Увеличиваем размер окна
        self.resize(300, 200)
        layout = QVBoxLayout()

        # Метка с текстом
        self.info_label = QLabel("Производится оценка моделей...")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        # Таймер для анимации текста
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loading_text)
        self.timer.start(500)
        self.setLayout(layout)
        self.show()

    def update_loading_text(self):
        current_text = self.info_label.text()
        if '...' in current_text:
            new_text = current_text.replace('...', '')
        else:
            new_text = current_text + '.'
        self.info_label.setText(new_text)
