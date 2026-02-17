from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt

class HelpDialog(QDialog):
    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Scrollable area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        
        label = QLabel(content)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        content_layout.addWidget(label)
        
        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)
        
        layout.addWidget(scroll)
        
        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)


# Текст помощи для параметров моделей
MODEL_PARAM_HELP = {
    'Number of Trees': "Количество деревьев в ансамбле. Больше → точнее, но медленнее",
    'Max Depth': "Максимальная глубина дерева. None — без ограничений. Большое значение → переобучение",
    'Min Samples Split': "Минимальное количество объектов для разделения узла. Больше → проще модель",
    'Learning Rate': "Темп обучения в GB. Меньше → стабильнее, но медленнее",
    'C': "Сила регуляризации в Logistic Regression. Чем выше — тем слабее регуляризация",
    'Max Iterations': "Максимальное количество итераций обучения. Увеличьте, если модель не сходится",
    'Penalty': "Тип регуляризации: l1, l2, none",
    'Random State': "Фиксация случайности. Для воспроизводимости результатов",
    'Fit Intercept': "Добавить свободный член в линейную модель",
    'Normalize': "Нормировать признаки перед обучением"
}

# Текст помощи для n_jobs
N_JOBS_HELP = "Количество ядер CPU для параллельных вычислений.\n1 — последовательно (по умолчанию)\n-1 — использовать все ядра"

# Текст помощи для Plot
PLOT_HELP_TEXT = """
<b>Типы графиков:</b><br>
• <b>Summary Plot</b> — суммирует важность признаков и направление влияния<br>
• <b>Bar</b> — столбчатая диаграмма важности<br>
• <b>Beeswarm</b> — распределение вкладов признаков по объектам<br><br>
<b>Сортировка:</b><br>
• По убыванию — по среднему |SHAP значению|<br>
• По алфавиту — по имени признака<br>
• Исходный порядок — как в датасете
"""