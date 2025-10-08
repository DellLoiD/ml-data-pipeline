from preprocessing.dataset_processing_ui import DatasetProcessingUI  # Импортируем класс окна обработки датасета


class LogicManager:
    @staticmethod
    def processRawDataset():
        """
        Метод, открывающий окно обработки датасета.
        """
        processing_window = DatasetProcessingUI()  # Создаем объект окна обработки датасета
        processing_window.show()  # Показываем окно