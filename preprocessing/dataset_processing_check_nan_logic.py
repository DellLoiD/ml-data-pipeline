# preprocessing/dataset_processing_check_nan_logic.py

# === Экспериментальные возможности Scikit-learn ===
from sklearn.experimental import enable_iterative_imputer

# === Машинное обучение и математика ===
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import nan_euclidean_distances

# === PySide6 для GUI (только в методах с окнами) ===
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QApplication
from PySide6.QtCore import Qt

# === Остальные библиотеки ===
import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

# ===================================================================
# 🧩 ОБЩАЯ СИГНАТУРА ВСЕХ ФУНКЦИЙ:
# def impute_xxx(df, column, parent=None) -> (df, description)
# ===================================================================


def impute_mean(df: pd.DataFrame, column: str, parent=None) -> tuple[pd.DataFrame, str]:
    """Заполняет пропуски средним значением."""
    if df[column].dtype not in ['int64', 'float64']:
        raise ValueError("Среднее применимо только к числовым колонкам.")
    value = df[column].mean()
    df[column] = df[column].fillna(value)
    return df, f"Среднее: {value:.4f}"


def impute_median(df: pd.DataFrame, column: str, parent=None) -> tuple[pd.DataFrame, str]:
    """Заполняет пропуски медианой."""
    if df[column].dtype not in ['int64', 'float64']:
        raise ValueError("Медиана применима только к числовым колонкам.")
    value = df[column].median()
    df[column] = df[column].fillna(value)
    return df, f"Медиана: {value:.4f}"


def impute_mode(df: pd.DataFrame, column: str, parent=None) -> tuple[pd.DataFrame, str]:
    """Заполняет пропуски модой (наиболее частым значением)."""
    value = df[column].mode()
    if value.empty:
        value = df[column].dropna().iloc[0] if not df[column].dropna().empty else "Unknown"
    else:
        value = value[0]
    df[column] = df[column].fillna(value)
    return df, f"Мода: {value}"


def impute_interpolate(df: pd.DataFrame, column: str, parent=None) -> tuple[pd.DataFrame, str]:
    """Интерполяция для числовых колонок."""
    if df[column].dtype not in ['int64', 'float64']:
        raise ValueError("Интерполяция доступна только для числовых колонок.")
    df[column] = df[column].interpolate(method='linear', limit_direction='both')
    return df, "Интерполяция (линейная)"


def impute_knn(df: pd.DataFrame, column: str, parent=None, n_neighbors: int = 5) -> tuple[pd.DataFrame, str]:
    """KNN-Imputer: заполняет на основе похожих строк."""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if column not in numeric_df.columns:
        raise ValueError("KNN требует числовые данные. Используйте кодирование для категорий.")

    # Подготовка данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # KNN импутация
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform", metric="nan_euclidean")
    X_imputed = imputer.fit_transform(X_scaled)

    # Обратное масштабирование
    X_restored = scaler.inverse_transform(X_imputed)
    df_numeric_restored = pd.DataFrame(X_restored, columns=numeric_df.columns, index=df.index)

    # Обновляем только нужную колонку
    df[column] = df_numeric_restored[column]
    return df, f"KNN-Imputer (k={n_neighbors})"

#тут был mice до переноса

def impute_hot_deck(df: pd.DataFrame, column: str, parent=None) -> tuple[pd.DataFrame, str]:
    """
    Hot Deck с модальным окном-заглушкой, показывающим ход выполнения.
    
    Args:
        df: исходный датафрейм
        column: колонка для восстановления
        parent: родительское окно (для центрирования)

    Returns:
        (df, сообщение)
    """
    # Проверка колонки
    if column not in df.columns:
        raise ValueError(f"Колонка '{column}' не найдена в датасете.")

    missing_mask = df[column].isna()
    missing_idx = df[missing_mask].index

    if not missing_mask.any():
        return df, "Hot Deck: нет пропусков для заполнения"

    # === Создаём модальное окно-заглушку ===
    if parent:
        progress_dialog = QDialog(parent)
        progress_dialog.setModal(True)
        progress_dialog.setWindowTitle("Hot Deck — восстановление пропусков")
        progress_dialog.setWindowFlags(progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint) # type: ignore
        progress_dialog.resize(400, 150)
        progress_dialog.move(parent.geometry().center() - progress_dialog.rect().center())

        layout = QVBoxLayout()
        label = QLabel("1/4: Поиск похожих строк...\nПодготовка данных...")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter) # type: ignore
        layout.addWidget(label)
        progress_dialog.setLayout(layout)
        progress_dialog.show()
        QApplication.processEvents()  # Обновляем GUI
    else:
        progress_dialog = None
        label = None

    # === Разделяем признаки ===
    if progress_dialog:
        label.setText("2/4: Разделение признаков на числовые и категориальные...") # type: ignore
        QApplication.processEvents()

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    compare_numeric = [col for col in numeric_cols if col != column]
    compare_categorical = [col for col in categorical_cols if col != column]

    X_numeric = df[compare_numeric].copy() if compare_numeric else pd.DataFrame(index=df.index)
    X_categorical = df[compare_categorical].copy() if compare_categorical else pd.DataFrame(index=df.index)

    # === Нормализуем числовые ===
    if progress_dialog:
        label.setText("3/4: Нормализация числовых признаков...") # type: ignore
        QApplication.processEvents()

    if not X_numeric.empty:
        scaler = StandardScaler()
        X_numeric_scaled = pd.DataFrame(
            scaler.fit_transform(X_numeric),
            index=X_numeric.index,
            columns=X_numeric.columns
        )
    else:
        X_numeric_scaled = pd.DataFrame(index=df.index)

    # === Проверка: есть ли строки с заполненным значением ===
    complete_target_mask = df[column].notna()
    if complete_target_mask.sum() == 0:
        if progress_dialog:
            label.setText("❌ Ошибка: нет строк с заполненным значением.") # type: ignore
            QApplication.processEvents()
            progress_dialog.close()
        raise ValueError(f"Нет строк с заполненным значением в колонке '{column}' — невозможно применить Hot Deck.")

    donor_indices = df[complete_target_mask].index

    # === Заполнение пропусков ===
    filled_count = 0
    total_missing = len(missing_idx)

    if progress_dialog:
        label.setText(f"4/4: Поиск доноров...\nВосстановлено: 0 из {total_missing}") # type: ignore
        QApplication.processEvents()

    for i, idx in enumerate(missing_idx):
        row_numeric = X_numeric_scaled.loc[[idx]] if not X_numeric_scaled.empty else pd.DataFrame(index=[idx])
        row_categorical = X_categorical.loc[[idx]] if not X_categorical.empty else pd.DataFrame(index=[idx])

        min_dist = np.inf
        best_match_idx = None

        for donor_idx in donor_indices:
            if not X_numeric.empty and X_numeric.loc[donor_idx].isna().any():
                continue
            if not X_categorical.empty and X_categorical.loc[donor_idx].isna().any():
                continue

            num_dist = 0
            if not X_numeric.empty:
                donor_row_numeric = X_numeric_scaled.loc[[donor_idx]]
                num_dist = nan_euclidean_distances(row_numeric, donor_row_numeric)[0][0]

            cat_mismatches = 0
            if not X_categorical.empty:
                donor_row_categorical = X_categorical.loc[[donor_idx]]
                cat_mismatches = (row_categorical.iloc[0] != donor_row_categorical.iloc[0]).sum()

            total_dist = num_dist + cat_mismatches

            if total_dist < min_dist:
                min_dist = total_dist
                best_match_idx = donor_idx

        if best_match_idx is not None:
            df.loc[idx, column] = df.loc[best_match_idx, column]
            filled_count += 1

        # === Обновляем счётчик ===
        if progress_dialog and (i + 1) % 3 == 0:
            label.setText(f"4/4: Поиск доноров...\nВосстановлено: {filled_count} из {total_missing}") # type: ignore
            QApplication.processEvents()

    # === Завершение ===
    if progress_dialog:
        label.setText(f"✅ Готово!\nВосстановлено: {filled_count} значений") # type: ignore
        QApplication.processEvents()
        time.sleep(1.5)
        progress_dialog.close()

    return df, f"Hot Deck: заполнено {filled_count} значений"


def impute_em(df: pd.DataFrame, column: str, parent=None, max_iter: int = 100) -> tuple[pd.DataFrame, str]:
    """
    EM-Imputation: упрощённая версия на основе нормального распределения.
    Только для числовых колонок.
    """
    if df[column].dtype not in ['int64', 'float64']:
        raise ValueError("EM работает только с числовыми колонками.")

    data = df[column].copy()
    missing_mask = data.isna()

    if not missing_mask.any():
        return df, "EM: нет пропусков"

    # Начальные значения
    mu = data.mean()
    sigma = data.std()

    # EM-цикл
    for _ in range(max_iter):
        # E-step: оценка отсутствующих значений
        data[missing_mask] = np.random.normal(mu, sigma, size=missing_mask.sum())

        # M-step: пересчёт параметров
        mu_new = data.mean()
        sigma_new = data.std()
        if abs(mu - mu_new) < 1e-5:
            break
        mu, sigma = mu_new, sigma_new

    df[column] = data
    return df, f"EM (μ={mu:.4f}, σ={sigma:.4f})"
