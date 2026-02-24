import numpy as np
import pandas as pd
import shap


def prepare_shap_data(shap_values, X_train, sort_order="По убыванию"):   
    #print(f"АААААААААААААААААААААААА_X_train_ИМЕНА  ПРИНАКОВ_1: {X_train}")
    if not hasattr(X_train, 'columns'):
        raise ValueError("X_train должен быть pandas DataFrame для получения имен признаков.")
    feature_names = X_train.columns.tolist()
    print(f"ААААААААААААААААААААААА_feature_names_ИМЕНА  ПРИНАКОВ_2: {feature_names}")
    # Проверка типа и значений имен признаков
    if not all(isinstance(name, str) for name in feature_names):
        raise ValueError(f"Найдены нестроковые имена признаков: {[name for name in feature_names if not isinstance(name, str)]}")
    #print(f"АААААААААААААААААААААА_shap_values_ИМЕНА  ПРИНАКОВ_3: {getattr(shap_values, 'feature_names', 'Нет атрибута feature_names')}")
    # 2. Агрегация значений для категориальных признаков
    explanation = shap_values  # Стартуем с исходного объекта
    #print(f"АААААААААААААААААААААААААААААА_ИМЕНА  ПРИНАКОВ_4: {explanation}")
    #print(f"ААААААААААААААААААААА_explanation.feature_names_ИМЕНА ПРИЗНАКОВ_ИЗ_EXPLANATION: {explanation.feature_names}")
    # Определяем категориальные признаки по наличию подчеркивания
    cat_columns = []
    for col in X_train.columns:
        if '_' in col:  # Предполагаем, что закодированные категориальные признаки содержат подчеркивание
            base_name = col.split('_')[0]
            if base_name not in cat_columns:
                cat_columns.append(base_name)
    print(f"ААААААААААААААААААААА_cat_columns_ИМЕНА ПРИЗНАКОВ: {cat_columns}")
    
    # Если в данных нет закодированных признаков (без '_'), то cat_columns останется пустым
    # Это нормально — значит, агрегация не требуется
    if len(cat_columns) > 0:
        print(f"ААААААААААААААААААААА_Найдены закодированные признаки для агрегации: {cat_columns}")
        # Агрегируем значения и имена признаков
        # Пропускаем агрегацию по уникальным значениям из df
        explanation = shap.Explanation(
            values=shap_values.values if hasattr(shap_values, 'values') else np.array(shap_values),
            data=np.zeros((1, len(feature_names))) if not (hasattr(shap_values, 'data') and shap_values.data is not None) else shap_values.data.copy(),
            feature_names=feature_names
        )
    
    # 3. Вычисление важности признаков
    if hasattr(explanation, 'values'):
        values_for_importance = explanation.values
    else:
        values_for_importance = explanation
    
    # Убедимся, что значения двумерные
    if values_for_importance.ndim == 1:
        values_for_importance = values_for_importance.reshape(1, -1)
    
    # Принудительно устанавливаем explanation.feature_names, если они некорректны
    #if explanation.feature_names is None or len(explanation.feature_names) != values_for_importance.shape[1]:
    explanation.feature_names = X_train.columns.tolist()
    print(f"УСЛОВИЕ ВЫПОЛНЯЕТСЯ_xplanation.feature_names: {explanation.feature_names}")
        
    print(f"ААААААААААААААААААААА_xplanation.feature_names: {explanation.feature_names}")
    
    # Вычисляем среднее абсолютное значение по всем экземплярам (первое измерение)
    mean_abs_shap = np.abs(values_for_importance).mean(axis=0)
    
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.flatten()

    # 4. Определение порядка сортировки
    if sort_order == "По убыванию":
        feature_order = np.argsort(-mean_abs_shap).tolist()
    elif sort_order == "По алфавиту":
        feature_order = np.argsort(explanation.feature_names).tolist()
    else:  # По исходному порядку
        feature_order = np.arange(len(explanation.feature_names)).tolist()
    
    # 5. Создание отображаемых имен
    name_mapping = {}
    # Используем cat_columns, определенные ранее из X_train
    for col in cat_columns:
        if col in explanation.feature_names:
            name_mapping[col] = col  # Для агрегированных признаков
    
    # 6. Формирование финального списка отображаемых имен
    features_display_names = [name_mapping.get(explanation.feature_names[i], explanation.feature_names[i]) for i in feature_order]
    
    # 8. Возврат обработанных данных
    prepared_data = {
        'explanation': explanation,
        'feature_names': explanation.feature_names,
        'features_display_names': features_display_names,
        'mean_abs_shap': mean_abs_shap,
        'feature_order': feature_order,
        'name_mapping': name_mapping
    }
    
    return prepared_data