import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загрузка данных
df = pd.read_csv('dataset/diabetes_BRFSS2015.csv')
# Данные для статуса диабета
labels1 = ['Нет диабета', 'Преддиабет', 'Диабет']
sizes1 = df['Diabetes_012'].value_counts().sort_index().tolist()
explode1 = (0, 0.1, 0)

# Данные для пола
labels2 = ['Женщины', 'Мужчины']
sizes2 = df['Sex'].value_counts().sort_index().tolist()
explode2 = (0, 0.1)

# Создание двух диаграмм рядом друг с другом
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Первая круговая диаграмма по диабету
axs[0].pie(sizes1, labels=labels1, explode=explode1, autopct='%1.1f%%', shadow=True, startangle=90)
axs[0].set_title('Распределение по статусу диабета')
axs[0].axis('equal')  # Убедимся, что график круглый

# Вторая круговая диаграмма по полу
axs[1].pie(sizes2, labels=labels2, explode=explode2, autopct='%1.1f%%', shadow=True, startangle=90)
axs[1].set_title('Распределение по полу')
axs[1].axis('equal')  # Убедимся, что график круглый

# Подпись ниже диаграмм
plt.figtext(0.5, 0.02,
    "Вывод: Женщин и мужчин примерно поровну. Здоровых больше всего.\n"
    "Для определения значимости каждого параметра пациента будем сравнивать корреляцию со здоровыми людьми, так как их больше.\n"
    "Например, если количество людей с высоким давлением велико среди здоровых так же, как среди больных – то признак не значителен.",
    ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Оставить место для подписи

plt.tight_layout()
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Артериальное давление в разных группах пациентов", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['Низкое', 'Высокое']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['HighBP'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['HighBP'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['HighBP'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: У большинства  здоровых   людей давление низкое.  Высокое давление — симптом диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Холестерин в разных группах пациентов", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['норма', 'Высокое']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['HighChol'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['HighChol'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['HighChol'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: У большинства  здоровых   людей Холестерин в норме.  Высокий холестерин — симптом диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Выкурили ли вы за всю свою жизнь хотя бы 100 сигарет?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['Smoker'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['Smoker'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['Smoker'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях разница между курящими и некурящими не значительна. Не  принак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Употребляете ли вы крепкие спиртные напитки?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['HvyAlcoholConsump'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['HvyAlcoholConsump'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['HvyAlcoholConsump'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle(" Возникают ли у вас серьезные трудности при длительных пешеходных?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['DiffWalk'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['DiffWalk'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['DiffWalk'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Употребляете ли вы регулярно фрукты один или более раз в день?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['Fruits'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['Fruits'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['Fruits'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Употребляете ли вы регулярно овощи один или более раз в день?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['Veggies'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['Veggies'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['Veggies'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Проверялся ли уровень холестерина в течение последних 5 лет?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['CholCheck'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['CholCheck'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['CholCheck'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Был ли у вас когда-нибудь зафиксирован инсульт?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['Stroke'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['Stroke'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['Stroke'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Были ли у вас когда-нибудь зафиксированы болезни сердца или инфаркт?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['HeartDiseaseorAttack'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['HeartDiseaseorAttack'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['HeartDiseaseorAttack'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Занимались ли вы в последние 30 дней активными физическими упражнениями, не считая работы?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['PhysActivity'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['PhysActivity'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['PhysActivity'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Есть ли у вас какая-либо медицинская страховка?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['AnyHealthcare'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['AnyHealthcare'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['AnyHealthcare'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#СЛЕДУЮЩИЙ  ГРАФИК
df.columns = df.columns.str.strip()
plt.figure(figsize=(14, 5))
plt.suptitle("Xотели попасть на прием к доктору, но не могли финансово себе это позволить?", fontsize=18, y=1)
# Группы по статусу диабета
diabetes = df[df['Diabetes_012'] == 2]
prediabetes = df[df['Diabetes_012'] == 1]
notdiabetes = df[df['Diabetes_012'] == 0]
# Названия для оси X
param_labels = ['нет', 'Да']  # 0=низкое, 1=высокое
# Считаем в каждой группе
param_counts_diabetes = diabetes['NoDocbcCost'].value_counts().reindex([0,1], fill_value=0)
param_counts_prediabetes = prediabetes['NoDocbcCost'].value_counts().reindex([0,1], fill_value=0)
param_counts_notdiabetes = notdiabetes['NoDocbcCost'].value_counts().reindex([0,1], fill_value=0)
# Диабет
plt.subplot(1, 3, 1)
plt.bar(param_labels, param_counts_diabetes, color='red')
#plt.xlabel('Артериальное давление')
plt.ylabel('Кол-во пациетов')
plt.title('Диабет')
plt.ylim(0, param_counts_diabetes.max() * 1.1)
# Преддиабет
plt.subplot(1, 3, 2)
plt.bar(param_labels, param_counts_prediabetes, color='orange')
plt.title('Преддиабет')
plt.ylim(0, param_counts_prediabetes.max() * 1.1)
# Нет диабета
plt.subplot(1, 3, 3)
plt.bar(param_labels, param_counts_notdiabetes, color='green')
plt.title('Нет диабета')
plt.ylim(0, param_counts_notdiabetes.max() * 1.1)
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.show()
#-----------------
# Фильтруем группы
healthy = df[df["Diabetes_012"] == 0]["BMI"]
pre_diabetic = df[df["Diabetes_012"] == 1]["BMI"]
diabetic = df[df["Diabetes_012"] == 2]["BMI"]

# Создаем гистограмму
plt.hist([healthy, pre_diabetic, diabetic],
         bins=1,
         label=["Здоровые", "Преддиабет", "Диабет"],
         density=False, alpha=0.7)

plt.legend(loc="upper right")
plt.xlabel("BMI")
plt.ylabel("Количество пациентов")
plt.title("Распределение BMI по группе диабета")
plt.figtext(0.5, 0.01, "Вывод: Высокий и повышенный  BMI это  признак диабета.", ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#-----------------
# Фильтруем группы
healthy = df[df["Diabetes_012"] == 0]["Age"]
pre_diabetic = df[df["Diabetes_012"] == 1]["Age"]
diabetic = df[df["Diabetes_012"] == 2]["Age"]

# Создаем гистограмму
plt.hist([healthy, pre_diabetic, diabetic],
         bins=2,
         label=["Здоровые", "Преддиабет", "Диабет"],
         density=False, alpha=0.7)

plt.legend(loc="upper right")
plt.xlabel("Возраст")
plt.ylabel("Количество пациентов")
plt.title("Распределение возраста по группе диабета")
plt.figtext(0.5, 0.01, "Вывод: Чем больше возраст тем вероятнее диабед.", ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#-----------------
# Фильтруем группы
healthy = df[df["Diabetes_012"] == 0]["GenHlth"]
pre_diabetic = df[df["Diabetes_012"] == 1]["GenHlth"]
diabetic = df[df["Diabetes_012"] == 2]["GenHlth"]

# Создаем гистограмму
plt.hist([healthy, pre_diabetic, diabetic],
         bins=5,
         label=["Здоровые", "Преддиабет", "Диабет"],
         density=False, alpha=0.7)

plt.legend(loc="upper right")
plt.xlabel("Здоровье")
plt.ylabel("Количество пациентов")
plt.title("Как бы вы в общем оценили уровень вашего здоровья от 1 = прекрасное до 5 = плохое?")
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#-----------------
# Фильтруем группы
healthy = df[df["Diabetes_012"] == 0]["PhysHlth"]
pre_diabetic = df[df["Diabetes_012"] == 1]["PhysHlth"]
diabetic = df[df["Diabetes_012"] == 2]["PhysHlth"]

# Создаем гистограмму
plt.hist([healthy, pre_diabetic, diabetic],
         bins=5,
         label=["Здоровые", "Преддиабет", "Диабет"],
         density=False, alpha=0.7)

plt.legend(loc="upper right")
plt.xlabel("Дней здоровых")
plt.ylabel("Количество пациентов")
plt.title("За последние 30 дней сколько дней ваше здоровье было неудовлетворительным?")
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#-----------------
# Фильтруем группы
healthy = df[df["Diabetes_012"] == 0]["Education"]
pre_diabetic = df[df["Diabetes_012"] == 1]["Education"]
diabetic = df[df["Diabetes_012"] == 2]["Education"]

# Создаем гистограмму
plt.hist([healthy, pre_diabetic, diabetic],
         bins=5,
         label=["Здоровые", "Преддиабет", "Диабет"],
         density=False, alpha=0.7)

plt.legend(loc="upper right")
plt.xlabel("Образование")
plt.ylabel("Количество пациентов")
plt.title("Уровень образования от 1 = Никогда не посещал школу до 6 = Магистр")
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#-----------------
# Фильтруем группы
healthy = df[df["Diabetes_012"] == 0]["Income"]
pre_diabetic = df[df["Diabetes_012"] == 1]["Income"]
diabetic = df[df["Diabetes_012"] == 2]["Income"]

# Создаем гистограмму
plt.hist([healthy, pre_diabetic, diabetic],
         bins=5,
         label=["Здоровые", "Преддиабет", "Диабет"],
         density=False, alpha=0.7)

plt.legend(loc="upper right")
plt.xlabel("Доход")
plt.ylabel("Количество пациентов")
plt.title("Доход от 1 = менее 10 тыс. долларов в год до 8 = более 75 тыс. долларов в год")
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#-----------------
# Фильтруем группы
healthy = df[df["Diabetes_012"] == 0]["MentHlth"]
pre_diabetic = df[df["Diabetes_012"] == 1]["MentHlth"]
diabetic = df[df["Diabetes_012"] == 2]["MentHlth"]

# Создаем гистограмму
plt.hist([healthy, pre_diabetic, diabetic],
         bins=5,
         label=["Здоровые", "Преддиабет", "Диабет"],
         density=False, alpha=0.7)

plt.legend(loc="upper right")
plt.xlabel("Дней")
plt.ylabel("Количество пациентов")
plt.title("За последние 30 дней сколько дней ваше психическое здоровье было неудовлетворительным?")
plt.figtext(0.5, 0.01, "Вывод: Во  всех категориях не коррелирует с наличием  диабета. Не  признак диабета.", ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
