import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
# ------------------ 1. Пути файлов ------------------
input_path = "/home/adush/unic/recognition theory/laby/laba_4/data/student_performance_prediction.csv"
output_path = "/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv"

# ------------------ 2. Загрузка данных ------------------
df = pd.read_csv(input_path)
original_len = len(df)

# ------------------ 3. Обработка некорректных и пропущенных значений ------------------

# 3.1. Коррекция числовых данных: некорректные значения -> NaN
df['Attendance Rate'] = df['Attendance Rate'].apply(lambda x: x if 0 <= x <= 100 else None)
df['Study Hours per Week'] = df['Study Hours per Week'].apply(lambda x: x if x >= 0 else None)
df['Previous Grades'] = df['Previous Grades'].apply(lambda x: x if 0 <= x <= 100 else None)

# 3.2. Числовые колонки — заполнение средним
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Passed' in numeric_cols:
    numeric_cols.remove('Passed')

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 3.3. Бинарные категориальные колонки (Yes/No)
binary_cols = ['Participation in Extracurricular Activities', 'Passed']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    # Пропуски кодируем отдельным значением -1
    df[col] = df[col].fillna(-1)

# 3.4. Категориальные колонки (кроме ID и бинарных)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in binary_cols + ['Student ID']]
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# 3.5. One-Hot Encoding для категориальных признаков
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # drop_first=True для предотвращения мультиколлинеарности

# ------------------ 4. Обработка выбросов ------------------

# 4.1. Ограничение выбросов методом capping (IQR)
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
for col in numeric_cols:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    df[col] = df[col].clip(lower, upper)

# 4.2. Ограничение допустимых значений (дублируем для надежности)
df = df[(df['Study Hours per Week'] >= 0) & (df['Study Hours per Week'] <= 40)]
df = df[(df['Attendance Rate'] >= 0) & (df['Attendance Rate'] <= 100)]
df = df[(df['Previous Grades'] >= 0) & (df['Previous Grades'] <= 100)]

# 4.3. Удаление дубликатов
df = df.drop_duplicates()

print(f"Строк до очистки: {original_len}, после очистки: {len(df)}")

# ------------------ 5. Масштабирование числовых признаков ------------------
# Для алгоритмов на основе расстояний (DBSCAN, KMeans) удобно MinMaxScaler
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Альтернатива: StandardScaler()
# scaler = StandardScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ------------------ 6. Перемещаем Passed в конец и удаляем Student ID ------------------
passed_col = df.pop('Passed')
df['Passed'] = passed_col
if 'Student ID' in df.columns:
    df = df.drop(columns=['Student ID'])

# ------------------ 7. Сохранение итогового датасета ------------------
df.to_csv(output_path, index=False)
print(f"Данные успешно обработаны и сохранены в '{output_path}'")

# ------------------ 8. Проверка пропущенных значений ------------------
print("\nПропущенные значения после обработки:")
print(df.isnull().sum())

# ------------------ 9. Проверка мультиколлинеарности (корреляция) ------------------
cor_matrix = df.corr().abs()
# Получаем верхний треугольник корреляционной матрицы
upper = cor_matrix.where(~np.tril(np.ones(cor_matrix.shape, dtype=bool)))
# Находим колонки с высокой корреляцией (>0.95)
high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
if high_corr:
    print(f"Высокая корреляция между колонками: {high_corr}")
    # Можно удалить высоко коррелированные колонки, если нужно