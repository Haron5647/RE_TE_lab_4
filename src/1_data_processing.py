import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных

df = pd.read_csv("/home/adush/unic/recognition theory/laby/laba_4/data/student_performance_prediction.csv")

# 2. Обработка пропущенных значений

df['Attendance Rate'].fillna(df['Attendance Rate'].mean(), inplace=True)
df['Previous Grades'].fillna(df['Previous Grades'].mean(), inplace=True)

# 3. Преобразование категориальных данных

# Binary Encoding Yes/No

df['Participation in Extracurricular Activities'] = df['Participation in Extracurricular Activities'].map({'Yes':1, 'No':0})

# One-Hot Encoding для Parent Education Level

df = pd.get_dummies(df, columns=['Parent Education Level'])

# 4. Нормализация числовых признаков

scaler = StandardScaler()
df[['Study Hours per Week', 'Attendance Rate', 'Previous Grades']] = scaler.fit_transform(
df[['Study Hours per Week', 'Attendance Rate', 'Previous Grades']]
)

# 5. Преобразование целевой переменной

df['Passed'] = df['Passed'].map({'Yes':1, 'No':0})

# 6. Сохранение обработанного датасета

df.to_csv("student_perf_predic_proces_dataset.csv", index=False)
df.to_csv("/home/adush/unic/recognition theory/laby/laba_4/data/student_perf_predic_proces_dataset.csv", index=False)


print("Данные успешно обработаны и сохранены в 'processed_dataset.csv'")
