import pandas as pd
import numpy as np

# Загрузка вашего датасета
df = pd.read_csv("/home/adush/unic/recognition theory/laby/laba_4/data/student_performance_prediction.csv")

# Создание колонки Text
df['Text'] = (
    "Student " + df['Student ID'] +
    " studies " + df['Study Hours per Week'].astype(str) + " hours per week, " +
    "attendance " + df['Attendance Rate'].fillna("unknown").astype(str) + ", " +
    "previous grades " + df['Previous Grades'].astype(str) + ", " +
    np.where(df['Participation in Extracurricular Activities'] == "Yes", "participates", "does not participate") +
    " in extracurricular activities, parent's education " + df['Parent Education Level'] + ", " +
    np.where(df['Passed'] == "Yes", "passed.", "did not pass.")
)

# Опционально добавляем Rating
df['Rating'] = 3

# Выбираем только колонки для сайта
output_df = df[['Text', 'Rating']]

# Сохраняем CSV
output_df.to_csv("students_for_summarizer.csv", index=False)
