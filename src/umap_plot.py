import os
import joblib
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.metrics import adjusted_rand_score

class ClusteringWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X)
        return self

    def predict(self, X):
        if hasattr(self.estimator_, "predict"):
            return self.estimator_.predict(X)
        else:
            return self.estimator_.fit_predict(X)

# ------------------ Настройки ------------------
DATA_PATH = "/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv"
MODEL_PATH = "/home/adush/unic/recognition theory/laby/laba_4/src/clustering_results_BayesSearchCV_PCA/birch_bayes_best.pkl"   # путь к любому сохранённому .pkl
OUT_IMG = "clustering_results/umap_visualization.png"

os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)

# ------------------ Загрузка данных ------------------
print("Загрузка данных:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
X = df.iloc[:, :-1].copy()

# ------------------ Загрузка модели ------------------
print("Загрузка модели:", MODEL_PATH)
model = joblib.load(MODEL_PATH)

# ------------------ Получаем кластеры ------------------
print("Предсказание кластеров...")
labels = model.predict(X)

y = df.iloc[:, -1].copy()     # Истинные метки
ari_score = adjusted_rand_score(y, labels)
print(f"ARI оценка модели: {ari_score:.10f}")

# ------------------ Построение UMAP ------------------
print("Построение UMAP-проекции...")
umap_model = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
)

X_umap = umap_model.fit_transform(X)

# ------------------ Визуализация ------------------
print("Визуализация кластеров...")

plt.figure(figsize=(10, 8))
unique_labels = np.unique(labels)

for label in unique_labels:
    mask = labels == label
    if label == -1:
        color = "black"
        name = "Noise"
    else:
        color = None  # автоматический выбор
        name = f"Cluster {label}"

    plt.scatter(
        X_umap[mask, 0],
        X_umap[mask, 1],
        s=25,
        label=name,
        alpha=0.8
    )

plt.title("UMAP visualization of clusters", fontsize=16)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.grid(True)

plt.savefig(OUT_IMG, dpi=300)
plt.show()

print(f"\n✔ Визуализация сохранена как {OUT_IMG}")
