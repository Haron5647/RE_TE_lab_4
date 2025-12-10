import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.metrics import silhouette_score
import joblib
import umap
from sklearn.preprocessing import StandardScaler

# ------------------ Конфигурация ------------------
INPUT = "/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv"
OUT_DIR = "clustering_results_GridSearch_UMAP"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ Загрузка данных ------------------
print("Загрузка данных:", INPUT)
df = pd.read_csv(INPUT)
y = df.iloc[:, -1].copy()
X = df.iloc[:, :-1].copy()

# ------------------ Стандартизация и UMAP ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

umap_model = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=7,  # можно менять на 2 для визуализации
    metric='euclidean',
    random_state=RANDOM_STATE
)
X_umap = umap_model.fit_transform(X_scaled)
print(f"Исходные признаки: {X.shape[1]}, после UMAP: {X_umap.shape[1]}")

# ------------------ Обёртка ------------------
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

# ------------------ Silhouette scorer ------------------
def sil_scorer(estimator, X, y=None):
    labels = estimator.estimator_.fit_predict(X)
    n_labels = len(np.unique(labels[labels != -1]))
    if n_labels < 2 or n_labels > len(X) - 1:
        return -1.0
    return silhouette_score(X, labels)

# ------------------ GridSearchCV для DBSCAN ------------------
print("\nGridSearchCV для DBSCAN...")

dbscan_wrapper = ClusteringWrapper(DBSCAN(metric="euclidean"))

dbscan_param_grid = {
    "estimator__eps": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2],
    "estimator__min_samples": [2, 3, 5, 10, 15, 20]
}

dbscan_grid = GridSearchCV(
    dbscan_wrapper,
    dbscan_param_grid,
    scoring=sil_scorer,
    n_jobs=20,  # лучше 1 для экономии памяти
    cv=3
)

dbscan_grid.fit(X_umap)

print("Лучшие параметры DBSCAN:", dbscan_grid.best_params_)
print("Лучший silhouette score DBSCAN:", dbscan_grid.best_score_)

joblib.dump(dbscan_grid.best_estimator_, os.path.join(OUT_DIR, "dbscan_grid_best.pkl"))

labels_dbscan_best = dbscan_grid.best_estimator_.predict(X_umap)

# ------------------ Сохраняем метки ------------------
out_labels = pd.DataFrame({
    "DBSCAN_grid_best": labels_dbscan_best,
    "Passed": y
})

out_labels.to_csv(os.path.join(OUT_DIR, "clusters_labels_grid_umap.csv"), index=False)

print("\n✔ GridSearchCV для DBSCAN завершён. Итоговые метки сохранены в clusters_labels_grid_umap.csv")
