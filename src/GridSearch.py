import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.metrics import silhouette_score
import joblib
import umap

# ------------------ Конфигурация ------------------
INPUT = "/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv"
OUT_DIR = "clustering_results_GridSearch"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ Загрузка данных ------------------
print("Загрузка данных:", INPUT)
df = pd.read_csv(INPUT)
y = df.iloc[:, -1].copy()
X = df.iloc[:, :-1].copy()

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

# ------------------ Silhouette scorер ------------------
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
    n_jobs=10,
    cv=3
)

dbscan_grid.fit(X)

print("Лучшие параметры DBSCAN:", dbscan_grid.best_params_)
print("Лучший silhouette score DBSCAN:", dbscan_grid.best_score_)

joblib.dump(dbscan_grid.best_estimator_, os.path.join(OUT_DIR, "dbscan_grid_best.pkl"))

labels_dbscan_best = dbscan_grid.best_estimator_.predict(X)

# ------------------ Сохраняем метки ------------------
out_labels = pd.DataFrame({
    "DBSCAN_grid_best": labels_dbscan_best,
    "Passed": y
})

out_labels.to_csv(os.path.join(OUT_DIR, "clusters_labels_grid.csv"), index=False)


print("\n✔ GridSearchCV для DBSCAN завершён. Итоговые метки сохранены в clusters_labels_grid.csv")

# # ------------------ GridSearchCV для Birch ------------------
# print("\nGridSearchCV для Birch...")
#
# from sklearn.cluster import Birch
#
# birch_wrapper = ClusteringWrapper(
#     Birch(
#         n_clusters=None,      # пусть Birch сам определяет количество кластеров
#     )
# )
#
# birch_param_grid = {
#     "estimator__threshold": [0.1, 0.2, 0.3, 0.4, 0.5],
#     "estimator__branching_factor": [20, 30, 40, 50, 60],
#     "estimator__n_clusters": [None, 2, 3, 4, 5, 6, 8, 10]
# }
#
# birch_grid = GridSearchCV(
#     birch_wrapper,
#     birch_param_grid,
#     scoring=sil_scorer,
#     n_jobs=3,
#     cv=3
# )
#
# birch_grid.fit(X)
#
# print("Лучшие параметры Birch:", birch_grid.best_params_)
# print("Лучший silhouette score Birch:", birch_grid.best_score_)
#
# joblib.dump(birch_grid.best_estimator_, os.path.join(OUT_DIR, "birch_grid_best.pkl"))
#
# labels_birch_best = birch_grid.best_estimator_.predict(X)
#
# # ------------------ Сохраняем метки ------------------
# out_labels_birch = pd.DataFrame({
#     "Birch_grid_best": labels_birch_best,
#     "Passed": y
# })
#
# out_labels_birch.to_csv(os.path.join(OUT_DIR, "clusters_labels_birch_grid.csv"), index=False)
#
# print("\n✔ GridSearchCV для Birch завершён. Итоговые метки сохранены в clusters_labels_birch_grid.csv")
