import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClusterMixin, clone

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


# ------------------ Конфигурация ------------------
INPUT = "/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv"
birch_model_path = "/home/adush/unic/recognition theory/laby/laba_4/src/clustering_results/birch_random_best.pkl"
dbscan_model_path = "/home/adush/unic/recognition theory/laby/laba_4/src/clustering_results_GridSearch/dbscan_grid_best.pkl"
RANDOM_STATE = 42

# ------------------ Загрузка данных ------------------
print("Загрузка данных:", INPUT)
df = pd.read_csv(INPUT)
X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()  # если нужен оригинальный target

# ------------------ Загрузка моделей ------------------
print("Загрузка моделей...")
birch_model = joblib.load(birch_model_path)
dbscan_model = joblib.load(dbscan_model_path)

# ------------------ Получение меток кластеров ------------------
labels_birch = birch_model.predict(X)
labels_dbscan = dbscan_model.predict(X)

# ------------------ PCA для снижения размерности ------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

# ------------------ Функция для визуализации ------------------
def plot_clusters(X_2d, labels, title):
    plt.figure(figsize=(8,6))
    unique_labels = set(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:  # шум в DBSCAN
            color = 'k'
            label_name = 'Noise'
        else:
            color = plt.cm.tab20(lbl % 20)
            label_name = f'Cluster {lbl}'
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=20, c=[color], label=label_name, alpha=0.6)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ------------------ Визуализация кластеров ------------------
plot_clusters(X_pca, labels_birch, "Birch RandomizedSearchCV Clusters")
plot_clusters(X_pca, labels_dbscan, "DBSCAN RandomizedSearchCV Clusters")
