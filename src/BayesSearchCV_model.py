import os
import pandas as pd
import numpy as np
from sklearn.cluster import Birch, DBSCAN
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.metrics import silhouette_score
import joblib
from scipy.stats import randint, uniform

# ------------------ Конфигурация ------------------
INPUT = "/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv"
OUT_DIR = "clustering_results_BayesSearchCV"
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

# ------------------ Silhouette scorer ------------------
def sil_scorer(estimator, X, y=None):
    labels = estimator.estimator_.fit_predict(X) if hasattr(estimator.estimator_, 'fit_predict') else estimator.estimator_.predict(X)
    n_labels = len(np.unique(labels[labels != -1]))
    if n_labels < 2 or n_labels > len(X) - 1:
        return -1.0
    return silhouette_score(X, labels)

# ------------------ Bayesian Optimization для Birch ------------------
print("\nBayesian Optimization для Birch...")
birch_wrapper = ClusteringWrapper(Birch(n_clusters=None))

birch_param_space = {
    'estimator__threshold': (0.05, 2.0, 'uniform'),  # диапазон
    'estimator__branching_factor': (10, 200)          # дискретный
}

birch_search = BayesSearchCV(
    birch_wrapper,
    birch_param_space,
    n_iter=25,
    scoring=sil_scorer,
    n_jobs=25,
    cv=3,
    random_state=RANDOM_STATE
)

birch_search.fit(X)
print("Лучшие параметры Birch:", birch_search.best_params_)
print("Лучший silhouette score Birch:", birch_search.best_score_)

joblib.dump(birch_search.best_estimator_, os.path.join(OUT_DIR, "birch_bayes_best.pkl"))
labels_birch_best = birch_search.best_estimator_.predict(X)

# ------------------ Bayesian Optimization для DBSCAN ------------------
print("\nBayesian Optimization для DBSCAN...")
dbscan_wrapper = ClusteringWrapper(DBSCAN(metric='euclidean'))

dbscan_param_space = {
    'estimator__eps': (0.1, 1.5, 'uniform'),
    'estimator__min_samples': (2, 20)
}

dbscan_search = BayesSearchCV(
    dbscan_wrapper,
    dbscan_param_space,
    n_iter=25,
    scoring=sil_scorer,
    n_jobs=25,
    cv=3,
    random_state=RANDOM_STATE
)

dbscan_search.fit(X)
print("Лучшие параметры DBSCAN:", dbscan_search.best_params_)
print("Лучший silhouette score DBSCAN:", dbscan_search.best_score_)

joblib.dump(dbscan_search.best_estimator_, os.path.join(OUT_DIR, "dbscan_bayes_best.pkl"))
labels_dbscan_best = dbscan_search.best_estimator_.predict(X)

# ------------------ Сохраняем все метки кластеров ------------------
out_labels = pd.DataFrame({
    'Birch_base': Birch(n_clusters=None, threshold=0.5, branching_factor=50).fit(X).predict(X),
    'DBSCAN_base': DBSCAN(eps=0.5, min_samples=5).fit_predict(X),
    'Birch_bayes_best': labels_birch_best,
    'DBSCAN_bayes_best': labels_dbscan_best,
    'Passed': y
})
out_labels.to_csv(os.path.join(OUT_DIR, "clusters_labels_bayes.csv"), index=False)

print("\n✔️ Все модели обучены и сохранены. Результаты кластеров сохранены в clusters_labels_bayes.csv")
