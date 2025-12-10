import os
import pandas as pd
import numpy as np
from sklearn.cluster import Birch, DBSCAN
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from scipy.stats import randint, uniform

# ------------------ Конфигурация ------------------
INPUT = "/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv"
OUT_DIR = "clustering_results_BayesSearchCV_PCA"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ Загрузка данных ------------------
print("Загрузка данных:", INPUT)
df = pd.read_csv(INPUT)

y = df.iloc[:, -1].copy()
X = df.iloc[:, :-1].copy()

# ------------------ Стандартизация и PCA ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Сохраняем 95% дисперсии
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print(f"Исходные признаки: {X.shape[1]}, после PCA: {X_pca.shape[1]}")

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
    'estimator__threshold': (0.05, 2.0, 'uniform'),
    'estimator__branching_factor': (10, 200)
}

birch_search = BayesSearchCV(
    birch_wrapper,
    birch_param_space,
    n_iter=25,
    scoring=sil_scorer,
    n_jobs=20,
    cv=3,
    random_state=RANDOM_STATE
)

birch_search.fit(X_pca)
print("Лучшие параметры Birch:", birch_search.best_params_)
print("Лучший silhouette score Birch:", birch_search.best_score_)

joblib.dump(birch_search.best_estimator_, os.path.join(OUT_DIR, "birch_bayes_best.pkl"))
labels_birch_best = birch_search.best_estimator_.predict(X_pca)

