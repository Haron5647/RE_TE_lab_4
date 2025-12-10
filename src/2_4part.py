# clustering_analysis.py
"""
Кластерный анализ: Birch и DBSCAN + сравнение, подбор гиперпараметров, визуализации.
Вход: processed_dataset.csv (должен быть подготовлен первым скриптом).
Выход: папка clustering_results/ с CSV-таблицами и графиками.

Запуск:
    python clustering_analysis.py

Требования:
    pandas, numpy, scikit-learn, matplotlib, seaborn
Опционально (для Bayesian / UMAP):
    scikit-optimize (skopt)  -> BayesSearchCV
    umap-learn              -> UMAP визуализация
Если этих пакетов нет — секции с ними будут пропущены, но основной анализ выполнится.
"""

import os
import time
import numpy as np
import pandas as pd

from sklearn.cluster import Birch, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClusterMixin, clone

import matplotlib.pyplot as plt
import seaborn as sns

# Опциональные
try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ------------------ Конфигурация ------------------
INPUT = "/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv"  # файл, который создаёт preprocess_dataset.py
OUT_DIR = "clustering_results"
RANDOM_STATE = 42
SAMPLE_FOR_TSNE = 10000  # для t-SNE/UMAP используем подвыборку, чтобы ускорить
N_STABILITY_RUNS = 5

os.makedirs(OUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

# ------------------ Утилиты ------------------
def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def purity_score(y_true, y_pred):
    """
    Purity: для каждого кластера выбирается наиболее частая настоящая метка, затем считаем долю правильно кластеризованных.
    """
    labels = np.unique(y_pred)
    total = 0
    for lab in labels:
        if lab == -1:  # если есть noise, учитываем его как отдельный кластер
            mask = (y_pred == lab)
            if mask.sum() == 0:
                continue
            # для noise берём наиболее частую реальную метку среди его точек
            total += np.max(np.bincount(y_true[mask].astype(int)))
        else:
            mask = (y_pred == lab)
            if mask.sum() == 0:
                continue
            total += np.max(np.bincount(y_true[mask].astype(int)))
    return total / len(y_true)

class ClusteringWrapper(BaseEstimator, ClusterMixin):
    """
    Обёртка для использования GridSearchCV/RandomizedSearchCV с кластеризаторами.
    Будет вычислять score через silhouette внутри GridSearch (cv=1).
    """
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

# ------------------ Загрузка данных ------------------
print("Загрузка:", INPUT)
df = pd.read_csv(INPUT)

# Если Student ID остался — убираем из признаков, но сохраним в output для связи
if 'Student ID' in df.columns:
    ids = df['Student ID']
    df = df.drop(columns=['Student ID'])
else:
    ids = None

# Целевая переменная 'Passed' — если есть, используем как внешнюю метрику
if 'Passed' in df.columns:
    y = df['Passed'].copy().astype('Int64')
    X = df.drop(columns=['Passed']).copy()
else:
    y = None
    X = df.copy()

# Масштабируем числовые признаки (на случай, если не все уже масштабированы)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
if len(numeric_cols) > 0:
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print("Форма X:", X.shape, "меток y:", None if y is None else y.shape)

# Для безопасности: если есть строки с NaN — заменяем медианой (маловероятно)
if X.isnull().any().any():
    X = X.fillna(X.median())

# ------------------ Задача 1: запустить Birch и DBSCAN ------------------
def run_clustering_and_eval(X, algorithm_name, model):
    t0 = time.time()
    labels = model.fit_predict(X)
    t = time.time() - t0

    # Внутренние метрики
    unique_clusters = np.unique(labels[labels != -1])
    n_clusters = len(unique_clusters)
    metrics_internal = {
        'n_clusters_found': n_clusters,
        'time_s': t,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }
    if n_clusters > 1:
        try:
            metrics_internal['silhouette'] = silhouette_score(X, labels)
        except Exception:
            metrics_internal['silhouette'] = np.nan
        try:
            metrics_internal['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        except Exception:
            metrics_internal['calinski_harabasz'] = np.nan
        try:
            metrics_internal['davies_bouldin'] = davies_bouldin_score(X, labels)
        except Exception:
            metrics_internal['davies_bouldin'] = np.nan

    # Внешние метрики (если y есть)
    metrics_external = {}
    if y is not None and y.notna().sum() > 0:
        y_arr = y.fillna(0).astype(int).to_numpy()
        try:
            metrics_external['ARI'] = adjusted_rand_score(y_arr, labels)
        except Exception:
            metrics_external['ARI'] = np.nan
        try:
            metrics_external['NMI'] = normalized_mutual_info_score(y_arr, labels)
        except Exception:
            metrics_external['NMI'] = np.nan
        try:
            metrics_external['Purity'] = purity_score(y_arr, labels)
        except Exception:
            metrics_external['Purity'] = np.nan
        try:
            metrics_external['Fowlkes_Mallows'] = fowlkes_mallows_score(y_arr, labels)
        except Exception:
            metrics_external['Fowlkes_Mallows'] = np.nan

    return labels, metrics_internal, metrics_external

# Инициализация моделей с базовыми параметрами
birch = Birch(n_clusters=None, threshold=0.5, branching_factor=50)
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')

print("\nЗапуск базовой кластеризации Birch...")
labels_birch, metrics_birch_internal, metrics_birch_external = run_clustering_and_eval(X, 'Birch', birch)
print("Birch:", metrics_birch_internal)
if metrics_birch_external:
    print("Birch (внешние):", metrics_birch_external)

print("\nЗапуск базовой кластеризации DBSCAN...")
labels_db, metrics_db_internal, metrics_db_external = run_clustering_and_eval(X, 'DBSCAN', dbscan)
print("DBSCAN:", metrics_db_internal)
if metrics_db_external:
    print("DBSCAN (внешние):", metrics_db_external)

# Сохраняем базовые метрики
metrics_df = pd.DataFrame([
    {'algorithm': 'Birch', **metrics_birch_internal, **({k:v for k,v in metrics_birch_external.items()} if metrics_birch_external else {})},
    {'algorithm': 'DBSCAN', **metrics_db_internal, **({k:v for k,v in metrics_db_external.items()} if metrics_db_external else {})}
])
metrics_df.to_csv(os.path.join(OUT_DIR, "basic_clustering_metrics.csv"), index=False)

# Сохраняем метки в CSV для последующего изучения (включая ID, если был)
out_labels = pd.DataFrame({'index': X.index})
if ids is not None:
    out_labels['Student ID'] = ids.values
out_labels['Birch_cluster'] = labels_birch
out_labels['DBSCAN_cluster'] = labels_db
if y is not None:
    out_labels['Passed'] = y.values
out_labels.to_csv(os.path.join(OUT_DIR, "clusters_basic_labels.csv"), index=False)

# ------------------ Сравнение по устойчивости (несколько прогонов) ------------------
def stability_analysis(factory, X, n_runs=5):
    labels_list = []
    times = []
    for i in range(n_runs):
        start = time.time()
        model = factory(random_state=(RANDOM_STATE + i))
        labels = model.fit_predict(X)
        times.append(time.time() - start)
        labels_list.append(labels)
    # попарный ARI и NMI
    n = len(labels_list)
    aris = []
    nmis = []
    for i in range(n):
        for j in range(i+1, n):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
            nmis.append(normalized_mutual_info_score(labels_list[i], labels_list[j]))
    return {
        'labels_list': labels_list,
        'time_mean': np.mean(times),
        'ari_mean': np.mean(aris) if aris else np.nan,
        'nmi_mean': np.mean(nmis) if nmis else np.nan
    }

print("\nОценка устойчивости Birch (несколько прогонов)...")
birch_factory = lambda random_state=None: Birch(n_clusters=None, threshold=0.3, branching_factor=50)
stab_birch = stability_analysis(birch_factory, X, n_runs=N_STABILITY_RUNS)
print("Birch stability:", stab_birch)

print("\nОценка устойчивости DBSCAN (несколько прогонов)...")
dbscan_factory = lambda random_state=None: DBSCAN(eps=0.5, min_samples=5)
stab_db = stability_analysis(dbscan_factory, X, n_runs=N_STABILITY_RUNS)
print("DBSCAN stability:", stab_db)

pd.DataFrame([
    {'algorithm': 'Birch', 'time_mean': stab_birch['time_mean'], 'ari_mean': stab_birch['ari_mean'], 'nmi_mean': stab_birch['nmi_mean']},
    {'algorithm': 'DBSCAN', 'time_mean': stab_db['time_mean'], 'ari_mean': stab_db['ari_mean'], 'nmi_mean': stab_db['nmi_mean']},
]).to_csv(os.path.join(OUT_DIR, "stability_summary.csv"), index=False)

# ------------------ Задача 2: подбор гиперпараметров ------------------
# Используем GridSearchCV и RandomizedSearchCV. Для кластеров используем обёртку, score = silhouette.
def silhouette_scorer(est, X_val, y_val=None):
    # est — экземпляр ClusteringWrapper; внутри уже есть estimator_
    labs = est.estimator_.fit_predict(X_val) if hasattr(est.estimator_, 'fit_predict') else est.estimator_.predict(X_val)
    # если <=1 полезный кластер — возвращаем очень плохое значение
    if len(np.unique(labs[labs != -1])) <= 1:
        return -1.0
    return silhouette_score(X_val, labs)

from sklearn.metrics import make_scorer
sil_scorer = make_scorer(lambda est, X_val, y_val=None: silhouette_scorer(est, X_val), greater_is_better=True)

# Сетки параметров
birch_param_grid = {
    'estimator__threshold': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    'estimator__branching_factor': [25, 50, 100]
}
dbscan_param_grid = {
    'estimator__eps': [0.1, 0.2, 0.5, 1.0, 1.5],
    'estimator__min_samples': [3, 5, 8, 12, 20]
}

print("\nGridSearchCV для Birch (silhouette)...")
try:
    gs_birch = GridSearchCV(ClusteringWrapper(Birch()), birch_param_grid, scoring=sil_scorer, cv=1, n_jobs=-1)
    gs_birch.fit(X)
    print("Лучшие параметры (Birch Grid):", gs_birch.best_params_, "score:", gs_birch.best_score_)
    pd.to_pickle(gs_birch, os.path.join(OUT_DIR, "gridsearch_birch.pkl"))
except Exception as e:
    print("GridSearch Birch failed:", e)
    gs_birch = None

print("\nRandomizedSearchCV для DBSCAN (silhouette)...")
try:
    rs_db = RandomizedSearchCV(ClusteringWrapper(DBSCAN()), dbscan_param_grid, n_iter=12, scoring=sil_scorer, cv=1, n_jobs=-1, random_state=RANDOM_STATE)
    rs_db.fit(X)
    print("Лучшие параметры (DBSCAN Random):", rs_db.best_params_, "score:", rs_db.best_score_)
    pd.to_pickle(rs_db, os.path.join(OUT_DIR, "randomsearch_dbscan.pkl"))
except Exception as e:
    print("RandomizedSearch DBSCAN failed:", e)
    rs_db = None

# Bayesian (skopt) — если доступен
if SKOPT_AVAILABLE:
    print("\nBayesian optimization (skopt) для Birch (пример)...")
    try:
        bayes_search = BayesSearchCV(
            ClusteringWrapper(Birch()),
            {
                'estimator__threshold': (1e-3, 2.0, 'log-uniform'),
                'estimator__branching_factor': (10, 200)
            },
            n_iter=25,
            scoring=sil_scorer,
            cv=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        bayes_search.fit(X)
        print("Bayes best (Birch):", bayes_search.best_params_, bayes_search.best_score_)
        pd.to_pickle(bayes_search, os.path.join(OUT_DIR, "bayes_birch.pkl"))
    except Exception as e:
        print("BayesSearch failed:", e)
else:
    print("\nskopt не установлен — Bayesian optimization пропущена. Установите scikit-optimize для использования.")

# Сохраняем лучшие параметры (если найдены)
best_params_list = []
if gs_birch is not None:
    best_params_list.append({'algorithm': 'Birch_Grid', 'best_params': str(gs_birch.best_params_), 'best_score': gs_birch.best_score_})
if rs_db is not None:
    best_params_list.append({'algorithm': 'DBSCAN_Random', 'best_params': str(rs_db.best_params_), 'best_score': rs_db.best_score_})
pd.DataFrame(best_params_list).to_csv(os.path.join(OUT_DIR, "best_params_summary.csv"), index=False)

# ------------------ Задача 3 & 5: Визуализации (PCA, LDA, t-SNE, UMAP) ------------------
def save_scatter_2d(emb, labels, title, fname, palette=None, size=8):
    fig, ax = plt.subplots(figsize=(8,6))
    # Если метки есть числом, преобразуем в строку, чтобы seaborn не пытался делать cont. cmap
    lbls = labels.astype(str)
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=lbls, s=12, palette=palette, legend='brief', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.legend(title='cluster', bbox_to_anchor=(1.05,1), loc='upper left')
    save_fig(fig, fname)

# Для визуализаций используем выборку, если большой датасет
n_samples = X.shape[0]
sample_idx = None
if n_samples > SAMPLE_FOR_TSNE:
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(np.arange(n_samples), SAMPLE_FOR_TSNE, replace=False)
    X_vis = X.iloc[sample_idx]
    if y is not None:
        y_vis = y.iloc[sample_idx].to_numpy()
    labels_birch_vis = labels_birch[sample_idx] if isinstance(labels_birch, np.ndarray) else np.array(labels_birch)[sample_idx]
    labels_db_vis = labels_db[sample_idx] if isinstance(labels_db, np.ndarray) else np.array(labels_db)[sample_idx]
else:
    X_vis = X
    y_vis = y.to_numpy() if y is not None else None
    labels_birch_vis = labels_birch
    labels_db_vis = labels_db

# 1) PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_vis)
save_scatter_2d(X_pca, labels_birch_vis, "Birch clusters (PCA)", "pca_birch.png")
save_scatter_2d(X_pca, labels_db_vis, "DBSCAN clusters (PCA)", "pca_dbscan.png")
if y is not None:
    save_scatter_2d(X_pca, y_vis, "Реальные метки Passed (PCA)", "pca_passed.png")

# 2) LDA (если есть метки и >1 класса)
if y is not None and len(np.unique(y.dropna())) > 1:
    try:
        lda = LDA(n_components=2)
        X_lda = lda.fit_transform(X_vis, y_vis)
        save_scatter_2d(X_lda, labels_birch_vis, "Birch clusters (LDA proj)", "lda_birch.png")
        save_scatter_2d(X_lda, labels_db_vis, "DBSCAN clusters (LDA proj)", "lda_dbscan.png")
        save_scatter_2d(X_lda, y_vis, "Реальные метки Passed (LDA proj)", "lda_passed.png")
    except Exception as e:
        print("LDA failed:", e)
else:
    print("LDA пропущена (нет реальных меток или недостаточно классов).")

# 3) t-SNE
try:
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=RANDOM_STATE, init='pca')
    X_tsne = tsne.fit_transform(X_vis)
    save_scatter_2d(X_tsne, labels_birch_vis, "Birch clusters (t-SNE)", "tsne_birch.png")
    save_scatter_2d(X_tsne, labels_db_vis, "DBSCAN clusters (t-SNE)", "tsne_dbscan.png")
    if y is not None:
        save_scatter_2d(X_tsne, y_vis, "Реальные метки Passed (t-SNE)", "tsne_passed.png")
except Exception as e:
    print("t-SNE failed or too slow:", e)

# 4) UMAP (если установлен)
if UMAP_AVAILABLE:
    try:
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        X_umap = reducer.fit_transform(X_vis)
        save_scatter_2d(X_umap, labels_birch_vis, "Birch clusters (UMAP)", "umap_birch.png")
        save_scatter_2d(X_umap, labels_db_vis, "DBSCAN clusters (UMAP)", "umap_dbscan.png")
        if y is not None:
            save_scatter_2d(X_umap, y_vis, "Реальные метки Passed (UMAP)", "umap_passed.png")
    except Exception as e:
        print("UMAP failed:", e)
else:
    print("UMAP не установлен — пропущено (pip install umap-learn для включения).")

# ------------------ Задача 4: Метрики (предложение и сохранение) ------------------
# Предлагаемые метрики:
# Внутренние (internal):
#   1) Silhouette Score (средняя ширина силуэта) — интерпретируемость качества кластеров
#   2) Davies-Bouldin или Calinski-Harabasz (оба хороши) — здесь сохраняем оба
# Внешние (external) — требуют "реальных" меток:
#   1) Adjusted Rand Index (ARI)
#   2) Normalized Mutual Information (NMI)
# Дополнительно Purity и Fowlkes-Mallows

internal_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
external_metrics = ['ARI', 'NMI', 'Purity', 'Fowlkes_Mallows']

# Соберём их в один DataFrame по алгоритмам и вариантам (базовые + лучшие параметры)
summary_rows = []

# Базовые (уже рассчитаны)
row_birch = {'algorithm': 'Birch_base'}
row_birch.update(metrics_birch_internal)
if metrics_birch_external:
    row_birch.update(metrics_birch_external)
summary_rows.append(row_birch)

row_db = {'algorithm': 'DBSCAN_base'}
row_db.update(metrics_db_internal)
if metrics_db_external:
    row_db.update(metrics_db_external)
summary_rows.append(row_db)

# Если были лучшие модели — применим их и добавим
def apply_and_record(name, estimator):
    labs, mi, me = None, {}, {}
    try:
        labs, mi, me = run_clustering_and_eval(X, name, estimator)
    except Exception as e:
        print("Ошибка при apply_and_record", name, e)
    row = {'algorithm': name}
    row.update(mi)
    if me:
        row.update(me)
    summary_rows.append(row)
    # также сохраняем labels
    out = pd.DataFrame({'index': X.index})
    if ids is not None:
        out['Student ID'] = ids.values
    out[name + "_cluster"] = labs
    if y is not None:
        out['Passed'] = y.values
    out.to_csv(os.path.join(OUT_DIR, f"clusters_{name}.csv"), index=False)

# Применим лучшие из Grid/Random (если есть)
if 'gs_birch' in locals() and gs_birch is not None:
    best_birch = gs_birch.best_estimator_.estimator_
    apply_and_record("Birch_grid_best", best_birch)

if 'rs_db' in locals() and rs_db is not None:
    best_db = rs_db.best_estimator_.estimator_
    apply_and_record("DBSCAN_random_best", best_db)

# Если Bayes есть — применим её
if SKOPT_AVAILABLE and 'bayes_search' in locals():
    try:
        best_bayes = bayes_search.best_estimator_.estimator_
        apply_and_record("Birch_bayes_best", best_bayes)
    except Exception:
        pass

pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "final_metrics_summary.csv"), index=False)

# ------------------ Финал: краткий отчёт в stdout ------------------
print("\nГотово. Результаты сохранены в папке:", OUT_DIR)
print("— basic_clustering_metrics.csv: базовые внутренние/внешние метрики")
print("— clusters_basic_labels.csv: метки кластеров (базовые)")
print("— stability_summary.csv: устойчивость (ARI/NMI между прогонками)")
print("— best_params_summary.csv: найденные лучшие параметры (Grid/Random)")
print("— final_metrics_summary.csv: итоговая таблица метрик для всех применённых конфигураций")
print("— графики: pca_*.png, tsne_*.png, umap_*.png, lda_*.png (если применялись)")
