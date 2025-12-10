import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import joblib
import os

# ------------------ Конфигурация ------------------
OUT_DIR = "/home/adush/unic/recognition theory/laby/laba_4/src/clustering_results"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------ Загрузка данных ------------------
X = pd.read_csv("/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv").iloc[:, :-1]
y_true = pd.read_csv("/home/adush/unic/recognition theory/laby/laba_4/data/processed_dataset.csv").iloc[:, -1]

# ------------------ Загрузка моделей ------------------
birch_model = joblib.load(os.path.join(OUT_DIR, "birch_random_best.pkl"))
dbscan_model = joblib.load(os.path.join(OUT_DIR, "dbscan_random_best.pkl"))

# ------------------ Получение меток кластеров ------------------
labels_birch = birch_model.predict(X)
labels_dbscan = dbscan_model.predict(X)

# ------------------ t-SNE ------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# ------------------ UMAP ------------------
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X)

# ------------------ Сохраняем результаты ------------------
df_viz = pd.DataFrame({
    'TSNE1': X_tsne[:, 0],
    'TSNE2': X_tsne[:, 1],
    'UMAP1': X_umap[:, 0],
    'UMAP2': X_umap[:, 1],
    'Birch': labels_birch,
    'DBSCAN': labels_dbscan,
    'TrueLabel': y_true
})
df_viz.to_csv(os.path.join(OUT_DIR, "clusters_viz.csv"), index=False)
print("Результаты t-SNE и UMAP сохранены в clusters_viz.csv")

# ------------------ Визуализация ------------------
def plot_clusters(df, x_col, y_col, label_col, title, filename):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=label_col, palette='tab20', s=10, legend='full')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

# t-SNE
plot_clusters(df_viz, 'TSNE1', 'TSNE2', 'Birch', 't-SNE визуализация Birch', 'tsne_birch.png')
plot_clusters(df_viz, 'TSNE1', 'TSNE2', 'DBSCAN', 't-SNE визуализация DBSCAN', 'tsne_dbscan.png')

# UMAP
plot_clusters(df_viz, 'UMAP1', 'UMAP2', 'Birch', 'UMAP визуализация Birch', 'umap_birch.png')
plot_clusters(df_viz, 'UMAP1', 'UMAP2', 'DBSCAN', 'UMAP визуализация DBSCAN', 'umap_dbscan.png')

print("Визуализация завершена. Графики сохранены в папке:", PLOT_DIR)
