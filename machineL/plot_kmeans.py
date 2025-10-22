import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_kmeans_boxplots(clustered_data_path):
    """
    Gera box plots para cada feature numérica, agrupados pelos clusters K-Means.
    """
    df_clustered = pd.read_parquet(clustered_data_path)

    numeric_cols = [
        'Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 
        'Tempo', 'Duration_ms', 'Views', 'Likes', 'Comments', 'Stream'
    ]

    print("Gerando box plots para as features: ", numeric_cols)

    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster_KMeans', y=col, data=df_clustered)
        plt.title(f'Distribuição de {col} por Cluster K-Means')
        plt.xlabel('Cluster K-Means')
        plt.ylabel(col)
        plt.grid(True)
        plt.savefig(f'boxplot_kmeans_{col}.png')
        plt.close()
        print(f'Box plot para {col} (K-Means) salvo como boxplot_kmeans_{col}.png')

if __name__ == '__main__':
    plot_kmeans_boxplots(
        clustered_data_path='Spotify_Youtube_clustered_kmeans.parquet'
    )