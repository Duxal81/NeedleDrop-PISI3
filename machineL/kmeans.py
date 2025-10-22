import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def run_kmeans_clustering(scaled_features_path, cleaned_data_path, output_clustered_path):
    """
    Aplica o algoritmo K-Means, utiliza o método do cotovelo e coeficiente de silhueta
    para auxiliar na determinação do número ideal de clusters e salva os resultados.
    """
    X_scaled = pd.read_parquet(scaled_features_path)

    print(f"Usando o dataset completo de {len(X_scaled)} amostras para o Metodo do Cotovelo e Coeficiente de Silhueta.")

    # Método do Cotovelo para encontrar o número ideal de clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init="auto")
        kmeans.fit(X_scaled) # Fit no dataset COMPLETO
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title("Metodo do Cotovelo")
    plt.xlabel("Numero de Clusters (K)")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.savefig("elbow_method.png")
    plt.close()
    print("Grafico do Metodo do Cotovelo salvo como elbow_method.png")

    # Cálculo do coeficiente de silhueta para diferentes números de clusters
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init="auto")
        kmeans.fit(X_scaled) # Fit no dataset COMPLETO
        score = silhouette_score(X_scaled, kmeans.labels_) # Score no dataset COMPLETO
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    plt.title("Coeficiente de Silhueta")
    plt.xlabel("Numero de Clusters (K)")
    plt.ylabel("Coeficiente de Silhueta")
    plt.grid(True)
    plt.savefig("silhouette_score.png")
    plt.close()
    print("Grafico do Coeficiente de Silhueta salvo como silhouette_score.png")

    # --- Aplicação final do K-Means no dataset COMPLETO --- 
    # Exemplo de aplicação do K-Means com um número arbitrário de clusters (e.g., 7)
    # Este valor deve ser ajustado após a análise dos gráficos elbow_method.png e silhouette_score.png
    k_optimal = 7  # Ajuste este valor com base na análise dos gráficos
    kmeans_final = KMeans(n_clusters=k_optimal, init="k-means++", random_state=42, n_init="auto")
    clusters = kmeans_final.fit_predict(X_scaled) # Fit e predict no dataset COMPLETO

    df_original = pd.read_parquet(cleaned_data_path)
    df_original["Cluster_KMeans"] = clusters
    df_original.to_parquet(output_clustered_path, index=False)
    print(f"Dataset com clusters K-Means (k={k_optimal}) salvo como {output_clustered_path}")
    return df_original

if __name__ == '__main__':
    run_kmeans_clustering(
        scaled_features_path='Spotify_Youtube_scaled_features.parquet',
        cleaned_data_path='Spotify_Youtube_cleaned.parquet',
        output_clustered_path='Spotify_Youtube_clustered_kmeans.parquet'
    )