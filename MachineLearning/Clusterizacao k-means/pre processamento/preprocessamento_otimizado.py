import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import joblib

def carregar_e_preparar_dados(caminho_dataset="Spotify_Youtube.csv"):
    df = pd.read_csv(caminho_dataset)

    features_para_clusterizar = [
        'Danceability', 'Energy', 'Valence', 'Views', 'Likes', 'Stream', 'Duration_min',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Tempo'
    ]

    if 'Duration_ms' in df.columns and 'Duration_min' not in df.columns:
        df['Duration_min'] = df['Duration_ms'] / 60000

    df_clean = df.dropna(subset=features_para_clusterizar).copy()

    X = df_clean[features_para_clusterizar]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_clean, X_scaled, scaler, features_para_clusterizar

def encontrar_melhor_k(X_scaled, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.title('Metodo do Cotovelo para Determinar o Numero Otimo de Clusters (K)')
    plt.xlabel('Numero de Clusters (K)')
    plt.ylabel('WCSS (Inercia)')
    plt.grid(True)
    plt.savefig('cotovelo_otimizado.png')
    plt.close()
    print("Grafico do Metodo do Cotovelo salvo como 'cotovelo_otimizado.png'")

    print("Analisar o grafico 'cotovelo_otimizado.png' para escolher o K ideal.")

def aplicar_e_analisar_kmeans(df_clean, X_scaled, scaler, features_para_clusterizar, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean['Cluster'] = clusters

    cluster_medias = df_clean.groupby('Cluster')[features_para_clusterizar].mean()
    print("\nMedias das Features por Cluster (valores originais):")
    print(cluster_medias)
    cluster_medias.to_csv('cluster_medias_otimizado.csv')
    print("Medias dos clusters salvas em 'cluster_medias_otimizado.csv'")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=50)
    plt.title('Visualizacao 2D dos Clusters (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.savefig('clusters_pca.png')
    plt.close()
    print("Visualizacao 2D dos clusters salva como 'clusters_pca.png'")

    return df_clean, cluster_medias, kmeans

if __name__ == "__main__":
    df_clean, X_scaled, scaler, features = carregar_e_preparar_dados("Spotify_Youtube.csv")
    
    encontrar_melhor_k(X_scaled)

    n_clusters_escolhido = 4 # Substitua pelo K escolhido após analisar o gráfico cotovelo

    df_final_com_clusters, medias_finais, kmeans_model = aplicar_e_analisar_kmeans(df_clean, X_scaled, scaler, features, n_clusters_escolhido)
    
    df_final_com_clusters.to_csv('dataset_processado_com_ml_otimizado.csv', index=False)
    print("Dataset final com clusters salvo como 'dataset_processado_com_ml_otimizado.csv'")

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(kmeans_model, 'kmeans_model.pkl')
    joblib.dump(features, 'features_para_clusterizar.pkl')
    joblib.dump(n_clusters_escolhido, 'n_clusters_musicas.pkl')
    print("Scaler, modelo KMeans, features e numero de clusters salvos.")
