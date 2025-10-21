import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

def simular_e_clusterizar_usuarios(df_musicas_com_clusters, features_para_clusterizar_musicas, n_clusters_musicas, n_usuarios_simulados=100, n_musicas_por_usuario=10):
    # Carregar scaler e kmeans do modelo de músicas
    scaler_musicas = joblib.load("scaler.pkl")
    kmeans_musicas = joblib.load("kmeans_model.pkl")

    # 1. Simular perfis de usuários
    # Cada usuário terá um 'gosto' primário que se alinha a um cluster de músicas
    np.random.seed(42)
    
    perfis_usuarios = []
    for i in range(n_usuarios_simulados):
        # Cada usuário é 'atraído' por um cluster de músicas específico
        cluster_preferido = np.random.randint(0, n_clusters_musicas)
        
        # Selecionar aleatoriamente N músicas desse cluster para o usuário ter 'gostado'
        musicas_gostadas_ids = df_musicas_com_clusters[df_musicas_com_clusters["Cluster"] == cluster_preferido].sample(n=n_musicas_por_usuario, replace=True).index
        musicas_gostadas = df_musicas_com_clusters.loc[musicas_gostadas_ids]
        
        # Calcular o perfil médio do usuário com base nas músicas que ele 'gostou'
        perfil_medio_usuario = musicas_gostadas[features_para_clusterizar_musicas].mean().values
        perfis_usuarios.append(perfil_medio_usuario)

    df_perfis_usuarios = pd.DataFrame(perfis_usuarios, columns=features_para_clusterizar_musicas)
    df_perfis_usuarios.index.name = "UserID"
    df_perfis_usuarios.to_csv("perfis_usuarios_simulados.csv")
    print("Perfis de usuarios simulados salvos em 'perfis_usuarios_simulados.csv'")

    # 2. Clusterizar os usuários
    # Padronizar os perfis de usuários usando o mesmo scaler das músicas (opcional, mas boa prática se as features forem as mesmas)
    # Ou treinar um novo scaler se as features forem diferentes ou se quiser um escalonamento independente
    scaler_usuarios = StandardScaler()
    perfis_usuarios_scaled = scaler_usuarios.fit_transform(df_perfis_usuarios)

    # Encontrar o K ideal para usuários (Método do Cotovelo - opcional, mas recomendado)
    # encontrar_melhor_k(perfis_usuarios_scaled, max_clusters=5) # Você pode rodar isso separadamente

    n_clusters_usuarios = 3 # Exemplo: 3 grupos de usuários. Ajuste após analisar o cotovelo se rodar.
    kmeans_usuarios = KMeans(n_clusters=n_clusters_usuarios, init="k-means++", random_state=42, n_init=10)
    clusters_usuarios = kmeans_usuarios.fit_predict(perfis_usuarios_scaled)
    df_perfis_usuarios["User_Cluster"] = clusters_usuarios

    df_perfis_usuarios.to_csv("usuarios_com_clusters.csv")
    print("Usuarios com clusters atribuidos salvos em 'usuarios_com_clusters.csv'")

    # Analisar médias dos clusters de usuários
    cluster_medias_usuarios = df_perfis_usuarios.groupby("User_Cluster")[features_para_clusterizar_musicas].mean()
    print("\nMedias das Features por Cluster de Usuario:")
    print(cluster_medias_usuarios)
    cluster_medias_usuarios.to_csv("cluster_medias_usuarios.csv")
    print("Medias dos clusters de usuarios salvas em 'cluster_medias_usuarios.csv'")

    joblib.dump(scaler_usuarios, "scaler_usuarios.pkl")
    joblib.dump(kmeans_usuarios, "kmeans_usuarios.pkl")
    print("Scaler e modelo KMeans de usuarios salvos como 'scaler_usuarios.pkl' e 'kmeans_usuarios.pkl'")

    return df_perfis_usuarios, cluster_medias_usuarios

# Exemplo de uso:
if __name__ == "__main__":
    # Carregar o dataset de músicas já clusterizado
    df_musicas_com_clusters = pd.read_csv("dataset_processado_com_ml_otimizado.csv")
    features_para_clusterizar_musicas = joblib.load("features_para_clusterizar.pkl")
    n_clusters_musicas = joblib.load("n_clusters_musicas.pkl")

    df_usuarios_com_clusters, medias_clusters_usuarios = simular_e_clusterizar_usuarios(
        df_musicas_com_clusters,
        features_para_clusterizar_musicas,
        n_clusters_musicas,
        n_usuarios_simulados=100,
        n_musicas_por_usuario=10
    )

    print("\nSimulacao e Clusterizacao de Usuarios Concluidas.")
