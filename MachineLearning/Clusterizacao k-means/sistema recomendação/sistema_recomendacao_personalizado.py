import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import joblib
import numpy as np

# --- Opcoes de exibicao do Pandas para mostrar todas as colunas ---
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
# ------------------------------------------------------------------

# Carregar dados e modelos previamente salvos
# Certifique-se de que esses arquivos foram gerados pelos scripts anteriores
df_musicas_com_clusters = pd.read_csv("dataset_processado_com_ml_otimizado.csv")
df_usuarios_com_clusters = pd.read_csv("usuarios_com_clusters.csv", index_col="UserID")

scaler_musicas = joblib.load("scaler.pkl")
kmeans_musicas = joblib.load("kmeans_model.pkl")
kmeans_usuarios = joblib.load("kmeans_usuarios.pkl")
scaler_usuarios = joblib.load("scaler_usuarios.pkl")
features_para_clusterizar_musicas = joblib.load("features_para_clusterizar.pkl")

def recomendar_para_usuario(user_id, df_musicas, df_usuarios, kmeans_musicas, kmeans_usuarios, scaler_musicas, scaler_usuarios, features_musicas, num_recomendacoes=10, max_musicas_por_artista=1):
    """
    Recomenda musicas para um usuario especifico, considerando o cluster do usuario
    e a similaridade dentro dos clusters de musicas.

    Args:
        user_id (int): ID do usuario para quem fazer a recomendacao.
        df_musicas (pd.DataFrame): DataFrame de musicas com seus clusters.
        df_usuarios (pd.DataFrame): DataFrame de usuarios com seus clusters.
        kmeans_musicas (KMeans): Modelo KMeans treinado para musicas.
        kmeans_usuarios (KMeans): Modelo KMeans treinado para usuarios.
        scaler_musicas (StandardScaler): Scaler usado para as features das musicas.
        scaler_usuarios (StandardScaler): Scaler usado para as features dos usuarios.
        features_musicas (list): Lista de features usadas para clusterizar musicas.
        num_recomendacoes (int): Numero de recomendacoes a serem retornadas.
        max_musicas_por_artista (int): Numero maximo de musicas permitidas por artista.

    Returns:
        pd.DataFrame: Musicas recomendadas para o usuario.
    """
    if user_id not in df_usuarios.index:
        print(f"Erro: Usuario com ID {user_id} nao encontrado.")
        return pd.DataFrame()

    user_cluster = int(df_usuarios.loc[user_id]["User_Cluster"]) # Garante que o indice seja um inteiro

    # Encontrar o centroide do cluster de usuarios
    centroide_user_cluster_scaled = kmeans_usuarios.cluster_centers_[user_cluster]
    
    # Encontrar o cluster de musicas mais 'proximo' ao perfil medio do cluster de usuarios
    # Isso pode ser feito calculando a distancia do centroide do cluster de usuarios
    # para os centroides de TODOS os clusters de musicas.
    
    # Centroides dos clusters de musicas ja estao em escala normalizada
    distancias_para_clusters_musicas = euclidean_distances(
        centroide_user_cluster_scaled.reshape(1, -1),
        kmeans_musicas.cluster_centers_
    )
    cluster_musicas_mais_proximo = np.argmin(distancias_para_clusters_musicas)
    print(f"Usuario {user_id} pertence ao Cluster de Usuario {user_cluster}.")
    print(f"O cluster de musicas mais proximo ao perfil medio do Cluster de Usuario {user_cluster} e o Cluster {cluster_musicas_mais_proximo}.")

    # Filtrar musicas do cluster de musicas mais proximo
    candidatas_a_recomendacao = df_musicas[df_musicas["Cluster"] == cluster_musicas_mais_proximo].copy()

    # Para recomendacoes iniciais, podemos pegar as mais populares desse cluster
    # ou as que estao mais proximas do centroide do cluster de musicas.
    # Vamos pegar as mais populares (maior 'Stream' ou 'Views') como exemplo.
    if "Stream" in candidatas_a_recomendacao.columns:
        candidatas_a_recomendacao = candidatas_a_recomendacao.sort_values(by="Stream", ascending=False)
    elif "Views" in candidatas_a_recomendacao.columns:
        candidatas_a_recomendacao = candidatas_a_recomendacao.sort_values(by="Views", ascending=False)
    
    # Aplicar logica de diversidade (ja implementada na versao anterior)
    recomendacoes_finais = pd.DataFrame()
    artistas_contagem = {}
    titulos_ja_recomendados = set()

    for index, row in candidatas_a_recomendacao.iterrows():
        artista = row["Artist"]
        titulo = row["Track"]
        
        if titulo in titulos_ja_recomendados:
            continue

        if artistas_contagem.get(artista, 0) < max_musicas_por_artista:
            recomendacoes_finais = pd.concat([recomendacoes_finais, pd.DataFrame([row])])
            artistas_contagem[artista] = artistas_contagem.get(artista, 0) + 1
            titulos_ja_recomendados.add(titulo)

        if len(recomendacoes_finais) >= num_recomendacoes:
            break

    # Adicionar uma coluna 'Distancia' com NaN ou um valor placeholder, pois nao e calculada diretamente aqui
    # para manter a consistencia com o formato de saida anterior.
    recomendacoes_finais['Distancia'] = np.nan

    return recomendacoes_finais.head(num_recomendacoes)[["Artist", "Track", "Cluster", "Distancia"]]

# Exemplo de uso:
if __name__ == "__main__":
    # Para este exemplo, simula-se que o user_id 0 e o usuario atual.
    # Podem ser feitos testes com outros user_ids do 'usuarios_com_clusters.csv'
    user_id_exemplo = 0 

    print(f"\nGerando recomendacoes personalizadas para o Usuario ID: {user_id_exemplo}")
    recomendacoes_personalizadas = recomendar_para_usuario(
        user_id_exemplo, 
        df_musicas_com_clusters, 
        df_usuarios_com_clusters,
        kmeans_musicas,
        kmeans_usuarios,
        scaler_musicas,
        scaler_usuarios,
        features_para_clusterizar_musicas,
        num_recomendacoes=10,
        max_musicas_por_artista=1
    )

    if not recomendacoes_personalizadas.empty:
        print("\nMusicas recomendadas (personalizadas para o usuario):")
        print(recomendacoes_personalizadas)
    else:
        print("Nao foi possivel gerar recomendacoes personalizadas para este usuario.")