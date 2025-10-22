import pandas as pd
import joblib
import random

def simulate_recommendation(
    scaled_features_path,
    cleaned_data_path,
    rf_model_path,
    et_model_path,
    lr_model_path,
    scaler_path,
    cluster_profiles_path,
    clustered_data_path # Adicionado este argumento
):
    """
    Simula um sistema de recomendação baseado nos clusters e modelos treinados.
    """
    print("\n--- Simulador de Recomendacao de Musicas ---")

    # Carregar dados e modelos
    df_cleaned = pd.read_parquet(cleaned_data_path)
    # X_scaled = pd.read_parquet(scaled_features_path) # Não é necessário carregar aqui, pois usaremos o scaler
    rf_model = joblib.load(rf_model_path)
    et_model = joblib.load(et_model_path)
    lr_model = joblib.load(lr_model_path)
    scaler = joblib.load(scaler_path)
    cluster_profiles = pd.read_csv(cluster_profiles_path, index_col='Cluster_KMeans') # Linha corrigida

    # Selecionar um modelo para previsão (usaremos Extra Trees por ter a maior acurácia de teste)
    prediction_model = et_model
    print(f"Usando o modelo Extra Trees para previsao de clusters.")

    # --- Parte 1: Simular a preferência do usuário com base em uma música de exemplo ---
    print("\n### Cenario 1: Usuario gosta de uma musica existente ###")
    # Escolher uma música aleatória do dataset para simular o gosto do usuário
    random_music_index = random.randint(0, len(df_cleaned) - 1)
    user_liked_music = df_cleaned.iloc[random_music_index]
    
    print(f"Musica que o usuario gostou (exemplo): ")
    print(f"  Titulo: {user_liked_music['Title']}")
    print(f"  Artista: {user_liked_music['Artist']}")

    # Prever o cluster da música que o usuário gostou
    # Precisamos escalar as features da música antes de prever
    # Pegar as colunas numéricas que foram usadas para o escalonamento original
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Remover 'Cluster_KMeans' se estiver presente e outras colunas que não são features
    if 'Cluster_KMeans' in numeric_cols: numeric_cols.remove('Cluster_KMeans')
    if 'Unnamed: 0' in numeric_cols: numeric_cols.remove('Unnamed: 0')
    
    user_music_features_raw = user_liked_music[numeric_cols].values.reshape(1, -1)
    user_music_features_scaled = scaler.transform(user_music_features_raw)
    
    # Prever o cluster usando o modelo treinado
    predicted_cluster = prediction_model.predict(user_music_features_scaled)[0]
    print(f"  Cluster previsto para esta musica: {predicted_cluster}")

    # Recomendar outras músicas do mesmo cluster
    # Carregar o dataset clusterizado para ter os rótulos de cluster
    df_clustered_with_labels = pd.read_parquet(clustered_data_path)
    
    recommended_musics = df_clustered_with_labels[df_clustered_with_labels["Cluster_KMeans"] == predicted_cluster]
    
    # Remover a própria música que o usuário gostou da lista de recomendações
    # Para isso, precisamos de um identificador único. Usaremos o índice original do DataFrame.
    # Se o DataFrame original não tiver um índice único persistente, isso pode ser problemático.
    # Por simplicidade, vamos usar o Título e Artista para evitar a própria música, mas idealmente seria um ID único.
    recommended_musics = recommended_musics[
        (recommended_musics['Title'] != user_liked_music['Title']) |
        (recommended_musics['Artist'] != user_liked_music['Artist'])
    ]
    
    if len(recommended_musics) > 5:
        recommended_musics = recommended_musics.sample(5, random_state=42) # random_state para reprodutibilidade
    
    print(f"\nRecomendacoes para o usuario (do Cluster {predicted_cluster}):")
    if not recommended_musics.empty:
        for i, row in recommended_musics.iterrows():
            print(f"  - {row['Title']} por {row['Artist']}")
    else:
        print("  Nenhuma outra musica encontrada neste cluster para recomendar.")

    # --- Parte 2: Simular a preferência do usuário com base em um perfil de cluster ---
    print("\n### Cenario 2: Usuario tem preferencia por um perfil de cluster (sem musica inicial) ###")
    # O usuário pode indicar preferência por um tipo de música (ex: "gosto de músicas com alta energia")
    # Ou podemos simplesmente escolher um cluster aleatoriamente para simular uma preferência inicial
    # Corrigido: converter o array numpy para lista antes de passar para random.choice
    preferred_cluster_id = random.choice(df_clustered_with_labels['Cluster_KMeans'].unique().tolist())
    print(f"Simulando que o usuario prefere o perfil do Cluster {preferred_cluster_id}.")
    print("Caracteristicas medias deste cluster:")
    print(cluster_profiles.loc[preferred_cluster_id])

    # Recomendar músicas diretamente deste cluster
    recommended_musics_from_profile = df_clustered_with_labels[
        df_clustered_with_labels["Cluster_KMeans"] == preferred_cluster_id
    ]
    if len(recommended_musics_from_profile) > 5:
        recommended_musics_from_profile = recommended_musics_from_profile.sample(5, random_state=43)

    print(f"\nRecomendacoes para o usuario (do Cluster {preferred_cluster_id} baseado no perfil):")
    if not recommended_musics_from_profile.empty:
        for i, row in recommended_musics_from_profile.iterrows():
            print(f"  - {row['Title']} por {row['Artist']}")
    else:
        print("  Nenhuma musica encontrada neste cluster para recomendar.")

    # --- Parte 3: Previsão de avaliação (conceitual) ---
    print("\n### Cenario 3: Previsao de Avaliacao (Conceitual) ###")
    print("Se o usuario tem uma forte afinidade com o Cluster X (identificado pelos cenarios acima),")
    print("e uma nova musica eh classificada pelo modelo de previsao como pertencente ao Cluster X,")
    print("entao a previsao de avaliacao para essa musica seria alta (provavel que o usuario goste).")
    print("Caso contrario, a previsao de avaliacao seria baixa ou neutra.")
    print("Isso eh inferido pela compatibilidade da musica com o perfil de cluster do usuario.")

if __name__ == '__main__':
    simulate_recommendation(
        scaled_features_path='Spotify_Youtube_scaled_features.parquet',
        cleaned_data_path='Spotify_Youtube_cleaned.parquet',
        rf_model_path='random_forest_model.joblib',
        et_model_path='extra_trees_model.joblib',
        lr_model_path='logistic_regression_model.joblib',
        scaler_path='scaler.joblib',
        clustered_data_path='Spotify_Youtube_clustered_kmeans.parquet',
        cluster_profiles_path='cluster_profiles.csv'
    )

