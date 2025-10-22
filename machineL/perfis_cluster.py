import pandas as pd

def analyze_cluster_profiles(clustered_data_path, output_profile_path):
    """
    Calcula o perfil médio de cada cluster e o salva em um arquivo CSV.
    """
    df = pd.read_parquet(clustered_data_path)

    # Selecionar apenas colunas numéricas e a coluna do cluster
    # Excluir 'Unnamed: 0' se presente, pois não é uma feature
    numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns.tolist() if col != 'Unnamed: 0']
    
    # Garantir que 'Cluster_KMeans' está nas colunas numéricas para o groupby, mas não para o cálculo da média das features
    if 'Cluster_KMeans' not in numeric_cols:
        features_to_profile = numeric_cols
    else:
        features_to_profile = [col for col in numeric_cols if col != 'Cluster_KMeans']

    # Calcular a média das features para cada cluster
    cluster_profiles = df[features_to_profile + ['Cluster_KMeans']].groupby('Cluster_KMeans').mean()

    print("\n--- Perfil dos Clusters (Valores Medios das Features) ---")
    print(cluster_profiles)

    # Salvar os perfis para consulta
    cluster_profiles.to_csv(output_profile_path)
    print(f"\nPerfil dos clusters salvo em: {output_profile_path}")

    return cluster_profiles

if __name__ == '__main__':
    analyze_cluster_profiles(
        clustered_data_path='Spotify_Youtube_clustered_kmeans.parquet',
        output_profile_path='cluster_profiles.csv'
    )

