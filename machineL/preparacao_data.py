import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib # Para salvar o scaler

def prepare_data(input_parquet_path, output_scaled_parquet_path, output_cleaned_parquet_path, output_scaler_path):
    """
    Carrega o dataset, limpa valores nulos e escala as features numéricas.
    """
    df = pd.read_parquet(input_parquet_path)

    # Remover a coluna 'Unnamed: 0' se existir
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("Coluna 'Unnamed: 0' removida.")

    print('Valores nulos antes da limpeza de features numéricas:')
    print(df.isnull().sum())

    # Colunas numéricas essenciais para clusterização
    numeric_cols = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Comments', 'Stream']

    # Remove linhas onde essas colunas numéricas têm valores nulos
    df.dropna(subset=numeric_cols, inplace=True)

    print('\nValores nulos depois da limpeza de features numéricas:')
    print(df.isnull().sum())

    df.to_parquet(output_cleaned_parquet_path, index=False)
    print(f'Dataset limpo salvo como {output_cleaned_parquet_path}')

    # Selecionar features numéricas para clusterização
    X = df[numeric_cols]

    # Escalonar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Salvar o scaler para uso posterior (ex: para escalar novas músicas)
    joblib.dump(scaler, output_scaler_path)
    print(f'Scaler salvo como {output_scaler_path}')

    pd.DataFrame(X_scaled, columns=numeric_cols).to_parquet(output_scaled_parquet_path, index=False)
    print(f'Features escalonadas salvas em {output_scaled_parquet_path}')
    
    return df, X_scaled, numeric_cols

if __name__ == '__main__':
    original_df, scaled_data, features = prepare_data(
        input_parquet_path='Spotify_Youtube.parquet',
        output_scaled_parquet_path='Spotify_Youtube_scaled_features.parquet',
        output_cleaned_parquet_path='Spotify_Youtube_cleaned.parquet',
        output_scaler_path='scaler.joblib'
    )