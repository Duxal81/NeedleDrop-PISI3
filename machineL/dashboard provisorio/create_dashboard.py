import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 

PARQUET_PATH = 'Spotify_Youtube.parquet' 
KMEANS_MODEL_PATH = 'kmeans_k6.pkl'
SCALER_KMEANS_PATH = 'scaler_kmeans.pkl'
K_VALUE = 6
RANDOM_STATE = 42

try:
    print("Carregando dados...")
    df_raw = pd.read_parquet(PARQUET_PATH)

    AUDIO_POPULARITY_FEATURES = [
        'Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence',
        'Tempo', 'Duration_ms', 'Views', 'Likes', 'Comments', 'Stream'
    ]

    print("Carregando modelos...")
    with open(KMEANS_MODEL_PATH, 'rb') as f:
        kmeans_model = pickle.load(f)

    with open(SCALER_KMEANS_PATH, 'rb') as f:
        scaler = pickle.load(f)

    df_features = df_raw[AUDIO_POPULARITY_FEATURES].copy()

    print("Tratando valores NaN (Imputacao por Media)...")
    
    imputer = SimpleImputer(strategy='mean')
    
    X_imputed = imputer.fit_transform(df_features)

    X_scaled = scaler.transform(X_imputed)
        
    df_raw['Cluster'] = kmeans_model.predict(X_scaled)
    df_raw['Cluster'] = df_raw['Cluster'].astype(str)

except FileNotFoundError as e:
    print(f"ERRO: Arquivo não encontrado. Certifique-se de que '{e.filename}' esta no diretorio correto.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro durante o carregamento/preparacao dos dados: {e}")
    exit()

print("Calculando perfis dos clusters...")
df_cluster_profile = df_raw.groupby('Cluster')[AUDIO_POPULARITY_FEATURES].mean().reset_index()

X_scaled_df = pd.DataFrame(X_scaled, columns=AUDIO_POPULARITY_FEATURES)
X_scaled_df['Cluster'] = df_raw['Cluster']
df_scaled_profile = X_scaled_df.groupby('Cluster')[AUDIO_POPULARITY_FEATURES].mean().reset_index()

dashboard_data = {
    'df_raw': df_raw,
    'df_cluster_profile': df_cluster_profile,
    'df_scaled_profile': df_scaled_profile,
    'AUDIO_POPULARITY_FEATURES': AUDIO_POPULARITY_FEATURES,
    'K_VALUE': K_VALUE,
    'X_scaled': X_scaled
}

print("Salvando dados processados...")
with open('dashboard_data.pkl', 'wb') as f:
    pickle.dump(dashboard_data, f)

print("SUCESSO: Dados salvos como 'dashboard_data.pkl'!")
print("PROXIMO PASSO: Execute -> python run_dashboard.py")