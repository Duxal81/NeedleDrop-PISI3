import pandas as pd
import pickle
import numpy as np
from supabase import create_client, Client

SUPABASE_URL = "https://qxkfkthihhlajmbiahqq.supabase.com"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF4a2ZrdGhpaGhsYWptYmlhaHFxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA5ODAwNjcsImV4cCI6MjA3NjU1NjA2N30.Uv573OkhjVTAl0kniltycnF1uQtqW32G4KTXX2nYnBU"
PARQUET_PATH = 'Spotify_Youtube.parquet'
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Carregando arquivos...")
df = pd.read_parquet(PARQUET_PATH)
audio_features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 
                  'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms']
df = df.dropna(subset=audio_features).reset_index(drop=True)
with open('kmeans_k6.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('scaler_kmeans.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Aplicando modelo K-Means...")
X_scaled = scaler.transform(df[audio_features])
clusters = kmeans.predict(X_scaled)
df['cluster_id'] = clusters
df['cluster_id'] = df['cluster_id'].astype(int)
print("Preparando dados para envio...")
data_to_upload = df.rename(columns={
    'Title': 'title',
    'Artist': 'artist',
    'Url_youtube': 'youtube_url',
    'Url_spotify': 'spotify_url',
    'Valence': 'valence',
    'Energy': 'energy',
    'Danceability': 'danceability'
})
cols_final = ['title', 'artist', 'youtube_url', 'spotify_url', 'cluster_id', 'valence', 'energy', 'danceability']
records = data_to_upload[cols_final].to_dict(orient='records')
batch_size = 1000
total = len(records)
print(f"Iniciando upload de {total} músicas...")
for i in range(0, total, batch_size):
    batch = records[i : i + batch_size]
    try:
        response = supabase.table('songs').upsert(batch).execute()
        print(f"Lote {i} a {i+batch_size} enviado com sucesso.")
    except Exception as e:
        print(f"Erro no lote {i}: {e}")
print("Processo finalizado!")