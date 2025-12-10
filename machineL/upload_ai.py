import pandas as pd
import pickle
import numpy as np
from supabase import create_client, Client
import os

SUPABASE_URL = "https://qxkfkthihhlajmbiahqq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF4a2ZrdGhpaGhsYWptYmlhaHFxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA5ODAwNjcsImV4cCI6MjA3NjU1NjA2N30.Uv573OkhjVTAl0kniltycnF1uQtqW32G4KTXX2nYnBU"
PARQUET_PATH = 'Spotify_Youtube.parquet'

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
df = pd.read_parquet(PARQUET_PATH)
audio_features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 
                  'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms']
popularity_features = ['Views', 'Likes', 'Comments', 'Stream']
all_features = audio_features + popularity_features
df = df.dropna(subset=all_features).reset_index(drop=True)
with open('kmeans_k6.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('scaler_kmeans.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(df[all_features])
clusters = kmeans.predict(X_scaled)
df['cluster_id'] = clusters
df['cluster_id'] = df['cluster_id'].astype(int)
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
for i in range(0, total, batch_size):
    batch = records[i : i + batch_size]
    try:
        response = supabase.table('musicas').upsert(batch).execute()
        print(f"Lote {i} a {i+batch_size} enviado.")
    except Exception as e:
        print(f"Erro no lote {i}: {e}")
        break
print("Processo finalizado!")