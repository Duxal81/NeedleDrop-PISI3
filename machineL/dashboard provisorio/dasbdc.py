import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

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

df_cluster_profile = df_raw.groupby('Cluster')[AUDIO_POPULARITY_FEATURES].mean().reset_index()


app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    html.H1(" Análise de Clusters de Músicas (K=6) ",
            style={'textAlign': 'center', 'color': "#AF0C0C"}),

    html.Hr(style={'borderTop': '2px solid #ccc'}),

    html.Div([
        html.H3("Distribuição de Observações por Cluster", style={'textAlign': 'center', 'color': '#333'}),
        dcc.Graph(
            id='cluster-distribution',
            figure=px.bar(
                df_raw['Cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'count': 'count'}),
                x='Cluster',
                y='count',
                title='Contagem de Músicas em Cada Cluster',
                labels={'Cluster': 'ID do Cluster', 'count': 'Contagem'},
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Bold
            ).update_layout(showlegend=False)
        )
    ], className='row', style={'marginBottom': '20px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),

    html.Hr(style={'borderTop': '2px solid #ccc'}),

    html.Div([
        html.H3("Perfil Médio dos Clusters por Característica", style={'textAlign': 'center', 'color': '#333'}),

        html.Div([
            html.Label("Selecione a Característica:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in AUDIO_POPULARITY_FEATURES],
                value=AUDIO_POPULARITY_FEATURES[0],
                clearable=False,
                style={'width': '50%', 'margin': '10px auto'}
            )
        ], style={'textAlign': 'center'}),

        dcc.Graph(id='cluster-profile-bar')
    ], className='row', style={'marginBottom': '20px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),

    html.Hr(style={'borderTop': '2px solid #ccc'}),

    html.Div([
        html.H3("Perfil Normalizado (Radar Plot)", style={'textAlign': 'center', 'color': '#333'}),
        html.P("Este gráfico exibe o valor médio das características NORMALIZADAS para o perfil do cluster."),
        dcc.Graph(id='cluster-profile-radar')
    ], className='row', style={'marginBottom': '20px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
])

@app.callback(
    Output('cluster-profile-bar', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_profile_bar_graph(selected_feature):
    fig = px.bar(
        df_cluster_profile,
        x='Cluster',
        y=selected_feature,
        title=f'Valor Médio de {selected_feature} por Cluster',
        labels={'Cluster': 'ID do Cluster', selected_feature: f'Média de {selected_feature}'},
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    return fig

@app.callback(
    Output('cluster-profile-radar', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_profile_radar_graph(selected_feature):
    X_scaled_df = pd.DataFrame(X_scaled, columns=AUDIO_POPULARITY_FEATURES)
    X_scaled_df['Cluster'] = df_raw['Cluster']
    df_scaled_profile = X_scaled_df.groupby('Cluster')[AUDIO_POPULARITY_FEATURES].mean().reset_index()

    fig = go.Figure()

    for i in range(K_VALUE):
        cluster_id = str(i)
        if not df_scaled_profile[df_scaled_profile['Cluster'] == cluster_id].empty:
            cluster_data = df_scaled_profile[df_scaled_profile['Cluster'] == cluster_id].iloc[0]
            
            categories = AUDIO_POPULARITY_FEATURES[:10]
            values = cluster_data[categories].values
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'Cluster {cluster_id}'
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 3]
            )),
        showlegend=True,
        title='Perfil dos Clusters (Dados Padronizados - 10 Primeiras Características)'
    )
    
    return fig

if __name__ == '__main__':
    print("\nIniciando o Dashboard Dash...")
    print("Acesse em: http://127.0.0.1:8050/")
    app.run(debug=True, use_reloader=False)