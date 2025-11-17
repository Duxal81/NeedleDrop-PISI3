import dash
from dash import dcc, html, Input, Output, callback
from dash.dash_table import DataTable
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os

# --- Configurações Iniciais e Carregamento de Dados ---
# É necessário importar DataTable separadamente
try:
    from dash.dash_table import DataTable
except ImportError:
    print("AVISO: dash.dash_table não encontrado. A tabela de resumo não será exibida corretamente.")
    DataTable = None


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard de Análise de Clusters - Recomendação Musical"

df_full = pd.DataFrame()
df_clusters = pd.DataFrame()

try:
    # 1. Carrega o dataset completo com os clusters (Necessário para a distribuição)
    FULL_DATA_PATH = 'Spotify_Youtube_clustered_kmeans.parquet'
    if os.path.exists(FULL_DATA_PATH):
        df_full = pd.read_parquet(FULL_DATA_PATH)
        # Renomear para padronizar
        if 'Cluster_KMeans' in df_full.columns:
            df_full.rename(columns={'Cluster_KMeans': 'Cluster'}, inplace=True)
            df_full['Cluster'] = df_full['Cluster'].astype(str)
        print(f"SUCESSO: Dataset completo carregado de {FULL_DATA_PATH}.")
    else:
        print(f"AVISO: Arquivo '{FULL_DATA_PATH}' não encontrado. O gráfico de distribuição não funcionará.")
    
    # 2. Carrega os perfis dos clusters (Essencial para as médias)
    PROFILE_DATA_PATH = 'cluster_profiles.csv'
    if os.path.exists(PROFILE_DATA_PATH):
        df_clusters = pd.read_csv(PROFILE_DATA_PATH)
        df_clusters.rename(columns={'Cluster_KMeans': 'Cluster'}, inplace=True)
        df_clusters['Cluster'] = df_clusters['Cluster'].astype(str)
        print(f"SUCESSO: Perfis dos clusters carregados de {PROFILE_DATA_PATH}.")
    else:
        print(f"ERRO CRÍTICO: Arquivo '{PROFILE_DATA_PATH}' não encontrado. O dashboard não funcionará.")

except Exception as e:
    print(f"ERRO durante o carregamento dos dados: {e}")

# --- Definição das Colunas ---
if not df_clusters.empty:
    music_features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
                      'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Key']
    popularity_metrics = ['Views', 'Likes', 'Comments', 'Stream']
    
    available_music_features = [f for f in music_features if f in df_clusters.columns]
    available_popularity_metrics = [m for m in popularity_metrics if m in df_clusters.columns]

# --- Layout do Dashboard ---
if df_clusters.empty:
    app.layout = dbc.Container(html.H1("ERRO: Não foi possível carregar os dados do cluster ('cluster_profiles.csv').", className="text-center my-4 text-danger"), fluid=True)
else:
    # Cabeçalho
    header = dbc.Row(
        dbc.Col(
            html.Div([
                html.H1("Dashboard de Perfis de Clusters (K-Means)", className="text-center text-primary"),
                html.P("Análise detalhada das características e popularidade médias para cada grupo de recomendação musical.", className="text-center lead"),
                html.Hr()
            ]),
            width=12
        ),
        className="mb-4 mt-2"
    )

    # Gráfico de Distribuição dos Clusters (Novo)
    distribution_card = dbc.Card([
        dbc.CardHeader(html.H4("1. Distribuição de Frequência dos Clusters")),
        dbc.CardBody([
            html.P("A contagem de músicas em cada cluster do dataset completo ('Spotify_Youtube_clustered_kmeans.parquet').", className="card-text text-muted"),
            dcc.Graph(id='cluster-distribution-chart')
        ])
    ], className="mb-4 shadow")

    # Tabela de Perfis dos Clusters (Novo)
    if DataTable:
        profile_table = dbc.Card([
            dbc.CardHeader(html.H4("2. Tabela de Perfis (Valores Médios Brutos)")),
            dbc.CardBody([
                html.P("Valores médios das características e popularidade por cluster para referência.", className="card-text text-muted"),
                DataTable(
                    id='cluster-profile-table',
                    columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ".3f"}} for i in df_clusters.columns if i != 'Cluster'] + [{"name": "Cluster", "id": "Cluster"}],
                    data=df_clusters.to_dict('records'),
                    style_table={'overflowX': 'auto', 'margin': '10px'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                )
            ])
        ], className="mb-4 shadow")
    else:
        profile_table = html.Div(html.P("Tabela de Perfis não disponível (dash.dash_table não importado)."), className="mb-4 alert alert-warning")

    # Gráficos de Análise
    analysis_row_1 = dbc.Row([
        # Radar Chart
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("3. Perfis Sonoros (Radar Chart)")),
            dbc.CardBody([
                html.P("Comparação das features musicais (normalizadas 0-1) que definem o perfil sonoro de cada cluster.", className="card-text text-muted"),
                dcc.Graph(id='cluster-radar-chart')
            ])
        ], className="shadow"), md=6),
        
        # Popularity Bar Chart
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("4. Popularidade Média")),
            dbc.CardBody([
                html.P("Métricas de popularidade por cluster. O eixo Y está em escala logarítmica para facilitar a visualização.", className="card-text text-muted"),
                dcc.Graph(id='popularity-bar-chart')
            ])
        ], className="shadow"), md=6),
    ], className="mb-4")

    # Heatmap
    heatmap_card = dbc.Card([
        dbc.CardHeader(html.H4("5. Heatmap de Perfil Completo (Normalizado 0-1)")),
        dbc.CardBody([
            html.P("Visão geral da intensidade de todas as características e métricas (normalizadas) por cluster.", className="card-text text-muted"),
            dcc.Graph(id='cluster-profile-heatmap')
        ])
    ], className="mb-4 shadow")

    app.layout = dbc.Container([
        header,
        distribution_card,
        profile_table,
        analysis_row_1,
        heatmap_card
    ], fluid=True)

    # --- Callbacks ---

    # Callback 1: Distribuição dos Clusters (Novo)
    @app.callback(
        Output('cluster-distribution-chart', 'figure'),
        Input('cluster-distribution-chart', 'id')
    )
    def update_cluster_distribution_chart(_):
        if 'Cluster' in df_full.columns and not df_full.empty:
            cluster_counts = df_full['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Contagem']
            
            fig = px.bar(cluster_counts, 
                         x='Cluster', 
                         y='Contagem', 
                         color='Cluster',
                         title='Distribuição de Músicas por Cluster',
                         labels={'Contagem': 'Número de Músicas', 'Cluster': 'Cluster K-Means'})
            fig.update_layout(xaxis={'categoryorder':'category ascending'})
            return fig
        
        return go.Figure().add_annotation(text="Dados de distribuição (Spotify_Youtube_clustered_kmeans.parquet) não disponíveis.", x=0.5, y=0.5, showarrow=False)

    # Callback 2: Cluster Radar Chart (Mantido)
    @app.callback(
        Output('cluster-radar-chart', 'figure'),
        Input('cluster-radar-chart', 'id')
    )
    def update_cluster_radar_chart(_):
        fig = go.Figure()
        
        radar_features = [f for f in available_music_features if f in ['Danceability', 'Energy', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence']]
        df_plot = df_clusters[['Cluster'] + radar_features].set_index('Cluster')
        
        # Normalização por coluna
        df_plot_normalized = df_plot.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0).fillna(0)

        for cluster in df_plot_normalized.index:
            r_values = df_plot_normalized.loc[cluster].tolist()
            original_values = df_plot.loc[cluster].tolist()
            
            hover_text = [f'{feat}: {orig_val:.3f}' for feat, orig_val in zip(radar_features, original_values)]
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=radar_features,
                fill='toself',
                name=f'Cluster {cluster}',
                hoverinfo='text',
                text=hover_text
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=20, l=20, r=20)
        )
        return fig

    # Callback 3: Bar Chart de Popularidade (Mantido)
    @app.callback(
        Output('popularity-bar-chart', 'figure'),
        Input('popularity-bar-chart', 'id')
    )
    def update_popularity_bar_chart(_):
        df_pop = df_clusters[['Cluster'] + available_popularity_metrics]
        df_pop_melt = df_pop.melt(id_vars='Cluster', value_vars=available_popularity_metrics,
                                  var_name='Métrica', value_name='Valor Médio')
        
        fig = px.bar(df_pop_melt, 
                     x='Cluster', 
                     y='Valor Médio', 
                     color='Métrica',
                     barmode='group',
                     log_y=True,
                     title='Métricas de Popularidade Média por Cluster',
                     labels={'Valor Médio': 'Valor Médio (Log Scale)', 'Cluster': 'Cluster K-Means'})
        
        return fig

    # Callback 4: Heatmap do Perfil (Mantido)
    @app.callback(
        Output('cluster-profile-heatmap', 'figure'),
        Input('cluster-profile-heatmap', 'id')
    )
    def update_cluster_profile_heatmap(_):
        heatmap_cols = available_music_features + available_popularity_metrics
        df_heatmap = df_clusters[['Cluster'] + heatmap_cols].set_index('Cluster')

        # Normalização por coluna (Feature Scaling Min-Max)
        df_heatmap_normalized = df_heatmap.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0).fillna(0)
        
        fig = px.imshow(df_heatmap_normalized,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale='Viridis',
                        title='Perfil de Clusters Normalizado (0 a 1)',
                        labels={'x': 'Feature/Métrica', 'y': 'Cluster'})
        
        fig.update_yaxes(tickvals=df_heatmap_normalized.index.tolist(), ticktext=[f'Cluster {c}' for c in df_heatmap_normalized.index])
        fig.update_layout(xaxis_title="Características e Métricas", yaxis_title="Cluster K-Means", height=600)
        
        return fig

# --- Inicia o Servidor ---
if __name__ == '__main__':
    print("\n*** Dashboard Pronto ***")
    print("Para acessar: abra seu navegador e vá para http://127.0.0.1:8050/")
    # Desative o debug no ambiente de produção
    app.run(debug=True)