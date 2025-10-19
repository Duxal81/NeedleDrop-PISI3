import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Análise Exploratória Spotify & YouTube"

try:
    df = pd.read_parquet('data_parquet/Spotify_Youtube.parquet')
    
    if df.empty:
        raise ValueError("Dataset está vazio")
        
except Exception as e:
    print(f"Erro ao carregar dataset: {e}")
    df = pd.DataFrame()

def safe_clean_text(text):
    try:
        if isinstance(text, str):
            return text.replace('$', 'S').replace('\\', '/').replace('_', ' ')
        return text
    except:
        return text

if not df.empty:
    text_columns = ['Artist', 'Track', 'Album', 'Title', 'Channel']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_clean_text)
    
    if 'Duration_ms' in df.columns:
        df['Duration_min'] = df['Duration_ms'] / 60000

def safe_value_counts(series, top_n=10):
    """Contagem segura de valores"""
    try:
        if series is not None and not series.empty:
            return series.value_counts().head(top_n)
        return pd.Series()
    except:
        return pd.Series()

def safe_mean(series):
    try:
        if series is not None and not series.empty:
            return series.mean()
        return 0
    except:
        return 0

def safe_nunique(series):
    try:
        if series is not None and not series.empty:
            return series.nunique()
        return 0
    except:
        return 0


COLOR_PALETTE = {
    'sequential': 'Teal',        
    'diverging': 'RdBu',        
    'categorical': 'Viridis',     
    'heatmap': 'Blues',        
    'single_color': "#2E758B"     
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Análise Exploratória - Spotify & YouTube", 
                       className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Informações do Dataset"),
                dbc.CardBody([
                    html.P(f"Shape: {df.shape if not df.empty else 'N/A'}"),
                    html.P(f"Total de músicas: {len(df) if not df.empty else 0}"),
                    html.P(f"Total de artistas: {safe_nunique(df.get('Artist')) if not df.empty else 0}"),
                    html.P(f"Colunas: {len(df.columns) if not df.empty else 0}")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Estatísticas Numéricas"),
                dbc.CardBody([
                    html.P(f"Média de visualizações: {safe_mean(df.get('Views')):,.0f}" if not df.empty and 'Views' in df.columns else "Média de visualizações: N/A"),
                    html.P(f"Média de streams: {safe_mean(df.get('Stream')):,.0f}" if not df.empty and 'Stream' in df.columns else "Média de streams: N/A"),
                    html.P(f"Média de duração: {safe_mean(df.get('Duration_min')):.2f} min" if not df.empty and 'Duration_min' in df.columns else "Média de duração: N/A")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Artista"),
                dbc.CardBody([
                    html.P(f"Artista com mais músicas: {df['Artist'].value_counts().index[0] if not df.empty and 'Artist' in df.columns and not df['Artist'].empty else 'N/A'}"),
                    html.P(f"Total: {df['Artist'].value_counts().iloc[0] if not df.empty and 'Artist' in df.columns and not df['Artist'].empty else 0} músicas")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top 10 Artistas com Mais Músicas"),
                dbc.CardBody([
                    dcc.Graph(id='top-artists-chart')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribuição de Características Musicais"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='feature-selector',
                        options=[
                            {'label': 'Danceability', 'value': 'Danceability'},
                            {'label': 'Energy', 'value': 'Energy'},
                            {'label': 'Loudness', 'value': 'Loudness'},
                            {'label': 'Speechiness', 'value': 'Speechiness'},
                            {'label': 'Acousticness', 'value': 'Acousticness'},
                            {'label': 'Instrumentalness', 'value': 'Instrumentalness'},
                            {'label': 'Liveness', 'value': 'Liveness'},
                            {'label': 'Valence', 'value': 'Valence'},
                            {'label': 'Tempo', 'value': 'Tempo'}
                        ],
                        value='Danceability',
                        clearable=False
                    ),
                    dcc.Graph(id='feature-distribution')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Relação entre Variáveis Musicais (Heatmap 2D)"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='x-axis-selector',
                                options=[
                                    {'label': 'Danceability', 'value': 'Danceability'},
                                    {'label': 'Energy', 'value': 'Energy'},
                                    {'label': 'Loudness', 'value': 'Loudness'},
                                    {'label': 'Valence', 'value': 'Valence'},
                                    {'label': 'Speechiness', 'value': 'Speechiness'},
                                    {'label': 'Acousticness', 'value': 'Acousticness'}
                                ],
                                value='Danceability',
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            dcc.Dropdown(
                                id='y-axis-selector',
                                options=[
                                    {'label': 'Energy', 'value': 'Energy'},
                                    {'label': 'Danceability', 'value': 'Danceability'},
                                    {'label': 'Loudness', 'value': 'Loudness'},
                                    {'label': 'Valence', 'value': 'Valence'},
                                    {'label': 'Speechiness', 'value': 'Speechiness'},
                                    {'label': 'Acousticness', 'value': 'Acousticness'}
                                ],
                                value='Energy',
                                clearable=False
                            )
                        ], width=6)
                    ]),
                    dcc.Graph(id='variables-heatmap')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Correlação entre Métricas de Engajamento"),
                dbc.CardBody([
                    dcc.Graph(id='engagement-heatmap')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Músicas por Engajamento"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='metric-selector',
                        options=[
                            {'label': 'Visualizações no YouTube', 'value': 'Views'},
                            {'label': 'Likes no YouTube', 'value': 'Likes'},
                            {'label': 'Streams no Spotify', 'value': 'Stream'},
                            {'label': 'Comentários no YouTube', 'value': 'Comments'}
                        ],
                        value='Views',
                        clearable=False
                    ),
                    dcc.Graph(id='top-songs-chart-horizontal')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribuição de Chaves Musicais"),
                dbc.CardBody([
                    dcc.Graph(id='key-distribution-horizontal')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top 10 Álbuns com Mais Músicas"),
                dbc.CardBody([
                    dcc.Graph(id='top-albums-chart-horizontal')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Densidade: Visualizações vs Streams"),
                dbc.CardBody([
                    dcc.Graph(id='views-streams-heatmap')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Densidade: Energy vs Danceability"),
                dbc.CardBody([
                    dcc.Graph(id='energy-danceability-heatmap')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Densidade: Valence vs Energy"),
                dbc.CardBody([
                    dcc.Graph(id='valence-energy-heatmap')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Correlação entre Características Musicais"),
                dbc.CardBody([
                    dcc.Graph(id='music-features-heatmap')
                ])
            ])
        ], width=6)
    ], className="mb-4")
], fluid=True)


@app.callback(
    Output('top-artists-chart', 'figure'),
    Input('top-artists-chart', 'id')
)
def update_top_artists(_):
    try:
        if 'Artist' in df.columns and not df['Artist'].empty:
            artist_counts = safe_value_counts(df['Artist'], 10)
            if not artist_counts.empty:
                fig = px.bar(x=artist_counts.values, y=artist_counts.index, 
                           orientation='h',
                           labels={'x': 'Número de Músicas', 'y': 'Artista'},
                           title='Top 10 Artistas com Mais Músicas',
                           color=artist_counts.values,
                           color_continuous_scale=COLOR_PALETTE['sequential'])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                return fig
    except Exception as e:
        print(f"Erro em update_top_artists: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('feature-distribution', 'figure'),
    Input('feature-selector', 'value')
)
def update_feature_distribution(feature):
    try:
        if not df.empty and feature in df.columns:
            fig = px.histogram(df, x=feature, nbins=30, 
                              title=f'Distribuição de {feature}',
                              color_discrete_sequence=[COLOR_PALETTE['single_color']])
            return fig
    except Exception as e:
        print(f"Erro em update_feature_distribution: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('variables-heatmap', 'figure'),
    [Input('x-axis-selector', 'value'),
     Input('y-axis-selector', 'value')]
)
def update_variables_heatmap(x_axis, y_axis):
    try:
        if not df.empty and x_axis in df.columns and y_axis in df.columns:
            fig = px.density_heatmap(df, x=x_axis, y=y_axis,
                                    title=f'Densidade: {x_axis} vs {y_axis}',
                                    nbinsx=20, nbinsy=20,
                                    color_continuous_scale=COLOR_PALETTE['heatmap'])
            return fig
    except Exception as e:
        print(f"Erro em update_variables_heatmap: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('engagement-heatmap', 'figure'),
    Input('engagement-heatmap', 'id')
)
def update_engagement_heatmap(_):
    try:
        if not df.empty:
            engagement_metrics = ['Views', 'Likes', 'Comments', 'Stream']
            engagement_metrics = [m for m in engagement_metrics if m in df.columns]
            
            if engagement_metrics:
                corr_data = df[engagement_metrics].replace([np.inf, -np.inf], np.nan).dropna()
                if not corr_data.empty:
                    correlation_matrix = corr_data.corr()
                    fig = px.imshow(correlation_matrix, 
                                   text_auto=True, 
                                   title='Correlação entre Métricas de Engajamento',
                                   aspect="auto",
                                   color_continuous_scale=COLOR_PALETTE['diverging'],
                                   zmin=-1, zmax=1)
                    return fig
    except Exception as e:
        print(f"Erro em update_engagement_heatmap: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('top-songs-chart-horizontal', 'figure'),
    Input('metric-selector', 'value')
)
def update_top_songs_horizontal(metric):
    try:
        if not df.empty and metric in df.columns:
            top_songs = df.nlargest(10, metric)[['Artist', 'Track', metric]].dropna()
            if not top_songs.empty:
                fig = px.bar(top_songs, x=metric, y='Track', orientation='h',
                            hover_data=['Artist'],
                            title=f'Top 10 Músicas por {metric}',
                            color=metric,
                            color_continuous_scale=COLOR_PALETTE['sequential'])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                return fig
    except Exception as e:
        print(f"Erro em update_top_songs_horizontal: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('key-distribution-horizontal', 'figure'),
    Input('key-distribution-horizontal', 'id')
)
def update_key_distribution_horizontal(_):
    try:
        if 'Key' in df.columns and not df.empty:
            key_data = df['Key'].dropna()
            if not key_data.empty:
                key_counts = key_data.value_counts()
                
                key_labels = {
                    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
                }
                
                labels = [key_labels.get(k, str(k)) for k in key_counts.index]
                
                fig = px.bar(x=key_counts.values, y=labels, orientation='h',
                            labels={'x': 'Contagem', 'y': 'Key'},
                            title='Distribuição de Chaves Musicais',
                            color=key_counts.values,
                            color_continuous_scale=COLOR_PALETTE['sequential'])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                return fig
    except Exception as e:
        print(f"Erro em update_key_distribution_horizontal: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('top-albums-chart-horizontal', 'figure'),
    Input('top-albums-chart-horizontal', 'id')
)
def update_top_albums_horizontal(_):
    try:
        if 'Album' in df.columns and not df.empty:
            album_counts = safe_value_counts(df['Album'], 10)
            if not album_counts.empty:
                fig = px.bar(x=album_counts.values, y=album_counts.index, 
                           orientation='h',
                           labels={'x': 'Número de Músicas', 'y': 'Álbum'},
                           title='Top 10 Álbuns com Mais Músicas',
                           color=album_counts.values,
                           color_continuous_scale=COLOR_PALETTE['sequential'])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                return fig
    except Exception as e:
        print(f"Erro em update_top_albums_horizontal: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('views-streams-heatmap', 'figure'),
    Input('views-streams-heatmap', 'id')
)
def update_views_streams_heatmap(_):
    try:
        if not df.empty and 'Views' in df.columns and 'Stream' in df.columns:
            valid_data = df[['Views', 'Stream']].replace([np.inf, -np.inf], np.nan).dropna()
            if not valid_data.empty:
                valid_data_log = valid_data.copy()
                valid_data_log['Views_log'] = np.log10(valid_data_log['Views'] + 1)
                valid_data_log['Stream_log'] = np.log10(valid_data_log['Stream'] + 1)
                
                fig = px.density_heatmap(valid_data_log, x='Views_log', y='Stream_log',
                                        title='Densidade: Visualizações vs Streams (Escala Log)',
                                        nbinsx=20, nbinsy=20,
                                        color_continuous_scale=COLOR_PALETTE['heatmap'],
                                        labels={'Views_log': 'Log(Views)', 'Stream_log': 'Log(Streams)'})
                return fig
    except Exception as e:
        print(f"Erro em update_views_streams_heatmap: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('energy-danceability-heatmap', 'figure'),
    Input('energy-danceability-heatmap', 'id')
)
def update_energy_danceability_heatmap(_):
    try:
        if not df.empty and 'Energy' in df.columns and 'Danceability' in df.columns:
            fig = px.density_heatmap(df, x='Energy', y='Danceability',
                                    title='Densidade: Energy vs Danceability',
                                    nbinsx=20, nbinsy=20,
                                    color_continuous_scale=COLOR_PALETTE['heatmap'])
            return fig
    except Exception as e:
        print(f"Erro em update_energy_danceability_heatmap: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('valence-energy-heatmap', 'figure'),
    Input('valence-energy-heatmap', 'id')
)
def update_valence_energy_heatmap(_):
    try:
        if not df.empty and 'Valence' in df.columns and 'Energy' in df.columns:
            fig = px.density_heatmap(df, x='Valence', y='Energy',
                                    title='Densidade: Valence vs Energy',
                                    nbinsx=20, nbinsy=20,
                                    color_continuous_scale=COLOR_PALETTE['heatmap'])
            return fig
    except Exception as e:
        print(f"Erro em update_valence_energy_heatmap: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('music-features-heatmap', 'figure'),
    Input('music-features-heatmap', 'id')
)
def update_music_features_heatmap(_):
    try:
        if not df.empty:
            music_features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
                            'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
            music_features = [m for m in music_features if m in df.columns]
            
            if music_features:
                corr_data = df[music_features].replace([np.inf, -np.inf], np.nan).dropna()
                if not corr_data.empty:
                    correlation_matrix = corr_data.corr()
                    fig = px.imshow(correlation_matrix, 
                                   text_auto=True, 
                                   title='Correlação entre Características Musicais',
                                   aspect="auto",
                                   color_continuous_scale=COLOR_PALETTE['diverging'],
                                   zmin=-1, zmax=1)
                    return fig
    except Exception as e:
        print(f"Erro em update_music_features_heatmap: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

if __name__ == '__main__':
    app.run(debug=True)