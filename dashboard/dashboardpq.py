import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard Analítico - Spotify & YouTube"

try:
  
    df = pd.read_parquet('Spotify_Youtube.parquet')
    
    if df.empty:
        raise ValueError("Dataset está vazio")
        
except Exception as e:
    print(f"Erro ao carregar dataset: {e}")
    
    try:
        df = pd.read_csv('Spotify_Youtube.csv')
        print("CSV carregado como fallback")
    except:
        print("Não foi possível carregar nem Parquet nem CSV")
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

def safe_isnull_sum(series):
    try:
        if series is not None:
            return series.isnull().sum()
        return 0
    except:
        return 0

COLOR_PALETTE = {
    'technical_primary': "#6A4C93",
    'technical_secondary': "#8B5FBF",
    'technical_tertiary': "#A67FCC",
    'technical_sequential': 'Purples',
    
    'musical_primary': "#2E758B",
    'musical_secondary': "#3A8CA8",
    'musical_tertiary': "#4ECDC4",
    'musical_sequential': 'Teal',
    
    'diverging': 'RdBu',
    'categorical': 'Viridis',
    'heatmap': 'Blues',
    
    'neutral_light': "#F8F9FA",
    'neutral_medium': "#E9ECEF",
    'neutral_dark': "#6C757D"
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard Analítico - Spotify & YouTube", 
                       className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Metadados do Dataset", 
                                                 style={'backgroundColor': COLOR_PALETTE['technical_primary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        html.P(f"Shape: {df.shape if not df.empty else 'N/A'}"),
                                        html.P(f"Total de registros: {len(df) if not df.empty else 0}"),
                                        html.P(f"Total de colunas: {len(df.columns) if not df.empty else 0}"),
                                        html.P(f"Memória utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB" if not df.empty else "N/A")
                                    ])
                                ], className="rounded-3")
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Tipos de Dados", 
                                                 style={'backgroundColor': COLOR_PALETTE['technical_secondary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        html.Div([
                                            html.P(f"Numéricas: {len(df.select_dtypes(include=[np.number]).columns)}"),
                                            html.P(f"Texto: {len(df.select_dtypes(include=['object']).columns)}"),
                                            html.P(f"Booleanas: {len(df.select_dtypes(include=['bool']).columns)}"),
                                            html.P(f"Datas: {len(df.select_dtypes(include=['datetime']).columns)}")
                                        ])
                                    ])
                                ], className="rounded-3")
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Qualidade dos Dados", 
                                                 style={'backgroundColor': COLOR_PALETTE['technical_tertiary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        html.P(f"Valores nulos totais: {df.isnull().sum().sum() if not df.empty else 0}"),
                                        html.P(f"Colunas com valores nulos: {(df.isnull().sum() > 0).sum() if not df.empty else 0}"),
                                        html.P(f"Valores duplicados: {df.duplicated().sum() if not df.empty else 0}"),
                                        html.P(f"Completude média: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%" if not df.empty else "N/A")
                                    ])
                                ], className="rounded-3")
                            ], width=4)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Distribuição de Valores Nulos por Coluna", 
                                                 style={'backgroundColor': COLOR_PALETTE['technical_primary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        dcc.Graph(id='null-values-chart')
                                    ])
                                ], className="rounded-3")
                            ], width=12)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Detalhamento das Colunas - Guia de Interpretação", 
                                                 style={'backgroundColor': COLOR_PALETTE['technical_secondary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        dbc.Alert([
                                            html.H5("Como interpretar esta tabela:", className="alert-heading"),
                                            html.P("• COLUNA: Nome da característica analisada"),
                                            html.P("• TIPO: Formato dos dados (número, texto, etc)"),
                                            html.P("• VALORES ÚNICOS: Quantidade de valores diferentes"),
                                            html.P("• PREENCHIMENTO: Porcentagem de dados preenchidos"),
                                            html.P("• QUALIDADE: Avaliação da completude dos dados")
                                        ], color="info", className="mb-3 rounded-3"),
                                        
                                        html.Div(id='columns-details', style={'maxHeight': '400px', 'overflowY': 'auto'})
                                    ])
                                ], className="rounded-3")
                            ], width=12)
                        ])
                    ])
                ], label="INFORMAÇÕES TÉCNICAS",
                tab_style={
                    'margin': '5px', 
                    'border': '2px solid #6A4C93',
                    'borderRadius': '15px',
                    'fontWeight': 'bold', 
                    'fontSize': '14px', 
                    'padding': '10px'
                },
                active_tab_style={
                    'margin': '5px', 
                    'border': '2px solid #6A4C93', 
                    'borderRadius': '15px',
                    'backgroundColor': '#6A4C93', 
                    'color': 'white',
                    'fontWeight': 'bold', 
                    'fontSize': '14px', 
                    'padding': '10px'
                }),
                
                dbc.Tab([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Resumo Artístico", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_primary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        html.P(f"Total de artistas: {safe_nunique(df.get('Artist')) if not df.empty else 0}"),
                                        html.P(f"Total de álbuns: {safe_nunique(df.get('Album')) if not df.empty else 0}"),
                                        html.P(f"Músicas únicas: {safe_nunique(df.get('Track')) if not df.empty else 0}"),
                                        html.P(f"Músicas por artista (média): {len(df)/safe_nunique(df.get('Artist')):.1f}" if not df.empty and 'Artist' in df.columns and safe_nunique(df.get('Artist')) > 0 else "N/A")
                                    ])
                                ], className="rounded-3")
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Estatísticas de Engajamento", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_secondary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        html.P(f"Média de visualizações: {safe_mean(df.get('Views')):,.0f}" if not df.empty and 'Views' in df.columns else "N/A"),
                                        html.P(f"Média de streams: {safe_mean(df.get('Stream')):,.0f}" if not df.empty and 'Stream' in df.columns else "N/A"),
                                        html.P(f"Média de likes: {safe_mean(df.get('Likes')):,.0f}" if not df.empty and 'Likes' in df.columns else "N/A"),
                                        html.P(f"Média de comentários: {safe_mean(df.get('Comments')):,.0f}" if not df.empty and 'Comments' in df.columns else "N/A")
                                    ])
                                ], className="rounded-3")
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Características Musicais", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_tertiary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        html.P(f"Duração média: {safe_mean(df.get('Duration_min')):.2f} min" if not df.empty and 'Duration_min' in df.columns else "N/A"),
                                        html.P(f"Energia média: {safe_mean(df.get('Energy')):.2f}" if not df.empty and 'Energy' in df.columns else "N/A"),
                                        html.P(f"Dançabilidade média: {safe_mean(df.get('Danceability')):.2f}" if not df.empty and 'Danceability' in df.columns else "N/A"),
                                        html.P(f"Tom emocional médio: {safe_mean(df.get('Valence')):.2f}" if not df.empty and 'Valence' in df.columns else "N/A")
                                    ])
                                ], className="rounded-3")
                            ], width=4)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Top 10 Artistas com Mais Músicas", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_primary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        dcc.Graph(id='top-artists-chart', style={'height': '400px'})
                                    ])
                                ], className="rounded-3 h-100")
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Distribuição de Características Musicais", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_secondary'], 'color': 'white'}),
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
                                            clearable=False,
                                            className="rounded-3 mb-3"
                                        ),
                                        dcc.Graph(id='feature-distribution', style={'height': '350px'})
                                    ])
                                ], className="rounded-3 h-100")
                            ], width=6)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Relação entre Variáveis Musicais", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_tertiary'], 'color': 'white'}),
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
                                                    clearable=False,
                                                    className="rounded-3"
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
                                                    clearable=False,
                                                    className="rounded-3"
                                                )
                                            ], width=6)
                                        ]),
                                        dcc.Graph(id='variables-heatmap', style={'height': '350px'})
                                    ])
                                ], className="rounded-3 h-100")
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Correlação entre Métricas de Engajamento", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_primary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        dcc.Graph(id='engagement-heatmap', style={'height': '400px'})
                                    ])
                                ], className="rounded-3 h-100")
                            ], width=6)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Top Músicas por Engajamento", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_secondary'], 'color': 'white'}),
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
                                            clearable=False,
                                            className="rounded-3 mb-3"
                                        ),
                                        dcc.Graph(id='top-songs-chart-horizontal', style={'height': '400px'})
                                    ])
                                ], className="rounded-3")
                            ], width=12)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Distribuição de Chaves Musicais", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_tertiary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        dcc.Graph(id='key-distribution-horizontal', style={'height': '400px'})
                                    ])
                                ], className="rounded-3 h-100")
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Correlação entre Características Musicais", 
                                                 style={'backgroundColor': COLOR_PALETTE['musical_primary'], 'color': 'white'}),
                                    dbc.CardBody([
                                        dcc.Graph(id='music-features-heatmap', style={'height': '400px'})
                                    ])
                                ], className="rounded-3 h-100")
                            ], width=6)
                        ], className="mb-4")
                    ])
                ], label="ANÁLISE MUSICAL",
                tab_style={
                    'margin': '5px', 
                    'border': '2px solid #2E758B',
                    'borderRadius': '15px',
                    'fontWeight': 'bold', 
                    'fontSize': '14px', 
                    'padding': '10px'
                },
                active_tab_style={
                    'margin': '5px', 
                    'border': '2px solid #2E758B', 
                    'borderRadius': '15px',
                    'backgroundColor': '#2E758B', 
                    'color': 'white',
                    'fontWeight': 'bold', 
                    'fontSize': '14px', 
                    'padding': '10px'
                })
            ])
        ], width=12)
    ])
], fluid=True, style={'backgroundColor': COLOR_PALETTE['neutral_light']})

@app.callback(
    Output('null-values-chart', 'figure'),
    Input('null-values-chart', 'id')
)
def update_null_values_chart(_):
    try:
        if not df.empty:
            null_counts = df.isnull().sum()
            null_counts = null_counts[null_counts > 0].sort_values(ascending=True)
            
            if not null_counts.empty:
                fig = px.bar(x=null_counts.values, y=null_counts.index, 
                           orientation='h',
                           labels={'x': 'Quantidade de Valores Nulos', 'y': 'Coluna'},
                           title='Distribuição de Valores Nulos por Coluna',
                           color=null_counts.values,
                           color_continuous_scale=COLOR_PALETTE['technical_sequential'])
                return fig
    except Exception as e:
        print(f"Erro em update_null_values_chart: {e}")
    
    return go.Figure().add_annotation(text="Nenhum valor nulo encontrado!", x=0.5, y=0.5, showarrow=False)

@app.callback(
    Output('columns-details', 'children'),
    Input('columns-details', 'id')
)
def update_columns_details(_):
    try:
        if not df.empty:
            details = []
            for col in df.columns:
                if col.startswith('Unnamed'):
                    continue
                    
                null_count = safe_isnull_sum(df[col])
                null_percentage = (null_count / len(df)) * 100
                data_type = df[col].dtype
                unique_count = safe_nunique(df[col])
                completeness = 100 - null_percentage
                
                if completeness == 100:
                    quality_badge = dbc.Badge("Excelente", color="success", className="ms-2")
                elif completeness >= 90:
                    quality_badge = dbc.Badge("Boa", color="primary", className="ms-2")
                elif completeness >= 70:
                    quality_badge = dbc.Badge("Regular", color="warning", className="ms-2")
                else:
                    quality_badge = dbc.Badge("Crítica", color="danger", className="ms-2")
                
                example_value = "N/A"
                if not df[col].empty and not pd.isna(df[col].iloc[0]):
                    example_str = str(df[col].iloc[0])
                    example_value = example_str[:20] + "..." if len(example_str) > 20 else example_str
                
                details.append(
                    dbc.Row([
                        dbc.Col([
                            html.Strong(col),
                            html.Br(),
                            html.Small(f"Tipo: {data_type}", className="text-muted")
                        ], width=3),
                        dbc.Col(f"{unique_count:,}", width=2),
                        dbc.Col([
                            f"{completeness:.1f}%",
                            html.Br(),
                            html.Small(f"({null_count} nulos)", className="text-muted")
                        ], width=3),
                        dbc.Col(quality_badge, width=2),
                        dbc.Col(html.Small(example_value, className="text-muted"), width=2)
                    ], className="mb-2 border-bottom p-2")
                )
            
            header = dbc.Row([
                dbc.Col(html.Strong("COLUNA"), width=3),
                dbc.Col(html.Strong("VALORES ÚNICOS"), width=2),
                dbc.Col(html.Strong("PREENCHIMENTO"), width=3),
                dbc.Col(html.Strong("QUALIDADE"), width=2),
                dbc.Col(html.Strong("EXEMPLO"), width=2)
            ], className="mb-3 border-bottom p-2", style={'backgroundColor': COLOR_PALETTE['neutral_medium']})
            
            return [header] + details
    except Exception as e:
        print(f"Erro em update_columns_details: {e}")
    
    return html.P("Detalhes das colunas não disponíveis")

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
                           color_continuous_scale=COLOR_PALETTE['musical_sequential'])
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
                              color_discrete_sequence=[COLOR_PALETTE['musical_primary']])
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
                                    color_continuous_scale='Blues')
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
                                   color_continuous_scale='Blues',
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
                            color_continuous_scale=COLOR_PALETTE['musical_sequential'])
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
                            color_continuous_scale=COLOR_PALETTE['musical_sequential'])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                return fig
    except Exception as e:
        print(f"Erro em update_key_distribution_horizontal: {e}")
    
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
                                   color_continuous_scale='Blues',
                                   zmin=-1, zmax=1)
                    return fig
    except Exception as e:
        print(f"Erro em update_music_features_heatmap: {e}")
    
    return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5, showarrow=False)

if __name__ == '__main__':
    app.run(debug=True)