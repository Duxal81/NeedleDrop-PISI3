import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.preprocessing import MinMaxScaler
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "NeedleDrop - Análise de Clusters (Vibes)"

AUDIO_FEATURES = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms']
ALL_FEATURES = AUDIO_FEATURES + ['Views', 'Likes', 'Comments', 'Stream']
WINE_STYLE = {'backgroundColor': '#800020', 'color': 'white', 'fontWeight': 'bold'}
CARD_STYLE = {"border": "none", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"}
HIGH_CONTRAST_COLORS = px.colors.qualitative.Dark2

def load_data_and_predict():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        parquet_path = os.path.join(base_path, 'Spotify_Youtube.parquet')
        scaler_path = os.path.join(base_path, 'scaler_kmeans.pkl')
        kmeans_path = os.path.join(base_path, 'kmeans_k6.pkl')

        if not os.path.exists(parquet_path): return pd.DataFrame()
        df = pd.read_parquet(parquet_path)
        df = df.dropna(subset=ALL_FEATURES).reset_index(drop=True)

        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        with open(kmeans_path, 'rb') as f: kmeans = pickle.load(f)

        X_scaled = scaler.transform(df[ALL_FEATURES])
        clusters = kmeans.predict(X_scaled)
        df['Cluster'] = clusters
        df['Rotulo_Cluster'] = df['Cluster'].apply(lambda x: f"Cluster {x}")
        return df
    except Exception as e:
        print(f"Erro: {e}")
        return pd.DataFrame()

df = load_data_and_predict()

def draw_cards():
    if df.empty: return html.Div("Erro ao carregar dados.")
    total = len(df)
    n_cl = df['Cluster'].nunique()
    avg_views = df['Views'].mean()
    cards = []
    metrics = [("Total de Faixas", f"{total:,}"), ("Total de Grupos (Clusters)", f"{n_cl}"), ("Média de Visualizações", f"{int(avg_views):,}")]
    
    for t, v in metrics:
        cards.append(dbc.Col(dbc.Card([
            dbc.CardHeader(t, style=WINE_STYLE), 
            dbc.CardBody([html.H3(v, className="card-title", style={'color': '#333', 'fontWeight': 'bold'})])
        ], style=CARD_STYLE), width=4))
    return dbc.Row(cards, className="mb-4")

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("NeedleDrop: Análise de Clusters", className="text-center my-4", style={'color': '#333', 'fontWeight': '800'}), width=12)]),
    draw_cards(),
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardHeader("Distribuição de Músicas", style=WINE_STYLE), dbc.CardBody(dcc.Graph(id='dist-graph'))], style=CARD_STYLE)], width=6),
        dbc.Col([dbc.Card([dbc.CardHeader("Assinatura Sonora (Radar)", style=WINE_STYLE), dbc.CardBody(dcc.Graph(id='radar-graph'))], style=CARD_STYLE)], width=6)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardHeader("Popularidade por Grupo", style=WINE_STYLE), dbc.CardBody(dcc.Graph(id='box-graph'))], style=CARD_STYLE)], width=6),
        dbc.Col([dbc.Card([dbc.CardHeader("Mapa de Humor (Energia x Valência)", style=WINE_STYLE), dbc.CardBody(dcc.Graph(id='scatter-graph'))], style=CARD_STYLE)], width=6)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.H4("Explorar Músicas por Grupo", style={'marginTop': '20px', 'color': '#333'}), 
            dcc.Dropdown(id='cl-drop', options=[{'label':f'Cluster {i}','value':i} for i in sorted(df['Cluster'].unique())] if not df.empty else [], value=0, clearable=False, style={'marginBottom': '10px'}), 
            html.Div(id='table-div')
        ], width=12)
    ])
], fluid=True, style={'padding':'30px', 'backgroundColor':'#e6e6e6', 'minHeight': '100vh'})

@app.callback(
    [Output('dist-graph','figure'), Output('radar-graph','figure'), Output('box-graph','figure'), Output('scatter-graph','figure')], 
    [Input('cl-drop','value')]
)
def update_graphs(dummy):
    if df.empty: return {},{},{},{}
    
    dist_data = df['Rotulo_Cluster'].value_counts().reset_index()
    dist_data.columns = ['Grupo', 'Quantidade']
    fig_d = px.bar(dist_data, x='Grupo', y='Quantidade', color='Grupo', template='plotly_white', color_discrete_sequence=HIGH_CONTRAST_COLORS)
    fig_d.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    feats = ['Danceability','Energy','Speechiness','Acousticness','Valence','Liveness']
    means = df.groupby('Cluster')[feats].mean().reset_index()
    sc_viz = MinMaxScaler()
    means[feats] = sc_viz.fit_transform(means[feats])
    df_r = means.melt(id_vars='Cluster', var_name='Caracteristica', value_name='Valor')
    
    fig_r = px.line_polar(df_r, r='Valor', theta='Caracteristica', color='Cluster', line_close=True, template='plotly_white', color_discrete_sequence=HIGH_CONTRAST_COLORS)
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor='white'), paper_bgcolor='rgba(0,0,0,0)')

    fig_b = px.box(df, x='Rotulo_Cluster', y='Views', color='Rotulo_Cluster', log_y=True, template='plotly_white', color_discrete_sequence=HIGH_CONTRAST_COLORS)
    fig_b.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig_s = px.scatter(df.sample(min(2000, len(df))), x='Valence', y='Energy', color='Rotulo_Cluster', template='plotly_white', title="Energia vs Valência", color_discrete_sequence=HIGH_CONTRAST_COLORS)
    fig_s.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig_s.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig_d, fig_r, fig_b, fig_s

@app.callback(Output('table-div','children'), [Input('cl-drop','value')])
def update_table(cl):
    if df.empty: return html.Div()
    dff = df[df['Cluster']==cl].sort_values('Views', ascending=False).head(10)
    return dbc.Table.from_dataframe(dff[['Title','Artist','Views','Energy','Valence']], striped=True, bordered=True, hover=True, index=False, style={'backgroundColor': 'white'})

if __name__ == '__main__':
    app.run(debug=True, port=8051)