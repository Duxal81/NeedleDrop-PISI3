import pickle
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

print("Carregando dados processados...")


with open('dashboard_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Extrair dados
df_raw = data['df_raw']
df_cluster_profile = data['df_cluster_profile']
df_scaled_profile = data['df_scaled_profile']
AUDIO_POPULARITY_FEATURES = data['AUDIO_POPULARITY_FEATURES']
K_VALUE = data['K_VALUE']

print("Criando aplicacao Dash...")

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    html.H1(" Analise de Clusters de Musicas (K=6) ",
            style={'textAlign': 'center', 'color': "#AF0C0C"}),

    html.Hr(style={'borderTop': '2px solid #ccc'}),

    html.Div([
        html.H3("Distribuicao de Observacoes por Cluster", style={'textAlign': 'center', 'color': '#333'}),
        dcc.Graph(
            id='cluster-distribution',
            figure=px.bar(
                df_raw['Cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'count': 'count'}),
                x='Cluster',
                y='count',
                title='Contagem de Musicas em Cada Cluster',
                labels={'Cluster': 'ID do Cluster', 'count': 'Contagem'},
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Bold
            ).update_layout(showlegend=False)
        )
    ], className='row', style={'marginBottom': '20px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),

    html.Hr(style={'borderTop': '2px solid #ccc'}),

    html.Div([
        html.H3("Perfil Medio dos Clusters por Caracteristica", style={'textAlign': 'center', 'color': '#333'}),

        html.Div([
            html.Label("Selecione a Caracteristica:", style={'fontWeight': 'bold'}),
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
        html.P("Este grafico exibe o valor medio das caracteristicas NORMALIZADAS para o perfil do cluster."),
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
        title=f'Valor Medio de {selected_feature} por Cluster',
        labels={'Cluster': 'ID do Cluster', selected_feature: f'Media de {selected_feature}'},
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    return fig

@app.callback(
    Output('cluster-profile-radar', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_profile_radar_graph(selected_feature):
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
        title='Perfil dos Clusters (Dados Padronizados - 10 Primeiras Caracteristicas)'
    )
    
    return fig


if __name__ == '__main__':
    print("\nDASHBOARD: Carregado com sucesso!")
    print("ACESSO: http://127.0.0.1:8050/")
    print("PARAR: Ctrl+C")
    app.run(debug=True, use_reloader=False)