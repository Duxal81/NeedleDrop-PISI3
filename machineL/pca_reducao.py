import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def run_pca_reduction(scaled_features_path, output_pca_features_path, n_components=5):
    """
    Aplica PCA para redução de dimensionalidade e salva os componentes principais.
    Gera um gráfico da variância explicada acumulada.
    """
    X_scaled = pd.read_parquet(scaled_features_path)

    # Aplicar PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Salvar os componentes principais
    pca_df = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components)])
    pca_df.to_parquet(output_pca_features_path, index=False)
    print(f'Features reduzidas por PCA (n_components={n_components}) salvas como {output_pca_features_path}')

    # Plotar a variância explicada acumulada
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Numero de Componentes Principais')
    plt.ylabel('Variancia Explicada Acumulada')
    plt.title('Variancia Explicada pelo PCA')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()
    print('Grafico da Variancia Explicada pelo PCA salvo como pca_explained_variance.png')

if __name__ == '__main__':
    run_pca_reduction(
        scaled_features_path='Spotify_Youtube_scaled_features.parquet',
        output_pca_features_path='Spotify_Youtube_pca_features.parquet',
        n_components=5 # Número de componentes a reter, pode ser ajustado
    )