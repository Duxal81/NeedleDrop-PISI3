import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

def train_and_evaluate_models(clustered_data_path, scaled_features_path):
    """
    Treina e avalia modelos de classificação para prever os clusters K-Means.
    """
    df_clustered = pd.read_parquet(clustered_data_path)
    X_scaled = pd.read_parquet(scaled_features_path)

    # Garantir que X_scaled e df_clustered têm o mesmo número de linhas
    if len(X_scaled) != len(df_clustered):
        raise ValueError("O número de linhas em X_scaled e df_clustered não corresponde.")

    X = X_scaled # Features escalonadas
    y = df_clustered["Cluster_KMeans"] # Target: os clusters

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Treinando e avaliando {name} ---")
        
        # Treinamento
        model.fit(X_train, y_train)
        
        # Previsão no conjunto de TREINO
        y_pred_train = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        print(f"Acurácia no Treino: {train_accuracy:.4f}")

        # Previsão no conjunto de TESTE
        y_pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test)
        
        print(f"Acurácia no Teste: {test_accuracy:.4f}")
        print("Relatório de Classificação no Teste:\n", report)
        
        # Validação Cruzada
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"Acurácia Média da Validação Cruzada (5-fold): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        results[name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean_accuracy': np.mean(cv_scores),
            'cv_std_accuracy': np.std(cv_scores),
            'report': report
        }
        
        # Salvar o modelo treinado
        model_filename = f'{name.lower().replace(" ", "_")}_model.joblib'
        joblib.dump(model, model_filename)
        print(f"Modelo {name} salvo como {model_filename}")

    return results

if __name__ == '__main__':
    model_results = train_and_evaluate_models(
        clustered_data_path='Spotify_Youtube_clustered_kmeans.parquet',
        scaled_features_path='Spotify_Youtube_scaled_features.parquet'
    )

