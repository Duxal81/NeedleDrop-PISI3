import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tabulate import tabulate
import shap
import logging

PARQUET_PATH = 'Spotify_Youtube.parquet'
K_VALUE = 6
RANDOM_STATE = 42
TARGET_COL = 'Rating'

def load_and_prepare_data():
    print("1. Carregando e preparando dados...")
    
    df = pd.read_parquet(PARQUET_PATH)
    
    audio_features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 
                      'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms']
    popularity_features = ['Views', 'Likes', 'Comments', 'Stream']
    
    agg_funcs = {col: 'mean' for col in audio_features}
    agg_funcs.update({col: 'sum' for col in popularity_features})
    
    df_albums = df.groupby(['Artist', 'Album']).agg(agg_funcs).reset_index()
    df_albums = df_albums.fillna(0)
    
    scaler_temp = StandardScaler()
    df_albums_scaled = df_albums.copy()
    df_albums_scaled[audio_features + popularity_features] = scaler_temp.fit_transform(df_albums[audio_features + popularity_features])
    
    num_users = 100
    user_ids = [f'User_{i}' for i in range(num_users)]
    interaction_data = []
    
    for index, row in df_albums_scaled.iterrows():
        album_interactions = np.random.randint(1, 5)
        selected_users = np.random.choice(user_ids, size=album_interactions, replace=True)
        
        pop_score = row['Views'] + row['Likes'] + row['Stream']
        valence_score = row['Valence']
        base_rating = 2 + (pop_score * 0.1) + (valence_score * 0.5)
        
        for user in selected_users:
            rating = int(np.clip(np.round(base_rating + np.random.normal(0, 0.5)), 0, 4))
            interaction_data.append({
                'Artist': row['Artist'],
                'Album': row['Album'],
                'UserID': user,
                TARGET_COL: rating
            })
            
    df_ratings = pd.DataFrame(interaction_data)
    final_df = df_ratings.merge(df_albums, on=['Artist', 'Album'], how='inner')
    
    print(f"2. Aplicando K-Means com K={K_VALUE}...")
    cluster_features = audio_features + popularity_features
    
    scaler_kmeans = StandardScaler()
    X_cluster = scaler_kmeans.fit_transform(final_df[cluster_features])
    
    kmeans = KMeans(n_clusters=K_VALUE, random_state=RANDOM_STATE, n_init=10)
    final_df['Cluster_ID'] = kmeans.fit_predict(X_cluster)
    
    cluster_distribution = final_df['Cluster_ID'].value_counts().sort_index()
    print("\nDistribuicao dos Clusters:")
    print(cluster_distribution.to_markdown())
    
    cluster_analysis = final_df.groupby('Cluster_ID')[audio_features[:5]].mean()
    print("\nCaracteristicas medias por cluster:")
    print(cluster_analysis.round(3).to_markdown())
    
    plt.figure(figsize=(12, 6))
    plt.bar(cluster_distribution.index, cluster_distribution.values, color='skyblue', edgecolor='black')
    plt.xlabel('Cluster ID')
    plt.ylabel('Numero de Albuns')
    plt.title(f'Distribuicao dos Clusters (K={K_VALUE})')
    plt.xticks(range(K_VALUE))
    for i, v in enumerate(cluster_distribution.values):
        plt.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'cluster_distribution_k{K_VALUE}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    with open(f'kmeans_k{K_VALUE}.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    with open('scaler_kmeans.pkl', 'wb') as f:
        pickle.dump(scaler_kmeans, f)
        
    final_df = pd.get_dummies(final_df, columns=['Cluster_ID'], prefix='Cluster')
    le = LabelEncoder()
    final_df['UserID_Encoded'] = le.fit_transform(final_df['UserID'])
    
    feature_cols = [col for col in final_df.columns if col not in ['Artist', 'Album', 'UserID', TARGET_COL]]
    X = final_df[feature_cols]
    y = final_df[TARGET_COL]
    
    print("3. Convertendo para classificacao binaria...")
    y_binary = y.apply(lambda x: 1 if x >= 3 else 0)
    
    print("Distribuicao do target binario:")
    print(y_binary.value_counts().sort_index().to_markdown())
    
    return X, y_binary, feature_cols

def apply_cross_validation_balance(X, y, test_size=0.2):
    print("4. Aplicando validacao cruzada e balanceamento...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Distribuicao original - Treino: {y_train.value_counts().to_dict()}")
    
    class_counts = y_train.value_counts()
    min_class = class_counts.idxmin()
    max_class = class_counts.idxmax()
    
    target_ratio = 0.7
    n_minority = class_counts[min_class]
    n_majority_target = int(n_minority / target_ratio)
    
    if n_majority_target < class_counts[max_class]:
        rus = RandomUnderSampler(
            sampling_strategy={max_class: n_majority_target},
            random_state=RANDOM_STATE
        )
        X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
    else:
        X_train_bal, y_train_bal = X_train, y_train
    
    smote = SMOTE(
        sampling_strategy='auto',
        random_state=RANDOM_STATE,
        k_neighbors=min(5, n_minority - 1)
    )
    X_train_final, y_train_final = smote.fit_resample(X_train_bal, y_train_bal)
    
    print(f"Distribuicao balanceada - Treino: {pd.Series(y_train_final).value_counts().to_dict()}")
    print(f"Distribuicao real - Teste: {y_test.value_counts().to_dict()}")
    
    numeric_features = [col for col in X_train_final.columns if col not in [f'Cluster_{i}' for i in range(K_VALUE)]]
    cluster_cols = [col for col in X_train_final.columns if col.startswith('Cluster_')]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    
    X_train_scaled = preprocessor.fit_transform(X_train_final)
    X_test_scaled = preprocessor.transform(X_test)
    
    feature_names = numeric_features + cluster_cols
    X_train_final_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_final_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
        
    return X_train_final_df, X_test_final_df, y_train_final, y_test, feature_names

def train_with_cross_validation(X_train, y_train, models):
    print("5. Treinamento com validacao cruzada...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    
    for name, model in models.items():
        print(f"--- Validacao Cruzada: {name} ---")
        
        recall_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall', n_jobs=-1)
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        
        cv_results[name] = {
            'model': model,
            'cv_recall_mean': np.mean(recall_scores),
            'cv_recall_std': np.std(recall_scores),
            'cv_accuracy_mean': np.mean(accuracy_scores),
            'cv_accuracy_std': np.std(accuracy_scores),
            'cv_f1_mean': np.mean(f1_scores),
            'cv_f1_std': np.std(f1_scores)
        }
        
        print(f"Recall: {cv_results[name]['cv_recall_mean']:.4f} (+/- {cv_results[name]['cv_recall_std']*2:.4f})")
        print(f"Acuracia: {cv_results[name]['cv_accuracy_mean']:.4f} (+/- {cv_results[name]['cv_accuracy_std']*2:.4f})")
        print(f"F1-Score: {cv_results[name]['cv_f1_mean']:.4f} (+/- {cv_results[name]['cv_f1_std']*2:.4f})")
    
    return cv_results

def evaluate_final_models(X_train, X_test, y_train, y_test, cv_results):
    print("6. Avaliacao final dos modelos...")
    
    results = []
    
    for name, cv_data in cv_results.items():
        model = cv_data['model']
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Train_Accuracy': train_accuracy,
            'Test_Accuracy': test_accuracy,
            'Train_Recall': train_recall,
            'Test_Recall': test_recall,
            'Train_F1': train_f1,
            'Test_F1': test_f1,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Overfitting_Gap': train_accuracy - test_accuracy,
            'CV_Recall': cv_data['cv_recall_mean'],
            'Model_Object': model
        })
        
        print(f"\n--- {name} ---")
        table_data = [
            ["Acuracia", f"{train_accuracy*100:.2f}%", f"{test_accuracy*100:.2f}%", f"{(train_accuracy - test_accuracy)*100:.2f} p.p."],
            ["MAE", f"{train_mae:.4f}", f"{test_mae:.4f}", f"{(train_mae - test_mae):.4f}"],
            ["Recall", f"{train_recall:.4f}", f"{test_recall:.4f}", f"{(train_recall - test_recall):.4f}"],
            ["F1-Score", f"{train_f1:.4f}", f"{test_f1:.4f}", f"{(train_f1 - test_f1):.4f}"]
        ]
        print(tabulate(table_data, headers=["Metrica", "Treino", "Teste", "Diferenca"], tablefmt="simple"))
        
        print(f"\nRelatorio de classificacao (Teste):")
        print(classification_report(y_test, y_test_pred, digits=4))
    
    best_model_result = max(results, key=lambda x: x['Test_Recall'])
    
    print("\n=== MELHOR MODELO SELECIONADO ===")
    print(f"Modelo: {best_model_result['Model']}")
    print(f"Recall (Teste): {best_model_result['Test_Recall']:.4f}")
    print(f"Acuracia (Teste): {best_model_result['Test_Accuracy']*100:.2f}%")
    print(f"F1-Score (Teste): {best_model_result['Test_F1']:.4f}")
    print(f"Overfitting: {(best_model_result['Overfitting_Gap'])*100:.2f} p.p.")
    
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model_result['Model_Object'], f)
    
    return results, best_model_result

def create_shap_explanations(model, X_train, X_test, y_test, feature_names, model_name="Model"):
    """
    Create SHAP explanations for the given model with enhanced flexibility and robustness.
    
    Parameters:
    -----------
    model : sklearn model
        Trained machine learning model
    X_train : pd.DataFrame
        Training data for background samples
    X_test : pd.DataFrame
        Test data to explain
    y_test : pd.Series
        Test labels
    feature_names : list
        List of feature names
    model_name : str
        Name of the model for labeling outputs
        
    Returns:
    --------
    dict : Dictionary containing SHAP values and explainer object
    """
    # Configure logging for better error tracking
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting SHAP explanations for {model_name}...")
    
    try:
        # 1. Dynamic handling for background size to optimize SHAP calculations
        # Use first 10 samples for faster computations
        background_size = min(10, len(X_train))
        X_background = X_train.iloc[:background_size]
        logger.info(f"Using background size of {background_size} samples for SHAP calculations")
        
        # Determine number of test samples to explain (limit for performance)
        explain_size = min(100, len(X_test))
        X_explain = X_test.iloc[:explain_size]
        logger.info(f"Explaining {explain_size} test samples")
        
        # 2. Validate model compatibility and select appropriate explainer
        # Models that work well with TreeExplainer
        tree_based_models = (RandomForestClassifier, GradientBoostingClassifier, 
                           HistGradientBoostingClassifier)
        
        # Models that need KernelExplainer (Logistic Regression, SVC, KNN)
        kernel_based_models = (LogisticRegression, SVC, KNeighborsClassifier)
        
        explainer = None
        shap_values = None
        
        if isinstance(model, tree_based_models):
            logger.info(f"Using TreeExplainer for {model_name}")
            try:
                # TreeExplainer works well for tree-based models
                explainer = shap.TreeExplainer(model, X_background)
                shap_values = explainer(X_explain)
                logger.info("TreeExplainer succeeded")
            except Exception as e:
                logger.warning(f"TreeExplainer failed: {str(e)}. Falling back to KernelExplainer")
                explainer = None
        
        elif isinstance(model, kernel_based_models):
            logger.info(f"Model {model_name} requires KernelExplainer - using automatic fallback")
            # Automatically fallback to KernelExplainer for unsupported models
            explainer = None
        
        # 3. Fallback to KernelExplainer if TreeExplainer not applicable or failed
        if explainer is None:
            logger.info(f"Using KernelExplainer for {model_name}")
            try:
                # Use reduced background for better performance
                # KernelExplainer can be slow, so we limit background size even more
                kernel_background_size = min(5, len(X_train))
                X_kernel_background = X_train.iloc[:kernel_background_size]
                logger.info(f"Using reduced background size of {kernel_background_size} for KernelExplainer")
                
                # Create prediction function for KernelExplainer
                if hasattr(model, 'predict_proba'):
                    predict_fn = lambda x: model.predict_proba(x)[:, 1]
                else:
                    predict_fn = model.predict
                
                explainer = shap.KernelExplainer(predict_fn, X_kernel_background, 
                                                link="identity")
                
                # 4. Limit the number of features analyzed when computation is resource-intensive
                # Select only top 10 most important features dynamically
                if hasattr(model, 'feature_importances_'):
                    # For tree-based models with feature importances
                    feature_importance = model.feature_importances_
                    top_indices = np.argsort(feature_importance)[-10:]
                    important_features = [feature_names[i] for i in top_indices]
                    
                    # Create reduced dataset with only top features
                    X_explain_reduced = pd.DataFrame(
                        X_explain.iloc[:, top_indices].values,
                        columns=[feature_names[i] for i in top_indices]
                    )
                    logger.info(f"Using top 10 features for KernelExplainer: {important_features}")
                    
                    # Calculate SHAP values for reduced feature set
                    shap_values_reduced = explainer.shap_values(X_explain_reduced)
                    
                    # Create full SHAP values array with zeros for non-selected features
                    if isinstance(shap_values_reduced, list):
                        # For multi-class classification
                        shap_values_reduced = shap_values_reduced[1] if len(shap_values_reduced) > 1 else shap_values_reduced[0]
                    
                    shap_values = np.zeros((len(X_explain), len(feature_names)))
                    for i, idx in enumerate(top_indices):
                        shap_values[:, idx] = shap_values_reduced[:, i]
                    
                else:
                    # For models without feature importances, use all features but with warning
                    logger.warning(f"Model {model_name} has no feature_importances_. Using all features (may be slow)")
                    shap_values = explainer.shap_values(X_explain)
                
                logger.info("KernelExplainer succeeded")
                
            except Exception as e:
                logger.error(f"KernelExplainer failed: {str(e)}")
                raise
        
        # 5. Generate and save SHAP visualizations
        logger.info("Generating SHAP visualizations...")
        
        try:
            # Summary plot
            plt.figure(figsize=(12, 8))
            if hasattr(shap_values, 'values'):
                # For SHAP 0.40+ with Explanation objects
                shap.summary_plot(shap_values, X_explain, feature_names=feature_names, 
                                show=False, max_display=15)
            else:
                # For older SHAP versions or numpy arrays
                shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                                show=False, max_display=15)
            plt.tight_layout()
            summary_plot_path = f'shap_summary_{model_name.replace(" ", "_")}.png'
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP summary plot to {summary_plot_path}")
            
            # Bar plot showing mean absolute SHAP values
            plt.figure(figsize=(12, 8))
            if hasattr(shap_values, 'values'):
                shap.plots.bar(shap_values, max_display=15, show=False)
            else:
                shap.summary_plot(shap_values, X_explain, plot_type="bar",
                                feature_names=feature_names, show=False, max_display=15)
            plt.tight_layout()
            bar_plot_path = f'shap_bar_{model_name.replace(" ", "_")}.png'
            plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP bar plot to {bar_plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate some SHAP plots: {str(e)}")
        
        # 6. Save SHAP objects for reuse
        # Note: We avoid saving the explainer if it contains unpicklable objects (like lambda functions)
        shap_artifacts = {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'model_name': model_name
        }
        
        # Try to save the explainer, but don't fail if it can't be pickled
        try:
            shap_artifacts['explainer'] = explainer
            pickle_path = f'shap_explainer_{model_name.replace(" ", "_")}.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(shap_artifacts, f)
            logger.info(f"Saved SHAP artifacts with explainer to {pickle_path}")
        except (TypeError, AttributeError) as e:
            # If explainer can't be pickled (e.g., contains lambda), save without it
            logger.warning(f"Could not pickle explainer due to: {str(e)}")
            shap_artifacts_no_explainer = {
                'shap_values': shap_values,
                'feature_names': feature_names,
                'model_name': model_name
            }
            pickle_path = f'shap_values_{model_name.replace(" ", "_")}.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(shap_artifacts_no_explainer, f)
            logger.info(f"Saved SHAP values (without explainer) to {pickle_path}")
        
        logger.info(f"SHAP explanations completed successfully for {model_name}")
        
        return shap_artifacts
        
    except Exception as e:
        # 5. Enhanced error handling for scenarios where SHAP calculations fail entirely
        logger.error(f"SHAP explanation failed for {model_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"This may be due to:")
        logger.error("  - Model incompatibility with available SHAP explainers")
        logger.error("  - Insufficient memory for the calculation")
        logger.error("  - Issues with the data format")
        logger.error("Continuing without SHAP explanations for this model...")
        return None

def main():
    X, y, feature_cols = load_and_prepare_data()
    X_train, X_test, y_train, y_test, feature_names = apply_cross_validation_balance(X, y)
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, C=0.5, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=100, max_depth=10,
            min_samples_split=20, min_samples_leaf=10, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=RANDOM_STATE, n_estimators=100, max_depth=6,
            learning_rate=0.1, subsample=0.8
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, max_iter=100, max_depth=6,
            learning_rate=0.1, early_stopping=True
        ),
        'SVM': SVC(
            random_state=RANDOM_STATE, C=1.0, kernel='rbf',
            class_weight='balanced', probability=True
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=15, weights='distance'
        )
    }
    
    cv_results = train_with_cross_validation(X_train, y_train, models)
    results, best_model_result = evaluate_final_models(X_train, X_test, y_train, y_test, cv_results)
    
    print("\n=== RESUMO FINAL ===")
    summary_data = []
    for result in results:
        summary_data.append([
            result['Model'],
            f"{result['Test_Accuracy']*100:.2f}%",
            f"{result['Test_Recall']:.4f}",
            f"{result['Test_F1']:.4f}",
            f"{result['Overfitting_Gap']*100:.2f} p.p."
        ])
    
    print(tabulate(summary_data, 
                   headers=["Modelo", "Acuracia Teste", "Recall Teste", "F1-Score Teste", "Overfitting"],
                   tablefmt="grid"))
    
    # Generate SHAP explanations for the best model
    print("\n=== GERANDO EXPLICACOES SHAP ===")
    shap_result = create_shap_explanations(
        model=best_model_result['Model_Object'],
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        model_name=best_model_result['Model']
    )
    
    if shap_result is not None:
        print(f"SHAP explanations geradas com sucesso para {best_model_result['Model']}")
    else:
        print(f"Nao foi possivel gerar explicacoes SHAP para {best_model_result['Model']}")
    
    print(f"\nPipeline concluido. Melhor modelo: {best_model_result['Model']}")
    print(f"K utilizado: {K_VALUE}")

if __name__ == "__main__":
    main()