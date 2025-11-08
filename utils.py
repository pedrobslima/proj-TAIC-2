
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json
import pandas as pd
import numpy as np
import re
import mlflow
import mlflow.sklearn
import optuna
#import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC#, SVR # kernels: 'linear', 'poly' e 'rbf'
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

# =================================================

# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv
# loading variables from .env file
load_dotenv() 

# accessing and printing value
#print(os.getenv("MY_KEY"))

#with open('config.json', 'w') as f:
#    os.getenv\(.+\) = randint(0, 4294967295)
#    json.dump(CONFIG, f)
#    print(int(os.getenv('SEED')))

# ==================================================

def log_model_to_mlflow(model, model_name, hyperparams, X_train, y_train, X_test, y_test):
    """
    Treina, avalia e registra um modelo no MLflow.
    
    Parameters:
    -----------
    model : estimator
        Modelo do scikit-learn ou compatível
    model_name : str
        Nome do modelo para registro
    hyperparams : dict
        Dicionário com os hiperparâmetros do modelo
    X_train, y_train : array-like
        Dados de treino
    X_test, y_test : array-like
        Dados de teste
    """
    with mlflow.start_run(run_name=model_name):
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_test)
        
        # Calcular probabilidades para AUROC (se disponível)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                # Para classificação binária
                if y_proba.shape[1] == 2:
                    auroc = roc_auc_score(y_test, y_proba[:, 1])
                # Para classificação multiclasse
                else:
                    auroc = roc_auc_score(y_test, y_proba)
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
                if len(np.unique(y_test)) == 2:
                    auroc = roc_auc_score(y_test, y_scores)
                else:
                    auroc = roc_auc_score(y_test, y_scores)
            else:
                auroc = None
        except Exception as e:
            #print(f"Aviso: Não foi possível calcular AUROC para {model_name}: {e}")
            auroc = None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        if(len(set(y_test)) > 2):
            precision = precision_score(y_test, y_pred, zero_division=0, average='macro')
            recall = recall_score(y_test, y_pred, zero_division=0, average='macro')
            f1 = f1_score(y_test, y_pred, zero_division=0, average='macro')
        else:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Registrar hiperparâmetros
        mlflow.log_params(hyperparams)
        
        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        if auroc is not None:
            mlflow.log_metric("auroc", auroc)
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, name=model_name)
        
        #print(f"\n{'='*60}")
        #print(f"Modelo: {model_name}")
        #print(f"{'='*60}")
        #print(f"Acurácia:  {accuracy:.4f}")
        #print(f"Precisão:  {precision:.4f}")
        #print(f"Recall:    {recall:.4f}")
        #print(f"F1-Score:  {f1:.4f}")
        #if auroc is not None:
            #print(f"AUROC:     {auroc:.4f}")
        #print(f"{'='*60}\n")
        
        return model
    
def load_model_by_name(experiment_name, run_name):
    """
    Carrega um modelo específico do MLflow pelo nome da run.
    
    Parameters:
    -----------
    experiment_name : str
        Nome do experimento MLflow
    run_name : str
        Nome da run (ex: "Random_Forest", "XGBoost")
    
    Returns:
    --------
    model : estimator
        Modelo carregado
    run_info : dict
        Informações da run (hiperparâmetros e métricas)
    """
    # Buscar experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado!")
    
    # Buscar runs do experimento
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    if runs.empty:
        raise ValueError(f"Run '{run_name}' não encontrada no experimento '{experiment_name}'!")
    
    # Pegar a run mais recente se houver múltiplas
    run = runs.iloc[0]
    run_id = run.run_id
    print(run_id)
    # Carregar modelo
    model_uri = f"runs:/{run_id}/{run_name}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Extrair informações da run
    #run_info = {
    #    'run_id': run_id,
    #    'run_name': run_name,
    #    'start_time': run.start_time,
    #    'params': {k.replace('params.', ''): v for k, v in run.items() if k.startswith('params.')},
    #    'metrics': {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}
    #}
    
    #print(f"✓ Modelo '{run_name}' carregado com sucesso!")
    #print(f"  Run ID: {run_id}")
    ##print(f"  Métricas: F1={run_info['metrics'].get('f1_score', 'N/A'):.4f}")
    
    return model#, run_info

def transform_property(x):
    x = re.sub('(Private room( in )?)|(Shared room( in )?)|(Entire )|(Room in )', '', x).lower()
    if(x=='casa particular'):
        x='home'
    
    if(x not in ['rental unit','home','condo','loft','serviced apartment']):
        x='other'

    return x

def get_data():
    df = pd.read_csv('listings.csv')

    bathrooms = df['bathrooms_text'].str.extract('([0-9\.]+)?([- A-Za-z]+)')#[[0,2]]
    bathrooms[1] = bathrooms[1].apply(lambda x: x if pd.isna(x) else x.strip().lower().replace('baths','bath'))
    bathrooms.columns = ['n_baths', 'bath_type']

    for i in range(len(bathrooms)):
        bt = bathrooms.at[i,'bath_type']
        if(pd.notna(bt)):
            if(re.search('half', bt)):
                bt = re.sub('half-', '', bt)
                bathrooms.loc[i,:] = [0.5, bt]

            if(bt=='bath'):
                bathrooms.at[i,'bath_type'] = 'regular bath'
            #else:
            #    bathrooms.at[i,'bath_type'] = re.sub(' bath', '', bt)

    df['bathrooms'] = bathrooms['n_baths'].astype(float)
    df['bathroom_type'] = bathrooms['bath_type']

    df = df[[
        'host_response_time', #ok
        'host_response_rate', #ok
        'host_is_superhost', #ok
        'host_total_listings_count', #ok
        'host_identity_verified', #ok
        'latitude', #ok
        'longitude', #ok
        'property_type',
        'room_type', #ok
        'accommodates', #ok
        'bathrooms', #ok (o atualizado, vindo de bathrooms_text)
        'bathroom_type', #ok
        'bedrooms', #ok
        'beds', #ok
        'number_of_reviews', #ok
        #'number_of_reviews_l30d', #ok
        'review_scores_rating', #ok
        'review_scores_checkin', #ok
        'review_scores_communication', #ok
        'review_scores_location', #ok
        'minimum_nights',#ok (como o preço é apenas no momento, então vou deixar as noites apenas do momento também)
        'maximum_nights',#ok (como o preço é apenas no momento, então vou deixar as noites apenas do momento também)
        #'has_availability',#ok
        'availability_30',#ok
        #'availability_60',#ok
        #'availability_90',#ok
        #'availability_365',#ok
        'price'
    ]].dropna()

    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)
    df['host_response_time'] = df['host_response_time'].astype('category').cat.reorder_categories(['within an hour', 'within a few hours', 'within a day', 'a few days or more']).cat.codes
    df[['host_is_superhost','host_identity_verified']] = df[['host_is_superhost','host_identity_verified']].map(lambda x: x=='t')
    df['property_type'] = df['property_type'].apply(transform_property)
    df['price'] = df['price'].str.replace('[,\$]','', regex=True).astype(float)>300
    
    X, y = df.drop(columns=['price']), df['price'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=int(os.getenv('SEED')))

    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler() 
    onehot = onehot.set_output(transform='pandas')
    X_train = pd.concat([X_train.drop(columns=['property_type','room_type','bathroom_type']), onehot.fit_transform(X_train[['property_type','room_type','bathroom_type']], y_train)], axis=1)
    X_test = pd.concat([X_test.drop(columns=['property_type','room_type','bathroom_type']), onehot.transform(X_test[['property_type','room_type','bathroom_type']])], axis=1)

    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()

    nmrc_cols = ['host_response_time','host_response_rate','host_total_listings_count',
                'latitude','longitude','accommodates','bathrooms','bedrooms','beds',
                'number_of_reviews','review_scores_rating','review_scores_checkin',
                'review_scores_communication','review_scores_location',
                'minimum_nights','maximum_nights','availability_30']

    X_train_norm.loc[:,nmrc_cols] = scaler.fit_transform(X_train_norm[nmrc_cols])
    X_test_norm.loc[:,nmrc_cols] = scaler.transform(X_test_norm[nmrc_cols])

    return X_train, X_train_norm, X_test, X_test_norm, y_train, y_test

def searchAndTrain(experiment_name, num_trials, load=False):

    mlflow.set_experiment(experiment_name=experiment_name)

    X_train, X_train_norm, X_test, X_test_norm, y_train, y_test = get_data()

    scorer_string = 'f1_macro' if len(set(y_train))>2 else 'f1'

    # 1. Define an objective function to be maximized.
    def dtree_objective(trial:optuna.trial._trial.Trial):
        
        # 2. Suggest values for the hyperparameters using a trial object.
        max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',1, 30)
        
        clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf).fit(X_train, y_train)
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=10).mean()
        
        return score

    def svm_rbf_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        max_iter = trial.suggest_int('max_iter', 500, 3000)
        
        clf = SVC(
            kernel='rbf', C=C, gamma=gamma, max_iter=max_iter, random_state=int(os.getenv('SEED'))
        ).fit(X_train_norm, y_train)
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
        return score.mean()

    def gb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, subsample=subsample,
            max_features=max_features, random_state=int(os.getenv('SEED'))
        ).fit(X_train, y_train)
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
        return score.mean()

    # ============================================

    # 3. Create a study object and optimize the objective function.
    try:
        dtree_model = load_model_by_name(experiment_name=experiment_name, run_name='Decision_Tree')
    except ValueError:
        dtree_study = optuna.create_study(direction='maximize')
        dtree_study.optimize(dtree_objective, n_trials=num_trials)
        #print(dtree_study.best_trial)

        dtree_params = dtree_study.best_params
        dtree_model = DecisionTreeClassifier(**dtree_params, random_state=int(os.getenv('SEED')))
        dtree_model = log_model_to_mlflow(
            dtree_model, "Decision_Tree", dtree_params, 
            X_train, y_train, X_test, y_test
        )

    try:
        svm_rbf_model = load_model_by_name(experiment_name=experiment_name, run_name='SVM_RBF')
    except ValueError:
        svm_rbf_study = optuna.create_study(direction='maximize')
        svm_rbf_study.optimize(svm_rbf_objective, n_trials=num_trials)
        #print("SVM RBF Best Trial:", svm_rbf_study.best_trial)
        svm_rbf_params = svm_rbf_study.best_params
        svm_rbf_model = SVC(kernel='rbf', **svm_rbf_params, random_state=int(os.getenv('SEED')), probability=True)
        svm_rbf_model = log_model_to_mlflow(
            svm_rbf_model, "SVM_RBF", svm_rbf_params,
            X_train_norm, y_train, X_test_norm, y_test
        )

    try:
        gb_model = load_model_by_name(experiment_name=experiment_name, run_name='Gradient_Boosting')
    except ValueError:
        gb_study = optuna.create_study(direction='maximize')
        gb_study.optimize(gb_objective, n_trials=num_trials)
        #print("Gradient Boosting Best Trial:", gb_study.best_trial)
        gb_params = gb_study.best_params
        gb_model = GradientBoostingClassifier(**gb_params, random_state=int(os.getenv('SEED')))
        gb_model = log_model_to_mlflow(
            gb_model, "Gradient_Boosting", gb_params,
            X_train, y_train, X_test, y_test
        )

    if(load):
        return {'Decision_Tree': dtree_model,
                'SVM_RBF':svm_rbf_model,
                'Grandient_Boosting':gb_model}

def getExpName(dataset):
    global CONFIG
    dataset = re.sub('[-_ ]', '', dataset).lower()
    return f"{dataset}_{os.getenv('VERSION')}_{int(os.getenv('SEED'))}"

if(__name__=='__main__'):
    NUM_TRIALS = 20
    #DATASET = 'circles'
    for DATASET in ['airbnb']:
        experiment_name = getExpName(DATASET)

        searchAndTrain(experiment_name=experiment_name, num_trials=NUM_TRIALS)