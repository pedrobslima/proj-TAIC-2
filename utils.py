
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
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
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
    
    return model#, run_info

def load_model_metrics_by_name(experiment_name, run_name):
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
    
    # Extrair informações da run
    run_info = {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}
    
    return run_info

def load_all_models_metrics(experiments:str|list, model_names:list=['Decision_Tree', 'Logistic_Regression', 'Gradient_Boosting', 'XGBoost']) -> dict:
    models = {}

    if(isinstance(experiments, str)):
        experiments = [experiments]*len(model_names)
    else:
        if(len(experiments) != len(model_names)):
            raise ValueError('Variável "experiments" deve ser uma lista com tamanho igual à "models", ou ser apenas uma string')

    for i in range(len(model_names)):
        metrics = load_model_metrics_by_name(experiments[i], model_names[i])
        models[model_names[i]] = metrics

    return models

def pfi(model, x, y, name=None):
  result = permutation_importance(model, x, y,n_repeats=30, random_state=int(os.getenv('SEED')))

  cols = [f"[{i}] - {x.columns[i]}" for i in range(len(x.columns))]

  importances = pd.Series(result.importances_mean, index=cols)

  _, ax = plt.subplots(figsize=(18,9))

  importances.plot.bar(yerr=result.importances_std, ax=ax)

  if(name):
    ax.set_title(f"Feature importances on {name} model")
  else:
    name = str(model)
    i = name.find('(')
    ax.set_title(f"Feature importances on {name[:i]} model\n{name[i:]}")#, fontsize=10)
  ax.set_ylabel("Mean accuracy decrease", fontsize=12)
  plt.xticks(rotation=85, fontsize=9)
  plt.show()
  return result

def dms_to_decimal(dms_string):
    """
    Converte coordenada em graus, minutos e segundos (DMS) para decimal
    
    Args:
        dms_string: String no formato "23° 33' 1.80\" S" ou variações
    
    Returns:
        Valor decimal da coordenada
    """
    # Extrair números e direção usando regex
    # Padrão: captura graus, minutos, segundos e direção (N/S/E/W)
    pattern = r"(\d+)º?\s*(\d+)'?\s*([\d.]+)\"?\s*([NSLO])"
    match = re.search(pattern, dms_string.upper())
    
    if not match:
        raise ValueError(f"Formato inválido: {dms_string}")
    
    degrees = float(match.group(1))
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)
    
    # Calcular decimal
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    
    # Aplicar sinal negativo para Sul e Oeste
    if direction in ['S', 'O']:
        decimal = -decimal
    
    return decimal

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

    def logreg_objective(trial):
        #penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        
        # Ajusta solver baseado no penalty
        params = {
            'penalty': 'l2', 'C': C, 'solver': solver, 
            'max_iter': max_iter, 'random_state': int(os.getenv('SEED'))
        }
        
        clf = LogisticRegression(**params).fit(X_train_norm, y_train)
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
    
    def xgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        gamma = trial.suggest_float('gamma', 0, 5)
        #subsample = trial.suggest_float('subsample', 0.5, 1.0)
        #colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-5, 100, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 100, log=True)
        
        clf = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, min_child_weight=min_child_weight,
            gamma=gamma, #1subsample=subsample, colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, 
            random_state=int(os.getenv('SEED')), n_jobs=-1, eval_metric='logloss'
        ).fit(X_train, y_train)
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=1, cv=10)
        return score.mean()

    # ============================================

    loaded_models = {}

    # 3. Create a study object and optimize the objective function.
    try:
        loaded_models['Decision_Tree'] = load_model_by_name(experiment_name=experiment_name, run_name='Decision_Tree')
    except ValueError:
        dtree_study = optuna.create_study(direction='maximize')
        dtree_study.optimize(dtree_objective, n_trials=num_trials)
        dtree_params = dtree_study.best_params
        loaded_models['Decision_Tree'] = DecisionTreeClassifier(**dtree_params, random_state=int(os.getenv('SEED')))
        loaded_models['Decision_Tree'] = log_model_to_mlflow(
            loaded_models['Decision_Tree'], "Decision_Tree", dtree_params, 
            X_train, y_train, X_test, y_test
        )

    try:
        loaded_models['Logistic_Regression'] = load_model_by_name(experiment_name=experiment_name, run_name='Logistic_Regression')
    except ValueError:
        logreg_study = optuna.create_study(direction='maximize')
        logreg_study.optimize(logreg_objective, n_trials=num_trials)
        logreg_params = logreg_study.best_params
        loaded_models['Logistic_Regression'] = LogisticRegression(**logreg_params, random_state=int(os.getenv('SEED')))
        loaded_models['Logistic_Regression'] = log_model_to_mlflow(
            loaded_models['Logistic_Regression'], "Logistic_Regression", logreg_params,
            X_train, y_train, X_test_norm, y_test
        )

    try:
        loaded_models['Gradient_Boosting'] = load_model_by_name(experiment_name=experiment_name, run_name='Gradient_Boosting')
    except ValueError:
        gb_study = optuna.create_study(direction='maximize')
        gb_study.optimize(gb_objective, n_trials=num_trials)
        gb_params = gb_study.best_params
        loaded_models['Gradient_Boosting'] = GradientBoostingClassifier(**gb_params, random_state=int(os.getenv('SEED')))
        loaded_models['Gradient_Boosting'] = log_model_to_mlflow(
            loaded_models['Gradient_Boosting'], "Gradient_Boosting", gb_params,
            X_train, y_train, X_test, y_test
        )

    try:
        loaded_models['XGBoost'] = load_model_by_name(experiment_name=experiment_name, run_name='XGBoost')
    except ValueError:
        xgb_study = optuna.create_study(direction='maximize')
        xgb_study.optimize(xgb_objective, n_trials=num_trials)
        xgb_params = xgb_study.best_params
        loaded_models['XGBoost'] = XGBClassifier(**xgb_params, random_state=int(os.getenv('SEED')), n_jobs=-1, eval_metric='logloss', enable_categorical=True)
        loaded_models['XGBoost'] = log_model_to_mlflow(
            loaded_models['XGBoost'], "XGBoost", xgb_params,
            X_train, y_train, X_test, y_test
        )

    if(load):
        return loaded_models

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