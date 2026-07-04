"""
Wrapper de training.py con tracking de MLflow y gestión de candidatos/produccion.

Flujo:
  1. Entrena en models/candidatos/{timestamp}_{model_name}/
  2. Registra el modelo en MLflow Model Registry como 'modelo-{model_name}'
  3. Promueve (copia) al directorio de produccion si supera al actual

Equivalencia notebook:
    # Notebook — solo guarda en disco
    aml.main(train_path, test_path, model_dir)

    # Proyecto — candidatos + produccion + MLflow Registry
    training_mlflow.main(train_path, test_path, model_dir, model_dir_candidatos, payload)
"""
import glob
import json
import os
import pickle
import shutil
from datetime import datetime

import mlflow
import mlflow.sklearn

import training as aml


def main(train_path: str, test_path: str, model_dir: str, model_dir_candidatos: str,
         payload: dict, run_name: str = 'training') -> str:
    """
    Entrena el modelo con MLflow tracking activo.

    Registra en MLflow:
    - Parámetros: rutas, config del modelo
    - Métricas: AUC train/test, decay %, tiempo de entrenamiento
    - Artefactos: modelo pkl, metadata json, config YAML
    - Registry: modelo registrado como 'modelo-{model_name}' (versionado por MLflow)

    Retorna la ruta al directorio de produccion con el modelo promovido.
    """
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME', 'pipeline-extrac'))

    model_name = payload['params']['model_name']
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate_dir = os.path.join(model_dir_candidatos, f'{timestamp}_{model_name}')
    os.makedirs(candidate_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            'model_name':       model_name,
            'training_periods': str(payload['params']['training_periods']),
            'psi_quantils':     payload['params']['psi_quantils'],
            'train_path':       train_path,
            'test_path':        test_path,
            'candidate_dir':    candidate_dir,
        })

        # Entrena en el directorio candidato
        aml.main(train_path, test_path, candidate_dir)

        # Leer metadata guardada por training.py y registrar métricas
        metadata_files = glob.glob(os.path.join(candidate_dir, '**', '*_metadata.json'),
                                   recursive=True)
        best_candidate_subdir = None
        auc_test = 0.0

        for mf in metadata_files:
            with open(mf) as f:
                meta = json.load(f)

            perf = meta.get('performance', {})
            auc_test = perf.get('auc_test', 0)
            mlflow.log_metrics({
                'auc_train':    perf.get('auc_train',    0),
                'auc_test':     auc_test,
                'decay_pct':    perf.get('decay_percent', 100),
                'train_time_s': perf.get('training_time_segs', 0),
            })
            mlflow.log_param('best_model_type', meta.get('ml_name', 'unknown'))
            best_candidate_subdir = os.path.dirname(mf)

        # Registrar artefactos en MLflow y en el Model Registry
        if best_candidate_subdir:
            mlflow.log_artifacts(best_candidate_subdir, artifact_path='model')

            pkl_files = glob.glob(os.path.join(best_candidate_subdir, '*.pkl'))
            if pkl_files:
                try:
                    with open(pkl_files[-1], 'rb') as f:
                        fitted_model = pickle.load(f)
                    mlflow.sklearn.log_model(
                        fitted_model,
                        artifact_path='model_registry',
                        registered_model_name=f'modelo-{model_name}',
                    )
                    print(f'Modelo registrado en MLflow Registry: modelo-{model_name}')
                except Exception as e:
                    print(f'MLflow Registry skipped (no es sklearn puro): {e}')

        run_id = mlflow.active_run().info.run_id
        print(f'MLflow run_id: {run_id}')
        print(f'Candidato guardado en: {candidate_dir}')

    # Promover candidato a produccion
    _promote_to_produccion(candidate_dir, model_dir, auc_test)

    return model_dir


def _promote_to_produccion(candidate_dir: str, produccion_dir: str, auc_test: float):
    """
    Copia el candidato a models/produccion/, reemplazando el anterior.
    Siempre promueve (el reentrenamiento es intencional).
    """
    for item in os.listdir(produccion_dir):
        item_path = os.path.join(produccion_dir, item)
        if item_path.endswith('.gitkeep'):
            continue
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    for item in os.listdir(candidate_dir):
        src = os.path.join(candidate_dir, item)
        dst = os.path.join(produccion_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)

    print(f'Modelo promovido a produccion (AUC test={auc_test:.4f}): {produccion_dir}')
