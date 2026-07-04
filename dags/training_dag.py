"""
DAG de entrenamiento — Equivale a la FASE A del notebook 11. Pipeline monitoreo.ipynb

Construye la referencia fija de monitoreo y entrena el modelo con MLflow tracking.

Equivalencia Prefect → Airflow:
  @flow monitoring_pipeline (Fase A)  →  DAG 'training_pipeline'
  @task setup_directories             →  task 'setup_directories'
  @task preprocess_data               →  task 'preprocess_*'
  @task train_model                   →  task 'train_model'  (+ MLflow)
  .submit() paralelo                  →  tasks sin dependencia entre sí
  ConcurrentTaskRunner                →  LocalExecutor (paralelismo a nivel DAG)

Flujo:
  setup
    └─ preprocess_p1 (baseline)
         ├─ [monitor_raw_training] ─── paralelo ─── [preprocess_training_periods]
         └─ build_reference
              ├─ [preprocess_training_mode] ─── paralelo ─── [monitor_pre_reference]
              └─ train_model
                   └─ run_inference_reference
                        ├─ [monitor_scores_reference] ─── paralelo ─── [postprocess_reference]
                        └─ monitor_grupos_reference
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup

# src/ y scripts/ están en PYTHONPATH (ver docker-compose.yml)
import monitoring_utils as mon
import preprocessing as prep
import inference as inf
import posprocessing as posp
from config_loader import load_payload
import training_mlflow


DEFAULT_ARGS = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='training_pipeline',
    description='Fase A: construcción de referencia + entrenamiento con MLflow',
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training', 'mlops', 'referencia'],
)
def training_pipeline():

    # ── Task 0: Setup de directorios ──────────────────────────────────────────
    # Equivale a: setup_directories(payload) en el notebook
    @task(task_id='setup_directories')
    def setup_directories() -> dict:
        payload = load_payload()
        dirs = [
            f'{payload["DIR_PROCESSED"]}/preprocessed',
            f'{payload["DIR_PROCESSED"]}/postprocessed',
            f'{payload["DIR_PROCESSED"]}/training_data',
            f'{payload["DIR_PROCESSED"]}/training_data/preprocessed',
            f'{payload["DIR_PROCESSED"]}/training_data/postprocessed',
            payload['MODEL_DIR'],
            payload['MODEL_DIR_CANDIDATOS'],
            payload['SCORE_DIR'],
            payload['DIR_OUTPUT'],
            payload['DIR_MONITORING'],
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f'{len(dirs)} directorios listos')
        # Retornar payload serializable (XCom solo acepta JSON-serializables)
        return payload

    # ── Task A1: Preprocesar período de referencia (p1) ───────────────────────
    # Equivale a: dir_pre_p1, dir_post_p1 = preprocess_data(payload, trn_ps[0])
    @task(task_id='preprocess_baseline')
    def preprocess_baseline(payload: dict) -> dict:
        model   = payload['params']['model_name']
        period  = payload['params']['training_periods'][0]
        raw_dir = payload['DIR_RAWDATA_TRAIN']
        prep.main(model, raw_dir, payload['DIR_PROCESSED'], 'inference', period)
        return {
            'dir_pre':  f'{payload["DIR_PROCESSED"]}/preprocessed/vars_{period}_{model}.csv',
            'dir_post': f'{payload["DIR_PROCESSED"]}/postprocessed/post_{period}_{model}.csv',
            'period':   period,
        }

    # ── Tasks A2 (paralelos): monitor_raw + preprocess para p2-p4 ─────────────
    # Equivale a:
    #   raw_mon_train_futs[p] = monitor_stage_task.submit('raw', ...)
    #   pre_train_futs[p]     = preprocess_data.submit(...)
    # Ambos se ejecutan sin dependencia entre sí → Airflow los lanza en paralelo

    @task(task_id='monitor_raw_training')
    def monitor_raw_training(payload: dict, p1_paths: dict) -> list:
        """
        Monitorea datos crudos de p2-p4 vs p1 (baseline).
        Equivale a los monitor_stage_task.submit('raw', ...) de la Fase A.
        Retorna lista de dicts result con campo 'ops' para operacional.
        """
        trn_ps  = payload['params']['training_periods']
        raw_dir = payload['DIR_RAWDATA_TRAIN']
        df_ref  = pd.read_csv(f'{raw_dir}/p{trn_ps[0]}_extrac.csv')

        results = []
        for period in trn_ps[1:]:
            raw_path = f'{raw_dir}/p{period}_extrac.csv'
            result   = mon.monitor_stage('raw', payload, period, raw_path, df_ref)
            results.append({
                'period':    result['period'],
                'psi_medio': result.get('psi_medio'),
                'ops':       result.get('ops', {}),
            })
        return results

    @task(task_id='preprocess_training_periods')
    def preprocess_training_periods(payload: dict) -> dict:
        """
        Preprocesa p2-p4 en modo inference para construir la referencia pre.
        Corre EN PARALELO con monitor_raw_training (sin dependencia entre sí).
        """
        model   = payload['params']['model_name']
        trn_ps  = payload['params']['training_periods']
        raw_dir = payload['DIR_RAWDATA_TRAIN']
        paths   = {}
        for period in trn_ps[1:]:
            prep.main(model, raw_dir, payload['DIR_PROCESSED'], 'inference', period)
            paths[str(period)] = {
                'dir_pre':  f'{payload["DIR_PROCESSED"]}/preprocessed/vars_{period}_{model}.csv',
                'dir_post': f'{payload["DIR_PROCESSED"]}/postprocessed/post_{period}_{model}.csv',
            }
        return paths

    # ── Task: Construir referencia (concat p1-p4 raw + pre) ───────────────────
    # Equivale a: df_ref_raw = pd.concat(all_raw_train) + save
    @task(task_id='build_reference')
    def build_reference(payload: dict, p1_paths: dict, all_pre_paths: dict) -> dict:
        """
        Concatena p1-p4 y guarda los DataFrames de referencia en disco.
        En el notebook, estos se pasaban como objetos en memoria.
        En Airflow, se persisten en disco y se cargan desde la ruta.
        """
        trn_ps  = payload['params']['training_periods']
        raw_dir = payload['DIR_RAWDATA_TRAIN']
        proc    = payload['DIR_PROCESSED']

        all_raw = [pd.read_csv(f'{raw_dir}/p{p}_extrac.csv') for p in trn_ps]
        df_ref_raw = pd.concat(all_raw, ignore_index=True)

        all_pre = [pd.read_csv(p1_paths['dir_pre'])]
        for period in trn_ps[1:]:
            all_pre.append(pd.read_csv(all_pre_paths[str(period)]['dir_pre']))
        df_ref_pre = pd.concat(all_pre, ignore_index=True)

        ref_raw_path = f'{proc}/reference_raw.pkl'
        ref_pre_path = f'{proc}/reference_pre.pkl'
        df_ref_raw.to_pickle(ref_raw_path)
        df_ref_pre.to_pickle(ref_pre_path)

        print(f'Referencia guardada: {ref_raw_path}, {ref_pre_path}')
        return {
            'ref_raw_path': ref_raw_path,
            'ref_pre_path': ref_pre_path,
            'p1_pre_path':  p1_paths['dir_pre'],
            'p1_post_path': p1_paths['dir_post'],
        }

    # ── Tasks A3 (paralelos): preprocess_training_mode + monitor_pre_ref ──────
    # Equivale a:
    #   prep.main(model, DIR_RAWDATA_TRAIN, DIR_PROCESSED, 'training')  (síncrono)
    #   pre_mon_ref_futs = [monitor_stage_task.submit('preprocessed', ...) ...]

    @task(task_id='preprocess_training_mode')
    def preprocess_training_mode(payload: dict) -> dict:
        """
        Modo training: genera train_vars/test_vars para entrenar el modelo.
        Corre EN PARALELO con monitor_pre_reference.
        """
        model = payload['params']['model_name']
        prep.main(model, payload['DIR_RAWDATA_TRAIN'], payload['DIR_PROCESSED'], 'training')
        proc = payload['DIR_PROCESSED']
        return {
            'train_path': f'{proc}/training_data/preprocessed/train_vars_{model}.csv',
            'test_path':  f'{proc}/training_data/preprocessed/test_vars_{model}.csv',
        }

    @task(task_id='monitor_pre_reference')
    def monitor_pre_reference(payload: dict, ref_paths: dict) -> list:
        """
        Monitorea datos preprocesados de referencia (p2-p4) vs concat.
        Corre EN PARALELO con preprocess_training_mode.
        """
        df_ref_pre = pd.read_pickle(ref_paths['ref_pre_path'])
        trn_ps     = payload['params']['training_periods']
        proc       = payload['DIR_PROCESSED']
        model      = payload['params']['model_name']

        results = []
        for period in trn_ps[1:]:
            pre_path = f'{proc}/preprocessed/vars_{period}_{model}.csv'
            result   = mon.monitor_stage('preprocessed', payload, period, pre_path, df_ref_pre)
            results.append({
                'period':    result['period'],
                'psi_medio': result.get('psi_medio'),
                'ops':       result.get('ops', {}),
            })
        return results

    # ── Task A4: Entrenar modelo con MLflow ───────────────────────────────────
    # Equivale a: train_model(payload)  +  MLflow tracking (nuevo en el proyecto)
    @task(task_id='train_model')
    def train_model(payload: dict, train_test_paths: dict) -> str:
        """
        Entrena el modelo y registra en MLflow Model Registry.
        Guarda candidato en MODEL_DIR_CANDIDATOS y promueve a MODEL_DIR (produccion).
        """
        import glob as gl
        model_dir = payload['MODEL_DIR']
        modelos   = gl.glob(f'{model_dir}/**/*.pkl', recursive=True)
        if modelos:
            print(f'Modelo en produccion: {os.path.basename(modelos[-1])} — omitiendo reentrenamiento')
            return model_dir

        model_dir = training_mlflow.main(
            train_path=train_test_paths['train_path'],
            test_path=train_test_paths['test_path'],
            model_dir=model_dir,
            model_dir_candidatos=payload['MODEL_DIR_CANDIDATOS'],
            payload=payload,
            run_name=f'training_{payload["params"]["model_name"]}',
        )
        return model_dir

    # ── Task A5: Inferencia sobre período de referencia (p1) ──────────────────
    @task(task_id='run_inference_reference')
    def run_inference_reference(payload: dict, ref_paths: dict) -> str:
        model  = payload['params']['model_name']
        period = payload['params']['training_periods'][0]
        score_path = f'{payload["SCORE_DIR"]}/inference_{model}_{period}.csv'
        if not os.path.exists(score_path):
            inf.main(payload['MODEL_DIR'], ref_paths['p1_pre_path'], payload['SCORE_DIR'])
        return score_path

    # ── Tasks A6 (paralelos): monitor_scores_ref + postprocess_ref ────────────
    @task(task_id='monitor_scores_reference')
    def monitor_scores_reference(payload: dict, score_ref_path: str) -> dict:
        """Monitorea distribución de scores de referencia."""
        df_ref_scores = pd.read_csv(score_ref_path)
        period = payload['params']['training_periods'][0]
        result = mon.monitor_stage('score', payload, period, score_ref_path, df_ref_scores)
        # Guardar df_ref_scores en disco (lo necesita monitoring_dag)
        ref_score_path = f'{payload["DIR_PROCESSED"]}/reference_scores.pkl'
        df_ref_scores.to_pickle(ref_score_path)
        return {'psi_medio': result.get('psi_medio'), 'ops': result.get('ops', {}),
                'ref_score_path': ref_score_path}

    @task(task_id='postprocess_reference')
    def postprocess_reference(payload: dict, ref_paths: dict, score_ref_path: str) -> str:
        """Posprocesa scores de referencia. Corre EN PARALELO con monitor_scores_reference."""
        model  = payload['params']['model_name']
        period = payload['params']['training_periods'][0]
        posp.main(ref_paths['p1_post_path'], score_ref_path, payload['DIR_OUTPUT'])
        replica_path = f'{payload["DIR_OUTPUT"]}/scr_{model}_{period}.txt'
        # Guardar df_ref_replica (lo necesita monitoring_dag)
        df_replica = pd.read_csv(replica_path, sep='|')
        ref_replica_path = f'{payload["DIR_PROCESSED"]}/reference_replica.pkl'
        df_replica.to_pickle(ref_replica_path)
        return ref_replica_path

    # ── Task A7: Monitor grupos de referencia ─────────────────────────────────
    @task(task_id='monitor_grupos_reference')
    def monitor_grupos_reference(payload: dict, ref_replica_path: str) -> dict:
        df_ref_replica = pd.read_pickle(ref_replica_path)
        period = payload['params']['training_periods'][0]
        result = mon.monitor_stage('grupo_ejec', payload, period, ref_replica_path, df_ref_replica)
        return {'psi_medio': result.get('psi_medio'), 'ops': result.get('ops', {})}

    # ════════════════════════════════════════════════════════════════════════════
    # DEFINICIÓN DEL DAG: encadenamiento de tasks
    # En Airflow, la dependencia se expresa PASANDO outputs como inputs.
    # Las tasks que reciben el mismo upstream pero no dependen entre sí
    # son lanzadas EN PARALELO por el scheduler.
    # ════════════════════════════════════════════════════════════════════════════

    payload     = setup_directories()
    p1_paths    = preprocess_baseline(payload)

    # A2: PARALELO — raw_mon y preprocess no tienen dependencia entre sí
    raw_mon_res  = monitor_raw_training(payload, p1_paths)
    all_pre      = preprocess_training_periods(payload)

    # build_reference necesita ambos A2 completados
    ref_paths    = build_reference(payload, p1_paths, all_pre)

    # A3: PARALELO — mode_training y monitor_pre no dependen entre sí
    train_paths  = preprocess_training_mode(payload)
    pre_mon_res  = monitor_pre_reference(payload, ref_paths)

    # A4: train_model necesita ambos A3
    model_dir    = train_model(payload, train_paths)

    # A5: inferencia de referencia
    score_ref    = run_inference_reference(payload, ref_paths)

    # A6: PARALELO — monitor_scores y postprocess no dependen entre sí
    score_mon    = monitor_scores_reference(payload, score_ref)
    replica_ref  = postprocess_reference(payload, ref_paths, score_ref)

    # A7: grupos necesita postprocess completado
    monitor_grupos_reference(payload, replica_ref)


# Instanciar el DAG
training_pipeline()
