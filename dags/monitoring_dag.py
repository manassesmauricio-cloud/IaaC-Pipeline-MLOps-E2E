"""
DAG de monitoreo OOT — Equivale a la FASE B del notebook 11. Pipeline monitoreo.ipynb

Ejecuta el pipeline de inferencia + monitoreo para cada período OOT,
usando la referencia construida por el training_dag.

Equivalencia Prefect → Airflow:
  monitoring_pipeline (Fase B, for period in mon_ps)  →  DAG 'monitoring_pipeline'
  monitor_stage_task.submit('raw', ...)               →  task 'monitor_raw_p{N}'
  preprocess_data.submit(...)                         →  task 'preprocess_p{N}'
  (los pares submit sin dependencia)                  →  (tasks sin dependencia → paralelo)
  generate_dashboard(...)                             →  task 'generate_dashboard'

Flujo por período OOT:
  load_reference
      ├─ [monitor_raw_p{N}] ─── paralelo ─── [preprocess_p{N}]
      └─ B2:
          ├─ [monitor_pre_p{N}] ─── paralelo ─── [run_inference_p{N}]
          └─ B3:
              ├─ [monitor_scores_p{N}] ─── paralelo ─── [postprocess_p{N}]
              └─ monitor_grupos_p{N}
  generate_dashboard  (después de todos los períodos)
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta

import mlflow
import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup

import monitoring_utils as mon
import dashboard_utils as dash
import preprocessing as prep
import inference as inf
import posprocessing as posp
from config_loader import load_payload, load_config


DEFAULT_ARGS = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='monitoring_pipeline',
    description='Fase B: inferencia + monitoreo PSI y operacional para períodos OOT',
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['monitoring', 'mlops', 'oot'],
)
def monitoring_pipeline():

    cfg = load_config()

    # ── Task: Cargar referencia desde disco (producida por training_dag) ───────
    # En el notebook, df_ref_raw/pre/scores/replica vivían en memoria del flow.
    # En Airflow, los leemos desde el disco donde training_dag los guardó.
    @task(task_id='load_reference')
    def load_reference() -> dict:
        payload = load_payload()
        proc    = payload['DIR_PROCESSED']
        required = {
            'ref_raw':     f'{proc}/reference_raw.pkl',
            'ref_pre':     f'{proc}/reference_pre.pkl',
            'ref_scores':  f'{proc}/reference_scores.pkl',
            'ref_replica': f'{proc}/reference_replica.pkl',
        }
        for name, path in required.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f'Referencia "{name}" no encontrada en {path}. '
                    'Ejecuta primero el DAG training_pipeline.'
                )
        print('Referencia cargada OK.')
        return {**required, 'payload': payload}

    # ════════════════════════════════════════════════════════════════════════════
    # POR PERÍODO OOT: grupo de tasks paralelos
    # En el notebook: for period in mon_ps: [submit raw] [submit preprocess] ...
    # En Airflow: TaskGroup por período, tasks sin dependencia entre sí = paralelo
    # ════════════════════════════════════════════════════════════════════════════

    def make_period_tasks(period: int):
        """
        Crea el grupo de tasks para un período OOT.
        Retorna dict con los resultados de monitoreo para el dashboard.
        """
        with TaskGroup(group_id=f'periodo_{period}') as tg:

            # B1a: monitor_raw — no depende de preprocess
            @task(task_id=f'monitor_raw_p{period}')
            def monitor_raw(ref_data: dict) -> dict:
                payload  = ref_data['payload']
                df_ref   = pd.read_pickle(ref_data['ref_raw'])
                raw_path = f'{payload["DIR_RAWDATA_OOT"]}/p{period}_extrac.csv'
                result   = mon.monitor_stage('raw', payload, period, raw_path, df_ref)
                return _serializable(result)

            # B1b: preprocess — no depende de monitor_raw (PARALELO)
            @task(task_id=f'preprocess_p{period}')
            def preprocess(ref_data: dict) -> dict:
                payload = ref_data['payload']
                model   = payload['params']['model_name']
                prep.main(model, payload['DIR_RAWDATA_OOT'], payload['DIR_PROCESSED'],
                          'inference', period)
                return {
                    'dir_pre':  f'{payload["DIR_PROCESSED"]}/preprocessed/vars_{period}_{model}.csv',
                    'dir_post': f'{payload["DIR_PROCESSED"]}/postprocessed/post_{period}_{model}.csv',
                }

            # B2a: monitor_pre — depende de preprocess (necesita dir_pre)
            @task(task_id=f'monitor_pre_p{period}')
            def monitor_pre(ref_data: dict, pre_paths: dict) -> dict:
                payload  = ref_data['payload']
                df_ref   = pd.read_pickle(ref_data['ref_pre'])
                result   = mon.monitor_stage('preprocessed', payload, period,
                                             pre_paths['dir_pre'], df_ref)
                return _serializable(result)

            # B2b: run_inference — depende de preprocess, no de monitor_pre (PARALELO)
            @task(task_id=f'run_inference_p{period}')
            def run_inference(ref_data: dict, pre_paths: dict) -> str:
                payload = ref_data['payload']
                model   = payload['params']['model_name']
                inf.main(payload['MODEL_DIR'], pre_paths['dir_pre'], payload['SCORE_DIR'])
                return f'{payload["SCORE_DIR"]}/inference_{model}_{period}.csv'

            # B3a: monitor_scores — depende de inference, no de postprocess (PARALELO)
            @task(task_id=f'monitor_scores_p{period}')
            def monitor_scores(ref_data: dict, score_path: str) -> dict:
                payload       = ref_data['payload']
                df_ref_scores = pd.read_pickle(ref_data['ref_scores'])
                result        = mon.monitor_stage('score', payload, period,
                                                  score_path, df_ref_scores)
                return _serializable(result)

            # B3b: postprocess — depende de inference, no de monitor_scores (PARALELO)
            @task(task_id=f'postprocess_p{period}')
            def postprocess(ref_data: dict, pre_paths: dict, score_path: str) -> str:
                payload = ref_data['payload']
                model   = payload['params']['model_name']
                posp.main(pre_paths['dir_post'], score_path, payload['DIR_OUTPUT'])
                return f'{payload["DIR_OUTPUT"]}/scr_{model}_{period}.txt'

            # B4: monitor_grupos — depende de postprocess (secuencial)
            @task(task_id=f'monitor_grupos_p{period}')
            def monitor_grupos(ref_data: dict, replica_path: str) -> dict:
                payload        = ref_data['payload']
                df_ref_replica = pd.read_pickle(ref_data['ref_replica'])
                result         = mon.monitor_stage('grupo_ejec', payload, period,
                                                   replica_path, df_ref_replica)
                return _serializable(result)

            # ── Encadenamiento dentro del período ─────────────────────────────
            ref = load_reference()

            raw_res  = monitor_raw(ref)        # B1a
            pre_p    = preprocess(ref)          # B1b  (paralelo con B1a)

            pre_res  = monitor_pre(ref, pre_p)  # B2a
            score_p  = run_inference(ref, pre_p) # B2b (paralelo con B2a)

            scr_res  = monitor_scores(ref, score_p)   # B3a
            rep_p    = postprocess(ref, pre_p, score_p) # B3b (paralelo con B3a)

            grp_res  = monitor_grupos(ref, rep_p)     # B4

        return raw_res, pre_res, scr_res, grp_res

    # ── Task: Generar dashboard consolidado ───────────────────────────────────
    # Equivale a: generate_dashboard(payload, all_raw, all_pre, all_scores, all_grupos)
    @task(task_id='generate_dashboard')
    def generate_dashboard(*period_results) -> str:
        """
        Consolida resultados de todos los períodos y genera el dashboard.
        También registra PSI medios en MLflow.

        DIFERENCIA con el notebook: aquí se registra PSI en MLflow para
        trazabilidad histórica del drift por run.
        """
        payload = load_payload()

        all_raw, all_pre, all_scores, all_grupos = [], [], [], []
        for raw_r, pre_r, scr_r, grp_r in period_results:
            all_raw.append(raw_r)
            all_pre.append(pre_r)
            all_scores.append(scr_r)
            all_grupos.append(grp_r)

        # Dashboard PNG
        output_path = dash.generate_dashboard(
            payload, all_raw, all_pre, all_scores, all_grupos
        )

        # Registrar PSI en MLflow para trazabilidad histórica
        _log_psi_to_mlflow(payload, all_raw, all_pre, all_scores, all_grupos)

        # Resumen en stdout (igual que el notebook)
        print('\n' + '=' * 60)
        print('RESUMEN DE MONITOREO')
        print('=' * 60)
        for stage_label, res_list in [('RAW',   all_raw),   ('PRE',   all_pre),
                                       ('SCORE', all_scores), ('GRUPO', all_grupos)]:
            for r in res_list:
                psi = r.get('psi_medio')
                if psi is not None:
                    icon, _ = mon.estado_psi(psi)
                    print(f'  {stage_label:6s} p{r["period"]}: {icon} PSI={psi:.4f}')
        print('=' * 60)

        return output_path

    # ── Encadenamiento principal del DAG ──────────────────────────────────────
    mon_periods = cfg['model']['monitoring_periods']

    all_period_results = [make_period_tasks(p) for p in mon_periods]
    generate_dashboard(*all_period_results)


# ── Helpers internos ──────────────────────────────────────────────────────────

def _serializable(result: dict) -> dict:
    """
    Convierte un result dict a formato JSON-serializable para XCom de Airflow.
    DataFrames y arrays numpy no son serializables → convertir a listas/dicts.
    """
    import numpy as np
    out = {
        'period':    result.get('period'),
        'stage':     result.get('stage'),
        'psi_medio': result.get('psi_medio'),
        'ops':       result.get('ops', {}),
    }
    if 'scores' in result and result['scores'] is not None:
        out['scores'] = result['scores'].tolist() if isinstance(result['scores'], np.ndarray) \
                        else result['scores']
    # report como dict (feature → psi) — omitir si vacío
    report = result.get('report')
    if report is not None and len(report) > 0:
        out['report'] = report['psi'].to_dict() if 'psi' in report.columns else {}
    else:
        out['report'] = {}
    return out


def _log_psi_to_mlflow(payload: dict, all_raw, all_pre, all_scores, all_grupos):
    """Registra PSI medios en MLflow como métricas de monitoreo."""
    try:
        import os
        mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME', 'pipeline-extrac'))

        with mlflow.start_run(run_name=f'monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
            for stage_label, res_list in [('raw',   all_raw),   ('pre',   all_pre),
                                           ('score', all_scores), ('grupo', all_grupos)]:
                for r in res_list:
                    if r.get('psi_medio') is not None:
                        mlflow.log_metric(
                            f'psi_{stage_label}_p{r["period"]}',
                            r['psi_medio']
                        )
    except Exception as e:
        print(f'MLflow logging skipped: {e}')


# Instanciar el DAG
monitoring_pipeline()
