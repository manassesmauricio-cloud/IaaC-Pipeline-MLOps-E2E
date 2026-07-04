"""
DAG orquestador — equivale al parámetro run_mode del notebook

Este DAG es el PUNTO DE ENTRADA único para los estudiantes.
Lee run_mode de config/pipeline_config.yml y decide qué sub-DAGs ejecutar:

  run_mode: training  -> training_pipeline (Fase A) + monitoring_pipeline (Fase B)
  run_mode: inference -> monitoring_pipeline (Fase B directamente)

Equivalencia con el notebook:

  Notebook:
    payload['params']['run_mode'] = 'training'
    results = monitoring_pipeline(payload)   # corre todo

  Proyecto Airflow:
    # Cambiar run_mode en config/pipeline_config.yml
    # Trigger DAG 'pipeline_orchestrator' desde la UI o:
    #   make trigger-pipeline

Flujo visual:

  run_mode=training:
    check_run_mode → trigger_training → trigger_monitoring
                     (training_pipeline DAG, espera)  (monitoring_pipeline DAG)

  run_mode=inference:
    check_run_mode → skip_training → trigger_monitoring
                                      (monitoring_pipeline DAG)
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import DagRun
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from config_loader import load_config


DEFAULT_ARGS = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 0,
}


@dag(
    dag_id='pipeline_orchestrator',
    description='Punto de entrada único: lee run_mode del config y orquesta training + monitoring',
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['orchestrator', 'mlops'],
)
def pipeline_orchestrator():
    """
    DAG raíz que decide el modo de ejecución.

    Para cambiar de modo:
      1. Editar config/pipeline_config.yml → model.run_mode: training|inference
      2. Hacer Trigger de este DAG (no de los sub-DAGs directamente)
    """
    cfg      = load_config()
    run_mode = cfg['model'].get('run_mode', 'training')

    @task.branch(task_id='check_run_mode')
    def check_run_mode() -> str:
        _cfg      = load_config()
        _run_mode = _cfg['model'].get('run_mode', 'training')
        print(f'[ORCHESTRATOR] run_mode = {_run_mode}')
        if _run_mode not in ('training', 'inference'):
            raise ValueError(
                f'run_mode invalido: "{_run_mode}". Valores permitidos: training | inference'
            )
        return 'trigger_training' if _run_mode == 'training' else 'skip_training'

    # Rama training: dispara el DAG de entrenamiento y espera a que termine
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training',
        trigger_dag_id='training_pipeline',
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True,
    )

    # Rama inference: no hay nada que hacer en Fase A
    skip_training = EmptyOperator(
        task_id='skip_training',
    )

    # Ambas ramas convergen en monitoreo
    # trigger_rule='none_failed_min_one_success' permite que la rama que NO se ejecutó
    # no bloquee al siguiente task (comportamiento correcto para BranchPythonOperator)
    trigger_monitoring = TriggerDagRunOperator(
        task_id='trigger_monitoring',
        trigger_dag_id='monitoring_pipeline',
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True,
        trigger_rule='none_failed_min_one_success',
    )

    branch = check_run_mode()
    branch >> [trigger_training, skip_training] >> trigger_monitoring


pipeline_orchestrator()
