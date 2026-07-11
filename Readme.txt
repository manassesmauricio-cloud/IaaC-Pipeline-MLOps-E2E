Pipeline MLOps E2E con Airflow + MLflow
========================================

ESTRUCTURA DEL PROYECTO
-----------------------

.
├── config/
│   └── pipeline_config.yml   <- UNICO archivo a editar para cambiar parametros
│                                (equivale al dict `payload` del notebook)
├── dags/
│   ├── pipeline_dag.py       <- Orquestador raiz (punto de entrada)
│   ├── training_dag.py       <- Fase A: construccion de referencia + entrenamiento
│   └── monitoring_dag.py     <- Fase B: inferencia + monitoreo OOT
├── src/
│   ├── config_loader.py      <- Convierte pipeline_config.yml -> payload del notebook
│   └── training_mlflow.py    <- Wrapper de training.py con MLflow Registry
├── scripts/                  <- Scripts del pipeline (preprocessing, training, etc.)
│   ├── preprocessing.py      <- Preprocesamiento de datos
│   ├── training.py           <- Entrenamiento del modelo
│   ├── hpo_training.py       <- Busqueda de hiperparametros
│   ├── inference.py          <- Inferencia / scoring
│   ├── posprocessing.py      <- Posprocesamiento de scores
│   ├── monitoring_utils.py   <- Calculo PSI y metricas de monitoreo
│   └── dashboard_utils.py    <- Generacion del dashboard HTML
├── data/                     <- Datos intermedios del pipeline (generados en runtime)
│   ├── preprocessed/         <- Variables preprocesadas por periodo
│   ├── posprocesada/         <- Scores posprocesados
│   ├── scores/               <- Scores puros del modelo
│   └── referencia/           <- DataFrames de referencia (.pkl) para monitoreo
├── models/
│   ├── produccion/           <- Modelo en produccion (el ultimo entrenado/promovido)
│   └── candidatos/           <- Modelos entrenados no promovidos (con timestamp en nombre)
│       └── {timestamp}_{model_name}/
├── monitoreo/                <- Resultados de monitoreo (PSI CSVs, dashboard HTML)
├── mlflow/
│   └── artifacts/            <- Artefactos MLflow (modelos registrados en Registry)
├── docker/
│   └── init_mlflow_db.sql    <- Inicializa la BD de MLflow en Postgres
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example              <- Plantilla de configuracion (copiar a .env)
└── .gitignore


MODO DE EJECUCION (run_mode)
-----------------------------

Editar config/pipeline_config.yml -> model.run_mode:

  run_mode: training    Entrena el modelo desde cero:
                          1. Construye datos de referencia (periodos 1-4)
                          2. Entrena y registra en MLflow Model Registry
                          3. Guarda candidato en models/candidatos/
                          4. Promueve a models/produccion/
                          5. Ejecuta monitoreo OOT completo

  run_mode: inference   Solo monitoreo (modelo de produccion ya existe):
                          1. Carga referencias del disco (data/referencia/)
                          2. Ejecuta monitoreo OOT completo

  En AMBOS modos se ejecuta el monitoreo OOT con PSI en las 4 etapas del
  pipeline (raw, preprocesada, score puro, score posprocesado).


PREREQUISITOS
-------------
- Docker Desktop instalado y corriendo
- Datos crudos disponibles en alguna carpeta del HOST (data de entrenamiento + OOT)


CONFIGURACION INICIAL (una sola vez)
--------------------------------------

1. Copiar y editar el archivo de entorno:

     cp .env.example .env

   Solo hay un parametro obligatorio a ajustar:

     HOST_DATASET_PATH=/ruta/a/tu/carpeta/Dataset

   La carpeta Dataset debe contener:
     Dataset/
       data de entrenamiento/   <- CSVs de entrenamiento (p1_extrac.csv, p2_extrac.csv, ...)
       OOT/                     <- CSVs de periodos OOT  (p11_extrac.csv, p12_extrac.csv, ...)

2. Construir la imagen Docker (primera vez, ~10-15 min):

     docker compose build

   Las veces siguientes es instantaneo si no cambiaste Dockerfile ni requirements.txt.

3. Inicializar la base de datos de Airflow (primera vez):

     docker compose up airflow-init


LEVANTAR LOS SERVICIOS
-----------------------

     docker compose up -d

Esperar ~60 segundos y verificar:

     docker compose ps

Deben aparecer 4 servicios en estado "running (healthy)":
  postgres, mlflow, airflow-webserver, airflow-scheduler


INTERFACES WEB
--------------

  MLflow (experimentos y modelo registry):  http://localhost:5000
    - Experimento: pipeline-extrac
    - Model Registry: modelo-extrac (versiones por cada entrenamiento)

  Airflow (orquestacion y logs):            http://localhost:8080
    - Usuario: admin / Contrasena: admin
    - DAG de entrada: pipeline_orchestrator


EJECUTAR EL PIPELINE (punto de entrada recomendado)
----------------------------------------------------

El DAG orquestador "pipeline_orchestrator" es el punto de entrada unico.
Lee run_mode de config/pipeline_config.yml y decide automaticamente que fases correr.

Opcion A - terminal:
  docker compose exec airflow-scheduler airflow dags trigger pipeline_orchestrator

Opcion B - Airflow UI:
  http://localhost:8080 -> DAG "pipeline_orchestrator" -> boton Trigger

Flujo segun run_mode:
  training  -> training_pipeline (Fase A) -> monitoring_pipeline (Fase B)
  inference -> monitoring_pipeline (Fase B) directamente


PIPELINE DE ENTRENAMIENTO (Fase A — training_pipeline)
-------------------------------------------------------

Cuando ejecutar:
  - Primera vez (no hay modelo en models/produccion/)
  - Cuando quieres reentrenar con datos nuevos

Paso 1. Asegurarte de que run_mode sea "training" en config/pipeline_config.yml:

    model:
      run_mode: training

Paso 2. Disparar el DAG de entrenamiento (directamente, sin pasar por el orquestador):

  Opcion A - terminal:
    docker compose exec airflow-scheduler airflow dags unpause training_pipeline
    docker compose exec airflow-scheduler airflow dags trigger training_pipeline

  Opcion B - Airflow UI:
    http://localhost:8080 -> DAG "training_pipeline" -> activar toggle -> boton Trigger

Que hace este DAG (en orden):
  1. setup_directories       Crea las carpetas necesarias en data/, models/, monitoreo/
  2. preprocess_baseline     Preprocesa el periodo p1 (referencia base)
  3. [PARALELO]
     - monitor_raw_training       PSI de datos crudos p2-p4 vs p1
     - preprocess_training_periods  Preprocesa p2-p4 en modo inference
  4. build_reference         Concatena p1-p4 y guarda referencia en disco
  5. [PARALELO]
     - preprocess_training_mode   Preprocesa todos los datos en modo training (train/test split)
     - monitor_pre_reference      PSI de datos preprocesados p2-p4 vs referencia
  6. train_model             Entrena el modelo (busca el mejor entre los configurados),
                             registra en MLflow Registry y promueve a models/produccion/
  7. run_inference_reference Genera scores sobre el periodo de referencia (p1)
  8. [PARALELO]
     - monitor_scores_reference   PSI de scores de referencia
     - postprocess_reference      Posprocesa scores de referencia
  9. monitor_grupos_reference    PSI de grupos operacionales de referencia

Resultados:
  - Modelo en produccion:    models/produccion/
  - Candidato con timestamp: models/candidatos/{timestamp}_{modelo}/
  - Experimento MLflow:      http://localhost:5000 -> pipeline-extrac
  - Registry MLflow:         http://localhost:5000 -> Models -> modelo-extrac


PIPELINE DE INFERENCIA / MONITOREO OOT (Fase B — monitoring_pipeline)
-----------------------------------------------------------------------

Cuando ejecutar:
  - El modelo ya esta entrenado y en models/produccion/
  - Quieres monitorear nuevos periodos OOT sin reentrenar

Paso 1. Cambiar run_mode a "inference" en config/pipeline_config.yml:

    model:
      run_mode: inference

Paso 2. Disparar el DAG de monitoreo (directamente):

  Opcion A - terminal:
    docker compose exec airflow-scheduler airflow dags unpause monitoring_pipeline
    docker compose exec airflow-scheduler airflow dags trigger monitoring_pipeline

  Opcion B - Airflow UI:
    http://localhost:8080 -> DAG "monitoring_pipeline" -> activar toggle -> boton Trigger

  Opcion C - via orquestador (automatico):
    Asegurarte de que run_mode = "inference" en el config y disparar pipeline_orchestrator.
    El orquestador salta el entrenamiento y ejecuta directamente la Fase B.

Que hace este DAG (por cada periodo OOT configurado):
  1. load_reference          Carga desde disco las referencias generadas en el entrenamiento
                             (data/reference_raw.pkl, reference_pre.pkl, reference_scores.pkl, etc.)
  2. [PARALELO por periodo]
     - monitor_raw_p{N}      PSI de datos crudos del periodo OOT vs referencia raw
     - preprocess_p{N}       Preprocesa datos del periodo OOT
  3. [PARALELO por periodo]
     - monitor_pre_p{N}      PSI de datos preprocesados vs referencia pre
     - run_inference_p{N}    Genera scores con el modelo de produccion
  4. [PARALELO por periodo]
     - monitor_scores_p{N}   PSI de distribucion de scores
     - postprocess_p{N}      Posprocesa scores
  5. monitor_grupos_p{N}     PSI de grupos operacionales
  6. generate_dashboard      Genera el dashboard HTML con todos los resultados PSI

Resultados:
  - Dashboard HTML:          monitoreo/dashboard_monitoring_{periodo}.html
  - PSI por etapa:           monitoreo/drift_{etapa}_{periodo}.csv
  - Scores:                  data/scores/
  - Scores posprocesados:    data/posprocesada/

NOTA: Para que la Fase B funcione, la Fase A debe haberse ejecutado al menos una vez
      (las referencias en data/ deben existir en disco).


VER RESULTADOS
--------------

  Modelos entrenados:  models/candidatos/{timestamp}_{model}/
  Modelo en uso:       models/produccion/
  MLflow Registry:     http://localhost:5000 -> Models -> modelo-extrac
  Datos intermedios:   data/
  Dashboard:           monitoreo/dashboard_monitoring_{periodo}.html
  PSI por etapa:       monitoreo/drift_{stage}_{periodo}.csv


MODIFICAR EL PIPELINE
---------------------

Para cambiar la logica de preprocesamiento, entrenamiento o monitoreo:
  - Editar los .py en scripts/
  - Los contenedores montan scripts/ en modo read-only; reiniciar el scheduler
    para que tome los cambios:
      docker compose restart airflow-scheduler airflow-webserver

Para cambiar el flujo de orquestacion (orden, paralelismo, nuevos tasks):
  - Editar los DAGs en dags/
  - Airflow detecta cambios automaticamente (polling cada 30s)

Para cambiar parametros (periodos, modelo, umbrales):
  - Editar config/pipeline_config.yml
  - No requiere reiniciar servicios (se lee en cada run del DAG)


DETENER LOS SERVICIOS
---------------------

  docker compose down        # detiene contenedores, conserva datos y modelos
  docker compose down -v     # detiene Y borra volumenes de Postgres (reset completo)
