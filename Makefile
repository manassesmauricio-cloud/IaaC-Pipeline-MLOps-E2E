# ── Comandos del proyecto MLOps ──────────────────────────────────────────────
# Uso: make <comando>

.PHONY: help up down init logs ps trigger-pipeline trigger-training trigger-monitoring

help:
	@echo "Comandos disponibles:"
	@echo "  make init               - Primera vez: crea .env y construye imagen"
	@echo "  make up                 - Levantar todos los servicios"
	@echo "  make down               - Detener todos los servicios"
	@echo "  make logs               - Ver logs del scheduler"
	@echo "  make ps                 - Estado de contenedores"
	@echo "  make trigger-pipeline   - PUNTO DE ENTRADA: lee run_mode del config y orquesta"
	@echo "  make trigger-training   - Ejecutar SOLO pipeline de entrenamiento (Fase A)"
	@echo "  make trigger-monitoring - Ejecutar SOLO pipeline de monitoreo (Fase B)"
	@echo "  make mlflow-ui          - Abrir MLflow UI"
	@echo "  make airflow-ui         - Abrir Airflow UI"
	@echo ""
	@echo "Flujo normal:"
	@echo "  1a vez: editar config/pipeline_config.yml → run_mode: training  → make trigger-pipeline"
	@echo "  Luego : editar config/pipeline_config.yml → run_mode: inference → make trigger-pipeline"

init:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Archivo .env creado. Edita HOST_DATASET_PATH y HOST_SCRIPTS_PATH."; fi
	docker compose build

up:
	docker compose up -d airflow-init
	@sleep 10
	docker compose up -d airflow-webserver airflow-scheduler mlflow

down:
	docker compose down

logs:
	docker compose logs -f airflow-scheduler

ps:
	docker compose ps

trigger-pipeline:
	docker compose exec airflow-scheduler airflow dags trigger pipeline_orchestrator

trigger-training:
	docker compose exec airflow-scheduler airflow dags trigger training_pipeline

trigger-monitoring:
	docker compose exec airflow-scheduler airflow dags trigger monitoring_pipeline

mlflow-ui:
	@echo "MLflow UI: http://localhost:5000"

airflow-ui:
	@echo "Airflow UI: http://localhost:8080  (admin / admin)"
