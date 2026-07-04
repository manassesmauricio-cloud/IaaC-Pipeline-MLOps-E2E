"""
Carga pipeline_config.yml y lo convierte al formato `payload` del notebook.

Equivalencia notebook:
    payload = { 'DIR_RAWDATA_TRAIN': ..., 'params': {'run_mode': ..., ...} }

Equivalencia proyecto:
    payload = load_payload()   # lee config/pipeline_config.yml
"""
import yaml
import os


CONFIG_PATH = os.environ.get(
    'PIPELINE_CONFIG_PATH',
    '/opt/airflow/config/pipeline_config.yml'
)


def load_config() -> dict:
    """Retorna el YAML completo como dict."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_payload() -> dict:
    """
    Construye el dict `payload` en el mismo formato que usa el notebook.
    Permite reusar sin cambios monitoring_utils y dashboard_utils.
    """
    cfg = load_config()
    p   = cfg['paths']
    m   = cfg['model']

    return {
        'DIR_RAWDATA_TRAIN':   p['raw_train'],
        'DIR_RAWDATA_OOT':     p['raw_oot'],
        'DIR_PROCESSED':       p['processed'],
        'MODEL_DIR':           p['models'],
        'MODEL_DIR_CANDIDATOS': p['candidatos'],
        'SCORE_DIR':           p['scores'],
        'DIR_OUTPUT':          p['output'],
        'DIR_MONITORING':      p['monitoring'],
        'params': {
            'run_mode':           m.get('run_mode', 'training'),
            'model_name':         m['name'],
            'training_periods':   m['training_periods'],
            'monitoring_periods': m['monitoring_periods'],
            'moving_avg_window':  m['moving_avg_window'],
            'top_n_vars':         m['top_n_vars'],
            'psi_quantils':       m['psi_quantils'],
            'score_col':          m['score_col'],
            'puntuacion_col':     m['puntuacion_col'],
            'grupo_col':          m['grupo_col'],
        },
    }
