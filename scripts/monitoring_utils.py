import glob
import time
import psutil
import pandas as pd
import numpy as np

# LEER VARIABLES CRUDAS =======================================================

def read_csv_files(DIR_RAWDATA):
    csv_files = glob.glob(f'{DIR_RAWDATA}/*.csv')
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        print("Successfully unified dataframes.")
    else:
        df = pd.DataFrame()
        print("No CSV files found or read errors occurred.")
    return df

# CÁLCULO DE MÉTRICAS =======================================================

def calculate_drift(dd_metric, df_ref, df_actual, quantils):#buckets o quantiles
    breakpoints = np.percentile(df_ref, np.linspace(0, 100, quantils + 1))
    breakpoints[0] = -np.inf  # para incluir todos los valores
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(df_ref, bins=breakpoints)[0] / len(df_ref)
    actual_counts = np.histogram(df_actual, bins=breakpoints)[0] / len(df_actual)

    # Evita divisiones por cero o log(0)
    ref_counts = np.where(ref_counts == 0, 1e-6, ref_counts)
    actual_counts = np.where(actual_counts == 0, 1e-6, actual_counts)

    # calculo de la métrica de drift
    if dd_metric == 'PSI':
        dd_values = (ref_counts - actual_counts) * np.log(ref_counts / actual_counts)
    elif dd_metric == 'KL':
        dd_values = ref_counts * np.log(ref_counts / actual_counts)

    dd_value = np.sum(dd_values)
    return (dd_value, dd_values, ref_counts, actual_counts, breakpoints)

def data_drift(dd_metric, df_actual, df_ref, quantils): #dataframe actual, dataframe referencia, cant particiones
  results = []
  for col in df_actual.columns: #el input pudieron ser las columnas
    if pd.api.types.is_numeric_dtype(df_actual[col]) and pd.api.types.is_numeric_dtype(df_ref[col]): # Esto está para variables númericas
      dd_vector = calculate_drift(dd_metric, df_ref[col].dropna().values, df_actual[col].dropna().values, quantils)
      results.append({'feature': col,'metric_value': dd_vector[0]})
  return pd.DataFrame(results).set_index('feature')


# HELPERS DE MONITOREO PSI ====================================================

def psi_report(df_actual, df_ref, quantils):
    """PSI por columna numérica común. Devuelve DataFrame ordenado por PSI desc."""
    cols = [
        c for c in df_actual.columns
        if c in df_ref.columns
        and pd.api.types.is_numeric_dtype(df_actual[c])
        and pd.api.types.is_numeric_dtype(df_ref[c])
        and df_actual[c].notna().sum() > 10
        and df_ref[c].notna().sum() > 10
    ]
    rows = []
    for col in cols:
        ref_vals = df_ref[col].dropna().astype(float).values
        act_vals = df_actual[col].dropna().astype(float).values
        val, *_ = calculate_drift('PSI', ref_vals, act_vals, quantils)
        rows.append({'feature': col, 'psi': round(val, 5)})
    return pd.DataFrame(rows).set_index('feature').sort_values('psi', ascending=False)


def estado_psi(psi_val):
    """Retorna (estado, color_hex) según umbrales PSI estándar."""
    if psi_val < 0.10:
        return ('OK',    '#27ae60')
    if psi_val < 0.25:
        return ('WARN',  '#f39c12')
    return ('ALARM', '#e74c3c')


def compute_rolling_psi(history_psi: dict, period: int, window: int) -> float:
    """PSI medio de los últimos `window` períodos anteriores al período dado."""
    prior          = sorted([p for p in history_psi if p < period])
    window_periods = prior[-window:]
    if not window_periods:
        return 0.0
    return float(np.mean([history_psi[p] for p in window_periods]))


# MONITOREO OPERACIONAL =======================================================

def capture_resources() -> dict:
    """Snapshot de recursos del sistema en el momento de la llamada."""
    proc = psutil.Process()
    ram  = psutil.virtual_memory()
    return {
        'cpu_pct':      psutil.cpu_percent(interval=0.1),
        'ram_mb':       round(proc.memory_info().rss / 1024**2, 1),
        'ram_sys_pct':  ram.percent,
        'ram_sys_mb':   round(ram.used / 1024**2, 1),
    }


# FUNCIONES PRIVADAS POR ETAPA ================================================

def _drift_tabular(payload: dict, period: int,
                   actual_path: str, df_ref: pd.DataFrame):
    """PSI para datos tabulares (raw o preprocesados)."""
    Q         = payload['params']['psi_quantils']
    df_actual = pd.read_csv(actual_path)
    report    = psi_report(df_actual, df_ref, Q)
    psi_medio = float(report['psi'].mean()) if len(report) > 0 else 0.0
    return report, psi_medio


def _drift_score(payload: dict, period: int,
                 actual_path: str, df_ref: pd.DataFrame):
    """PSI de la distribución de scores."""
    Q  = payload['params']['psi_quantils']
    sc = payload['params']['score_col']
    df_actual = pd.read_csv(actual_path)
    if sc not in df_actual.columns:
        raise ValueError(f'columna "{sc}" no encontrada en {actual_path}')
    report  = psi_report(df_actual[[sc]], df_ref[[sc]], Q)
    psi_val = float(report['psi'].iloc[0]) if len(report) > 0 else 0.0
    return report, psi_val, df_actual[sc].values


def _drift_grupo(payload: dict, period: int,
                 actual_path: str, df_ref: pd.DataFrame):
    """PSI de score por grupo de ejecución."""
    Q    = payload['params']['psi_quantils']
    gcol = payload['params']['grupo_col']
    pcol = payload['params']['puntuacion_col']
    df_actual = pd.read_csv(actual_path, sep='|')
    grupos    = sorted(df_actual[gcol].dropna().unique())
    rows = []
    for g in grupos:
        mask_act = df_actual[gcol] == g
        mask_ref = df_ref[gcol] == g
        if mask_act.sum() < 10 or mask_ref.sum() < 10:
            continue
        for col in ['score', pcol]:
            if col in df_actual.columns and col in df_ref.columns:
                psi_val, *_ = calculate_drift(
                    'PSI',
                    df_ref.loc[mask_ref, col].dropna().values,
                    df_actual.loc[mask_act, col].dropna().values,
                    Q
                )
                rows.append({'grupo_ejec': g, 'columna': col,
                             'psi': round(psi_val, 5), 'n_actual': int(mask_act.sum())})
    report    = pd.DataFrame(rows)
    psi_medio = float(report['psi'].mean()) if len(report) > 0 else 0.0
    return report, psi_medio


# FUNCIÓN PÚBLICA UNIFICADA ===================================================

def monitor_stage(stage_type: str, payload: dict, period: int,
                  actual_path: str, df_ref: pd.DataFrame) -> dict:
    """
    Función unificada de monitoreo: drift PSI + métricas operacionales.

    stage_type : 'raw' | 'preprocessed' | 'score' | 'grupo_ejec'
    actual_path: ruta al archivo de datos del período actual
    df_ref     : DataFrame de referencia para el cálculo de PSI

    Retorna dict con claves:
      period, stage, report, psi_medio, [scores]   <- drift
      ops: duration_s, ram_proc_mb, ram_delta_mb,  <- operacional
           cpu_pct, ram_sys_pct, ram_sys_mb
    """
    t0         = time.time()
    res_before = capture_resources()

    result = {'period': period, 'stage': stage_type,
              'report': pd.DataFrame(), 'psi_medio': None}

    try:
        if stage_type in ('raw', 'preprocessed'):
            report, psi_medio = _drift_tabular(payload, period, actual_path, df_ref)
            result.update({'report': report, 'psi_medio': psi_medio})

        elif stage_type == 'score':
            report, psi_val, scores = _drift_score(payload, period, actual_path, df_ref)
            result.update({'report': report, 'psi_medio': psi_val, 'scores': scores})

        elif stage_type == 'grupo_ejec':
            report, psi_medio = _drift_grupo(payload, period, actual_path, df_ref)
            result.update({'report': report, 'psi_medio': psi_medio})

        else:
            raise ValueError(f'stage_type desconocido: {stage_type}')

    except Exception as e:
        print(f'  [monitor_{stage_type}] p{period}: SKIP ({e})')

    res_after = capture_resources()
    duration  = round(time.time() - t0, 2)

    result['ops'] = {
        'stage':        stage_type,
        'period':       period,
        'duration_s':   duration,
        'ram_proc_mb':  res_after['ram_mb'],
        'ram_delta_mb': round(res_after['ram_mb'] - res_before['ram_mb'], 1),
        'cpu_pct':      res_after['cpu_pct'],
        'ram_sys_pct':  res_after['ram_sys_pct'],
        'ram_sys_mb':   res_after['ram_sys_mb'],
    }

    psi = result['psi_medio']
    if psi is not None:
        icon, _ = estado_psi(psi)
        print(f'  [monitor_{stage_type:12s}] p{period}: {icon} PSI={psi:.4f} '
              f'| {duration:.1f}s | RAM {res_after["ram_mb"]:.0f}MB '
              f'(+{result["ops"]["ram_delta_mb"]:.0f}) | CPU {res_after["cpu_pct"]:.0f}%')
    else:
        print(f'  [monitor_{stage_type:12s}] p{period}: SKIP '
              f'| {duration:.1f}s | RAM {res_after["ram_mb"]:.0f}MB | CPU {res_after["cpu_pct"]:.0f}%')

    return result
