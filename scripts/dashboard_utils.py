import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from monitoring_utils import estado_psi, compute_rolling_psi


# ── HELPERS DE MATRIZ PSI ─────────────────────────────────────────────────────

def build_psi_matrix(results_list: list, top_n: int) -> pd.DataFrame:
    """Construye matriz feature x periodo de PSI para el heatmap del dashboard."""
    all_reports = {r['period']: r['report'] for r in results_list if len(r['report']) > 0}
    if not all_reports:
        return pd.DataFrame()
    matrix = pd.DataFrame({f'p{p}': rpt['psi'] for p, rpt in all_reports.items()})
    matrix.fillna(0, inplace=True)
    matrix['max_psi'] = matrix.max(axis=1)
    matrix = matrix.sort_values('max_psi', ascending=False).drop(columns='max_psi')
    return matrix.head(top_n)


# ── HELPERS OPERACIONALES ─────────────────────────────────────────────────────

def build_ops_timeline(all_results: dict) -> pd.DataFrame:
    """
    Extrae métricas operacionales de los resultados de monitoreo.

    all_results: {'raw': [...], 'pre': [...], 'score': [...], 'grupo': [...]}
    Cada elemento debe tener una clave 'ops' con duration_s, ram_delta_mb, etc.

    Retorna DataFrame con una fila por (stage, period).
    """
    rows = []
    for stage_label, results in all_results.items():
        for r in results:
            ops = r.get('ops')
            if ops:
                rows.append({
                    'stage':        stage_label,
                    'period':       r['period'],
                    'duration_s':   ops.get('duration_s',   0.0),
                    'ram_proc_mb':  ops.get('ram_proc_mb',  0.0),
                    'ram_delta_mb': ops.get('ram_delta_mb', 0.0),
                    'cpu_pct':      ops.get('cpu_pct',      0.0),
                    'ram_sys_pct':  ops.get('ram_sys_pct',  0.0),
                    'ram_sys_mb':   ops.get('ram_sys_mb',   0.0),
                    'psi_medio':    r.get('psi_medio'),
                })
    cols = ['stage', 'period', 'duration_s', 'ram_proc_mb', 'ram_delta_mb',
            'cpu_pct', 'ram_sys_pct', 'ram_sys_mb', 'psi_medio']
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def ops_alerts(ops_df: pd.DataFrame,
               thresh_duration_s: float = 30.0,
               thresh_ram_delta_mb: float = 200.0,
               thresh_ram_sys_pct: float = 80.0) -> list:
    """
    Detecta condiciones de alerta operacional.

    Retorna lista de strings con mensajes de alerta.
    Útil para diagnosticar OOM, steps trabados o workers sobrecargados.
    """
    alerts = []
    if ops_df.empty:
        return alerts

    slow = ops_df[ops_df['duration_s'] > thresh_duration_s]
    for _, row in slow.iterrows():
        alerts.append(
            f'LENTO  [{row["stage"]} p{row["period"]}]: '
            f'{row["duration_s"]:.1f}s > umbral {thresh_duration_s}s'
        )

    high_ram = ops_df[ops_df['ram_delta_mb'] > thresh_ram_delta_mb]
    for _, row in high_ram.iterrows():
        alerts.append(
            f'RAM+   [{row["stage"]} p{row["period"]}]: '
            f'+{row["ram_delta_mb"]:.0f}MB > umbral {thresh_ram_delta_mb}MB'
        )

    high_sys = ops_df[ops_df['ram_sys_pct'] > thresh_ram_sys_pct]
    for _, row in high_sys.iterrows():
        alerts.append(
            f'SYSRAM [{row["stage"]} p{row["period"]}]: '
            f'Sistema al {row["ram_sys_pct"]:.0f}% RAM > umbral {thresh_ram_sys_pct}%'
        )

    return alerts


# ── DASHBOARD PRINCIPAL ───────────────────────────────────────────────────────

def generate_dashboard(payload, raw_results, pre_results, score_results, grupo_results):
    """
    Genera el tablero de monitoreo E2E de 8 paneles y lo guarda como PNG.

    Paneles de drift (A-F):
      A: Heatmap PSI features crudas
      B: Heatmap PSI features preprocesadas
      C: Distribución de scores
      D: PSI por etapa + media móvil (serie temporal)
      E: Heatmap PSI por grupo_ejec
      F: Semáforo E2E

    Paneles operacionales (G-H):
      G: Tiempo de ejecución por etapa y período
      H: RAM delta + uso del sistema
    """
    W     = payload['params']['moving_avg_window']
    top_n = payload['params']['top_n_vars']
    model = payload['params']['model_name']
    DIR_M = payload['DIR_MONITORING']

    cmap_psi = mcolors.LinearSegmentedColormap.from_list(
        'psi', ['#27ae60', '#f39c12', '#e74c3c'], N=256)
    norm_psi = mcolors.Normalize(vmin=0, vmax=0.5)

    # 4 filas: drift (3) + operacional (1)
    fig = plt.figure(figsize=(20, 28))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)
    periods       = payload['params']['monitoring_periods']
    period_labels = [f'p{p}' for p in periods]

    # ── Panel A — PSI datos crudos ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    mat_raw = build_psi_matrix(raw_results, top_n)
    if not mat_raw.empty:
        im = ax_a.imshow(mat_raw.values, aspect='auto', cmap=cmap_psi, norm=norm_psi)
        ax_a.set_xticks(range(len(mat_raw.columns)))
        ax_a.set_xticklabels(mat_raw.columns, fontsize=9)
        ax_a.set_yticks(range(len(mat_raw.index)))
        ax_a.set_yticklabels(mat_raw.index, fontsize=7)
        for i in range(mat_raw.shape[0]):
            for j in range(mat_raw.shape[1]):
                v = mat_raw.iloc[i, j]
                ax_a.text(j, i, f'{v:.2f}', ha='center', va='center',
                          fontsize=7, color='white' if v > 0.2 else 'black')
        plt.colorbar(im, ax=ax_a, shrink=0.8)
    ax_a.set_title('A. PSI — Datos Crudos', fontweight='bold', fontsize=11)
    ax_a.set_xlabel('Período')

    # ── Panel B — PSI datos preprocesados ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    mat_pre = build_psi_matrix(pre_results, top_n)
    if not mat_pre.empty:
        im2 = ax_b.imshow(mat_pre.values, aspect='auto', cmap=cmap_psi, norm=norm_psi)
        ax_b.set_xticks(range(len(mat_pre.columns)))
        ax_b.set_xticklabels(mat_pre.columns, fontsize=9)
        ax_b.set_yticks(range(len(mat_pre.index)))
        ax_b.set_yticklabels(mat_pre.index, fontsize=7)
        for i in range(mat_pre.shape[0]):
            for j in range(mat_pre.shape[1]):
                v = mat_pre.iloc[i, j]
                ax_b.text(j, i, f'{v:.2f}', ha='center', va='center',
                          fontsize=7, color='white' if v > 0.2 else 'black')
        plt.colorbar(im2, ax=ax_b, shrink=0.8)
    ax_b.set_title('B. PSI — Datos Preprocesados', fontweight='bold', fontsize=11)
    ax_b.set_xlabel('Período')

    # ── Panel C — Distribución de scores ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    COLORES = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    for idx, r in enumerate(score_results):
        if 'scores' in r and r['scores'] is not None:
            ax_c.hist(r['scores'], bins=40, alpha=0.55, density=True,
                      label=f'p{r["period"]}', color=COLORES[idx % len(COLORES)], edgecolor='none')
    ax_c.set_title('C. Distribución de Scores', fontweight='bold', fontsize=11)
    ax_c.set_xlabel('Probabilidad predicha')
    ax_c.set_ylabel('Densidad')
    ax_c.legend(fontsize=9)

    # ── Panel D — PSI por etapa + media móvil ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, :])
    history = {
        'raw':   {r['period']: r['psi_medio'] for r in raw_results   if r['psi_medio'] is not None},
        'pre':   {r['period']: r['psi_medio'] for r in pre_results   if r['psi_medio'] is not None},
        'score': {r['period']: r['psi_medio'] for r in score_results if r['psi_medio'] is not None},
        'grupo': {r['period']: r['psi_medio'] for r in grupo_results if r['psi_medio'] is not None},
    }
    x = np.arange(len(periods))
    width = 0.18
    stage_cfg = [('raw', '#3498db', 'Raw'), ('pre', '#2ecc71', 'Preprocesado'),
                 ('score', '#f39c12', 'Score puro'), ('grupo', '#e74c3c', 'Score x Grupo')]
    for i, (stage, color, label) in enumerate(stage_cfg):
        vals   = [history[stage].get(p, 0) for p in periods]
        offset = (i - 1.5) * width
        ax_d.bar(x + offset, vals, width=width, color=color, alpha=0.85, label=label, edgecolor='white')

    rolling_psi = [compute_rolling_psi(history['score'], p, W) for p in periods]
    ax_d.plot(x, rolling_psi, 'k--o', linewidth=2, markersize=7,
              label=f'Media móvil Score (X={W}m)', zorder=5)
    ax_d.axhline(0.10, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.8)
    ax_d.axhline(0.25, color='#e74c3c', linestyle=':', linewidth=2,   alpha=0.8)
    ax_d.text(len(periods) - 0.5, 0.11, 'WARN 0.10', fontsize=9, color='#f39c12')
    ax_d.text(len(periods) - 0.5, 0.26, 'ALARM 0.25', fontsize=9, color='#e74c3c')
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(period_labels, fontsize=10)
    ax_d.set_ylabel('PSI medio')
    ax_d.set_title(
        f'D. PSI por etapa del pipeline — Barras: vs. ref. fija | Línea: media móvil últimos {W} meses',
        fontweight='bold', fontsize=11)
    ax_d.legend(fontsize=9, loc='upper left')

    # ── Panel E — Heatmap PSI por grupo_ejec ──────────────────────────────────
    ax_e = fig.add_subplot(gs[2, :2])
    grupo_data = {}
    for r in grupo_results:
        if len(r['report']) > 0:
            for _, row in r['report'].iterrows():
                if row['columna'] == 'score':
                    grupo_data[(int(row['grupo_ejec']), r['period'])] = row['psi']
    if grupo_data:
        all_grupos  = sorted(set(k[0] for k in grupo_data))
        all_periods = sorted(set(k[1] for k in grupo_data))
        mat_g = np.array([[grupo_data.get((g, p), 0) for p in all_periods] for g in all_grupos])
        im3 = ax_e.imshow(mat_g, aspect='auto', cmap=cmap_psi, norm=norm_psi)
        ax_e.set_xticks(range(len(all_periods)))
        ax_e.set_xticklabels([f'p{p}' for p in all_periods], fontsize=9)
        ax_e.set_yticks(range(len(all_grupos)))
        ax_e.set_yticklabels([f'Grupo {g}' for g in all_grupos], fontsize=9)
        for i in range(mat_g.shape[0]):
            for j in range(mat_g.shape[1]):
                v = mat_g[i, j]
                ax_e.text(j, i, f'{v:.3f}', ha='center', va='center',
                          fontsize=8, color='white' if v > 0.2 else 'black', fontweight='bold')
        plt.colorbar(im3, ax=ax_e, shrink=0.8)
    ax_e.set_title('E. PSI de Score por Grupo de Ejecución x Período', fontweight='bold', fontsize=11)
    ax_e.set_xlabel('Período')

    # ── Panel F — Semáforo E2E ────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 2])
    ax_f.axis('off')
    filas = [['Período', 'Raw', 'Pre', 'Score', 'Grupos']]
    for p in periods:
        fila = [f'p{p}']
        for stage in ['raw', 'pre', 'score', 'grupo']:
            psi_v = history[stage].get(p)
            fila.append('—' if psi_v is None else f'{estado_psi(psi_v)[0]} {psi_v:.3f}')
        filas.append(fila)
    tbl = ax_f.table(cellText=filas[1:], colLabels=filas[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(range(5))
    ax_f.set_title('F. Semáforo E2E\nOK <0.10 | WARN 0.10-0.25 | ALARM >0.25',
                   fontweight='bold', fontsize=11)

    # ── Paneles operacionales G + H ───────────────────────────────────────────
    ops_df = build_ops_timeline({
        'raw': raw_results, 'pre': pre_results,
        'score': score_results, 'grupo': grupo_results
    })

    # Panel G — Tiempos de ejecución
    ax_g = fig.add_subplot(gs[3, :2])
    if not ops_df.empty:
        stages_ops  = list(ops_df['stage'].unique())
        periods_ops = sorted(ops_df['period'].unique())
        x_g  = np.arange(len(periods_ops))
        w_g  = 0.8 / max(len(stages_ops), 1)
        op_colors = {'raw': '#3498db', 'preprocessed': '#2ecc71',
                     'score': '#f39c12', 'grupo_ejec': '#e74c3c'}
        for i, stage in enumerate(stages_ops):
            sub  = ops_df[ops_df['stage'] == stage]
            vals = []
            for p in periods_ops:
                row = sub[sub['period'] == p]
                vals.append(float(row['duration_s'].values[0]) if len(row) > 0 else 0.0)
            ax_g.bar(x_g + i * w_g, vals, width=w_g,
                     label=stage, color=op_colors.get(stage, '#7f8c8d'), alpha=0.85)

        ax_g2 = ax_g.twinx()
        sys_ram = [float(ops_df[ops_df['period'] == p]['ram_sys_pct'].mean())
                   for p in periods_ops]
        ax_g2.plot(x_g + w_g * len(stages_ops) / 2, sys_ram, 'k--s',
                   linewidth=1.5, markersize=7, label='RAM sistema %')
        ax_g2.set_ylabel('RAM sistema (%)', fontsize=9)
        ax_g2.set_ylim(0, 110)
        ax_g2.axhline(80, color='#e74c3c', linestyle=':', linewidth=1, alpha=0.6)

        ax_g.set_xticks(x_g + w_g * len(stages_ops) / 2)
        ax_g.set_xticklabels([f'p{p}' for p in periods_ops], fontsize=10)
        ax_g.set_ylabel('Segundos', fontsize=10)
        ax_g.set_title(
            'G. Tiempo de ejecución de monitoreo por etapa\n'
            '(línea punteada: % RAM del sistema — umbral rojo: 80%)',
            fontweight='bold', fontsize=11)
        lines1, labels1 = ax_g.get_legend_handles_labels()
        lines2, labels2 = ax_g2.get_legend_handles_labels()
        ax_g.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    else:
        ax_g.text(0.5, 0.5, 'Sin datos operacionales', ha='center', va='center',
                  transform=ax_g.transAxes, fontsize=12, color='gray')
        ax_g.set_title('G. Tiempo de ejecución', fontweight='bold', fontsize=11)

    # Panel H — RAM delta + alertas
    ax_h = fig.add_subplot(gs[3, 2])
    if not ops_df.empty:
        pivot_ram = ops_df.pivot_table(
            index='stage', columns='period', values='ram_delta_mb', aggfunc='mean')
        cmap_ram = mcolors.LinearSegmentedColormap.from_list(
            'ram', ['#27ae60', '#f39c12', '#e74c3c'], N=256)
        norm_ram = mcolors.Normalize(
            vmin=pivot_ram.values.min(), vmax=max(pivot_ram.values.max(), 1))
        im_h = ax_h.imshow(pivot_ram.values, aspect='auto', cmap=cmap_ram, norm=norm_ram)
        ax_h.set_xticks(range(len(pivot_ram.columns)))
        ax_h.set_xticklabels([f'p{c}' for c in pivot_ram.columns], fontsize=9)
        ax_h.set_yticks(range(len(pivot_ram.index)))
        ax_h.set_yticklabels(pivot_ram.index, fontsize=9)
        for i in range(pivot_ram.shape[0]):
            for j in range(pivot_ram.shape[1]):
                v = pivot_ram.iloc[i, j]
                ax_h.text(j, i, f'{v:+.0f}MB', ha='center', va='center',
                          fontsize=9, fontweight='bold',
                          color='white' if abs(v) > (norm_ram.vmax * 0.5) else 'black')
        plt.colorbar(im_h, ax=ax_h, shrink=0.8)

        # Mini-resumen de alertas debajo del heatmap
        alerts = ops_alerts(ops_df)
        if alerts:
            alert_txt = '\n'.join(alerts[:4])
        else:
            alert_txt = 'Sin alertas operacionales'
        ax_h.set_xlabel(alert_txt, fontsize=7, color='#c0392b' if alerts else '#27ae60',
                        labelpad=8)
    else:
        ax_h.text(0.5, 0.5, 'Sin datos operacionales', ha='center', va='center',
                  transform=ax_h.transAxes, fontsize=11, color='gray')
    ax_h.set_title('H. Delta RAM por etapa (MB)\nrojo = presión de memoria',
                   fontweight='bold', fontsize=11)

    # ── Título general ────────────────────────────────────────────────────────
    fig.suptitle(
        f'TABLERO DE MONITOREO E2E — Modelo: {model.upper()}\n'
        f'Media móvil: últimos {W} meses | Ref. fija: p{payload["params"]["training_periods"]}',
        fontsize=14, fontweight='bold', y=0.995
    )

    output_path = f'{DIR_M}/dashboard_monitoreo_{model}.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f'Dashboard guardado en: {output_path}')
    return output_path
