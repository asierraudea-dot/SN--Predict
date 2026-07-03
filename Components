"""
components.py  —  SN Predict v2
=================================
Componentes de UI reutilizables para la app Streamlit.
Centraliza HTML/CSS custom para mantener app.py limpio.

Autor:  [nombre del estudiante]
Fecha:  Junio 2026
"""
from __future__ import annotations

import streamlit as st

# ── Paleta institucional ───────────────────────────────────────────────────────
C_BLUE   = "#185FA5"
C_GREEN  = "#0F6E56"
C_AMBER  = "#BA7517"
C_RED    = "#A32D2D"
C_GRAY   = "#888780"
C_BG     = "#f8f8f7"
C_BORDER = "rgba(11,11,11,0.09)"


def inject_css() -> None:
    """Inyecta el CSS global de la aplicación."""
    st.markdown(f"""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {{ background:{C_BG}; }}
[data-testid="stSidebar"] {{
    background:#ffffff;
    border-right:0.5px solid {C_BORDER};
}}
[data-testid="stSidebar"] .stMarkdown p {{
    font-size:12px; color:#52514e;
}}

/* ── KPI row ── */
.kpi-row {{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:10px; margin-bottom:1.25rem;
}}
.kpi-card {{
    background:#fff;
    border:0.5px solid {C_BORDER};
    border-radius:10px; padding:12px 16px;
}}
.kpi-label {{ font-size:11px; color:{C_GRAY}; margin-bottom:5px; }}
.kpi-value {{ font-size:22px; font-weight:600; color:#0b0b0b; line-height:1; }}
.kpi-delta {{ font-size:10px; margin-top:4px; color:{C_GREEN}; }}

/* ── Cards ── */
.mod-card {{
    background:#fff;
    border:0.5px solid {C_BORDER};
    border-radius:12px;
    padding:1.1rem 1.2rem;
    margin-bottom:.9rem;
}}
.mod-title {{
    font-size:13px; font-weight:600; color:#0b0b0b;
    display:flex; align-items:center; gap:7px; margin-bottom:12px;
}}

/* ── Resultado predicción ── */
.result-box {{
    background:#E6F1FB;
    border:0.5px solid #B5D4F4;
    border-radius:12px; padding:1.1rem;
}}
.res-number {{ font-size:40px; font-weight:600; color:{C_BLUE}; line-height:1; }}
.res-label  {{ font-size:11px; color:#378ADD; margin-top:3px; }}

/* ── Viabilidad barra ── */
.viab-track {{
    height:8px; background:#f1f0ea;
    border-radius:4px; overflow:hidden; margin:5px 0;
}}
.viab-fill {{ height:100%; border-radius:4px; transition:width .4s; }}

/* ── Progress bars ── */
.pb-wrap {{ margin-bottom:6px; }}
.pb-meta {{
    display:flex; justify-content:space-between;
    font-size:11px; margin-bottom:2px;
}}
.pb-name {{ color:#52514e; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:74%; }}
.pb-val  {{ color:#0b0b0b; font-weight:600; }}
.pb-track {{ height:5px; background:#f1f0ea; border-radius:2px; overflow:hidden; }}
.pb-fill  {{ height:100%; border-radius:2px; }}

/* ── Badges ── */
.badge {{
    display:inline-flex; align-items:center; gap:3px;
    font-size:10px; font-weight:600;
    padding:2px 8px; border-radius:20px;
}}
.badge-s {{ background:#EAF3DE; color:{C_GREEN}; }}
.badge-w {{ background:#FAEEDA; color:{C_AMBER}; }}
.badge-d {{ background:#FCEBEB; color:{C_RED};   }}
.badge-i {{ background:#E6F1FB; color:{C_BLUE};  }}
.badge-g {{ background:#f1f0ea; color:{C_GRAY};  }}

/* ── Tabla ── */
.tbl-sn {{ width:100%; border-collapse:collapse; font-size:12px; }}
.tbl-sn th {{
    text-align:left; padding:7px 9px;
    font-size:10px; font-weight:600; color:{C_GRAY};
    border-bottom:0.5px solid {C_BORDER};
    background:#fafaf9;
}}
.tbl-sn td {{
    padding:7px 9px; color:#52514e;
    border-bottom:0.5px solid {C_BORDER};
}}
.tbl-sn td:first-child {{ color:#0b0b0b; font-weight:600; }}
.tbl-sn tr:last-child td {{ border-bottom:none; }}
.tbl-sn tr:hover td {{ background:{C_BG}; }}

/* ── Alert boxes ── */
.al-s {{ background:#EAF3DE; border:.5px solid #C0DD97; border-radius:8px; padding:9px 12px; margin-bottom:7px; font-size:11px; color:{C_GREEN}; line-height:1.6; }}
.al-w {{ background:#FAEEDA; border:.5px solid #FAC775; border-radius:8px; padding:9px 12px; margin-bottom:7px; font-size:11px; color:{C_AMBER}; line-height:1.6; }}
.al-i {{ background:#E6F1FB; border:.5px solid #B5D4F4; border-radius:8px; padding:9px 12px; margin-bottom:7px; font-size:11px; color:{C_BLUE}; line-height:1.6; }}
.al-r {{ background:#FCEBEB; border:.5px solid #F7C1C1; border-radius:8px; padding:9px 12px; margin-bottom:7px; font-size:11px; color:{C_RED}; line-height:1.6; }}

/* ── Footer ── */
.app-footer {{
    text-align:center; font-size:11px; color:{C_GRAY};
    padding:.8rem 0; border-top:.5px solid {C_BORDER}; margin-top:1.5rem;
}}

/* ── Instructor bar ── */
.inst-row {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
.inst-name {{ font-size:11px; color:#52514e; width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex-shrink:0; }}
.inst-track {{ flex:1; height:6px; background:#f1f0ea; border-radius:3px; overflow:hidden; }}
.inst-fill  {{ height:100%; border-radius:3px; }}
.inst-val   {{ font-size:11px; font-weight:600; width:38px; text-align:right; flex-shrink:0; }}

/* ── Ruta formativa ── */
.ruta-step {{
    border-radius:8px; padding:10px 14px; margin-bottom:6px;
    font-size:12px; line-height:1.6;
}}
.ruta-step b {{ font-size:12px; display:block; margin-bottom:3px; }}

/* ── Rubro card ── */
.rub-card {{
    background:#fff; border:.5px solid {C_BORDER};
    border-radius:8px; padding:.9rem;
    cursor:pointer; transition:border-color .12s;
    margin-bottom:8px;
}}
.rub-card-name {{ font-size:12px; font-weight:600; margin-bottom:2px; }}
.rub-card-meta {{ font-size:10px; color:{C_GRAY}; }}
</style>
""", unsafe_allow_html=True)


def kpi_row_html(fichas: int, aprendices: int, municipios: int, tasa: float) -> str:
    """Genera el HTML de la fila de KPIs globales."""
    return f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Fichas históricas</div>
    <div class="kpi-value">{fichas:,}</div>
    <div class="kpi-delta">▲ 2020–2025</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Aprendices formados</div>
    <div class="kpi-value">{aprendices:,}</div>
    <div class="kpi-delta">▲ prom. {round(aprendices/max(fichas,1),1)}/ficha</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Municipios cubiertos</div>
    <div class="kpi-value">{municipios}</div>
    <div class="kpi-delta">Occidente Antioquia</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Tasa alto impacto</div>
    <div class="kpi-value">{tasa:.1f}%</div>
    <div class="kpi-delta">▲ >25 aprendices/ficha</div>
  </div>
</div>"""


def prog_bars_html(items: list[dict], key_label: str, key_val: str,
                   max_val: float, color: str = C_BLUE) -> str:
    """
    Genera barras de progreso horizontales para listas de programas/sectores.

    Args:
        items:     Lista de dicts.
        key_label: Clave para la etiqueta (ej. 'prog').
        key_val:   Clave para el valor numérico.
        max_val:   Valor máximo para normalizar la barra.
        color:     Color de la barra.

    Returns:
        String HTML.
    """
    html = ""
    for item in items:
        pct = round(item[key_val] / max(max_val, 1) * 100)
        html += f"""
<div class="pb-wrap">
  <div class="pb-meta">
    <span class="pb-name">{item[key_label]}</span>
    <span class="pb-val">{item[key_val]}</span>
  </div>
  <div class="pb-track">
    <div class="pb-fill" style="width:{pct}%;background:{color}"></div>
  </div>
</div>"""
    return html


def result_box_html(pred: dict) -> str:
    """
    Genera el cuadro de resultado de la predicción.

    Args:
        pred: Resultado de predecir_heuristico() o intentar_modelo_ml().

    Returns:
        String HTML.
    """
    ap   = pred["aprendices"]
    viab = pred["viabilidad_pct"]
    alto = pred["alto_impacto"]
    color = pred["color_viab"]
    badge = (
        '<span class="badge badge-s">▲ Alto impacto</span>'
        if alto else
        '<span class="badge badge-w">▼ Impacto normal</span>'
    )
    return f"""
<div class="result-box">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <div>
      <div class="res-number">{ap}</div>
      <div class="res-label">Aprendices predichos</div>
    </div>
    {badge}
  </div>
  <div style="font-size:10px;color:#378ADD;display:flex;justify-content:space-between;margin-bottom:3px">
    <span>Viabilidad de apertura</span>
    <span style="font-weight:700">{viab}%</span>
  </div>
  <div class="viab-track">
    <div class="viab-fill" style="width:{viab}%;background:{color}"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:9px;color:#378ADD;margin-top:2px">
    <span>Baja</span><span>Media</span><span>Alta</span>
  </div>
  <div style="font-size:11px;color:#52514e;margin-top:10px;line-height:1.6">
    {pred['interpretacion']}
  </div>
</div>"""


def badge_html(texto: str, tipo: str = "i") -> str:
    """Genera un badge HTML. tipo: s=success, w=warning, d=danger, i=info, g=gray."""
    return f'<span class="badge badge-{tipo}">{texto}</span>'


def tabla_html(headers: list[str], rows: list[list]) -> str:
    """
    Genera una tabla HTML con estilo consistente.

    Args:
        headers: Lista de encabezados.
        rows:    Lista de filas (cada fila es una lista de strings/HTML).

    Returns:
        String HTML.
    """
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f'<div style="overflow-x:auto"><table class="tbl-sn"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def footer_html() -> str:
    return '<div class="app-footer">SN Predict © 2026 · Centro SENA Occidente de Antioquia · v2.0</div>'
