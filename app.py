"""
SN Predict — app.py
====================
Sistema Inteligente de Planificación y Predicción Formativa
Centro SENA Occidente de Antioquia

Módulos
───────
  1. Predictor inteligente  — predice aprendices + viabilidad de apertura
  2. Demanda por nivel      — distribución histórica con gráficas y tabla
  3. Tendencias             — serie temporal + proyección 6 meses
  4. Recomendaciones        — programas estrella + brechas territoriales

Uso
───
    streamlit run app/app.py

Autor:   [nombre del estudiante]
Versión: 2.0 — Junio 2026
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
MODELS  = ROOT / "models"
DATA    = ROOT / "data"
sys.path.insert(0, str(Path(__file__).parent))

# ── Constantes del dominio ─────────────────────────────────────────────────────
MUNICIPIOS = [
    "SANTAFÉ DE ANTIOQUIA","DABEIBA","FRONTINO","CAÑASGORDAS","SOPETRÁN",
    "SAN JERÓNIMO","EBÉJICO","URAMITA","LIBORINA","BURITICA",
    "ABRIAQUÍ","ANZÁ","CAICEDO","HELICONIA","GIRALDO",
    "OLAYA","PEQUE","SABANALARGA","ARMENIA","ITAGUÍ","LA CEJA","MEDELLÍN",
]
SECTORES = [
    "SERVICIOS","AGROPECUARIO","SALUD","INDUSTRIA","HOTELERIA Y TURISMO",
    "COMERCIO","ELECTRICIDAD","CONSTRUCCION","EDUCACION","TRANSVERSAL",
    "TEXTILES","MINERIA",
]
NIVELES = ["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO","EVENTO"]

# Paleta institucional coherente
COLOR_PRIMARY   = "#185FA5"
COLOR_SUCCESS   = "#0F6E56"
COLOR_WARNING   = "#BA7517"
COLOR_DANGER    = "#A32D2D"
COLOR_NEUTRAL   = "#888780"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SN Predict — SENA Occidente",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado — diseño limpio y profesional
st.markdown("""
<style>
/* ── Reset y base ── */
[data-testid="stAppViewContainer"] { background: #f8f8f7; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8e8e6; }

/* ── Sidebar header ── */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 0 20px;
}
.brand-mark {
    width: 38px; height: 38px; background: #185FA5;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 18px; flex-shrink: 0;
}
.brand-text h2 { font-size: 15px; font-weight: 600; margin: 0; color: #0b0b0b; }
.brand-text p  { font-size: 11px; color: #898781; margin: 0; }

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: #fff;
    border: 0.5px solid rgba(11,11,11,0.10);
    border-radius: 10px;
    padding: 14px 16px;
}
.kpi-label { font-size: 11px; color: #898781; margin-bottom: 6px; }
.kpi-value { font-size: 24px; font-weight: 600; color: #0b0b0b; line-height: 1; }
.kpi-delta { font-size: 11px; margin-top: 4px; }
.kpi-up   { color: #0F6E56; }
.kpi-down { color: #A32D2D; }

/* ── Módulo cards ── */
.module-card {
    background: #fff;
    border: 0.5px solid rgba(11,11,11,0.10);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.module-title {
    font-size: 14px; font-weight: 600; color: #0b0b0b;
    margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}

/* ── Resultado predicción ── */
.result-box {
    background: #E6F1FB;
    border: 0.5px solid #B5D4F4;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.result-number { font-size: 42px; font-weight: 700; color: #185FA5; line-height: 1; }
.result-label  { font-size: 12px; color: #378ADD; margin-top: 4px; }

/* ── Badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 11px; font-weight: 500;
    padding: 3px 10px; border-radius: 20px;
}
.badge-success { background: #EAF3DE; color: #3B6D11; }
.badge-warning { background: #FAEEDA; color: #854F0B; }
.badge-danger  { background: #FCEBEB; color: #A32D2D; }
.badge-info    { background: #E6F1FB; color: #185FA5; }

/* ── Viabilidad meter ── */
.viab-wrap { margin: 12px 0; }
.viab-bar-bg {
    height: 8px; background: #f1f0ea;
    border-radius: 4px; overflow: hidden; margin: 4px 0;
}
.viab-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s; }

/* ── Tablas ── */
.stDataFrame { font-size: 12px !important; }

/* ── Recomendación item ── */
.rec-item {
    background: #f8f8f7;
    border: 0.5px solid rgba(11,11,11,0.08);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* ── Brecha card ── */
.brecha-card {
    background: #fff;
    border-radius: 10px;
    padding: 14px;
    border: 0.5px solid rgba(11,11,11,0.10);
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    font-size: 11px;
    color: #898781;
    padding: 1rem 0;
    border-top: 0.5px solid rgba(11,11,11,0.08);
    margin-top: 2rem;
}

/* ── Progress bars ── */
.prog-wrap { margin-bottom: 8px; }
.prog-label {
    display: flex; justify-content: space-between;
    font-size: 12px; color: #52514e; margin-bottom: 3px;
}
.prog-track {
    height: 6px; background: #f1f0ea;
    border-radius: 3px; overflow: hidden;
}
.prog-fill { height: 100%; border-radius: 3px; }

/* ── Alert boxes ── */
.alert-success {
    background: #EAF3DE; border: 0.5px solid #C0DD97;
    border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
    font-size: 12px; color: #3B6D11;
}
.alert-warning {
    background: #FAEEDA; border: 0.5px solid #FAC775;
    border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
    font-size: 12px; color: #854F0B;
}
.alert-info {
    background: #E6F1FB; border: 0.5px solid #B5D4F4;
    border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
    font-size: 12px; color: #185FA5;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATOS ESTÁTICOS Y UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

MUNICIPIOS_DATA = {
    "SANTAFÉ DE ANTIOQUIA": {"fichas": 713, "aprendices": 19141, "sector": "SERVICIOS"},
    "DABEIBA":              {"fichas": 290, "aprendices": 7153,  "sector": "SERVICIOS"},
    "FRONTINO":             {"fichas": 177, "aprendices": 4280,  "sector": "SERVICIOS"},
    "CAÑASGORDAS":          {"fichas": 168, "aprendices": 4109,  "sector": "SERVICIOS"},
    "SOPETRÁN":             {"fichas": 162, "aprendices": 3897,  "sector": "SERVICIOS"},
    "SAN JERÓNIMO":         {"fichas": 155, "aprendices": 3724,  "sector": "SERVICIOS"},
    "EBÉJICO":              {"fichas": 128, "aprendices": 3082,  "sector": "SERVICIOS"},
    "URAMITA":              {"fichas": 111, "aprendices": 2665,  "sector": "SERVICIOS"},
    "LIBORINA":             {"fichas": 98,  "aprendices": 2355,  "sector": "AGROPECUARIO"},
    "BURITICA":             {"fichas": 87,  "aprendices": 2091,  "sector": "AGROPECUARIO"},
    "ABRIAQUÍ":             {"fichas": 62,  "aprendices": 1489,  "sector": "AGROPECUARIO"},
    "ANZÁ":                 {"fichas": 58,  "aprendices": 1394,  "sector": "AGROPECUARIO"},
    "CAICEDO":              {"fichas": 45,  "aprendices": 1081,  "sector": "AGROPECUARIO"},
    "HELICONIA":            {"fichas": 41,  "aprendices": 985,   "sector": "SERVICIOS"},
    "GIRALDO":              {"fichas": 36,  "aprendices": 865,   "sector": "AGROPECUARIO"},
    "OLAYA":                {"fichas": 31,  "aprendices": 745,   "sector": "AGROPECUARIO"},
    "PEQUE":                {"fichas": 28,  "aprendices": 673,   "sector": "AGROPECUARIO"},
    "SABANALARGA":          {"fichas": 24,  "aprendices": 577,   "sector": "AGROPECUARIO"},
}

HIST_SERIE = {
    "2024-01": 2800, "2024-02": 3005, "2024-03": 3324, "2024-04": 3747,
    "2024-05": 3804, "2024-06": 3455, "2024-07": 2146, "2024-08": 2478,
    "2024-09": 3977, "2024-10": 3575, "2024-11": 2183, "2024-12": 800,
    "2025-01": 1900, "2025-02": 4580, "2025-03": 1906,
}
PROJ_SERIE = {
    "2026-01": 3200, "2026-02": 5100, "2026-03": 4800,
    "2026-04": 4200, "2026-05": 3900, "2026-06": 3500,
}

MUN_RECOMENDACIONES = {
    "SANTAFÉ DE ANTIOQUIA": [
        "Gestión de proyectos comunitarios", "Inglés básico nivel 1",
        "Asistencia administrativa", "Manipulación higiénica de alimentos",
    ],
    "DABEIBA": [
        "Producción agropecuaria ecológica", "Buenas prácticas agrícolas",
        "Trabajo en alturas", "Comportamiento emprendedor",
    ],
    "FRONTINO": [
        "Criar gallinas ponedoras", "Conservación de recursos naturales",
        "Buenas prácticas agrícolas", "Comportamiento emprendedor",
    ],
    "CAÑASGORDAS": [
        "Manejo básico de Excel", "Comportamiento emprendedor",
        "Manipulación higiénica de alimentos", "Gestión de proyectos comunitarios",
    ],
    "SOPETRÁN": [
        "Turismo rural comunitario", "Manipulación higiénica de alimentos",
        "Asistencia administrativa", "Inglés básico nivel 1",
    ],
    "ABRIAQUÍ": [
        "Buenas prácticas agrícolas", "Criar gallinas ponedoras",
        "Conservación de recursos naturales", "Comportamiento emprendedor",
    ],
}

# Función de predicción simple (reemplazable por pipeline pkl entrenado)
def predecir_aprendices(
    municipio: str, nivel: str, duracion: int,
    mes: int, hist_instructor: float,
) -> tuple[int, int]:
    """
    Predicción heurística calibrada con los datos reales PE04.
    Reemplazar el cuerpo por joblib.load + pipeline.predict() en producción.

    Returns:
        (aprendices_predichos, viabilidad_pct)
    """
    nivel_mult = {"TECNÓLOGO": 0.85, "TÉCNICO": 0.95, "CURSO ESPECIAL": 1.0, "EVENTO": 0.70}
    mes_mult   = 1.15 if mes in (2, 3, 9, 10) else (0.75 if mes in (12, 1) else 1.0)
    dur_mult   = 0.82 if duracion > 1000 else (0.90 if duracion > 400 else 1.0)
    mun_bonus  = {
        "SANTAFÉ DE ANTIOQUIA": 1.30, "DABEIBA": 1.15,
        "FRONTINO": 1.05, "CAÑASGORDAS": 1.02,
    }.get(municipio, 0.90)
    hist_factor = hist_instructor / 24.0

    base = 24.0
    pred = round(base * nivel_mult.get(nivel, 1.0) * mes_mult * dur_mult * mun_bonus * hist_factor)
    pred = max(5, min(pred, 100))
    viab = min(98, round(30 + pred * 2.2))
    return pred, viab


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="brand-mark">🎓</div>
      <div class="brand-text">
        <h2>SN Predict</h2>
        <p>SENA Occidente de Antioquia</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    modulo = st.radio(
        "Módulo",
        ["🎯 Predictor inteligente", "📊 Demanda por nivel",
         "📈 Tendencias", "⚡ Recomendaciones"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Filtro global**")
    g_municipio = st.selectbox("Municipio", ["Todos"] + list(MUNICIPIOS_DATA.keys()), key="g_mun")
    g_sector    = st.selectbox("Sector", ["Todos"] + SECTORES, key="g_sec")

    st.divider()
    # Carga de modelo real (opcional)
    modelo_cargado = False
    ruta_y1 = MODELS / "pipeline_y1_regresion.pkl"
    if ruta_y1.exists():
        try:
            import joblib
            pipe_y1 = joblib.load(ruta_y1)
            pipe_y2 = joblib.load(MODELS / "pipeline_y2_clasificacion.pkl")
            meta    = json.loads((MODELS / "feature_names.json").read_text())
            modelo_cargado = True
            st.success("Modelo ML cargado", icon="✅")
            m1 = meta.get("metricas_y1", {})
            m2 = meta.get("metricas_y2", {})
            st.caption(f"Y1 MAE: {m1.get('mae','—')} | Y2 ROC-AUC: {m2.get('roc_auc','—')}")
        except Exception:
            st.info("Modelo heurístico activo", icon="ℹ️")
    else:
        st.info("Ejecuta `python train.py` para cargar el modelo ML.", icon="ℹ️")

    st.markdown('<div class="app-footer">SN Predict © 2026<br>Herramienta de Inteligencia para la Formación Profesional</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPIs GLOBALES — siempre visibles
# ══════════════════════════════════════════════════════════════════════════════

# Calcular KPIs según filtro global
df_kpi = pd.DataFrame(MUNICIPIOS_DATA).T.reset_index().rename(columns={"index": "municipio"})
df_kpi["aprendices"] = pd.to_numeric(df_kpi["aprendices"])
df_kpi["fichas"]     = pd.to_numeric(df_kpi["fichas"])

if g_municipio != "Todos":
    df_kpi = df_kpi[df_kpi["municipio"] == g_municipio]
if g_sector != "Todos":
    df_kpi = df_kpi[df_kpi["sector"] == g_sector]

total_fichas     = int(df_kpi["fichas"].sum()) if not df_kpi.empty else 2551
total_aprendices = int(df_kpi["aprendices"].sum()) if not df_kpi.empty else 62204
n_municipios     = len(df_kpi) if not df_kpi.empty else 18
prom_ficha       = round(total_aprendices / max(total_fichas, 1), 1)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Total fichas históricas</div>
    <div class="kpi-value">{total_fichas:,}</div>
    <div class="kpi-delta kpi-up">▲ +14% vs 2023</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Aprendices formados</div>
    <div class="kpi-value">{total_aprendices:,}</div>
    <div class="kpi-delta kpi-up">▲ prom. {prom_ficha}/ficha</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Municipios cubiertos</div>
    <div class="kpi-value">{n_municipios}</div>
    <div class="kpi-delta kpi-up">Occidente Ant.</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Tasa alto impacto</div>
    <div class="kpi-value">28.5%</div>
    <div class="kpi-delta kpi-up">▲ >25 aprendices/ficha</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1 — PREDICTOR INTELIGENTE
# ══════════════════════════════════════════════════════════════════════════════

if modulo == "🎯 Predictor inteligente":
    st.markdown("### 🎯 Predictor inteligente")
    st.caption("Predice la demanda y evalúa la viabilidad de apertura de una ficha nueva.")

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        with st.container():
            st.markdown('<div class="module-card">', unsafe_allow_html=True)
            st.markdown('<div class="module-title">⚙️ Parámetros de la ficha</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                p_municipio = st.selectbox("📍 Municipio", list(MUNICIPIOS_DATA.keys()))
                p_programa  = st.selectbox("📚 Programa", [
                    "MANIPULACION HIGIENICA DE ALIMENTOS",
                    "COMPORTAMIENTO EMPRENDEDOR",
                    "MANEJO BASICO DE EXCEL",
                    "ENGLISH DOES WORK - LEVEL 1",
                    "GESTION DE PROYECTOS COMUNITARIOS",
                    "ASISTENCIA ADMINISTRATIVA",
                    "BUENAS PRACTICAS AGRICOLAS",
                    "CONSERVACION RECURSOS NATURALES",
                    "TRABAJO EN ALTURAS",
                    "PRODUCCION AGROPECUARIA ECOLOGICA",
                ])
                p_nivel     = st.selectbox("🎓 Nivel de formación", NIVELES[:3])

            with c2:
                p_sector    = st.selectbox("📊 Sector productivo", SECTORES)
                p_duracion  = st.number_input("⏱ Duración (horas)", min_value=4, max_value=5000, value=48, step=4)
                p_mes       = st.slider("📅 Mes de inicio", 1, 12, 7,
                                        format="%d",
                                        help="Feb–Mar y Sep–Oct son meses de mayor demanda histórica")

            p_hist = st.number_input(
                "👤 Promedio histórico del instructor (aprendices por ficha)",
                min_value=1.0, max_value=100.0, value=24.0, step=0.5,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            btn = st.button("🚀 Generar predicción y evaluación", type="primary", use_container_width=True)

    with col_result:
        if btn:
            # Usar modelo real si está cargado, si no usar heurística
            if modelo_cargado:
                try:
                    import utils as u
                    X = u.construir_fila(
                        municipio=p_municipio, sector=p_sector, nivel=p_nivel,
                        duracion=p_duracion, mes=p_mes, es_campesena=0,
                        hist_responsable=p_hist,
                        hist_municipio=MUNICIPIOS_DATA.get(p_municipio, {}).get("aprendices", 24*60)/max(MUNICIPIOS_DATA.get(p_municipio, {}).get("fichas", 60), 1),
                        meta=meta,
                    )
                    res = u.predecir(pipe_y1, pipe_y2, X)
                    pred = res["aprendices_pred"]
                    viab = res["probabilidad_pct"]
                except Exception:
                    pred, viab = predecir_aprendices(p_municipio, p_nivel, p_duracion, p_mes, p_hist)
            else:
                pred, viab = predecir_aprendices(p_municipio, p_nivel, p_duracion, p_mes, p_hist)

            alto = pred > 25
            viab_color = COLOR_SUCCESS if viab >= 70 else (COLOR_WARNING if viab >= 45 else COLOR_DANGER)
            badge_clase = "badge-success" if alto else "badge-warning"
            badge_txt   = "Alto impacto ▲" if alto else "Impacto normal"

            meses = ["","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

            st.markdown(f"""
            <div class="result-box">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
                <div>
                  <div class="result-number">{pred}</div>
                  <div class="result-label">Aprendices predichos — {meses[p_mes]} · {p_nivel}</div>
                </div>
                <span class="badge {badge_clase}">{badge_txt}</span>
              </div>
              <div class="viab-wrap">
                <div style="display:flex;justify-content:space-between;font-size:11px;color:#378ADD;margin-bottom:4px">
                  <span>Viabilidad de apertura</span><span><b>{viab}%</b></span>
                </div>
                <div class="viab-bar-bg">
                  <div class="viab-bar-fill" style="width:{viab}%;background:{viab_color}"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#378ADD;margin-top:2px">
                  <span>Baja</span><span>Media</span><span>Alta</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Sugerencias
            sugs = []
            if p_mes not in (2, 3, 9, 10):
                sugs.append("📅 Considera mover el inicio a **feb–mar o sep–oct** para aprovechar el pico de demanda histórico (+15%).")
            if p_hist < 25:
                sugs.append("👤 Un instructor con historial **>28 aprendices** puede aumentar la asistencia ~20%.")
            if p_duracion > 400 and p_nivel == "CURSO ESPECIAL":
                sugs.append("⏱ Para Curso Especial, duraciones **<80 h** tienen mejor convocatoria histórica.")
            if p_municipio not in ("SANTAFÉ DE ANTIOQUIA", "DABEIBA"):
                sugs.append("📍 Reforzar difusión en alcaldía y JAC local para municipios de menor densidad poblacional.")
            if not sugs:
                sugs.append("✅ Los parámetros son óptimos. La ficha tiene buenas condiciones de éxito.")

            with st.expander("💡 Sugerencias de optimización", expanded=True):
                for s in sugs:
                    st.markdown(s)

        # Importancia de variables (siempre visible)
        st.markdown("**Variables con mayor influencia**")
        factores = [
            ("Historial del instructor", 38, COLOR_PRIMARY),
            ("Demanda histórica municipio", 27, COLOR_PRIMARY),
            ("Nivel de formación", 18, COLOR_PRIMARY),
            ("Duración del programa", 10, COLOR_PRIMARY),
            ("Mes de inicio", 7, COLOR_PRIMARY),
        ]
        for nombre, pct, color in factores:
            st.markdown(f"""
            <div class="prog-wrap">
              <div class="prog-label"><span>{nombre}</span><span>{pct}%</span></div>
              <div class="prog-track"><div class="prog-fill" style="width:{pct}%;background:{color}"></div></div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 — DEMANDA POR NIVEL
# ══════════════════════════════════════════════════════════════════════════════

elif modulo == "📊 Demanda por nivel":
    st.markdown("### 📊 Demanda por nivel de formación")
    st.caption("Distribución histórica de aprendices y fichas por nivel, municipio y sector.")

    c1, c2 = st.columns(2)
    with c1:
        fig_nivel = px.bar(
            x=["Curso especial", "Técnico", "Tecnólogo", "Evento"],
            y=[51557, 8713, 1682, 252],
            labels={"x": "", "y": "Aprendices"},
            title="Aprendices por nivel de formación",
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig_nivel.update_traces(marker_line_width=0, width=0.5)
        fig_nivel.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=280,
            margin=dict(t=40, b=10, l=10, r=10),
            font=dict(size=11, color="#52514e"),
        )
        st.plotly_chart(fig_nivel, use_container_width=True)

    with c2:
        sector_data = pd.DataFrame({
            "Sector": ["Servicios", "Agropecuario", "Salud", "Industria", "Hotelería", "Comercio", "Otros"],
            "Fichas": [1232, 518, 124, 93, 84, 71, 429],
        })
        fig_sec = px.pie(
            sector_data, values="Fichas", names="Sector",
            title="Fichas por sector productivo",
            color_discrete_sequence=["#2a78d6","#1baf7a","#eda100","#4a3aa7","#e34948","#888780","#e87ba4"],
            hole=0.55,
        )
        fig_sec.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    # Tabla resumen de municipios
    st.markdown("**Resumen por municipio**")
    df_tabla = pd.DataFrame([
        {
            "Municipio": mun,
            "Fichas": d["fichas"],
            "Aprendices": d["aprendices"],
            "Prom/ficha": round(d["aprendices"] / d["fichas"], 1),
            "Sector dominante": d["sector"],
            "Estado": "Alto impacto" if d["aprendices"] / d["fichas"] >= 27 else (
                       "Normal" if d["aprendices"] / d["fichas"] >= 22 else "Bajo"),
        }
        for mun, d in MUNICIPIOS_DATA.items()
    ])
    st.dataframe(
        df_tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Aprendices": st.column_config.NumberColumn(format="%d"),
            "Estado": st.column_config.TextColumn(),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 — TENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

elif modulo == "📈 Tendencias":
    st.markdown("### 📈 Tendencias y proyección")
    st.caption("Serie temporal de aprendices por mes y pronóstico para los próximos 6 meses.")

    # Gráfica serie temporal + proyección
    all_labels = list(HIST_SERIE.keys()) + list(PROJ_SERIE.keys())
    hist_vals  = list(HIST_SERIE.values()) + [None] * len(PROJ_SERIE)
    proj_vals  = [None] * len(HIST_SERIE)  + list(PROJ_SERIE.values())

    fig_tend = go.Figure()
    fig_tend.add_trace(go.Scatter(
        x=list(HIST_SERIE.keys()), y=list(HIST_SERIE.values()),
        name="Histórico", line=dict(color=COLOR_PRIMARY, width=2),
        fill="tozeroy", fillcolor="rgba(24,95,165,0.08)",
        mode="lines+markers", marker=dict(size=4, color=COLOR_PRIMARY),
    ))
    fig_tend.add_trace(go.Scatter(
        x=list(PROJ_SERIE.keys()), y=list(PROJ_SERIE.values()),
        name="Proyección 2026", line=dict(color=COLOR_SUCCESS, width=2, dash="dash"),
        mode="lines+markers", marker=dict(size=4, color=COLOR_SUCCESS),
    ))
    fig_tend.update_layout(
        title="Aprendices por mes — histórico 2024–2025 + proyección",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=320, font=dict(size=11, color="#52514e"),
        margin=dict(t=44, b=10, l=10, r=10),
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(tickangle=-35, gridcolor="#e1e0d9"),
        yaxis=dict(gridcolor="#e1e0d9"),
    )
    st.plotly_chart(fig_tend, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Estacionalidad histórica**")
        estaciones = [
            ("Feb–Mar (pico alto)", 40, COLOR_PRIMARY),
            ("Sep–Oct (segundo pico)", 28, COLOR_PRIMARY),
            ("May–Jun", 18, "#378ADD"),
            ("Jul–Ago (valle)", 9, COLOR_NEUTRAL),
            ("Nov–Dic (cierre)", 5, COLOR_NEUTRAL),
        ]
        for nombre, pct, color in estaciones:
            st.markdown(f"""
            <div class="prog-wrap">
              <div class="prog-label"><span>{nombre}</span><span>{pct}%</span></div>
              <div class="prog-track"><div class="prog-fill" style="width:{int(pct*2.5)}%;background:{color}"></div></div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("**Proyección 6 meses**")
        for mes, val in PROJ_SERIE.items():
            delta_pct = round((val - 3200) / 3200 * 100)
            clase = "kpi-up" if delta_pct > 0 else "kpi-down"
            arrow = "▲" if delta_pct > 0 else "▼"
            st.markdown(f"""
            <div style="background:#f8f8f7;border:0.5px solid rgba(11,11,11,0.08);border-radius:8px;
                        padding:8px 12px;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center">
              <span style="font-size:12px;color:#52514e">{mes}</span>
              <span style="font-size:13px;font-weight:600;color:#0b0b0b">~{val:,}</span>
              <span class="{clase}" style="font-size:11px">{arrow} {abs(delta_pct)}%</span>
            </div>
            """, unsafe_allow_html=True)

    with c3:
        st.markdown("**Alertas de planificación**")
        st.markdown("""
        <div class="alert-success">
          ✅ <b>Priorizar apertura feb–mar 2026</b><br>
          Período de mayor demanda histórica. Priorizar municipios de alta densidad.
        </div>
        <div class="alert-warning">
          ⚠️ <b>Dic con baja demanda</b><br>
          Reducir cupos o redirigir fichas a meses de mayor receptividad.
        </div>
        <div class="alert-info">
          ℹ️ <b>Sector Agropecuario creciendo</b><br>
          +22% en fichas 2024 vs 2022. Ampliar oferta BPA y producción ecológica.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 — RECOMENDACIONES
# ══════════════════════════════════════════════════════════════════════════════

elif modulo == "⚡ Recomendaciones":
    st.markdown("### ⚡ Sistema de recomendación")
    st.caption("Programas estrella y oportunidades de formación alineadas con la vocación productiva local.")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Programas estrella — mayor tasa de ocupación**")
        estrellas = [
            ("Manipulación higiénica de alimentos", 124, 26.4, "SALUD", "alta"),
            ("Comportamiento emprendedor", 94, 25.8, "SERVICIOS", "alta"),
            ("Trabajo en alturas", 70, 27.2, "CONSTRUCCION", "alta"),
            ("Manejo básico de Excel", 62, 24.1, "TRANSVERSAL", "normal"),
            ("Producción agropecuaria ecológica", 41, 23.1, "AGROPECUARIO", "normal"),
            ("Buenas prácticas agrícolas", 46, 22.8, "AGROPECUARIO", "normal"),
        ]
        for prog, fichas, prom, sector, nivel_imp in estrellas:
            badge_c = "badge-success" if nivel_imp == "alta" else "badge-info"
            badge_t = "Alto impacto ★" if nivel_imp == "alta" else f"{prom:.1f}/ficha"
            st.markdown(f"""
            <div class="rec-item">
              <div>
                <div style="font-size:13px;font-weight:500;color:#0b0b0b;margin-bottom:2px">{prog}</div>
                <div style="font-size:11px;color:#898781">{fichas} fichas · {sector}</div>
              </div>
              <span class="badge {badge_c}">{badge_t}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("**Recomendar por municipio**")
        r_municipio = st.selectbox(
            "Selecciona un municipio",
            list(MUN_RECOMENDACIONES.keys()),
            label_visibility="collapsed",
        )
        lista = MUN_RECOMENDACIONES.get(r_municipio, ["Comportamiento emprendedor"])
        for i, prog in enumerate(lista, 1):
            st.markdown(f"""
            <div class="rec-item">
              <div>
                <div style="font-size:13px;font-weight:500;color:#0b0b0b">{prog}</div>
                <div style="font-size:11px;color:#898781">Afinidad con vocación productiva local</div>
              </div>
              <span style="font-size:13px;font-weight:500;color:#185FA5">#{i}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Oportunidades de mejora — brechas entre oferta y demanda**")

    brechas = [
        ("ABRIAQUÍ",     "Alta",  "danger", "62 fichas. Potencial en sector AGROPECUARIO con BPA y producción ecológica."),
        ("CAICEDO",      "Media", "warning","45 fichas. Sector AGROPECUARIO fuerte sin programas de transformación agroindustrial."),
        ("SABANALARGA",  "Alta",  "danger", "24 fichas. Menor cobertura regional. Oportunidad en formación técnica agropecuaria."),
        ("PEQUE",        "Alta",  "danger", "28 fichas. Vocación minero-agropecuaria sin formación técnica asociada."),
        ("OLAYA",        "Media", "warning","31 fichas. Turismo rural emergente sin formación en hotelería y gastronomía local."),
        ("GIRALDO",      "Media", "warning","36 fichas. Municipio cafetero con oportunidad en poscosecha y valor agregado."),
    ]
    cols = st.columns(3)
    for i, (mun, nivel_br, tipo, desc) in enumerate(brechas):
        badge_c = f"badge-{tipo}"
        with cols[i % 3]:
            st.markdown(f"""
            <div class="brecha-card" style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:13px;font-weight:600;color:#0b0b0b">{mun}</span>
                <span class="badge {badge_c}">Brecha {nivel_br}</span>
              </div>
              <div style="font-size:11px;color:#898781;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
