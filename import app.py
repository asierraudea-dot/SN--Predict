"""
app.py  —  SN Predict v2
==========================
Sistema Inteligente de Planificación y Predicción Formativa
Centro SENA Occidente de Antioquia

Módulos
───────
  1. Predictor inteligente  — filtros dinámicos, mapa de calor, instructor
  2. Demanda por nivel      — filtros cruzados nivel × sector × municipio
  3. Recomendaciones        — ruta formativa complementario→técnico→tecnólogo
  4. Oportunidades          — brechas por rubro productivo
  5. Manual de usuario      — guía completa + tour interactivo

Uso
───
    streamlit run app/app.py

Autor:  [nombre del estudiante]
Versión: 2.0 — Junio 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Asegurar imports locales
sys.path.insert(0, str(Path(__file__).parent))
from components import (
    C_AMBER, C_BLUE, C_BORDER, C_GRAY, C_GREEN, C_RED,
    badge_html, footer_html, inject_css, kpi_row_html,
    prog_bars_html, result_box_html, tabla_html,
)
from data_loader import (
    MUN_COORDS, RUBROS_KW,
    cargar_df, get_mun_stats, get_niv_sec, get_prog_mun,
    get_resp_stats, get_rubros, get_sec_prog, get_serie_temporal,
)
from predictor import (
    MESES_NOM, generar_sugerencias,
    intentar_modelo_ml, predecir_heuristico,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SN Predict — SENA Occidente",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ══════════════════════════════════════════════════════════════════════════════
# DATOS (cacheados)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Cargando datos históricos PE04…")
def load_all():
    try:
        df = cargar_df()
    except FileNotFoundError as e:
        st.error(f"Error cargando datos: {e}")
        return None, {}, {}, [], {}, pd.DataFrame(), {}, pd.DataFrame()
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None, {}, {}, [], {}, pd.DataFrame(), {}, pd.DataFrame()
    return (
        df,
        get_sec_prog(df),
        get_prog_mun(df),
        get_resp_stats(df),
        get_mun_stats(df),
        get_niv_sec(df),
        get_rubros(df),
        get_serie_temporal(df),
    )

df, SEC_PROG, PROG_MUN, RESP, MUN_STATS, NIV_SEC, RUBROS, SERIE = load_all()

TIENE_DATOS = df is not None

# Fallback con datos embebidos cuando no hay XLSX
if not TIENE_DATOS:
    from data_loader import _CANDIDATOS, DATA_RAW
    rutas = chr(10).join([f"  • {p} — {'✅ EXISTE' if p.exists() else '❌ no existe'}" for p in _CANDIDATOS])
    st.warning(
        f"⚠️ Dataset no cargado. Rutas buscadas:{chr(10)}{rutas}",
        icon="⚠️",
    )
    SEC_PROG = {
        "SERVICIOS": [
            {"prog": "MANIPULACION HIGIENICA DE ALIMENTOS", "nivel": "CURSO ESPECIAL", "fichas": 123, "prom": 27.8, "dur": 12, "ap": 3419},
            {"prog": "COMPORTAMIENTO EMPRENDEDOR",          "nivel": "CURSO ESPECIAL", "fichas": 93,  "prom": 22.3, "dur": 48, "ap": 2076},
            {"prog": "HIGIENE Y MANIPULACION DE ALIMENTOS", "nivel": "CURSO ESPECIAL", "fichas": 87,  "prom": 25.2, "dur": 10, "ap": 2195},
            {"prog": "ENGLISH DOES WORK - LEVEL 1",         "nivel": "CURSO ESPECIAL", "fichas": 53,  "prom": 22.8, "dur": 48, "ap": 1210},
            {"prog": "GESTION DE PROYECTOS COMUNITARIOS",   "nivel": "CURSO ESPECIAL", "fichas": 47,  "prom": 24.6, "dur": 40, "ap": 1156},
        ],
        "AGROPECUARIO": [
            {"prog": "CONSERVACION DE RECURSOS NATURALES",    "nivel": "TÉCNICO",        "fichas": 41, "prom": 24.8, "dur": 2205, "ap": 1015},
            {"prog": "PRODUCCION AGROPECUARIA ECOLOGICA",     "nivel": "CURSO ESPECIAL", "fichas": 41, "prom": 22.6, "dur": 192,  "ap": 925},
            {"prog": "BUENAS PRACTICAS AGRICOLAS",            "nivel": "CURSO ESPECIAL", "fichas": 30, "prom": 18.6, "dur": 96,   "ap": 557},
            {"prog": "CRIAR GALLINAS PONEDORAS",              "nivel": "CURSO ESPECIAL", "fichas": 27, "prom": 21.0, "dur": 60,   "ap": 568},
        ],
        "SALUD": [
            {"prog": "JEFES DE AREA PARA TRABAJO EN ALTURAS",  "nivel": "CURSO ESPECIAL", "fichas": 39, "prom": 7.9,  "dur": 8,  "ap": 309},
            {"prog": "TRABAJADOR AUTORIZADO TRABAJO EN ALTURAS","nivel": "CURSO ESPECIAL", "fichas": 31, "prom": 7.3,  "dur": 8,  "ap": 227},
            {"prog": "PRIMEROS AUXILIOS",                       "nivel": "CURSO ESPECIAL", "fichas": 21, "prom": 25.8, "dur": 40, "ap": 542},
        ],
        "HOTELERIA Y TURISMO": [
            {"prog": "SERVICIO DE RECEPCION HOTELERA", "nivel": "TÉCNICO",        "fichas": 11, "prom": 43.4, "dur": 1540, "ap": 477},
            {"prog": "OPERACION TURISTICA LOCAL",      "nivel": "CURSO ESPECIAL", "fichas": 12, "prom": 25.2, "dur": 60,   "ap": 302},
        ],
        "TRANSVERSAL": [
            {"prog": "INFORMATICA: WORD EXCEL E INTERNET", "nivel": "CURSO ESPECIAL", "fichas": 22, "prom": 23.0, "dur": 48, "ap": 506},
            {"prog": "SISTEMAS",                           "nivel": "TÉCNICO",        "fichas": 14, "prom": 28.4, "dur": 1870,"ap": 398},
        ],
    }
    RESP = [
        {"n": "TOL GRANJA",                   "fichas": 209, "prom": 40.2, "total": 8397, "max_ap": 320},
        {"n": "ADRIANA AGUIRRE LOPEZ",         "fichas": 200, "prom": 27.8, "total": 5564, "max_ap": 100},
        {"n": "CARLOS RUBIO ESCOBAR",          "fichas": 50,  "prom": 26.5, "total": 1327, "max_ap": 37},
        {"n": "JUAN DIEGO MARIN ORTIZ",        "fichas": 49,  "prom": 26.6, "total": 1304, "max_ap": 100},
        {"n": "JESUS TRUJILLO BERMEO",         "fichas": 43,  "prom": 26.4, "total": 1135, "max_ap": 46},
        {"n": "ALEJANDRO CONTRERAS",           "fichas": 31,  "prom": 31.5, "total": 978,  "max_ap": 415},
        {"n": "BERTA PASTRANA MADERA",         "fichas": 28,  "prom": 30.7, "total": 859,  "max_ap": 419},
        {"n": "ARLES VALOYES OBREGON",         "fichas": 72,  "prom": 18.6, "total": 1339, "max_ap": 33},
        {"n": "SEBASTIAN GUARIN FONSECA",      "fichas": 55,  "prom": 7.9,  "total": 435,  "max_ap": 10},
        {"n": "MARIBEL GALEANO ROJAS",         "fichas": 48,  "prom": 7.3,  "total": 352,  "max_ap": 10},
    ]
    MUN_STATS = {
        "SANTAFÉ DE ANTIOQUIA": {"ap": 21072, "fichas": 713, "prom": 29.6, "sector": "SERVICIOS",    "tend": 12.0, "lat": 6.5566, "lng": -75.8238},
        "DABEIBA":              {"ap": 6874,  "fichas": 320, "prom": 21.5, "sector": "SERVICIOS",    "tend": 8.0,  "lat": 7.0091, "lng": -76.2581},
        "FRONTINO":             {"ap": 3642,  "fichas": 158, "prom": 23.1, "sector": "AGROPECUARIO", "tend": 5.0,  "lat": 6.7854, "lng": -76.1358},
        "CAÑASGORDAS":          {"ap": 3143,  "fichas": 142, "prom": 22.1, "sector": "SERVICIOS",    "tend": 3.0,  "lat": 6.7393, "lng": -75.9944},
        "SOPETRÁN":             {"ap": 2865,  "fichas": 127, "prom": 22.6, "sector": "SERVICIOS",    "tend": 6.0,  "lat": 6.5058, "lng": -75.7383},
        "SAN JERÓNIMO":         {"ap": 3165,  "fichas": 135, "prom": 23.4, "sector": "SERVICIOS",    "tend": 4.0,  "lat": 6.4827, "lng": -75.7173},
        "EBÉJICO":              {"ap": 2940,  "fichas": 133, "prom": 22.1, "sector": "AGROPECUARIO", "tend": 2.0,  "lat": 6.3328, "lng": -75.7449},
        "URAMITA":              {"ap": 2544,  "fichas": 112, "prom": 22.7, "sector": "AGROPECUARIO", "tend": 1.0,  "lat": 6.9108, "lng": -76.1642},
        "LIBORINA":             {"ap": 2213,  "fichas": 98,  "prom": 22.6, "sector": "AGROPECUARIO", "tend": -1.0, "lat": 6.6944, "lng": -75.9336},
        "BURITICA":             {"ap": 1612,  "fichas": 84,  "prom": 19.2, "sector": "AGROPECUARIO", "tend": 0.0,  "lat": 6.7297, "lng": -75.9045},
        "ABRIAQUÍ":             {"ap": 746,   "fichas": 36,  "prom": 20.7, "sector": "AGROPECUARIO", "tend": -2.0, "lat": 6.6384, "lng": -76.0946},
        "ANZÁ":                 {"ap": 1552,  "fichas": 72,  "prom": 21.6, "sector": "AGROPECUARIO", "tend": 3.0,  "lat": 6.3185, "lng": -75.8800},
        "CAICEDO":              {"ap": 1946,  "fichas": 80,  "prom": 24.3, "sector": "AGROPECUARIO", "tend": 5.0,  "lat": 6.4141, "lng": -76.0204},
        "HELICONIA":            {"ap": 1005,  "fichas": 50,  "prom": 20.1, "sector": "SERVICIOS",    "tend": -3.0, "lat": 6.2066, "lng": -75.7487},
        "GIRALDO":              {"ap": 1900,  "fichas": 87,  "prom": 21.8, "sector": "AGROPECUARIO", "tend": 4.0,  "lat": 6.6207, "lng": -75.8819},
        "OLAYA":                {"ap": 834,   "fichas": 35,  "prom": 23.8, "sector": "AGROPECUARIO", "tend": -1.0, "lat": 6.5920, "lng": -75.8555},
        "PEQUE":                {"ap": 973,   "fichas": 40,  "prom": 24.3, "sector": "AGROPECUARIO", "tend": 2.0,  "lat": 6.9890, "lng": -75.9994},
        "SABANALARGA":          {"ap": 1118,  "fichas": 43,  "prom": 26.0, "sector": "AGROPECUARIO", "tend": 7.0,  "lat": 6.8878, "lng": -75.7083},
        "MEDELLÍN":             {"ap": 1389,  "fichas": 57,  "prom": 24.4, "sector": "TRANSVERSAL",  "tend": 1.0,  "lat": 6.2518, "lng": -75.5636},
    }
    RUBROS = {
        "Café":       {"fichas": 63,  "ap": 1391, "muns": {"ANZÁ": 322, "EBÉJICO": 177, "CAICEDO": 173}},
        "Ganadería":  {"fichas": 129, "ap": 3021, "muns": {"DABEIBA": 466, "FRONTINO": 421, "SAN JERÓNIMO": 325}},
        "Avicultura": {"fichas": 38,  "ap": 777,  "muns": {"HELICONIA": 152, "EBÉJICO": 90, "URAMITA": 88}},
        "Agricultura":{"fichas": 181, "ap": 3631, "muns": {"DABEIBA": 592, "URAMITA": 398, "EBÉJICO": 374}},
        "Alimentos":  {"fichas": 374, "ap": 9919, "muns": {"SANTAFÉ DE ANTIOQUIA": 3702, "SOPETRÁN": 764}},
        "Emprendimiento":{"fichas":341,"ap":7742, "muns":{"DABEIBA":1495,"CAÑASGORDAS":855,"EBÉJICO":490}},
        "Inglés":     {"fichas": 195, "ap": 7355, "muns": {"SANTAFÉ DE ANTIOQUIA": 6470, "DABEIBA": 204}},
        "TIC / Excel":{"fichas": 226, "ap": 5104, "muns": {"SANTAFÉ DE ANTIOQUIA": 1453, "DABEIBA": 914}},
        "Turismo":    {"fichas": 10,  "ap": 203,  "muns": {"LIBORINA": 46, "EBÉJICO": 42}},
        "Alturas":    {"fichas": 105, "ap": 798,  "muns": {"SANTAFÉ DE ANTIOQUIA": 553, "BURITICA": 148}},
    }

SECTORES  = sorted(SEC_PROG.keys())
MUNICIPIOS = sorted(MUN_STATS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:14px;border-bottom:.5px solid rgba(11,11,11,.09)">
      <div style="width:36px;height:36px;background:#185FA5;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px">🎓</div>
      <div>
        <div style="font-size:14px;font-weight:600;color:#0b0b0b">SN Predict</div>
        <div style="font-size:11px;color:#898781">SENA Occidente · v2.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    modulo = st.radio(
        "Módulo",
        ["🎯 Predictor inteligente", "📊 Demanda por nivel",
         "⚡ Recomendaciones", "🔭 Oportunidades de mejora", "📖 Manual de usuario"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Filtro global**", help="Aplica a los KPIs del encabezado.")
    g_mun = st.selectbox("Municipio", ["Todos"] + MUNICIPIOS, key="g_mun")
    g_sec = st.selectbox("Sector",    ["Todos"] + SECTORES,   key="g_sec")
    g_niv = st.selectbox("Nivel",     ["Todos", "CURSO ESPECIAL", "TÉCNICO", "TECNÓLOGO"], key="g_niv")

    st.divider()
    if st.button("▶ Tour rápido", use_container_width=True, type="primary"):
        st.session_state["tour_step"] = 0
        st.session_state["show_tour"] = True

    # Estado del modelo ML
    from pathlib import Path as _P
    ml_ok = (_P(__file__).parent.parent / "models" / "pipeline_y1_regresion.pkl").exists()
    if ml_ok:
        st.success("Modelo ML activo", icon="✅")
    else:
        st.info("Modo heurístico. Ejecuta `python train.py` para activar ML.", icon="ℹ️")

    st.markdown(footer_html(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOUR INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════

TOUR_STEPS = [
    ("Bienvenido a SN Predict 🎓",
     "Esta herramienta analiza 2.551 fichas históricas de 22 municipios del Centro SENA "
     "Occidente de Antioquia (2020–2025) para predecir demanda, detectar oportunidades "
     "y optimizar la apertura de programas de formación."),
    ("Predictor inteligente 🎯",
     "Selecciona un **Sector** → los programas se filtran automáticamente con datos reales. "
     "Al elegir un programa, se actualiza su duración, nivel y los municipios donde se ha "
     "ofertado. Usa la búsqueda por palabra clave para encontrar programas relacionados."),
    ("Mapa de calor y instructor 🗺️",
     "El **mapa de calor** muestra la intensidad de demanda por municipio. Cambia el modo "
     "entre 'Total aprendices', 'N.º fichas' y 'Prom/ficha'. El historial de **instructores** "
     "muestra quién tiene mayor capacidad de convocatoria."),
    ("Demanda por nivel y Recomendaciones 📊⚡",
     "En 'Demanda por nivel' filtra por Curso Especial, Técnico o Tecnólogo y compara sectores. "
     "En 'Recomendaciones' selecciona un municipio y ve la **ruta formativa estratégica**: "
     "Complementarios → Técnico → Tecnólogo."),
    ("Oportunidades de mejora 🔭",
     "Selecciona un **rubro productivo** (Café, Ganadería, Turismo…) para identificar "
     "qué municipios lideran la formación en ese rubro y cuáles tienen brechas de cobertura — "
     "candidatos ideales para nuevas fichas con alto impacto territorial."),
]

if st.session_state.get("show_tour"):
    step_idx = st.session_state.get("tour_step", 0)
    step = TOUR_STEPS[step_idx]
    with st.container():
        st.info(f"**Paso {step_idx + 1} de {len(TOUR_STEPS)} — {step[0]}**\n\n{step[1]}")
        col_p, col_n, col_c = st.columns([1, 1, 2])
        with col_p:
            if step_idx > 0 and st.button("← Anterior", key="tour_prev"):
                st.session_state["tour_step"] -= 1
                st.rerun()
        with col_n:
            lbl = "Siguiente →" if step_idx < len(TOUR_STEPS) - 1 else "Cerrar tour"
            if st.button(lbl, key="tour_next", type="primary"):
                if step_idx < len(TOUR_STEPS) - 1:
                    st.session_state["tour_step"] += 1
                    st.rerun()
                else:
                    st.session_state["show_tour"] = False
                    st.rerun()
        with col_c:
            st.caption(f"{'●' * (step_idx + 1)}{'○' * (len(TOUR_STEPS) - step_idx - 1)}")
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# KPIs GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

_mun_f = g_mun if g_mun != "Todos" else None
_stats = {k: v for k, v in MUN_STATS.items() if (not _mun_f or k == _mun_f)}
_total_f  = sum(v["fichas"] for v in _stats.values()) if _stats else 2551
_total_ap = sum(v["ap"]     for v in _stats.values()) if _stats else 62472
_total_m  = len(_stats) if _stats else 22
_tasa_ai  = 28.5

st.markdown(
    kpi_row_html(_total_f, _total_ap, _total_m, _tasa_ai),
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1 — PREDICTOR INTELIGENTE
# ══════════════════════════════════════════════════════════════════════════════

if modulo == "🎯 Predictor inteligente":
    st.markdown("### 🎯 Predictor inteligente")
    st.caption(
        "Selecciona sector → programa → municipio → instructor y genera la predicción "
        "de demanda con datos históricos reales."
    )

    # Búsqueda por palabra clave (sobre todos los programas disponibles)
    kw = st.text_input(
        "🔍 Buscar programa por palabra clave",
        placeholder="Ej: café, ganadería, emprendimiento, alturas…",
        key="kw_search",
    )
    if kw and len(kw) >= 2:
        todos_progs = [(p["prog"], sec, p["nivel"], p["dur"], p["prom"])
                       for sec, progs in SEC_PROG.items() for p in progs]
        matches = [(p, s, n, d, pm) for p, s, n, d, pm in todos_progs
                   if kw.lower() in p.lower()]
        if matches:
            cols_kw = st.columns(min(len(matches), 3))
            for i, (prog, sec, niv, dur, pm) in enumerate(matches[:6]):
                with cols_kw[i % 3]:
                    if st.button(
                        f"📌 {prog[:40]}…\n{niv} · {pm} ap/f",
                        key=f"kw_{i}", use_container_width=True,
                    ):
                        st.session_state["sel_sec"]  = sec
                        st.session_state["sel_prog"] = prog
                        st.session_state["sel_dur"]  = dur
                        st.session_state["sel_niv"]  = niv
        else:
            st.caption("Sin resultados para esa búsqueda.")

    col_form, col_result = st.columns([1.05, 0.95], gap="large")

    with col_form:
        # ── FORMULARIO ──────────────────────────────────────────────────────
        st.markdown('<div class="mod-card">', unsafe_allow_html=True)
        st.markdown('<div class="mod-title">⚙️ Parámetros de la ficha</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            idx_sec = SECTORES.index(st.session_state.get("sel_sec", SECTORES[0])) if st.session_state.get("sel_sec") in SECTORES else 0
            p_sec = st.selectbox("📊 Sector productivo", SECTORES, index=idx_sec, key="p_sec")

        progs_sec = SEC_PROG.get(p_sec, [])
        prog_names = [p["prog"] for p in progs_sec]

        with c2:
            idx_prog = prog_names.index(st.session_state.get("sel_prog")) if st.session_state.get("sel_prog") in prog_names else 0
            p_prog = st.selectbox("📚 Programa", prog_names, index=idx_prog, key="p_prog")

        # Auto-rellenar duración, nivel y sector desde los metadatos del programa
        prog_meta = next((p for p in progs_sec if p["prog"] == p_prog), {})
        auto_dur  = int(st.session_state.get("sel_dur", prog_meta.get("dur", 48)))
        auto_niv  = st.session_state.get("sel_niv", prog_meta.get("nivel", "CURSO ESPECIAL"))
        niveles_ord = ["CURSO ESPECIAL", "TÉCNICO", "TECNÓLOGO", "EVENTO"]
        idx_niv = niveles_ord.index(auto_niv) if auto_niv in niveles_ord else 0

        if prog_meta:
            st.markdown(
                f'<div class="al-i">📋 <b>Auto-rellenado:</b> '
                f'{prog_meta.get("fichas",0)} fichas históricas · '
                f'Prom. {prog_meta.get("prom",0)} ap/ficha · '
                f'Duración típica: {prog_meta.get("dur",0)} h</div>',
                unsafe_allow_html=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            p_mun = st.selectbox("📍 Municipio", MUNICIPIOS, key="p_mun")
            p_niv = st.selectbox("🎓 Nivel", niveles_ord, index=idx_niv, key="p_niv")
        with c4:
            p_dur = st.number_input("⏱ Duración (h)", min_value=4, max_value=5000, value=auto_dur, step=4, key="p_dur")
            p_mes = st.slider("📅 Mes de inicio", 1, 12, 7,
                               format="%d", key="p_mes",
                               help="Feb–Mar y Sep–Oct son meses de mayor demanda.")

        # Selector de instructor
        inst_opts = ["— Sin preferencia (prom. 24.5) —"] + [
            f"{r['n']} — {r['prom']} ap/f ({r['fichas']} fichas)" for r in RESP
        ]
        p_inst_idx = st.selectbox("👤 Instructor (historial)", range(len(inst_opts)),
                                   format_func=lambda i: inst_opts[i], key="p_inst")
        hist_inst = RESP[p_inst_idx - 1]["prom"] if p_inst_idx > 0 else 24.5

        p_campe = st.checkbox("⭐ Es programa CAMPE-SENA (convocatoria masiva)", key="p_campe")

        st.markdown('</div>', unsafe_allow_html=True)

        # Municipios donde se ha ofertado el programa seleccionado
        muns_prog = PROG_MUN.get(p_prog[:70], [])
        if muns_prog:
            st.markdown('<div class="mod-card">', unsafe_allow_html=True)
            st.markdown('<div class="mod-title">🗺️ Municipios donde se ha ofertado este programa</div>', unsafe_allow_html=True)
            rows = [
                [
                    m["mun"],
                    str(m["fichas"]),
                    f'{m["ap"]:,}',
                    str(m["prom"]),
                    badge_html("▲" if m["prom"] > 25 else "▼", "s" if m["prom"] > 25 else "w"),
                ]
                for m in muns_prog
            ]
            st.markdown(
                tabla_html(["Municipio", "Fichas", "Aprendices", "Prom/f", ""], rows),
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        btn = st.button("🚀 Generar predicción y evaluación", type="primary",
                        use_container_width=True, key="btn_pred")

        if btn or st.session_state.get("last_pred"):
            if btn:
                resultado = intentar_modelo_ml(p_mun, p_sec, p_niv, p_dur, p_mes, hist_inst)
                if not resultado:
                    resultado = predecir_heuristico(p_mun, p_niv, p_dur, p_mes, hist_inst, p_campe)
                resultado["municipio"] = p_mun
                resultado["mes"]       = p_mes
                resultado["nivel"]     = p_niv
                st.session_state["last_pred"] = resultado

            res = st.session_state.get("last_pred", {})
            if res:
                st.markdown(result_box_html(res), unsafe_allow_html=True)

                # Sugerencias
                sugs = generar_sugerencias(
                    res["mes"], hist_inst, p_dur, res["nivel"], res["municipio"]
                )
                with st.expander("💡 Sugerencias de optimización", expanded=True):
                    for s in sugs:
                        st.markdown(s)

                # Factores de influencia
                if res.get("factores"):
                    st.markdown("**Factores de influencia en la predicción**")
                    factores_items = [
                        {"label": "Historial instructor", "val": round(res["factores"].get("instructor", 1) * 38)},
                        {"label": "Municipio",            "val": round(res["factores"].get("municipio",  1) * 27)},
                        {"label": "Nivel formación",      "val": round(res["factores"].get("nivel",      1) * 18)},
                        {"label": "Duración",             "val": round(res["factores"].get("duracion",   1) * 10)},
                        {"label": "Mes de inicio",        "val": round(res["factores"].get("mes",        1) * 7)},
                    ]
                    max_f = max(f["val"] for f in factores_items)
                    st.markdown(
                        prog_bars_html(factores_items, "label", "val", max_f),
                        unsafe_allow_html=True,
                    )

        # Programas estrella del sector seleccionado
        if progs_sec:
            st.markdown("---")
            st.markdown(f"**Programas con mayor demanda en {p_sec}**")
            max_prom = max(p["prom"] for p in progs_sec)
            items_s = [{"prog": p["prog"][:50], "prom": p["prom"]} for p in progs_sec[:6]]
            st.markdown(prog_bars_html(items_s, "prog", "prom", max_prom), unsafe_allow_html=True)

    # ── MAPA DE CALOR ───────────────────────────────────────────────────────
    st.markdown("---")
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        st.markdown("**🗺 Mapa de calor — intensidad por municipio**")
        modo_mapa = st.radio(
            "Modo mapa", ["Total aprendices", "N.º fichas", "Prom/ficha"],
            horizontal=True, key="modo_mapa",
        )
        mapa_df = pd.DataFrame([
            {
                "Municipio": mun,
                "Latitud":   v["lat"],
                "Longitud":  v["lng"],
                "Aprendices": v["ap"],
                "Fichas":    v["fichas"],
                "Prom/ficha": v["prom"],
                "Sector":    v["sector"],
                "Tendencia %": v.get("tend", 0),
            }
            for mun, v in MUN_STATS.items()
        ])
        size_col = {"Total aprendices": "Aprendices", "N.º fichas": "Fichas", "Prom/ficha": "Prom/ficha"}[modo_mapa]
        fig_map = px.scatter_mapbox(
            mapa_df, lat="Latitud", lon="Longitud",
            size=size_col, color=size_col,
            hover_name="Municipio",
            hover_data={"Aprendices": True, "Fichas": True, "Prom/ficha": True,
                        "Sector": True, "Latitud": False, "Longitud": False},
            color_continuous_scale="Blues",
            size_max=40, zoom=8,
            mapbox_style="carto-positron",
            title=f"Municipios — {modo_mapa}",
        )
        fig_map.update_layout(
            height=400, margin=dict(t=36, b=0, l=0, r=0),
            coloraxis_showscale=True,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_m2:
        st.markdown("**Top municipios**")
        top_mun = sorted(MUN_STATS.items(), key=lambda x: -x[1]["ap"])[:8]
        max_ap = top_mun[0][1]["ap"]
        html_top = ""
        for mun, v in top_mun:
            tend_badge = (
                badge_html(f"▲ {v.get('tend', 0)}%", "s")
                if v.get("tend", 0) > 0
                else badge_html(f"▼ {abs(v.get('tend', 0))}%", "w") if v.get("tend", 0) < 0
                else badge_html("→ 0%", "g")
            )
            pct = round(v["ap"] / max_ap * 100)
            html_top += f"""
<div class="pb-wrap">
  <div class="pb-meta">
    <span class="pb-name">{mun[:22]}</span>
    <span style="display:flex;align-items:center;gap:4px">
      <span class="pb-val">{v['ap']:,}</span>{tend_badge}
    </span>
  </div>
  <div class="pb-track"><div class="pb-fill" style="width:{pct}%;background:{C_BLUE}"></div></div>
</div>"""
        st.markdown(html_top, unsafe_allow_html=True)

    # ── INSTRUCTORES ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**👥 Historial de instructores — promedio de aprendices por ficha**")
    max_prom_inst = max(r["prom"] for r in RESP)
    col_i1, col_i2 = st.columns(2)
    for i, r in enumerate(RESP[:14]):
        col = col_i1 if i < 7 else col_i2
        color = C_GREEN if r["prom"] >= 26 else (C_BLUE if r["prom"] >= 20 else C_AMBER)
        pct = round(r["prom"] / max_prom_inst * 100)
        with col:
            st.markdown(f"""
<div class="inst-row">
  <span class="inst-name">{r['n']}</span>
  <div class="inst-track">
    <div class="inst-fill" style="width:{pct}%;background:{color}"></div>
  </div>
  <span class="inst-val" style="color:{color}">{r['prom']}</span>
</div>""", unsafe_allow_html=True)
    st.caption(f"🟢 ≥26 ap/ficha — alto impacto  ·  🔵 20–25 — normal  ·  🟡 <20 — bajo")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 — DEMANDA POR NIVEL
# ══════════════════════════════════════════════════════════════════════════════

elif modulo == "📊 Demanda por nivel":
    st.markdown("### 📊 Demanda por nivel de formación")
    st.caption("Distribución histórica de aprendices por nivel, sector y municipio.")

    niv_filter = st.radio(
        "Filtrar por nivel",
        ["Todos", "CURSO ESPECIAL", "TÉCNICO", "TECNÓLOGO"],
        horizontal=True, key="dem_niv",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_niv = px.bar(
            x=["Curso especial", "Técnico", "Tecnólogo", "Evento"],
            y=[51557, 8713, 1682, 252],
            title="Aprendices totales por nivel",
            labels={"x": "", "y": "Aprendices"},
            color_discrete_sequence=[C_BLUE],
        )
        fig_niv.update_traces(marker_cornerradius=4, width=0.55)
        fig_niv.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=280, margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_niv, use_container_width=True)

    with c2:
        sec_data = pd.DataFrame([
            {"Sector": "Servicios",  "Fichas": 1232},
            {"Sector": "Agropecuario", "Fichas": 518},
            {"Sector": "Salud",      "Fichas": 124},
            {"Sector": "Industria",  "Fichas": 93},
            {"Sector": "Hotelería",  "Fichas": 84},
            {"Sector": "Comercio",   "Fichas": 71},
            {"Sector": "Otros",      "Fichas": 429},
        ])
        fig_sec = px.pie(
            sec_data, values="Fichas", names="Sector",
            title="Fichas por sector productivo",
            hole=0.55,
            color_discrete_sequence=["#2a78d6","#1baf7a","#eda100","#4a3aa7","#e34948","#888780","#e87ba4"],
        )
        fig_sec.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    # Contraste por sector dentro del nivel seleccionado
    NIV_SEC_STATIC = {
        "Todos":         [{"sec": "Servicios", "ap": 58281}, {"sec": "Agropecuario", "ap": 12330}, {"sec": "Salud", "ap": 3245}, {"sec": "Transversal", "ap": 5410}, {"sec": "Industria", "ap": 4212}, {"sec": "Hotelería", "ap": 4534}],
        "CURSO ESPECIAL":[{"sec": "Servicios", "ap": 19284}, {"sec": "Agropecuario", "ap": 9401}, {"sec": "Salud", "ap": 2473}, {"sec": "Transversal", "ap": 4280}, {"sec": "Industria", "ap": 3267}, {"sec": "Hotelería", "ap": 3277}],
        "TÉCNICO":       [{"sec": "Servicios", "ap": 3041}, {"sec": "Agropecuario", "ap": 1986}, {"sec": "Hotelería", "ap": 1061}, {"sec": "Construcción", "ap": 418}, {"sec": "Transversal", "ap": 735}, {"sec": "Comercio", "ap": 376}],
        "TECNÓLOGO":     [{"sec": "Servicios", "ap": 410}, {"sec": "Agropecuario", "ap": 321}, {"sec": "Transversal", "ap": 230}, {"sec": "Comercio", "ap": 269}, {"sec": "Construcción", "ap": 143}],
    }
    data_niv = NIV_SEC_STATIC.get(niv_filter, NIV_SEC_STATIC["Todos"])
    max_ap_niv = max(d["ap"] for d in data_niv)

    st.markdown(f"**Aprendices por sector — nivel: {niv_filter}**")
    items_niv = [{"sec": d["sec"], "ap": d["ap"]} for d in data_niv]
    st.markdown(prog_bars_html(items_niv, "sec", "ap", max_ap_niv), unsafe_allow_html=True)

    # Tabla de municipios
    st.markdown("**Avances por municipio**")
    rows_mun = []
    for mun, v in sorted(MUN_STATS.items(), key=lambda x: -x[1]["ap"]):
        tend = v.get("tend", 0)
        rows_mun.append([
            mun, str(v["fichas"]), f'{v["ap"]:,}',
            str(v["prom"]), v["sector"],
            badge_html(f"▲ {tend}%", "s") if tend > 0 else badge_html(f"▼ {abs(tend)}%", "w") if tend < 0 else badge_html("→", "g"),
            badge_html("Alto", "s") if v["prom"] >= 27 else badge_html("Normal", "i") if v["prom"] >= 22 else badge_html("Bajo", "w"),
        ])
    st.markdown(
        tabla_html(["Municipio", "Fichas", "Aprendices", "Prom/f", "Sector", "Tendencia", "Estado"], rows_mun),
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 — RECOMENDACIONES
# ══════════════════════════════════════════════════════════════════════════════

elif modulo == "⚡ Recomendaciones":
    st.markdown("### ⚡ Recomendaciones de programas")
    st.caption("Ruta formativa estratégica por municipio: Complementarios → Técnico → Tecnólogo.")

    REC_MUN = {
        "SANTAFÉ DE ANTIOQUIA": {
            "comp": ["Manipulación higiénica de alimentos", "Inglés básico nivel 1", "Coctelería tropical", "Gastronomía colombiana", "Comportamiento emprendedor"],
            "tec":  ["Técnico en cocina", "Técnico en turismo y hotelería", "Asistencia administrativa"],
            "tec2": ["Tecnólogo en gestión hotelera y turismo"],
        },
        "DABEIBA": {
            "comp": ["Producción agropecuaria ecológica", "Comportamiento emprendedor", "Buenas prácticas ganaderas", "Manejo básico de Excel", "Ganadería bovina sostenible"],
            "tec":  ["Técnico en producción agropecuaria", "Técnico en sistemas"],
            "tec2": ["Tecnólogo en gestión empresarial agropecuaria"],
        },
        "FRONTINO": {
            "comp": ["Buenas prácticas ganaderas", "Producción de cacao", "Criar gallinas ponedoras", "Conservación de recursos naturales"],
            "tec":  ["Técnico en producción agropecuaria"],
            "tec2": ["Tecnólogo en agroindustria"],
        },
        "CAÑASGORDAS": {
            "comp": ["Comportamiento emprendedor", "Manejo básico de Excel", "Buenas prácticas agrícolas", "Piscicultura básica"],
            "tec":  ["Técnico en sistemas", "Técnico en producción agropecuaria"],
            "tec2": ["Tecnólogo en administración agropecuaria"],
        },
        "EBÉJICO": {
            "comp": ["Producción agropecuaria ecológica", "Prácticas de beneficio del café", "Criar gallinas ponedoras", "Conservación de recursos naturales"],
            "tec":  ["Técnico en producción agropecuaria"],
            "tec2": ["Tecnólogo en agroindustria"],
        },
        "ANZÁ": {
            "comp": ["Prácticas de beneficio del café", "Criar gallinas ponedoras", "Comportamiento emprendedor", "Buenas prácticas agrícolas"],
            "tec":  ["Técnico en producción agropecuaria"],
            "tec2": ["Tecnólogo en café y agroindustria"],
        },
        "URAMITA": {
            "comp": ["Producción de cacao", "Ganadería bovina", "Manejo básico de Excel", "Comportamiento emprendedor"],
            "tec":  ["Técnico en producción agropecuaria"],
            "tec2": ["Tecnólogo en administración agropecuaria"],
        },
        "LIBORINA": {
            "comp": ["Turismo rural comunitario", "Buenas prácticas ganaderas", "Manipulación higiénica de alimentos"],
            "tec":  ["Técnico en producción agropecuaria"],
            "tec2": ["Tecnólogo en ecoturismo"],
        },
        "BURITICA": {
            "comp": ["Trabajo en alturas", "Primeros auxilios", "Comportamiento emprendedor"],
            "tec":  ["Técnico en minería", "Técnico en sistemas"],
            "tec2": ["Tecnólogo en gestión empresarial"],
        },
    }
    mun_rec = st.selectbox("Selecciona municipio", sorted(REC_MUN.keys()), key="rec_mun")
    tipo_filter = st.radio(
        "Tipo de programa",
        ["Todos", "Complementarios", "Técnico", "Tecnólogo"],
        horizontal=True, key="rec_tipo",
    )

    data_rec = REC_MUN.get(mun_rec, {"comp": [], "tec": [], "tec2": []})
    c_rec1, c_rec2 = st.columns([1, 1.2])

    with c_rec1:
        st.markdown("**Ruta formativa recomendada**")
        st.markdown("""
<div class="ruta-step" style="background:#EAF3DE;border:.5px solid #C0DD97">
  <b>1️⃣ Cursos Complementarios — Base estratégica</b>
  Forma capital humano local antes de invertir en niveles más altos. Genera confianza en la comunidad y valida la demanda. Costo de apertura bajo y convocatoria alta.
</div>
<div style="text-align:center;font-size:18px;margin:4px 0">↓</div>
<div class="ruta-step" style="background:#E6F1FB;border:.5px solid #B5D4F4">
  <b>2️⃣ Programas Técnicos</b>
  Consolida competencias con comunidad ya sensibilizada. Mayor impacto laboral y empleabilidad.
</div>
<div style="text-align:center;font-size:18px;margin:4px 0">↓</div>
<div class="ruta-step" style="background:#FAEEDA;border:.5px solid #FAC775">
  <b>3️⃣ Tecnólogos</b>
  Nivel superior. Requiere base técnica sólida y demanda laboral específica en el territorio.
</div>""", unsafe_allow_html=True)

    with c_rec2:
        st.markdown(f"**Programas recomendados — {mun_rec}**")
        items_rec = []
        if tipo_filter in ("Todos", "Complementarios"):
            items_rec += [(p, "Complementario", "s", "✅") for p in data_rec.get("comp", [])]
        if tipo_filter in ("Todos", "Técnico"):
            items_rec += [(p, "Técnico", "i", "📘") for p in data_rec.get("tec", [])]
        if tipo_filter in ("Todos", "Tecnólogo"):
            items_rec += [(p, "Tecnólogo", "w", "🎓") for p in data_rec.get("tec2", [])]

        for i, (prog, tipo, bc, emoji) in enumerate(items_rec, 1):
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;background:#fff;border:.5px solid {C_BORDER};border-radius:8px;margin-bottom:6px">
  <span style="font-size:16px">{emoji}</span>
  <div style="flex:1"><div style="font-size:12px;font-weight:600;color:#0b0b0b">{prog}</div></div>
  {badge_html(tipo, bc)}
  <span style="font-size:11px;font-weight:600;color:{C_GRAY}">#{i}</span>
</div>""", unsafe_allow_html=True)

    # Programas estrella
    st.markdown("---")
    st.markdown("**⭐ Programas estrella — mayor promedio histórico de aprendices**")
    estrellas = [
        ("Servicio de recepción hotelera",    "TÉCNICO",         "HOTELERIA Y TURISMO", 43.4, 11),
        ("Venta de productos en línea",        "CURSO ESPECIAL",  "COMERCIO",            43.0, 5),
        ("Manipulación higiénica de alimentos","CURSO ESPECIAL",  "SERVICIOS",           27.8, 123),
        ("Construcción de edificaciones",      "TÉCNICO",         "CONSTRUCCION",        26.7, 10),
        ("Carlos Rubio (instructor top)",      "VARIOS",          "TODOS",               26.5, 50),
        ("Sistemas",                           "TÉCNICO",         "TRANSVERSAL",         28.4, 14),
    ]
    cols_est = st.columns(3)
    for i, (prog, niv, sec, prom, fichas) in enumerate(estrellas):
        with cols_est[i % 3]:
            st.markdown(f"""
<div style="background:#fff;border:.5px solid {C_BORDER};border-radius:10px;padding:12px;margin-bottom:8px">
  <div style="font-size:12px;font-weight:600;margin-bottom:4px">⭐ {prog}</div>
  <div style="font-size:10px;color:{C_GRAY};margin-bottom:6px">{fichas} fichas · {sec}</div>
  <div style="font-size:18px;font-weight:600;color:{C_GREEN}">{prom} ap/f</div>
  {badge_html(niv, "i")}
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 — OPORTUNIDADES DE MEJORA
# ══════════════════════════════════════════════════════════════════════════════

elif modulo == "🔭 Oportunidades de mejora":
    st.markdown("### 🔭 Oportunidades de mejora")
    st.caption("Brechas entre oferta y demanda por rubro productivo y municipio.")

    EMOJIS_RUBROS = {
        "Café": "☕", "Cacao": "🍫", "Ganadería": "🐄", "Avicultura": "🐔",
        "Turismo": "🏕️", "Agricultura": "🌿", "Piscicultura": "🐟",
        "Alimentos": "🍽️", "Inglés": "🇺🇸", "Emprendimiento": "💡",
        "TIC / Excel": "💻", "Medio Ambiente": "🌳", "Artesanías": "🧶",
        "Alturas": "🏗️", "Aguacate": "🥑",
    }

    rub_names = list(RUBROS.keys())
    cols_rub = st.columns(min(len(rub_names), 5))
    sel_rub = st.session_state.get("sel_rub", rub_names[0])
    for i, rub in enumerate(rub_names):
        with cols_rub[i % 5]:
            emoji = EMOJIS_RUBROS.get(rub, "📌")
            d = RUBROS[rub]
            if st.button(
                f"{emoji} {rub}\n{d['fichas']} fichas",
                key=f"rb_{rub}",
                use_container_width=True,
                type="primary" if rub == sel_rub else "secondary",
            ):
                st.session_state["sel_rub"] = rub
                st.rerun()

    st.markdown("---")
    d_rub = RUBROS.get(sel_rub, {})
    if d_rub:
        col_o1, col_o2 = st.columns([1.2, 0.8])

        with col_o1:
            st.markdown(f"**{EMOJIS_RUBROS.get(sel_rub, '📌')} Municipios líderes en {sel_rub}**")
            muns_rub = sorted(d_rub["muns"].items(), key=lambda x: -x[1])
            max_rub = muns_rub[0][1] if muns_rub else 1
            for mun, ap in muns_rub:
                pct = round(ap / max_rub * 100)
                st.markdown(f"""
<div class="pb-wrap">
  <div class="pb-meta"><span class="pb-name">{mun}</span><span class="pb-val">{ap:,} ap</span></div>
  <div class="pb-track"><div class="pb-fill" style="width:{pct}%;background:{C_BLUE}"></div></div>
</div>""", unsafe_allow_html=True)

            # Gráfica de barras
            df_rub = pd.DataFrame(list(d_rub["muns"].items()), columns=["Municipio", "Aprendices"])
            fig_rub = px.bar(
                df_rub, x="Aprendices", y="Municipio",
                orientation="h", title=f"Aprendices en {sel_rub} por municipio",
                color="Aprendices", color_continuous_scale="Blues",
            )
            fig_rub.update_layout(
                height=300, showlegend=False, margin=dict(t=36, b=0, l=0, r=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_rub, use_container_width=True)

        with col_o2:
            st.markdown("**Análisis de brechas y potencial**")
            muns_con = set(d_rub["muns"].keys())
            muns_sin = [m for m in MUNICIPIOS if m not in muns_con
                        and MUN_STATS.get(m, {}).get("sector") == "AGROPECUARIO"][:5]

            st.markdown(f"""
<div class="al-s"><b>✅ Municipios líderes en {sel_rub}:</b><br>
{', '.join(list(d_rub["muns"].keys())[:3])}.<br>
Total: {d_rub['ap']:,} aprendices en {d_rub['fichas']} fichas.</div>""", unsafe_allow_html=True)

            if muns_sin:
                st.markdown(f"""
<div class="al-w"><b>⚠️ Municipios con potencial sin cobertura en {sel_rub}:</b><br>
{', '.join(muns_sin)}.<br>
Estos municipios tienen vocación agropecuaria pero no registran fichas históricas en este rubro.</div>""",
                    unsafe_allow_html=True)

            st.markdown(f"""
<div class="al-i"><b>💡 Recomendación estratégica:</b><br>
Abrir cursos complementarios de <b>{sel_rub}</b> en los municipios sin cobertura como primer paso. 
Validar la demanda con la comunidad antes de invertir en nivel técnico. 
Los municipios con tendencia creciente tienen mayor probabilidad de éxito.</div>""",
                unsafe_allow_html=True)

            # Métricas del rubro
            st.markdown("**Indicadores del rubro**")
            avg_ap_rub = round(d_rub["ap"] / max(d_rub["fichas"], 1), 1)
            st.metric("Total aprendices", f'{d_rub["ap"]:,}')
            st.metric("Total fichas",     f'{d_rub["fichas"]}')
            st.metric("Promedio ap/ficha", avg_ap_rub)
            st.metric("Municipios cubiertos", len(d_rub["muns"]))


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5 — MANUAL DE USUARIO
# ══════════════════════════════════════════════════════════════════════════════

elif modulo == "📖 Manual de usuario":
    st.markdown("### 📖 Manual de usuario — SN Predict v2.0")

    with st.expander("🎯 Módulo 1 — Predictor inteligente", expanded=True):
        st.markdown("""
**¿Qué hace?**
Predice cuántos aprendices participarán en una nueva ficha de formación antes de abrirla.

**Paso a paso:**
1. Usa la **búsqueda por palabra clave** para encontrar programas por tema (ej: "café", "ganadería").
2. Selecciona el **Sector productivo** → la lista de programas se actualiza automáticamente con los de mayor demanda histórica en ese sector.
3. Elige el **Programa** → la duración, nivel y estadísticas se auto-rellenan con datos reales.
4. Completa municipio, mes de inicio e instructor.
5. Haz clic en **Generar predicción** → obtienes:
   - Aprendices estimados
   - Viabilidad de apertura (0–100%)
   - Clasificación Alto/Normal impacto
   - Sugerencias de optimización contextuales
6. El **mapa de calor** muestra la distribución territorial de la demanda.
7. El **historial de instructores** ayuda a elegir al responsable con mayor capacidad de convocatoria.
""")

    with st.expander("📊 Módulo 2 — Demanda por nivel"):
        st.markdown("""
- Usa los **chips de nivel** (Curso Especial / Técnico / Tecnólogo) para filtrar la distribución.
- La gráfica de **sectores dentro del nivel** compara el peso de cada sector.
- La tabla de municipios incluye tendencia interanual (▲/▼) y clasificación de impacto.
""")

    with st.expander("⚡ Módulo 3 — Recomendaciones"):
        st.markdown("""
- Selecciona un municipio para ver la **ruta formativa**: Complementarios → Técnico → Tecnólogo.
- Los **Cursos Complementarios** son la base estratégica: forman capital humano local, tienen bajo costo de apertura y alta convocatoria. Son el primer paso antes de abrir programas técnicos.
- Filtra por tipo de programa para ver solo el nivel de interés.
""")

    with st.expander("🔭 Módulo 4 — Oportunidades de mejora"):
        st.markdown("""
- Selecciona un **rubro productivo** (Café, Ganadería, Turismo…) para ver:
  - Municipios líderes con más aprendices formados en ese rubro.
  - Municipios con vocación agropecuaria pero sin cobertura formativa (brechas).
  - Recomendación estratégica para cerrar la brecha.
""")

    with st.expander("⚙️ Consideraciones técnicas"):
        st.markdown("""
- **Datos:** Histórico PE04 Centro SENA Occidente (2020–2025), 2.551 fichas, 22 municipios.
- **Modelo:** Gradient Boosting entrenado con `python train.py`. Error típico Y1: ±8–11 aprendices. ROC-AUC Y2: ~0.78.
- **Modo heurístico:** Si el modelo ML no está disponible, la app usa patrones calibrados del dataset real.
- **Actualización:** Volver a ejecutar `python train.py` cuando lleguen nuevos datos PE04.
- Las predicciones son de **apoyo a la decisión** — el juicio del coordinador siempre prevalece.
""")

st.markdown(footer_html(), unsafe_allow_html=True)
