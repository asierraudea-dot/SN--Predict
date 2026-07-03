"""
oportunidades_mejora.py  —  SN Predict v2.1
============================================
Módulo completo de Oportunidades de Mejora con:
  - Barras apiladas por nivel de formación (colores diferenciados)
  - Filtro por años de análisis (2023 / 2024 / 2025)
  - Filtro por rubro productivo
  - Carga de nuevo archivo Excel con combinación automática
  - Análisis de brechas por municipio y nivel

INSTRUCCIONES DE USO:
  En app.py, dentro del bloque elif modulo == "🔭 Oportunidades de mejora":
  llama a render_oportunidades(df, MUN_STATS)

Autor:  [nombre del estudiante]
Fecha:  Junio 2026
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

# ── Paleta por nivel (fija, siempre la misma) ─────────────────────────────────
COLORES_NIVEL: dict[str, str] = {
    "CURSO ESPECIAL": "#185FA5",   # azul  → complementario
    "TÉCNICO":        "#1D9E75",   # verde → técnico
    "TECNÓLOGO":      "#BA7517",   # ámbar → tecnólogo
    "EVENTO":         "#888780",   # gris  → evento
}
LABEL_NIVEL: dict[str, str] = {
    "CURSO ESPECIAL": "Complementario",
    "TÉCNICO":        "Técnico",
    "TECNÓLOGO":      "Tecnólogo",
    "EVENTO":         "Evento",
}

# Palabras clave por rubro productivo
RUBROS_KW: dict[str, list[str]] = {
    "Café":          ["cafe", "café", "cafeto"],
    "Cacao":         ["cacao"],
    "Ganadería":     ["ganader", "bovino", "vacuno", "pecuari"],
    "Avicultura":    ["avicul", "gallina", "pollo", "ponedora"],
    "Turismo":       ["turismo", "ecoturismo", "aviturism"],
    "Agricultura":   ["agric", "cultivo", "sembr", "vegetal", "agropecuar"],
    "Piscicultura":  ["piscic", "tilapia", "trucha", "acuic"],
    "Alturas":       ["altura"],
    "Inglés":        ["ingles", "english"],
    "Emprendimiento":["emprend", "negocio", "empresar", "innovad"],
    "Alimentos":     ["alimento", "higien", "manipul", "cocina", "gastro"],
    "TIC / Excel":   ["excel", "ofimati", "sistemas", "software", "inform", "word"],
    "Medio Ambiente":["ambient", "ecolog", "conserv", "recurso natur"],
    "Artesanías":    ["bisuter", "artesanal", "tejid", "mostacilla", "chaquira"],
}


# ── Carga y combinación de múltiples archivos ─────────────────────────────────

def cargar_y_combinar(archivos_extra: list[Path] | None = None) -> pd.DataFrame:
    """
    Carga el PE04 base y opcionalmente combina archivos adicionales.

    Busca todos los .xlsx que empiecen con PE04 en la carpeta data/ y
    los combina eliminando duplicados por IDENTIFICADOR_FICHA.

    Args:
        archivos_extra: Lista de rutas adicionales subidas por el usuario.

    Returns:
        DataFrame combinado y limpio.
    """
    from data_loader import cargar_df

    df_base = cargar_df()
    dfs = [df_base]

    # Buscar archivos PE04 adicionales en data/
    data_dir = ROOT / "data"
    if data_dir.exists():
        for p in sorted(data_dir.glob("PE04*.xlsx")):
            nombre_base = "PE04_HISTORICO_PREVIOS.xlsx"
            if p.name not in (nombre_base, "PE04_HISTÓRICO_PREVIOS.xlsx"):
                try:
                    df_extra = pd.read_excel(p, sheet_name=0)
                    df_extra.columns = [c.strip() for c in df_extra.columns]
                    df_extra["TOTAL_APRENDICES"] = pd.to_numeric(
                        df_extra.get("TOTAL_APRENDICES", 0), errors="coerce"
                    ).fillna(0)
                    df_extra["AÑO"] = pd.to_numeric(
                        df_extra.get("AÑO", 0), errors="coerce"
                    )
                    dfs.append(df_extra)
                    st.toast(f"✅ Archivo adicional cargado: {p.name}", icon="📂")
                except Exception as e:
                    st.warning(f"No se pudo cargar {p.name}: {e}")

    if archivos_extra:
        for ruta in archivos_extra:
            try:
                df_extra = pd.read_excel(ruta)
                df_extra.columns = [c.strip() for c in df_extra.columns]
                df_extra["TOTAL_APRENDICES"] = pd.to_numeric(
                    df_extra.get("TOTAL_APRENDICES", 0), errors="coerce"
                ).fillna(0)
                df_extra["AÑO"] = pd.to_numeric(
                    df_extra.get("AÑO", 0), errors="coerce"
                )
                dfs.append(df_extra)
            except Exception as e:
                st.error(f"Error cargando {ruta.name}: {e}")

    df_total = pd.concat(dfs, ignore_index=True)

    # Eliminar duplicados por ficha si la columna existe
    if "IDENTIFICADOR_FICHA" in df_total.columns:
        antes = len(df_total)
        df_total = df_total.drop_duplicates(subset=["IDENTIFICADOR_FICHA"])
        duplicados = antes - len(df_total)
        if duplicados > 0:
            st.info(f"Se eliminaron {duplicados:,} registros duplicados al combinar archivos.", icon="ℹ️")

    return df_total


def filtrar_por_rubro(df: pd.DataFrame, rubro: str) -> pd.DataFrame:
    """
    Filtra el DataFrame por palabras clave del rubro productivo.

    Args:
        df:     DataFrame completo.
        rubro:  Nombre del rubro. "Todos" devuelve el df sin filtrar.

    Returns:
        DataFrame filtrado.
    """
    if rubro == "Todos los rubros" or not rubro:
        return df
    kws = RUBROS_KW.get(rubro, [])
    if not kws:
        return df
    mask = df["NOMBRE_PROGRAMA_FORMACION"].str.lower().str.contains(
        "|".join(kws), na=False
    )
    return df[mask]


def render_oportunidades(df: pd.DataFrame | None, mun_stats: dict) -> None:
    """
    Renderiza el módulo completo de Oportunidades de Mejora.

    Secciones:
      1. Carga de nuevo archivo Excel
      2. Filtros: año + rubro + nivel
      3. KPIs dinámicos según filtros
      4. Barras apiladas municipio × nivel (con colores)
      5. Distribución por nivel (donut/barras)
      6. Evolución anual por nivel
      7. Análisis de brechas

    Args:
        df:        DataFrame PE04 cargado (puede ser None si no hay datos).
        mun_stats: Dict municipio → estadísticas base.
    """
    st.markdown("### 🔭 Oportunidades de mejora")
    st.caption(
        "Barras apiladas por nivel de formación · Filtros de año y rubro · "
        "Carga de datos actualizados"
    )

    # ── 1. CARGA DE NUEVO ARCHIVO ─────────────────────────────────────────────
    with st.expander("📂 Cargar datos actualizados — nuevo archivo Excel", expanded=False):
        st.markdown("""
**¿Cómo funciona?**
Sube un archivo PE04 nuevo y se combina automáticamente con los datos históricos.
Los registros duplicados (mismo `IDENTIFICADOR_FICHA`) se descartan.

**Columnas requeridas:**
`NOMBRE_MUNICIPIO_CURSO` · `NOMBRE_PROGRAMA_FORMACION` · `NIVEL_FORMACION` ·
`NOMBRE_NUEVO_SECTOR` · `TOTAL_APRENDICES` · `FECHA_INICIO_FICHA` · `AÑO`
""")

        col_up1, col_up2 = st.columns([1.2, 0.8])

        with col_up1:
            archivo_nuevo = st.file_uploader(
                "Selecciona el nuevo archivo PE04",
                type=["xlsx", "xls"],
                key="uploader_pe04",
                help="Arrastra o haz clic. Máx. 200MB.",
            )

            if archivo_nuevo:
                try:
                    df_nuevo = pd.read_excel(archivo_nuevo)
                    df_nuevo.columns = [c.strip() for c in df_nuevo.columns]
                    df_nuevo["TOTAL_APRENDICES"] = pd.to_numeric(
                        df_nuevo.get("TOTAL_APRENDICES", 0), errors="coerce"
                    ).fillna(0)
                    df_nuevo["AÑO"] = pd.to_numeric(
                        df_nuevo.get("AÑO", 0), errors="coerce"
                    )

                    # Guardar en session_state para uso en la sesión actual
                    st.session_state["df_extra"] = df_nuevo
                    st.success(
                        f"✅ **{archivo_nuevo.name}** cargado: "
                        f"{len(df_nuevo):,} fichas · "
                        f"Años: {sorted(df_nuevo['AÑO'].dropna().unique().astype(int).tolist())}",
                        icon="✅",
                    )
                    st.info(
                        "Los gráficos ahora incluyen estos datos. "
                        "Para hacer permanente, copia el archivo a la carpeta `data/` del repositorio.",
                        icon="ℹ️",
                    )
                except Exception as e:
                    st.error(f"Error al leer el archivo: {e}")

        with col_up2:
            st.markdown("**Pasos para carga permanente en GitHub:**")
            st.code("""
# 1. Copia el nuevo archivo a data/
cp PE04_NUEVO_2026.xlsx data/

# 2. Sube al repositorio
git add data/PE04_NUEVO_2026.xlsx
git commit -m "datos: nuevo PE04 2026"
git push

# Streamlit Cloud recarga automáticamente
# en ~60 segundos al detectar el push
""", language="bash")
            st.markdown(
                "El código en `data_loader.py` detecta automáticamente "
                "todos los archivos `PE04*.xlsx` en `data/` y los combina."
            )

    # Combinar con archivo extra si existe en session_state
    df_extra = st.session_state.get("df_extra")
    if df is not None and df_extra is not None:
        df_trabajo = pd.concat([df, df_extra], ignore_index=True)
        if "IDENTIFICADOR_FICHA" in df_trabajo.columns:
            df_trabajo = df_trabajo.drop_duplicates(subset=["IDENTIFICADOR_FICHA"])
    elif df is not None:
        df_trabajo = df
    else:
        st.warning("Dataset no disponible. Carga un archivo para ver los análisis.", icon="⚠️")
        return

    # ── 2. FILTROS ────────────────────────────────────────────────────────────
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("**📅 Años de análisis**")
        años_disp = sorted(df_trabajo["AÑO"].dropna().unique().astype(int).tolist())
        años_sel  = st.multiselect(
            "Selecciona años",
            años_disp,
            default=años_disp,
            key="años_opp",
            label_visibility="collapsed",
        )

    with col_f2:
        st.markdown("**🌿 Rubro productivo**")
        rubro_sel = st.selectbox(
            "Rubro",
            ["Todos los rubros"] + list(RUBROS_KW.keys()),
            key="rubro_opp",
            label_visibility="collapsed",
        )

    with col_f3:
        st.markdown("**🎓 Nivel de formación**")
        nivel_sel = st.multiselect(
            "Nivel",
            ["CURSO ESPECIAL", "TÉCNICO", "TECNÓLOGO", "EVENTO"],
            default=["CURSO ESPECIAL", "TÉCNICO", "TECNÓLOGO"],
            key="nivel_opp",
            label_visibility="collapsed",
        )

    if not años_sel:
        st.warning("Selecciona al menos un año.", icon="⚠️")
        return

    # Aplicar filtros
    df_f = df_trabajo[df_trabajo["AÑO"].isin(años_sel)].copy()
    df_f = filtrar_por_rubro(df_f, rubro_sel)
    if nivel_sel:
        df_f = df_f[df_f["NIVEL_FORMACION"].isin(nivel_sel)]

    if df_f.empty:
        st.info("Sin datos para la combinación de filtros seleccionada.", icon="ℹ️")
        return

    # ── 3. KPIs dinámicos ────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    anos_txt = " · ".join(str(a) for a in sorted(años_sel))
    with k1:
        st.metric("Fichas filtradas",  f"{len(df_f):,}")
    with k2:
        st.metric("Aprendices",        f"{int(df_f['TOTAL_APRENDICES'].sum()):,}")
    with k3:
        st.metric("Municipios",        df_f["NOMBRE_MUNICIPIO_CURSO"].nunique())
    with k4:
        st.metric("Años analizados",   f"{len(años_sel)} ({anos_txt})")

    st.markdown("---")

    # ── 4. BARRAS APILADAS municipio × nivel ──────────────────────────────────
    col_bars, col_right = st.columns([1.4, 0.6])

    with col_bars:
        st.markdown(
            f"**Municipios líderes — aprendices por nivel · "
            f"{rubro_sel} · Años {anos_txt}**"
        )

        # Tabla pivote municipio × nivel
        pivot = (
            df_f.groupby(["NOMBRE_MUNICIPIO_CURSO", "NIVEL_FORMACION"])["TOTAL_APRENDICES"]
            .sum()
            .reset_index()
        )
        pivot_wide = pivot.pivot_table(
            index="NOMBRE_MUNICIPIO_CURSO",
            columns="NIVEL_FORMACION",
            values="TOTAL_APRENDICES",
            fill_value=0,
        ).reset_index()
        pivot_wide["TOTAL"] = pivot_wide.drop(columns=["NOMBRE_MUNICIPIO_CURSO"]).sum(axis=1)
        pivot_wide = pivot_wide.sort_values("TOTAL", ascending=True).tail(18)

        # Construir gráfica apilada con Plotly
        niveles_presentes = [n for n in ["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO","EVENTO"]
                             if n in pivot_wide.columns]

        fig = go.Figure()
        for nivel in niveles_presentes:
            if nivel in pivot_wide.columns:
                fig.add_trace(go.Bar(
                    name=LABEL_NIVEL.get(nivel, nivel),
                    y=pivot_wide["NOMBRE_MUNICIPIO_CURSO"],
                    x=pivot_wide[nivel],
                    orientation="h",
                    marker_color=COLORES_NIVEL[nivel],
                    hovertemplate=(
                        f"<b>%{{y}}</b><br>"
                        f"{LABEL_NIVEL.get(nivel,nivel)}: %{{x:,}} aprendices<extra></extra>"
                    ),
                ))

        fig.update_layout(
            barmode="stack",
            height=max(320, len(pivot_wide) * 28 + 80),
            margin=dict(t=10, b=30, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h", y=1.02, x=0,
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                gridcolor="#e1e0d9",
                title="Aprendices",
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                title="",
            ),
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Distribución por nivel
        st.markdown("**Distribución por nivel**")
        tot_niv = (
            df_f.groupby("NIVEL_FORMACION")["TOTAL_APRENDICES"]
            .sum()
            .reindex(["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO","EVENTO"])
            .dropna()
        )
        grand = tot_niv.sum() or 1
        for niv, val in tot_niv.items():
            pct = round(val / grand * 100)
            color = COLORES_NIVEL.get(niv, "#888780")
            st.markdown(f"""
<div style="margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">
    <span style="display:flex;align-items:center;gap:5px">
      <span style="width:9px;height:9px;border-radius:2px;background:{color};display:inline-block"></span>
      {LABEL_NIVEL.get(niv,niv)}
    </span>
    <span style="font-weight:500">{int(val):,} <span style="color:#898781;font-weight:400">({pct}%)</span></span>
  </div>
  <div style="height:6px;background:#f1f0ea;border-radius:3px;overflow:hidden">
    <div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div>
  </div>
</div>""", unsafe_allow_html=True)

        # Evolución anual
        st.markdown("**Evolución anual por nivel**")
        evol = (
            df_f.groupby(["AÑO","NIVEL_FORMACION"])["TOTAL_APRENDICES"]
            .sum()
            .reset_index()
        )
        if not evol.empty:
            fig_evol = px.bar(
                evol,
                x="AÑO", y="TOTAL_APRENDICES",
                color="NIVEL_FORMACION",
                color_discrete_map=COLORES_NIVEL,
                labels={"AÑO":"Año","TOTAL_APRENDICES":"Aprendices","NIVEL_FORMACION":"Nivel"},
                barmode="stack",
                category_orders={"NIVEL_FORMACION":["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO","EVENTO"]},
            )
            fig_evol.update_layout(
                height=220, margin=dict(t=10,b=20,l=10,r=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=9), orientation="h", y=1.08),
                xaxis=dict(tickfont=dict(size=10), dtick=1),
                yaxis=dict(tickfont=dict(size=10)),
                showlegend=True,
            )
            st.plotly_chart(fig_evol, use_container_width=True)

    # ── 5. ANÁLISIS DE BRECHAS ────────────────────────────────────────────────
    st.markdown("---")
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("**Brechas detectadas**")

        todos_mun = sorted(df_f["NOMBRE_MUNICIPIO_CURSO"].dropna().unique().tolist())
        mun_con_tecnologo = (
            df_f[df_f["NIVEL_FORMACION"]=="TECNÓLOGO"]
            ["NOMBRE_MUNICIPIO_CURSO"].dropna().unique().tolist()
        )
        mun_sin_tecnologo = [m for m in todos_mun if m not in mun_con_tecnologo]

        mun_crecimiento = []
        if "AÑO" in df_f.columns and len(años_sel) >= 2:
            año_min = min(años_sel)
            año_max = max(años_sel)
            ap_min = df_f[df_f["AÑO"]==año_min].groupby("NOMBRE_MUNICIPIO_CURSO")["TOTAL_APRENDICES"].sum()
            ap_max = df_f[df_f["AÑO"]==año_max].groupby("NOMBRE_MUNICIPIO_CURSO")["TOTAL_APRENDICES"].sum()
            crec = ((ap_max - ap_min) / ap_min.replace(0,1) * 100).dropna().sort_values(ascending=False)
            mun_crecimiento = [(m, round(float(v))) for m, v in crec.head(5).items() if v > 0]

        if mun_con_tecnologo:
            st.markdown(
                f'<div style="background:#EAF3DE;border:.5px solid #C0DD97;'
                f'border-radius:8px;padding:9px 12px;font-size:11px;color:#3B6D11;margin-bottom:7px">'
                f'<b>Con Tecnólogo ({len(mun_con_tecnologo)}):</b> '
                f'{", ".join(mun_con_tecnologo[:5])}{'…' if len(mun_con_tecnologo)>5 else ""}.</div>',
                unsafe_allow_html=True,
            )
        if mun_sin_tecnologo:
            st.markdown(
                f'<div style="background:#FAEEDA;border:.5px solid #FAC775;'
                f'border-radius:8px;padding:9px 12px;font-size:11px;color:#854F0B;margin-bottom:7px">'
                f'<b>Sin Tecnólogo aún ({len(mun_sin_tecnologo)}):</b> '
                f'{", ".join(mun_sin_tecnologo[:6])}{'…' if len(mun_sin_tecnologo)>6 else ""}. '
                f'Candidatos a escalar si la demanda laboral lo justifica.</div>',
                unsafe_allow_html=True,
            )
        if mun_crecimiento:
            crec_txt = " · ".join(
                f"{m} ({'+' if v>=0 else ''}{v}%)" for m, v in mun_crecimiento
            )
            st.markdown(
                f'<div style="background:#E6F1FB;border:.5px solid #B5D4F4;'
                f'border-radius:8px;padding:9px 12px;font-size:11px;color:#185FA5;margin-bottom:7px">'
                f'<b>Mayor crecimiento {min(años_sel)}→{max(años_sel)}:</b> {crec_txt}. '
                f'Priorizar apertura de fichas.</div>',
                unsafe_allow_html=True,
            )
        if rubro_sel != "Todos los rubros":
            mun_activos = df_f["NOMBRE_MUNICIPIO_CURSO"].dropna().unique().tolist()
            todos_22 = list(mun_stats.keys())
            mun_sin_rubro = [m for m in todos_22 if m not in mun_activos][:5]
            if mun_sin_rubro:
                st.markdown(
                    f'<div style="background:#FCEBEB;border:.5px solid #F7C1C1;'
                    f'border-radius:8px;padding:9px 12px;font-size:11px;color:#A32D2D;margin-bottom:7px">'
                    f'<b>Sin cobertura en {rubro_sel}:</b> '
                    f'{", ".join(mun_sin_rubro)}. '
                    f'Oportunidad para nuevas fichas complementarias.</div>',
                    unsafe_allow_html=True,
                )

    with col_b2:
        st.markdown("**Tabla resumen por municipio**")
        resumen = (
            df_f.groupby(["NOMBRE_MUNICIPIO_CURSO","NIVEL_FORMACION"])["TOTAL_APRENDICES"]
            .sum()
            .reset_index()
            .pivot_table(
                index="NOMBRE_MUNICIPIO_CURSO",
                columns="NIVEL_FORMACION",
                values="TOTAL_APRENDICES",
                fill_value=0,
            )
            .reset_index()
        )
        resumen.columns.name = None
        resumen["Total"] = resumen.drop(columns=["NOMBRE_MUNICIPIO_CURSO"]).sum(axis=1)
        resumen = resumen.sort_values("Total", ascending=False)
        resumen = resumen.rename(columns={
            "NOMBRE_MUNICIPIO_CURSO": "Municipio",
            "CURSO ESPECIAL":         "Complementario",
        })
        # Redondear todos los numéricos
        for col in resumen.columns:
            if col != "Municipio":
                resumen[col] = resumen[col].astype(int)

        st.dataframe(
            resumen,
            use_container_width=True,
            hide_index=True,
            height=300,
            column_config={
                "Complementario": st.column_config.NumberColumn(format="%d"),
                "TÉCNICO":        st.column_config.NumberColumn("Técnico",    format="%d"),
                "TECNÓLOGO":      st.column_config.NumberColumn("Tecnólogo",  format="%d"),
                "EVENTO":         st.column_config.NumberColumn("Evento",     format="%d"),
                "Total":          st.column_config.NumberColumn(format="%d"),
            },
        )
