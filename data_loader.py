"""
data_loader.py  —  SN Predict v2
=================================
Carga, limpia y preprocesa el dataset PE04 Histórico Previos.
Expone funciones que devuelven estructuras listas para consumir en la app.

Todas las funciones están decoradas con @st.cache_data cuando se llaman
desde la app, por lo que solo se ejecutan una vez por sesión.

Autor:  [nombre del estudiante]
Fecha:  Junio 2026
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "PE04_HISTORICO_PREVIOS.xlsx"

# Coordenadas de los 22 municipios del Centro Occidente de Antioquia
MUN_COORDS: dict[str, tuple[float, float]] = {
    "SANTAFÉ DE ANTIOQUIA": (6.5566, -75.8238),
    "DABEIBA":              (7.0091, -76.2581),
    "FRONTINO":             (6.7854, -76.1358),
    "CAÑASGORDAS":          (6.7393, -75.9944),
    "SOPETRÁN":             (6.5058, -75.7383),
    "SAN JERÓNIMO":         (6.4827, -75.7173),
    "EBÉJICO":              (6.3328, -75.7449),
    "URAMITA":              (6.9108, -76.1642),
    "LIBORINA":             (6.6944, -75.9336),
    "BURITICA":             (6.7297, -75.9045),
    "ABRIAQUÍ":             (6.6384, -76.0946),
    "ANZÁ":                 (6.3185, -75.8800),
    "CAICEDO":              (6.4141, -76.0204),
    "HELICONIA":            (6.2066, -75.7487),
    "GIRALDO":              (6.6207, -75.8819),
    "OLAYA":                (6.5920, -75.8555),
    "PEQUE":                (6.9890, -75.9994),
    "SABANALARGA":          (6.8878, -75.7083),
    "ARMENIA":              (5.0893, -75.6718),
    "ITAGUÍ":               (6.1849, -75.5991),
    "LA CEJA":              (6.0218, -75.4358),
    "MEDELLÍN":             (6.2518, -75.5636),
}

# Palabras clave para identificar rubros productivos en nombres de programas
RUBROS_KW: dict[str, list[str]] = {
    "Café":           ["cafe", "café", "cafeto", "beneficio del cafe"],
    "Cacao":          ["cacao"],
    "Aguacate":       ["aguacate"],
    "Ganadería":      ["ganader", "bovino", "vacuno", "pecuari"],
    "Avicultura":     ["avicul", "gallina", "pollo", "ponedora"],
    "Turismo":        ["turismo", "ecoturismo", "aviturism"],
    "Agricultura":    ["agric", "cultivo", "sembr", "vegetal", "agropecuar"],
    "Piscicultura":   ["piscic", "tilapia", "trucha", "acuic", "cachama"],
    "Alturas":        ["altura"],
    "Inglés":         ["ingles", "english"],
    "Emprendimiento": ["emprend", "negocio", "empresar", "innovad"],
    "Alimentos":      ["alimento", "higien", "manipul", "cocina", "gastro"],
    "TIC / Excel":    ["excel", "ofimati", "sistemas", "software", "inform", "word"],
    "Medio Ambiente": ["ambient", "ecolog", "conserv", "recurso natur"],
    "Artesanías":     ["bisuter", "artesanal", "tejid", "mostacilla", "chaquira"],
}


def cargar_df() -> pd.DataFrame:
    """
    Carga el archivo PE04 y aplica limpieza estructural.

    Returns:
        DataFrame limpio con tipos corregidos y columnas auxiliares.

    Raises:
        FileNotFoundError: Si el XLSX no existe en data/.
    """
    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {DATA_RAW}\n"
            "Copia PE04_HISTORICO_PREVIOS.xlsx a la carpeta data/"
        )
    df = pd.read_excel(DATA_RAW, sheet_name="Hoja1")
    df.columns = [c.strip() for c in df.columns]

    # Fechas
    for col in ["FECHA_INICIO_FICHA", "FECHA_TERMINACION_FICHA"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Numéricos
    df["TOTAL_APRENDICES"] = pd.to_numeric(df["TOTAL_APRENDICES"], errors="coerce").fillna(0)
    df["DURACION_PROGRAMA"] = pd.to_numeric(df["DURACION_PROGRAMA"], errors="coerce").fillna(0)
    df["AÑO"]               = pd.to_numeric(df["AÑO"], errors="coerce")

    # Auxiliares
    df["MES"]     = df["FECHA_INICIO_FICHA"].dt.month
    df["TRIMESTRE"] = df["FECHA_INICIO_FICHA"].dt.quarter

    # Texto limpio
    texto_cols = [
        "NOMBRE_MUNICIPIO_CURSO", "NOMBRE_PROGRAMA_FORMACION",
        "NOMBRE_RESPONSABLE", "NIVEL_FORMACION",
        "NOMBRE_NUEVO_SECTOR", "NOMBRE_PROGRAMA_ESPECIAL",
    ]
    for col in texto_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().replace("NAN", pd.NA)

    # Variable objetivo binaria
    df["ALTO_IMPACTO"] = (df["TOTAL_APRENDICES"] > 25).astype(int)

    return df


def get_sec_prog(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Devuelve un diccionario sector → lista de programas con estadísticas.

    Cada entrada de la lista contiene: prog, nivel, fichas, prom, dur, ap.

    Args:
        df: DataFrame limpio.

    Returns:
        Dict con 12 sectores y hasta 10 programas por sector.
    """
    out: dict[str, list[dict]] = {}
    for sec in df["NOMBRE_NUEVO_SECTOR"].dropna().unique():
        sub = df[df["NOMBRE_NUEVO_SECTOR"] == sec]
        top = (
            sub.groupby(["NOMBRE_PROGRAMA_FORMACION", "NIVEL_FORMACION"])
            .agg(
                fichas=("TOTAL_APRENDICES", "count"),
                prom=("TOTAL_APRENDICES", "mean"),
                dur=("DURACION_PROGRAMA", "mean"),
                ap=("TOTAL_APRENDICES", "sum"),
            )
            .reset_index()
            .sort_values("fichas", ascending=False)
            .head(10)
        )
        out[sec] = [
            {
                "prog":   r["NOMBRE_PROGRAMA_FORMACION"][:70],
                "nivel":  r["NIVEL_FORMACION"],
                "fichas": int(r["fichas"]),
                "prom":   round(float(r["prom"]), 1),
                "dur":    int(round(float(r["dur"]))),
                "ap":     int(r["ap"]),
            }
            for _, r in top.iterrows()
        ]
    return out


def get_prog_mun(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Devuelve un diccionario programa → municipios donde se ha ofertado.

    Args:
        df: DataFrame limpio.

    Returns:
        Dict con programa (truncado a 70 chars) → lista de municipios con stats.
    """
    # Solo programas con al menos 3 fichas
    progs_validos = (
        df.groupby("NOMBRE_PROGRAMA_FORMACION")
        .size()
        .reset_index(name="n")
        .query("n >= 3")["NOMBRE_PROGRAMA_FORMACION"]
        .tolist()
    )
    out: dict[str, list[dict]] = {}
    for prog in progs_validos:
        sub = df[df["NOMBRE_PROGRAMA_FORMACION"] == prog]
        muns = (
            sub.groupby("NOMBRE_MUNICIPIO_CURSO")
            .agg(ap=("TOTAL_APRENDICES", "sum"), fichas=("TOTAL_APRENDICES", "count"),
                 prom=("TOTAL_APRENDICES", "mean"))
            .reset_index()
            .sort_values("ap", ascending=False)
            .head(8)
        )
        out[prog[:70]] = [
            {"mun": r["NOMBRE_MUNICIPIO_CURSO"], "ap": int(r["ap"]),
             "fichas": int(r["fichas"]), "prom": round(float(r["prom"]), 1)}
            for _, r in muns.iterrows()
        ]
    return out


def get_resp_stats(df: pd.DataFrame) -> list[dict]:
    """
    Devuelve estadísticas por instructor/responsable.

    Args:
        df: DataFrame limpio.

    Returns:
        Lista de dicts con n, fichas, prom, total, max_ap — ordenada por fichas desc.
    """
    resp = (
        df.groupby("NOMBRE_RESPONSABLE")
        .agg(fichas=("TOTAL_APRENDICES", "count"), total=("TOTAL_APRENDICES", "sum"),
             prom=("TOTAL_APRENDICES", "mean"), max_ap=("TOTAL_APRENDICES", "max"))
        .reset_index()
        .sort_values("fichas", ascending=False)
        .head(25)
    )
    return [
        {"n": r["NOMBRE_RESPONSABLE"], "fichas": int(r["fichas"]),
         "prom": round(float(r["prom"]), 1), "total": int(r["total"]), "max_ap": int(r["max_ap"])}
        for _, r in resp.iterrows()
    ]


def get_mun_stats(df: pd.DataFrame) -> dict[str, dict]:
    """
    Devuelve estadísticas por municipio incluyendo tendencia interanual.

    Args:
        df: DataFrame limpio.

    Returns:
        Dict municipio → dict con ap, fichas, prom, sector, tend (% cambio 2022→2024),
        lat, lng.
    """
    out: dict[str, dict] = {}
    for mun in df["NOMBRE_MUNICIPIO_CURSO"].dropna().unique():
        sub = df[df["NOMBRE_MUNICIPIO_CURSO"] == mun]
        sec = (
            sub.groupby("NOMBRE_NUEVO_SECTOR")["TOTAL_APRENDICES"]
            .count().idxmax()
            if len(sub) > 0 else "SERVICIOS"
        )
        a22 = float(sub[sub["AÑO"] == 2022]["TOTAL_APRENDICES"].sum())
        a24 = float(sub[sub["AÑO"] == 2024]["TOTAL_APRENDICES"].sum())
        tend = round((a24 - a22) / max(a22, 1) * 100, 1) if a22 > 0 else 0.0
        lat, lng = MUN_COORDS.get(mun, (6.5, -75.9))
        out[mun] = {
            "ap":     int(sub["TOTAL_APRENDICES"].sum()),
            "fichas": int(len(sub)),
            "prom":   round(float(sub["TOTAL_APRENDICES"].mean()), 1),
            "sector": sec,
            "tend":   tend,
            "lat":    lat,
            "lng":    lng,
        }
    return out


def get_niv_sec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla cruzada nivel × sector con fichas y aprendices.

    Args:
        df: DataFrame limpio.

    Returns:
        DataFrame con columnas: NIVEL_FORMACION, NOMBRE_NUEVO_SECTOR, fichas, ap.
    """
    return (
        df.groupby(["NIVEL_FORMACION", "NOMBRE_NUEVO_SECTOR"])
        .agg(fichas=("TOTAL_APRENDICES", "count"), ap=("TOTAL_APRENDICES", "sum"))
        .reset_index()
    )


def get_rubros(df: pd.DataFrame) -> dict[str, dict]:
    """
    Clasifica fichas por rubro productivo detectando palabras clave en el nombre del programa.

    Args:
        df: DataFrame limpio.

    Returns:
        Dict rubro → dict con fichas, ap total y top municipios.
    """
    out: dict[str, dict] = {}
    for rub, kws in RUBROS_KW.items():
        mask = df["NOMBRE_PROGRAMA_FORMACION"].str.lower().str.contains(
            "|".join(kws), na=False
        )
        sub = df[mask]
        if len(sub) == 0:
            continue
        muns = (
            sub.groupby("NOMBRE_MUNICIPIO_CURSO")["TOTAL_APRENDICES"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .to_dict()
        )
        out[rub] = {
            "fichas": int(len(sub)),
            "ap":     int(sub["TOTAL_APRENDICES"].sum()),
            "muns":   {k: int(v) for k, v in muns.items()},
        }
    return out


def get_serie_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serie mensual de aprendices y fichas para la sección de Tendencias.

    Args:
        df: DataFrame limpio.

    Returns:
        DataFrame indexado por ANO_MES con columnas: ap, fichas.
    """
    df2 = df.copy()
    df2["ANO_MES"] = df2["FECHA_INICIO_FICHA"].dt.to_period("M").astype(str)
    return (
        df2.groupby("ANO_MES")
        .agg(ap=("TOTAL_APRENDICES", "sum"), fichas=("TOTAL_APRENDICES", "count"))
        .reset_index()
        .sort_values("ANO_MES")
    )
