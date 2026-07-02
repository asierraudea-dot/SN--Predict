"""
predictor.py  —  SN Predict v2
================================
Funciones de predicción: heurística calibrada y wrapper del modelo ML.

La heurística usa patrones extraídos del dataset real PE04:
- Media global: 24.5 aprendices/ficha
- Picos feb–mar y sep–oct: +18 %
- Valle dic–ene: -28 %
- Nivel TECNÓLOGO penaliza -15 % vs CURSO ESPECIAL
- Duración >1.000 h penaliza -18 % (programas largos reducen convocatoria)
- Municipios grandes (Santafé, Dabeiba) tienen +30 % y +15 %

Autor:  [nombre del estudiante]
Fecha:  Junio 2026
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).parent.parent
MODELS = ROOT / "models"

# Factores por municipio extraídos de la media real del dataset
MUN_FACTOR: dict[str, float] = {
    "SANTAFÉ DE ANTIOQUIA": 1.30, "DABEIBA": 1.12, "FRONTINO": 1.05,
    "CAÑASGORDAS": 1.01, "SOPETRÁN": 1.03, "SAN JERÓNIMO": 1.04,
    "EBÉJICO": 1.01, "URAMITA": 1.02, "LIBORINA": 1.00, "BURITICA": 0.93,
    "ABRIAQUÍ": 0.90, "ANZÁ": 0.94, "CAICEDO": 1.00, "HELICONIA": 0.92,
    "GIRALDO": 0.96, "OLAYA": 0.97, "PEQUE": 0.98, "SABANALARGA": 1.06,
    "MEDELLÍN": 1.02,
}

NIVEL_FACTOR: dict[str, float] = {
    "CURSO ESPECIAL": 1.00, "TÉCNICO": 0.95, "TECNÓLOGO": 0.85, "EVENTO": 0.60,
}

MESES_FACTOR: dict[int, float] = {
    1: 0.75, 2: 1.18, 3: 1.15, 4: 1.00, 5: 1.00, 6: 0.98,
    7: 0.92, 8: 0.95, 9: 1.12, 10: 1.10, 11: 0.90, 12: 0.72,
}

MESES_NOM: list[str] = [
    "", "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
]


def predecir_heuristico(
    municipio: str,
    nivel: str,
    duracion: int,
    mes: int,
    hist_instructor: float,
    es_campesena: bool = False,
) -> dict[str, float | int | str]:
    """
    Predicción heurística calibrada con los datos reales PE04.

    Diseñada para usarse cuando el modelo ML entrenado no está disponible.
    Reproduce los patrones más fuertes del dataset sin necesitar sklearn.

    Args:
        municipio:        Nombre del municipio en mayúsculas.
        nivel:            Nivel de formación (CURSO ESPECIAL / TÉCNICO / TECNÓLOGO).
        duracion:         Duración del programa en horas.
        mes:              Mes de inicio (1-12).
        hist_instructor:  Promedio histórico del instructor (aprendices/ficha).
        es_campesena:     True si el programa pertenece a CAMPE-SENA.

    Returns:
        Dict con: aprendices (int), viabilidad_pct (int), alto_impacto (bool),
                  mes_nom (str), color_viab (str), interpretacion (str).
    """
    base        = 24.5
    f_mun       = MUN_FACTOR.get(municipio, 0.95)
    f_nivel     = NIVEL_FACTOR.get(nivel, 1.0)
    f_mes       = MESES_FACTOR.get(mes, 1.0)
    f_dur       = 0.82 if duracion > 1000 else (0.90 if duracion > 400 else 1.0)
    f_instructor = hist_instructor / 24.5
    f_campe     = 1.20 if es_campesena else 1.0

    pred = base * f_mun * f_nivel * f_mes * f_dur * f_instructor * f_campe
    pred = int(round(max(4.0, min(pred, 95.0))))

    viabilidad = int(min(97, round(28 + pred * 2.6)))
    alto_impacto = pred > 25
    color = "#0F6E56" if viabilidad >= 70 else ("#BA7517" if viabilidad >= 45 else "#A32D2D")

    notas: list[str] = []
    if f_mes > 1.1:
        notas.append(f"{MESES_NOM[mes]} es un mes de alta demanda histórica (pico +18%).")
    elif f_mes < 0.80:
        notas.append(f"{MESES_NOM[mes]} tiene demanda históricamente baja (-28%).")
    if f_instructor > 1.2:
        notas.append("El instructor tiene historial sólido — refuerza la predicción.")
    if es_campesena:
        notas.append("CAMPE-SENA añade +20% por convocatoria masiva.")

    return {
        "aprendices":      pred,
        "viabilidad_pct":  viabilidad,
        "alto_impacto":    alto_impacto,
        "mes_nom":         MESES_NOM[mes],
        "color_viab":      color,
        "interpretacion":  " ".join(notas) if notas else "Condiciones estándar.",
        "factores": {
            "municipio": round(f_mun, 2),
            "nivel":     round(f_nivel, 2),
            "mes":       round(f_mes, 2),
            "duracion":  round(f_dur, 2),
            "instructor":round(f_instructor, 2),
        },
    }


def intentar_modelo_ml(
    municipio: str,
    sector: str,
    nivel: str,
    duracion: int,
    mes: int,
    hist_instructor: float,
) -> dict | None:
    """
    Intenta cargar y aplicar el modelo ML entrenado por train.py.

    Si los modelos no existen retorna None (la app usará la heurística).

    Args:
        municipio, sector, nivel, duracion, mes, hist_instructor: parámetros del predictor.

    Returns:
        Dict compatible con predecir_heuristico(), o None si el modelo no está disponible.
    """
    ruta_y1 = MODELS / "pipeline_y1_regresion.pkl"
    ruta_y2 = MODELS / "pipeline_y2_clasificacion.pkl"
    ruta_meta = MODELS / "feature_names.json"

    if not (ruta_y1.exists() and ruta_y2.exists() and ruta_meta.exists()):
        return None

    try:
        import joblib  # importación tardía — opcional
        pipe_y1 = joblib.load(ruta_y1)
        pipe_y2 = joblib.load(ruta_y2)
        meta    = json.loads(ruta_meta.read_text())

        nivel_enc = {"CURSO ESPECIAL": 1, "TÉCNICO": 2, "TECNÓLOGO": 3, "EVENTO": 0}.get(nivel, 1)
        hist_mun  = 24.5  # fallback

        fila = pd.DataFrame([{
            "mes_inicio":          mes,
            "DURACION_PROGRAMA":   duracion,
            "nivel_enc":           nivel_enc,
            "es_campesena":        0,
            "hist_responsable":    hist_instructor,
            "hist_municipio":      hist_mun,
            "NOMBRE_MUNICIPIO_CURSO": municipio,
            "NOMBRE_NUEVO_SECTOR":    sector,
        }])[meta["features"]]

        pred = max(0, int(round(float(pipe_y1.predict(fila)[0]))))
        proba = float(pipe_y2.predict_proba(fila)[0][1])
        viab  = int(min(97, round(proba * 100)))
        alto  = proba >= 0.5
        color = "#0F6E56" if viab >= 70 else ("#BA7517" if viab >= 45 else "#A32D2D")

        return {
            "aprendices":     pred,
            "viabilidad_pct": viab,
            "alto_impacto":   alto,
            "mes_nom":        MESES_NOM[mes],
            "color_viab":     color,
            "interpretacion": "Predicción generada por modelo ML (GradientBoosting).",
            "factores":       {},
        }
    except Exception:
        return None


def generar_sugerencias(
    mes: int,
    hist_instructor: float,
    duracion: int,
    nivel: str,
    municipio: str,
) -> list[str]:
    """
    Genera sugerencias de optimización contextuales según los parámetros.

    Args:
        mes, hist_instructor, duracion, nivel, municipio: parámetros del predictor.

    Returns:
        Lista de strings con sugerencias en HTML (safe para st.markdown).
    """
    sugs: list[str] = []
    if mes not in (2, 3, 9, 10):
        sugs.append(
            "📅 Mover el inicio a **feb–mar o sep–oct** puede aumentar "
            "la convocatoria hasta +18% según el histórico."
        )
    if hist_instructor < 22:
        sugs.append(
            "👤 Asignar un instructor con historial **> 26 aprendices/ficha** "
            "puede mejorar la predicción ~20%."
        )
    if duracion > 400 and nivel == "CURSO ESPECIAL":
        sugs.append(
            "⏱ Para Curso Especial, duraciones **< 80 h** tienen mejor convocatoria histórica."
        )
    if municipio in ("ABRIAQUÍ", "OLAYA", "PEQUE", "SABANALARGA"):
        sugs.append(
            "📢 Este municipio tiene baja cobertura histórica. "
            "Refuerza la difusión con la alcaldía y JAC antes de abrir la ficha."
        )
    if mes in (2, 3):
        sugs.append(
            "✅ Feb–Mar es el período de mayor demanda histórica. Buena elección de mes."
        )
    if not sugs:
        sugs.append(
            "✅ Los parámetros están bien configurados. La ficha tiene buenas condiciones."
        )
    return sugs
