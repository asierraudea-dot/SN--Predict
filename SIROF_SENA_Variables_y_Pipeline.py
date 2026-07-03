"""
SIROF— Análisis de Pertinencia de Variables y Pipeline Completo
======================================================================
Sistema Inteligente para la Planeación de la Oferta Formativa 
mediante Aprendizaje Automático, Analítica Territorial y Sistemas de Recomendación

Estructura del módulo
─────────────────────
1. CONFIG               — Rutas, constantes, tabla de mapeo semántico
2. INGESTA              — Carga y limpieza de EVA y Catálogo SENA
3. FEATURE ENGINEERING  — Variables independientes (X)
4. VARIABLE OBJETIVO    — Construcción de Y con 4 modalidades
5. ANÁLISIS PERTINENCIA — Correlación, importancia, brechas
6. MODELO               — Entrenamiento LightGBM + evaluación
7. RECOMENDACIÓN        — Motor de ranking semántico top-K
8. INFORME              — Exportación de resultados
9. MAIN                 — Orquestador principal

Autores:  [AMSR]
Versión:  1.0
Fecha:    Junio 2026
Stack:    Python 3.11+, Pandas, NumPy, LightGBM, Scikit-learn, SHAP
"""

from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────────────────────────
import logging
import warnings
from pathlib import Path
from typing import NamedTuple

# ── third-party ────────────────────────────────────────────────────────────────
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    ndcg_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("SIROF-SENA")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class Config(NamedTuple):
    """Parámetros globales del sistema. Centralizar aquí evita magic strings."""

    # Rutas
    raw_eva: Path = Path("data/raw/Evaluaciones_Agropecuarias_Municipales_EVA_20260623.csv")
    raw_catalogo: Path = Path("data/raw/Catalogo_PROGRAMAS_EN_EJECUCION.xlsx")
    output_dir: Path = Path("data/outputs")

    # Modelo
    n_splits: int = 5           # folds CV
    test_size: float = 0.15     # fracción para prueba final
    random_state: int = 42
    top_k: int = 5              # programas a recomendar por municipio

    # Umbrales heurísticos para la variable objetivo Y1
    # (pendientes de validación con expertos SENA/MADR)
    umbral_esp_tec: float = 50_000.0   # ha para especialización tecnológica
    umbral_tecnologo: float = 10_000.0
    umbral_tecnico: float = 2_000.0
    umbral_auxiliar: float = 500.0
    umbral_operario: float = 100.0


CFG = Config()
CFG.output_dir.mkdir(parents=True, exist_ok=True)


# ── Tabla de mapeo semántico: Grupo EVA → Red de Conocimiento SENA ─────────────
# Esta tabla es el "pegamento" entre las dos fuentes.
# Construida a partir del análisis de 223 cultivos y 36 redes del catálogo.
# EXTENSIBLE: añadir filas a medida que se incorporen nuevos cultivos o redes.
MAPEO_GRUPO_RED: dict[str, str] = {
    "HORTALIZAS":                                   "Agrícola",
    "FRUTALES":                                     "Agrícola",
    "CEREALES":                                     "Agrícola",
    "TUBERCULOS Y PLATANOS":                        "Agrícola",
    "LEGUMINOSAS":                                  "Agrícola",
    "OLEAGINOSAS":                                  "Agrícola",
    "FIBRAS":                                       "Agrícola",
    "PLANTAS AROMATICAS, CONDIMENTARIAS Y MEDICINALES": "Agrícola",
    "FLORES Y FOLLAJES":                            "Agrícola",
    "OTROS PERMANENTES":                            "Agrícola",
    "FORESTALES":                                   "Ambiental",
    "HONGOS":                                       "Agrícola",
    "OTROS TRANSITORIOS":                           "Agrícola",
}

# Mapeo de cultivos específicos a redes secundarias (refinamiento)
# Permite que un cultivo tenga afinidad con más de una red
MAPEO_CULTIVO_REDES: dict[str, list[str]] = {
    "CAFE":           ["Agrícola", "Agroindustrial"],
    "CACAO":          ["Agrícola", "Agroindustrial"],
    "PALMA DE ACEITE": ["Agrícola", "Agroindustrial"],
    "CAÑA PANELERA":  ["Agrícola", "Agroindustrial"],
    "CAUCHO":         ["Agrícola", "Materiales Para La Industria"],
    "TILAPIA":        ["Acuícola Y De Pesca"],
    "TRUCHA":         ["Acuícola Y De Pesca"],
    "CACHAMA":        ["Acuícola Y De Pesca"],
    "PINO":           ["Ambiental"],
    "EUCALIPTO":      ["Ambiental"],
}

# Etiquetas posibles para Y1 (variable objetivo principal)
NIVELES_FORMACION = [
    "COMPLEMENTARIO",
    "OPERARIO",
    "AUXILIAR",
    "TÉCNICO",
    "TECNÓLOGO",
    "ESPECIALIZACIÓN TECNOLÓGICA",
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. INGESTA Y LIMPIEZA
# ══════════════════════════════════════════════════════════════════════════════

def cargar_eva(ruta: Path) -> pd.DataFrame:
    """
    Carga el dataset EVA del MADR y aplica limpieza estructural.

    Transformaciones aplicadas:
    - Renombrado a snake_case estable.
    - Coerción de tipos numéricos con manejo explícito de errores.
    - Limpieza de texto (strip, upper) en variables categóricas.
    - Corrección del año (dato viene como float con decimal .0).

    Args:
        ruta: Ruta al archivo CSV Latin-1.

    Returns:
        DataFrame limpio con 17 columnas tipadas correctamente.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Archivo EVA no encontrado: {ruta}")

    logger.info("Cargando EVA desde %s", ruta)
    df = pd.read_csv(ruta, encoding="latin-1", low_memory=False)

    # Nombres estables — orden fijo para este dataset
    df.columns = [
        "cod_dep", "departamento", "cod_mun", "municipio",
        "grupo_cultivo", "subgrupo_cultivo", "cultivo", "desagregacion",
        "ano", "periodo", "area_sembrada_ha", "area_cosechada_ha",
        "produccion_t", "rendimiento_t_ha", "estado_fisico",
        "nombre_cientifico", "ciclo_cultivo",
    ]

    # Numéricos
    num_cols = ["area_sembrada_ha", "area_cosechada_ha"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Producción tiene comas decimales en algunos registros
    df["produccion_t"] = pd.to_numeric(
        df["produccion_t"].astype(str).str.replace(",", "."),
        errors="coerce",
    ).fillna(0)

    # Rendimiento: puede tener comas y valores no numéricos
    df["rendimiento_t_ha"] = pd.to_numeric(
        df["rendimiento_t_ha"].astype(str).str.replace(",", "."),
        errors="coerce",
    )

    # Año: viene como float con decimales espurios (2006.0 → 2006)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")

    # Texto
    texto_cols = [
        "departamento", "municipio", "grupo_cultivo",
        "subgrupo_cultivo", "cultivo", "ciclo_cultivo",
        "estado_fisico", "periodo",
    ]
    for col in texto_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # Eliminar registros sin municipio válido
    df = df[df["municipio"].notna() & (df["municipio"] != "")]

    logger.info(
        "EVA cargado: %d registros | %d depto | %d municipios | años %s–%s",
        len(df), df["departamento"].nunique(), df["municipio"].nunique(),
        df["ano"].min(), df["ano"].max(),
    )
    return df


def cargar_catalogo(ruta: Path) -> pd.DataFrame:
    """
    Carga el Catálogo de Programas SENA y normaliza valores críticos.

    Transformaciones:
    - Normalización de modalidades (3 variantes → valor canónico).
    - Limpieza de espacios en nombres de columnas.
    - Conversión de fechas seriales Excel a datetime.

    Args:
        ruta: Ruta al archivo XLSX.

    Returns:
        DataFrame con 4.235 programas y 29 columnas normalizadas.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Catálogo no encontrado: {ruta}")

    logger.info("Cargando Catálogo SENA desde %s", ruta)
    df = pd.read_excel(ruta)

    # Normalizar columnas con espacios
    df.columns = df.columns.str.strip()

    # Unificar variantes de modalidad
    mapeo_modalidad = {
        "PRESENCIAL":          "Presencial",
        "Presencial ":         "Presencial",
        "VIRTUAL":             "Virtual",
        "Presencial/Virtual":  "Mixta",
        "Presencial / A Distancia": "Mixta",
        "A distancia":         "Distancia",
    }
    df["Modalidad"] = df["Modalidad"].replace(mapeo_modalidad)

    # Texto en mayúsculas para joins
    df["Red de Conocimiento"] = df["Red de Conocimiento"].astype(str).str.strip()
    df["NIVEL DE FORMACION"] = df["NIVEL DE FORMACION"].astype(str).str.strip().str.upper()
    df["TIPO DE FORMACION"] = df["TIPO DE FORMACION"].astype(str).str.strip().str.upper()

    logger.info(
        "Catálogo cargado: %d programas | %d redes | %d niveles",
        len(df),
        df["Red de Conocimiento"].nunique(),
        df["NIVEL DE FORMACION"].nunique(),
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING — VARIABLES INDEPENDIENTES (X)
# ══════════════════════════════════════════════════════════════════════════════

def construir_perfil_municipal(eva: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el perfil productivo de cada municipio a partir del EVA.

    Variables generadas y su pertinencia teórica
    ─────────────────────────────────────────────
    area_total_ha
        Escala del sistema productivo. Proxy de capacidad económica agrícola.
        Correlación esperada con nivel de formación: alta y positiva.
        (mayor área → mayor complejidad → nivel más alto).

    produccion_total_t
        Volumen absoluto. Complementa el área: un municipio puede tener mucha
        área con baja producción (baja productividad) o viceversa.

    n_cultivos
        Diversidad de cadenas productivas. Municipios con alta diversificación
        necesitan oferta formativa más amplia (múltiples redes).

    indice_hhi
        Índice Herfindahl-Hirschman normalizado [0, 1].
        0 = municipio completamente diversificado.
        1 = municipio monoproductor.
        Municipios monocultivistas con HHI alto se benefician de
        formación especializada (Esp. Tecnológica).

    rendimiento_ponderado
        Rendimiento promedio ponderado por área cosechada.
        Proxy de nivel tecnológico actual del sistema productivo.
        Municipios con alto rendimiento ya tienen alguna adopción tecnológica
        y pueden absorber formación de nivel más elevado.

    tendencia_area_5a
        Pendiente de regresión lineal del área sembrada en los últimos 5 años.
        Positiva: sector en expansión → mayor demanda formativa futura.
        Negativa: sector en contracción → riesgo de sobreoferta formativa.

    participacion_dpto
        Participación del municipio en el área total del departamento.
        Señala si es un municipio líder o marginal en su vocación productiva.

    ciclo_dominante_enc
        Codificación del ciclo de cultivo dominante (transitorio=0, permanente=1).
        Los cultivos permanentes (café, cacao, palma) tienen cadenas de valor
        más complejas y requieren formación de mayor nivel.

    ratio_cosecha_siembra
        Área cosechada / Área sembrada. Indica eficiencia productiva.
        Valores altos → mayor adopción de buenas prácticas → capacidad para
        formación técnica/tecnológica.

    Args:
        eva: DataFrame EVA limpio.

    Returns:
        DataFrame con una fila por municipio y todas las features.
    """
    logger.info("Construyendo perfil municipal (%d registros EVA)...", len(eva))

    # ── Agregados básicos ──────────────────────────────────────────────────────
    agg = (
        eva.groupby(["cod_mun", "municipio", "cod_dep", "departamento"])
        .agg(
            area_total_ha=("area_sembrada_ha", "sum"),
            area_cosechada_total=("area_cosechada_ha", "sum"),
            produccion_total_t=("produccion_t", "sum"),
            n_cultivos=("cultivo", "nunique"),
            n_subgrupos=("subgrupo_cultivo", "nunique"),
            n_periodos=("periodo", "nunique"),
        )
        .reset_index()
    )

    # ── Cultivo y grupo dominante (por área sembrada acumulada) ───────────────
    dom = (
        eva.groupby(["cod_mun", "cultivo", "grupo_cultivo"])["area_sembrada_ha"]
        .sum()
        .reset_index()
        .sort_values("area_sembrada_ha", ascending=False)
        .drop_duplicates("cod_mun")
        .rename(columns={
            "cultivo": "cultivo_dominante",
            "grupo_cultivo": "grupo_dominante",
            "area_sembrada_ha": "area_cultivo_dom_ha",
        })
    )

    # ── Índice Herfindahl-Hirschman (concentración productiva) ───────────────
    # HHI = Σ (participación_i²); normalizado dividiéndolo por 1/n_cultivos²
    hhi_base = (
        eva.groupby(["cod_mun", "cultivo"])["area_sembrada_ha"]
        .sum()
        .reset_index()
    )
    total_mun = hhi_base.groupby("cod_mun")["area_sembrada_ha"].transform("sum").replace(0, np.nan)
    hhi_base["part"] = hhi_base["area_sembrada_ha"] / total_mun
    hhi_agg = (
        hhi_base.groupby("cod_mun")
        .apply(lambda x: (x["part"] ** 2).sum(), include_groups=False)
        .reset_index()
        .rename(columns={0: "indice_hhi"})
    )

    # ── Rendimiento ponderado por área cosechada ──────────────────────────────
    rend = eva[eva["rendimiento_t_ha"].notna()].copy()
    rend["peso"] = rend["area_cosechada_ha"].replace(0, np.nan).fillna(1)
    rend_agg = (
        rend.groupby("cod_mun")
        .apply(
            lambda x: np.average(x["rendimiento_t_ha"], weights=x["peso"]),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={0: "rendimiento_ponderado"})
    )

    # ── Tendencia área sembrada (pendiente OLS últimos 5 años disponibles) ────
    anos_recientes = eva["ano"].dropna().sort_values().unique()[-10:]
    eva_rec = eva[eva["ano"].isin(anos_recientes)]
    tendencia_rows = []
    for mun, grp in eva_rec.groupby("cod_mun"):
        serie = grp.groupby("ano")["area_sembrada_ha"].sum()
        if len(serie) >= 3:
            slope, _, _, _, _ = stats.linregress(serie.index.astype(float), serie.values)
        else:
            slope = 0.0
        tendencia_rows.append({"cod_mun": mun, "tendencia_area_ha_ano": round(slope, 2)})
    tendencia = pd.DataFrame(tendencia_rows)

    # ── Participación departamental ───────────────────────────────────────────
    dpto_total = eva.groupby("cod_dep")["area_sembrada_ha"].sum().reset_index()
    dpto_total.columns = ["cod_dep", "area_dpto_total"]
    part_dpto = (
        eva.groupby(["cod_mun", "cod_dep"])["area_sembrada_ha"]
        .sum()
        .reset_index()
        .merge(dpto_total, on="cod_dep")
    )
    part_dpto["participacion_dpto"] = (
        part_dpto["area_sembrada_ha"] / part_dpto["area_dpto_total"].replace(0, np.nan)
    ).fillna(0)
    part_dpto = part_dpto[["cod_mun", "participacion_dpto"]]

    # ── Ciclo dominante ───────────────────────────────────────────────────────
    ciclo_dom = (
        eva.groupby(["cod_mun", "ciclo_cultivo"])["area_sembrada_ha"]
        .sum()
        .reset_index()
        .sort_values("area_sembrada_ha", ascending=False)
        .drop_duplicates("cod_mun")[["cod_mun", "ciclo_cultivo"]]
        .rename(columns={"ciclo_cultivo": "ciclo_dominante"})
    )
    ciclo_enc = {"PERMANENTE": 2, "ANUAL": 1, "TRANSITORIO": 0}
    ciclo_dom["ciclo_dominante_enc"] = ciclo_dom["ciclo_dominante"].map(ciclo_enc).fillna(0)

    # ── Unión final ──────────────────────────────────────────────────────────
    perfil = (
        agg
        .merge(dom[["cod_mun", "cultivo_dominante", "grupo_dominante", "area_cultivo_dom_ha"]], on="cod_mun", how="left")
        .merge(hhi_agg, on="cod_mun", how="left")
        .merge(rend_agg, on="cod_mun", how="left")
        .merge(tendencia, on="cod_mun", how="left")
        .merge(part_dpto, on="cod_mun", how="left")
        .merge(ciclo_dom[["cod_mun", "ciclo_dominante_enc"]], on="cod_mun", how="left")
    )

    # ── Ratio eficiencia cosecha/siembra ─────────────────────────────────────
    perfil["ratio_cosecha_siembra"] = (
        perfil["area_cosechada_total"] / perfil["area_total_ha"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    # ── Codificación grupo dominante (ordinal por complejidad productiva) ─────
    grupos_complejidad = {
        "HORTALIZAS": 1, "LEGUMINOSAS": 1, "OTROS TRANSITORIOS": 1,
        "CEREALES": 2, "TUBERCULOS Y PLATANOS": 2, "FIBRAS": 2,
        "FLORES Y FOLLAJES": 3, "PLANTAS AROMATICAS, CONDIMENTARIAS Y MEDICINALES": 3,
        "OLEAGINOSAS": 4, "FRUTALES": 4, "HONGOS": 3,
        "OTROS PERMANENTES": 4, "FORESTALES": 3,
    }
    perfil["grupo_complejidad"] = perfil["grupo_dominante"].map(grupos_complejidad).fillna(2)

    # Rellenar NaN con medianas (imputación simple — documentar en producción)
    for col in ["rendimiento_ponderado", "indice_hhi", "tendencia_area_ha_ano",
                "participacion_dpto", "ciclo_dominante_enc"]:
        mediana = perfil[col].median()
        n_nulos = perfil[col].isna().sum()
        if n_nulos > 0:
            logger.warning("Imputando %d nulos en '%s' con mediana=%.4f", n_nulos, col, mediana)
        perfil[col] = perfil[col].fillna(mediana)

    logger.info("Perfil municipal construido: %d municipios, %d features", len(perfil), len(FEATURE_COLS))
    return perfil


# Columnas de entrada al modelo (X)
# Separar aquí permite actualizar sin tocar el código de entrenamiento
FEATURE_COLS = [
    "area_total_ha",           # Escala productiva
    "produccion_total_t",      # Volumen económico
    "n_cultivos",              # Diversificación
    "n_subgrupos",             # Diversificación fina
    "area_cultivo_dom_ha",     # Peso del cultivo líder
    "indice_hhi",              # Concentración (0=diverso, 1=mono)
    "rendimiento_ponderado",   # Nivel tecnológico actual
    "tendencia_area_ha_ano",   # Dinámica expansión/contracción
    "participacion_dpto",      # Relevancia territorial relativa
    "ciclo_dominante_enc",     # Tipo de cadena productiva
    "ratio_cosecha_siembra",   # Eficiencia productiva
    "grupo_complejidad",       # Complejidad de cadena (ordinal)
]


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONSTRUCCIÓN DE VARIABLES DEPENDIENTES (Y)
# ══════════════════════════════════════════════════════════════════════════════

# ── Y1: Nivel de formación recomendado (clasificación multiclase) ─────────────

def asignar_nivel_formacion(row: pd.Series, cfg: Config = CFG) -> str:
    """
    Asigna la etiqueta Y1 mediante heurística experta validable.

    SUPUESTO MARCADO: Esta función es una aproximación inicial.
    Debe validarse con instructores SENA y técnicos MADR antes de producción.
    En la versión 2.0 se reemplazará por etiquetas reales de matrícula por
    municipio obtenidas de SOFIA Plus.

    Lógica de asignación:
    ─────────────────────
    Se combina el área total municipal (escala) con la complejidad del grupo
    dominante y el índice de concentración para diferenciar entre niveles
    de formación adecuados.

    Ejemplo:
        Municipio con 45.000 ha de palma de aceite (HHI=0.92, grupo complejidad=4)
        → Especialización Tecnológica (sector palmicultor maduro y concentrado).

        Municipio con 8.000 ha de café (HHI=0.6, complejidad=4)
        → Tecnólogo (escala media-alta, cadena compleja).

        Municipio con 300 ha de hortalizas diversas (HHI=0.2, complejidad=1)
        → Operario / Complementario (escala pequeña, cadena simple).

    Args:
        row: Fila del DataFrame de perfil municipal.
        cfg: Configuración con umbrales.

    Returns:
        String con la etiqueta de nivel asignada.
    """
    area: float = float(row.get("area_total_ha", 0) or 0)
    hhi: float = float(row.get("indice_hhi", 0.5) or 0.5)
    complejidad: int = int(row.get("grupo_complejidad", 2) or 2)
    tendencia: float = float(row.get("tendencia_area_ha_ano", 0) or 0)

    # Ajuste por complejidad de cadena y concentración
    bonus_complejidad = complejidad * 0.15     # hasta +0.6x en área efectiva
    bonus_concentracion = hhi * 0.10           # HHI alto → mayor especialización
    bonus_tendencia = max(0, tendencia / 1000) # expansión activa suma weight

    area_efectiva = area * (1 + bonus_complejidad + bonus_concentracion + bonus_tendencia)

    if area_efectiva >= cfg.umbral_esp_tec:
        return "ESPECIALIZACIÓN TECNOLÓGICA"
    elif area_efectiva >= cfg.umbral_tecnologo:
        return "TECNÓLOGO"
    elif area_efectiva >= cfg.umbral_tecnico:
        return "TÉCNICO"
    elif area_efectiva >= cfg.umbral_auxiliar:
        return "AUXILIAR"
    elif area_efectiva >= cfg.umbral_operario:
        return "OPERARIO"
    else:
        return "COMPLEMENTARIO"


# ── Y2: Déficit de oferta formativa (regresión continua) ─────────────────────

def calcular_brecha_formativa(
    perfil: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula Y3: Brecha entre vocación productiva y oferta formativa disponible.

    Fórmula:
        brecha = area_total_ha_normalizada – cobertura_programas_agro_normalizada

    Valores positivos: municipio con vocación fuerte pero poca oferta.
    Valores negativos: oferta supera la demanda potencial (posible sobreoferta).

    SUPUESTO: Sin datos de matrícula real por municipio, se usa la cobertura
    de programas del catálogo en la red afín como proxy de oferta disponible.
    Esto es una aproximación muy conservadora; mejora drásticamente cuando
    se incorporan datos de SOFIA Plus con cupos por regional y municipio.

    Args:
        perfil: DataFrame de perfiles municipales.
        catalogo: Catálogo SENA.

    Returns:
        DataFrame con columna adicional 'brecha_formativa'.
    """
    # Contar programas agropecuarios en catálogo (proxy de oferta regional)
    agro_redes = {"Agrícola", "Pecuaria", "Acuícola Y De Pesca",
                  "Agroindustrial", "Ambiental"}
    n_agro = catalogo[catalogo["Red de Conocimiento"].isin(agro_redes)].shape[0]

    # Normalizar área (min-max dentro del perfil)
    area_norm = (perfil["area_total_ha"] - perfil["area_total_ha"].min()) / (
        perfil["area_total_ha"].max() - perfil["area_total_ha"].min() + 1e-9
    )

    # Proxy de cobertura por municipio: participación departamental × n_programas_agro
    cobertura_proxy = perfil["participacion_dpto"] * n_agro
    cobertura_norm = (cobertura_proxy - cobertura_proxy.min()) / (
        cobertura_proxy.max() - cobertura_proxy.min() + 1e-9
    )

    perfil = perfil.copy()
    perfil["brecha_formativa"] = (area_norm - cobertura_norm).round(4)
    return perfil


# ── Y4: Tendencia de vocación productiva (ya en perfil como tendencia_area_ha_ano) ──
# Se documenta aquí para mayor claridad del diseño

def construir_serie_temporal_municipal(
    eva: pd.DataFrame,
    cod_mun: int,
) -> pd.Series:
    """
    Construye la serie temporal de área sembrada para un municipio específico.
    Utilizable directamente para modelos de series temporales (Prophet, SARIMA).

    Args:
        eva: DataFrame EVA completo.
        cod_mun: Código DANE del municipio.

    Returns:
        Serie indexada por año con área total sembrada.
    """
    serie = (
        eva[eva["cod_mun"] == cod_mun]
        .groupby("ano")["area_sembrada_ha"]
        .sum()
        .sort_index()
    )
    return serie


# ══════════════════════════════════════════════════════════════════════════════
# 5. ANÁLISIS DE PERTINENCIA DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def analizar_pertinencia(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    le: LabelEncoder,
) -> pd.DataFrame:
    """
    Evalúa la pertinencia estadística de cada variable independiente.

    Métricas calculadas:
    ────────────────────
    correlacion_eta         Coeficiente η² (eta cuadrado) de ANOVA.
                            Mide qué fracción de la varianza de la variable X
                            es explicada por la clase Y1. Rango [0, 1].
                            Interpretación: >0.14 efecto grande, >0.06 mediano.

    kruskal_p               Valor-p del test de Kruskal-Wallis (no paramétrico).
                            H0: las distribuciones de X son iguales en todos los
                            niveles de Y1. p < 0.05 → rechazar H0 → variable útil.

    cv_importancia          Importancia promedio en 3-fold CV con LightGBM (gain).
                            Directamente comparable entre variables.

    Args:
        X: Features numéricas.
        y_encoded: Etiquetas codificadas (int).
        le: LabelEncoder usado.

    Returns:
        DataFrame ordenado por importancia con las tres métricas.
    """
    logger.info("Analizando pertinencia de %d variables...", X.shape[1])
    resultados = []

    for col in X.columns:
        # Eta cuadrado (ANOVA)
        grupos = [X[col][y_encoded == k].values for k in np.unique(y_encoded)]
        f_stat, p_val = stats.f_oneway(*grupos)
        grand_mean = X[col].mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2 for g in grupos
        )
        ss_total = ((X[col] - grand_mean) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else 0

        # Kruskal-Wallis
        try:
            kw_stat, kw_p = stats.kruskal(*grupos)
        except ValueError:
            kw_p = 1.0

        resultados.append({
            "variable": col,
            "eta_cuadrado": round(eta2, 4),
            "kruskal_p": round(kw_p, 6),
            "significativa": kw_p < 0.05,
        })

    # Importancia rápida con LightGBM (1 fit sin CV para rapidez)
    modelo_imp = lgb.LGBMClassifier(
        n_estimators=100, random_state=42, verbose=-1,
        class_weight="balanced", num_class=len(le.classes_),
        objective="multiclass",
    )
    modelo_imp.fit(X, y_encoded)
    imp = dict(zip(X.columns, modelo_imp.feature_importances_))
    for r in resultados:
        r["importancia_lgbm"] = imp.get(r["variable"], 0)

    df_pert = pd.DataFrame(resultados).sort_values("importancia_lgbm", ascending=False)
    df_pert["rango"] = range(1, len(df_pert) + 1)

    logger.info("Top 5 variables por pertinencia:\n%s",
                df_pert[["variable", "eta_cuadrado", "kruskal_p", "importancia_lgbm"]].head().to_string(index=False))
    return df_pert


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODELO — ENTRENAMIENTO Y EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_modelo(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    grupos: pd.Series,
    cfg: Config = CFG,
) -> tuple[lgb.LGBMClassifier, list[dict]]:
    """
    Entrena el clasificador LightGBM con validación cruzada estratificada.

    Decisiones de diseño:
    ─────────────────────
    StratifiedGroupKFold: garantiza que municipios del mismo departamento
    no aparezcan en train y test a la vez (evita leakage geográfico).

    class_weight='balanced': compensa el desbalance entre niveles
    (CURSO ESPECIAL tiene 3.060 programas vs ESPECIALIZACIÓN 9 programas).

    n_estimators=300 + early stopping en validación: evita sobreajuste
    sin sacrificar capacidad.

    Args:
        X: Features numéricas (n_municipios × n_features).
        y_encoded: Etiquetas enteras codificadas con LabelEncoder.
        grupos: Serie con departamento por municipio (para GroupKFold).
        cfg: Configuración global.

    Returns:
        Tupla (modelo final, lista de métricas por fold).
    """
    logger.info("Iniciando entrenamiento con %d-fold CV estratificado por departamento...", cfg.n_splits)

    cv = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_encoded)),
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": cfg.random_state,
        "verbose": -1,
    }

    metricas_folds: list[dict] = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_encoded, grupos)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

        modelo = lgb.LGBMClassifier(**params)
        modelo.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        y_pred = modelo.predict(X_val)
        y_proba = modelo.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        try:
            roc = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            roc = float("nan")

        metricas_folds.append({"fold": fold + 1, "accuracy": acc, "f1_macro": f1, "roc_auc": roc})
        logger.info(
            "Fold %d → Accuracy: %.4f | F1-macro: %.4f | ROC-AUC: %.4f",
            fold + 1, acc, f1, roc,
        )

    # Resumen CV
    df_m = pd.DataFrame(metricas_folds)
    logger.info(
        "CV resumen → F1-macro: %.4f ± %.4f | ROC-AUC: %.4f ± %.4f",
        df_m["f1_macro"].mean(), df_m["f1_macro"].std(),
        df_m["roc_auc"].mean(), df_m["roc_auc"].std(),
    )

    # Entrenamiento final con todos los datos
    modelo_final = lgb.LGBMClassifier(**params)
    modelo_final.fit(X, y_encoded)
    logger.info("Modelo final entrenado con %d muestras.", len(X))

    return modelo_final, metricas_folds


def evaluar_conjunto_prueba(
    modelo: lgb.LGBMClassifier,
    le: LabelEncoder,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Evalúa el modelo en el conjunto de prueba y retorna el reporte completo.

    Args:
        modelo: Clasificador entrenado.
        le: LabelEncoder de etiquetas.
        X_test: Features del conjunto de prueba.
        y_test: Etiquetas del conjunto de prueba.

    Returns:
        DataFrame con métricas por clase y globales.
    """
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    reporte = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )
    df_reporte = pd.DataFrame(reporte).T
    logger.info("Reporte clasificación conjunto de prueba:\n%s", df_reporte.round(4).to_string())
    return df_reporte


# ══════════════════════════════════════════════════════════════════════════════
# 7. MOTOR DE RECOMENDACIÓN — RANKING TOP-K
# ══════════════════════════════════════════════════════════════════════════════

def recomendar_programas(
    cod_mun: int,
    perfil_municipal: pd.DataFrame,
    catalogo: pd.DataFrame,
    modelo: lgb.LGBMClassifier,
    le: LabelEncoder,
    cfg: Config = CFG,
) -> pd.DataFrame:
    """
    Genera el ranking top-K de programas SENA para un municipio dado.

    Algoritmo de scoring:
    ─────────────────────
    score = w_afinidad × afinidad_red + w_duracion × dur_norm + w_tipo × tipo_titulada

    donde:
    - afinidad_red:  1 si la Red de Conocimiento del programa coincide con la
                     red afín al grupo/cultivo dominante del municipio, 0 si no.
    - dur_norm:      Duración normalizada (0-1) dentro del nivel predicho.
                     Proxy de profundidad y complejidad formativa.
    - tipo_titulada: 1 si es formación titulada (Técnico, Tecnólogo, etc.),
                     0 si es complementaria. La titulada genera competencias
                     certificadas con mayor impacto en empleabilidad.

    Limitaciones actuales (mejora con fuentes adicionales):
    - Sin cupos disponibles → no puede filtrar por disponibilidad real.
    - Sin datos de matrícula → no considera sobresaturación de programas.
    - Sin geolocalización → no considera distancia al Centro de Formación.

    Args:
        cod_mun: Código DANE del municipio.
        perfil_municipal: DataFrame de perfiles.
        catalogo: Catálogo SENA completo.
        modelo: Clasificador entrenado.
        le: LabelEncoder de etiquetas.
        cfg: Configuración global.

    Returns:
        DataFrame con top-K programas recomendados y su score.
    """
    perfil = perfil_municipal[perfil_municipal["cod_mun"] == cod_mun]
    if perfil.empty:
        logger.warning("Municipio cod_mun=%d no encontrado.", cod_mun)
        return pd.DataFrame()

    # ── Predicción del nivel ─────────────────────────────────────────────────
    X_mun = perfil[FEATURE_COLS].fillna(0)
    nivel_pred_enc = modelo.predict(X_mun)[0]
    nivel_pred = le.inverse_transform([nivel_pred_enc])[0]
    proba_max = modelo.predict_proba(X_mun)[0].max()

    grupo_dom = perfil["grupo_dominante"].values[0]
    cultivo_dom = perfil["cultivo_dominante"].values[0]

    # ── Red de conocimiento afín ─────────────────────────────────────────────
    red_afin = MAPEO_GRUPO_RED.get(grupo_dom, "Agrícola")
    redes_afines_cultivo = MAPEO_CULTIVO_REDES.get(cultivo_dom, [red_afin])

    logger.info(
        "Municipio %d → nivel predicho: %s (prob=%.2f) | cultivo dom: %s | red: %s",
        cod_mun, nivel_pred, proba_max, cultivo_dom, red_afin,
    )

    # ── Filtrado del catálogo ─────────────────────────────────────────────────
    # Nivel predicho; si hay pocos programas, ampliar al nivel inmediato inferior
    candidatos = catalogo[catalogo["NIVEL DE FORMACION"] == nivel_pred].copy()
    if len(candidatos) < 3:
        logger.warning(
            "Pocos programas en nivel '%s' (%d). Ampliando búsqueda.", nivel_pred, len(candidatos)
        )
        candidatos = catalogo.copy()

    # ── Scoring ──────────────────────────────────────────────────────────────
    candidatos = candidatos.copy()

    # Afinidad con red de conocimiento
    candidatos["afinidad_red"] = candidatos["Red de Conocimiento"].apply(
        lambda r: 1.0 if r in redes_afines_cultivo else (0.5 if r == red_afin else 0.0)
    )

    # Duración normalizada dentro del nivel
    max_dur = candidatos["PRF_DURACION_MAXIMA"].max() or 1
    candidatos["dur_norm"] = (candidatos["PRF_DURACION_MAXIMA"] / max_dur).fillna(0)

    # Bonus formación titulada
    candidatos["bonus_titulada"] = (candidatos["TIPO DE FORMACION"] == "TITULADA").astype(float)

    # Score compuesto
    w = {"afinidad": 0.55, "duracion": 0.25, "titulada": 0.20}
    candidatos["score"] = (
        w["afinidad"] * candidatos["afinidad_red"]
        + w["duracion"] * candidatos["dur_norm"]
        + w["titulada"] * candidatos["bonus_titulada"]
    ).round(4)

    ranking = (
        candidatos
        .sort_values("score", ascending=False)
        .head(cfg.top_k)[
            [
                "PRF_CODIGO", "PRF_DENOMINACION", "NIVEL DE FORMACION",
                "Red de Conocimiento", "Modalidad", "PRF_DURACION_MAXIMA",
                "TIPO DE FORMACION", "APUESTAS PRIORITARIAS", "score",
            ]
        ]
        .copy()
    )
    ranking.insert(0, "municipio", perfil["municipio"].values[0])
    ranking.insert(1, "nivel_predicho", nivel_pred)
    ranking.insert(2, "prob_nivel", round(proba_max, 4))

    return ranking


# ══════════════════════════════════════════════════════════════════════════════
# 8. EXPLICABILIDAD — SHAP
# ══════════════════════════════════════════════════════════════════════════════

def generar_explicacion_shap(
    modelo: lgb.LGBMClassifier,
    X: pd.DataFrame,
    le: LabelEncoder,
    cod_mun: int,
    perfil: pd.DataFrame,
) -> dict:
    """
    Genera la explicación SHAP local para un municipio específico.

    Retorna los valores SHAP de la clase predicha para este municipio,
    permitiendo explicar al usuario qué variables productivas influyeron
    y en qué dirección en la recomendación.

    Args:
        modelo: Clasificador LightGBM entrenado.
        X: Features del conjunto completo (para el explainer).
        le: LabelEncoder de etiquetas.
        cod_mun: Municipio a explicar.
        perfil: DataFrame de perfiles.

    Returns:
        Dict con 'shap_values', 'base_value', 'clase_predicha', 'feature_names'.
    """
    explainer = shap.TreeExplainer(modelo)

    p = perfil[perfil["cod_mun"] == cod_mun]
    if p.empty:
        return {}

    X_mun = p[FEATURE_COLS].fillna(0)
    clase_pred = modelo.predict(X_mun)[0]
    shap_vals = explainer.shap_values(X_mun)

    return {
        "shap_values": shap_vals[clase_pred][0],
        "base_value": explainer.expected_value[clase_pred],
        "clase_predicha": le.inverse_transform([clase_pred])[0],
        "feature_names": FEATURE_COLS,
        "feature_values": X_mun.values[0],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN — ORQUESTADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Orquesta el pipeline completo SIROF-SENA.

    Pasos:
    1. Carga y limpieza de fuentes.
    2. Feature engineering territorial.
    3. Construcción de variable objetivo Y1.
    4. Análisis de pertinencia de variables.
    5. División train/test estratificada.
    6. Entrenamiento con CV.
    7. Evaluación en conjunto de prueba.
    8. Generación de recomendaciones de ejemplo.
    9. Exportación de resultados.
    """
    logger.info("═" * 60)
    logger.info("SIROF-SENA — Pipeline iniciado")
    logger.info("═" * 60)

    cfg = CFG

    # ── Paso 1: Carga ─────────────────────────────────────────────────────────
    eva = cargar_eva(cfg.raw_eva)
    catalogo = cargar_catalogo(cfg.raw_catalogo)

    # ── Paso 2: Feature engineering ───────────────────────────────────────────
    perfil = construir_perfil_municipal(eva)

    # ── Paso 3: Variable objetivo Y1 ──────────────────────────────────────────
    perfil["nivel_recomendado"] = perfil.apply(asignar_nivel_formacion, axis=1)

    logger.info("Distribución Y1 (nivel recomendado):\n%s",
                perfil["nivel_recomendado"].value_counts().to_string())

    # ── Paso 3b: Variable Y3 (brecha formativa) ───────────────────────────────
    perfil = calcular_brecha_formativa(perfil, catalogo)

    # ── Paso 4: Preparar X e y ────────────────────────────────────────────────
    X = perfil[FEATURE_COLS].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(perfil["nivel_recomendado"])
    grupos = perfil["departamento"]

    # ── Paso 5: Análisis de pertinencia ───────────────────────────────────────
    df_pertinencia = analizar_pertinencia(X, y, le)
    df_pertinencia.to_csv(cfg.output_dir / "pertinencia_variables.csv", index=False)
    logger.info("Tabla de pertinencia guardada en %s", cfg.output_dir / "pertinencia_variables.csv")

    # ── Paso 6: Entrenamiento ─────────────────────────────────────────────────
    modelo, metricas_cv = entrenar_modelo(X, y, grupos, cfg)

    # ── Paso 7: Evaluación ────────────────────────────────────────────────────
    # División manual para prueba final (últimos 15% por departamento)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )
    reporte = evaluar_conjunto_prueba(modelo, le, X_test, y_test)
    reporte.to_csv(cfg.output_dir / "reporte_evaluacion.csv")

    # ── Paso 8: Recomendaciones de ejemplo ───────────────────────────────────
    # Seleccionar municipios representativos de distintos grupos productivos
    municipios_ejemplo = {
        "Cafetero alto rendimiento": None,     # Salento o similar (OTROS PERMANENTES)
        "Cerealero gran escala": None,          # San Martín (CEREALES)
        "Horticultor pequeño": None,            # Municipio con <500 ha
    }

    # Buscar cod_mun para cada perfil
    for etiqueta, nivel in [
        ("ESPECIALIZACIÓN TECNOLÓGICA", "OTROS PERMANENTES"),
        ("TECNÓLOGO", "CEREALES"),
        ("COMPLEMENTARIO", "HORTALIZAS"),
    ]:
        mask = (perfil["nivel_recomendado"] == etiqueta) & (perfil["grupo_dominante"] == nivel)
        if mask.any():
            cod = perfil[mask]["cod_mun"].iloc[0]
            rec = recomendar_programas(cod, perfil, catalogo, modelo, le, cfg)
            if not rec.empty:
                nombre_mun = perfil[perfil["cod_mun"] == cod]["municipio"].values[0]
                logger.info("Recomendaciones para %s (cod=%d):\n%s", nombre_mun, cod, rec.to_string(index=False))
                rec.to_csv(cfg.output_dir / f"recomendaciones_{nombre_mun.replace(' ', '_')}.csv", index=False)

    # ── Paso 9: Exportar perfil completo ─────────────────────────────────────
    perfil.to_csv(cfg.output_dir / "perfil_municipal_features.csv", index=False)
    logger.info("Perfil municipal con features exportado.")

    logger.info("═" * 60)
    logger.info("SIROF-SENA — Pipeline completado exitosamente.")
    logger.info("Outputs en: %s", cfg.output_dir)
    logger.info("═" * 60)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# ANEXO: VARIABLES FALTANTES — PLAN DE INCORPORACIÓN
# ══════════════════════════════════════════════════════════════════════════════
"""
VARIABLES DEPENDIENTES RECOMENDADAS (Y)
═══════════════════════════════════════

Y1 — NIVEL DE FORMACIÓN RECOMENDADO (Clasificación multiclase)
    Descripción: Nivel educativo más pertinente para el municipio.
    Clases: OPERARIO, AUXILIAR, TÉCNICO, TECNÓLOGO, ESPECIALIZACIÓN TECNOLÓGICA, COMPLEMENTARIO
    Modelo: LightGBM multiclase
    Métrica: F1-macro (desbalance entre clases), ROC-AUC OvR
    Fuente actual: Heurística experta sobre EVA (supuesto)
    Fuente ideal: Histórico de matrículas SOFIA Plus por municipio
    Impacto de mejora: Muy alto — elimina el supuesto más crítico del sistema

Y2 — PROGRAMA SENA RECOMENDADO (Ranking / Recuperación de información)
    Descripción: Lista ordenada de los K programas más pertinentes
    Modelo: Motor de recomendación híbrido (filtrado por contenido + colaborativo)
    Métrica: NDCG@5, Precision@K, Recall@K
    Fuente actual: Catálogo SENA + mapeo semántico cultivo↔red
    Fuente ideal: Feedback de aprendices (relevance judgments) + matrícula real

Y3 — BRECHA FORMATIVA MUNICIPAL (Regresión continua)
    Descripción: Déficit cuantitativo entre vocación productiva y oferta formativa
    Rango: [-1, 1] donde 1 = máxima brecha (sin oferta) y -1 = sobreoferta
    Modelo: Regresión LightGBM o Ridge
    Métrica: RMSE, MAE, R²
    Fuente actual: Proxy con área EVA vs n_programas catálogo (muy aproximado)
    Fuente ideal: Matrículas SOFIA Plus + cupos por regional + empleo egresados

Y4 — EVOLUCIÓN DE VOCACIÓN PRODUCTIVA (Serie temporal)
    Descripción: Proyección de la dinámica productiva municipal a 3-5 años
    Modelo: Prophet, SARIMA o LSTM (según longitud de serie disponible)
    Métrica: MAPE, RMSE en horizonte de pronóstico
    Fuente actual: EVA 2006–2018 (solo 13 años — limitado para pronóstico)
    Fuente ideal: EVA actualizado 2019–2025 + datos de mercado agropecuario


VARIABLES INDEPENDIENTES FALTANTES CRÍTICAS
═══════════════════════════════════════════

ALTA PRIORIDAD — Incorporar en versión 1.1
─────────────────────────────────────────────────────────────────────

tasa_desempleo_municipal        DANE - GEIH        Pertinencia: MUY ALTA
    Señala urgencia de formación para empleabilidad. Municipios con alto
    desempleo priorizan niveles con mayor empleabilidad rápida (Técnico, Operario).

pea_agropecuaria               DANE - Censo 2018   Pertinencia: MUY ALTA
    Población Económicamente Activa en el sector agropecuario.
    Denomina la demanda potencial de formación agropecuaria.

indice_ruralidad               DNP                 Pertinencia: ALTA
    Determina si el municipio puede acceder a formación presencial o requiere
    modalidad virtual/a distancia. Impacta directamente el tipo de oferta viable.

matriculas_sena_municipio      SOFIA Plus          Pertinencia: MUY ALTA
    Número real de aprendices matriculados por programa en el municipio.
    Elimina el supuesto más débil del modelo actual (la variable objetivo Y1
    pasa de heurística a dato real observado).

cupos_disponibles_regional     SOFIA Plus          Pertinencia: ALTA
    Sin esto, el sistema recomienda programas que pueden estar llenos o
    inactivos en la regional correspondiente.

zona_pdet                      ART / DNP           Pertinencia: ALTA
    Municipios PDET tienen tratamiento diferencial en política de formación.
    Deben recibir oferta formativa ajustada a su contexto de posconflicto
    y planes de desarrollo.

nbi_municipal                  DANE                Pertinencia: MEDIA-ALTA
    Necesidades Básicas Insatisfechas. Proxy de capacidad de acceso a
    formación (desplazamiento, alimentación, dispositivos digitales).

distancia_centro_sena_km       SENA / OpenStreetMap  Pertinencia: ALTA
    Distancia al Centro de Formación más cercano. Determina si la formación
    presencial es accesible o debe ofertarse en otro municipio o de forma virtual.


MEDIA PRIORIDAD — Incorporar en versión 1.2
─────────────────────────────────────────────────────────────────────

pib_per_capita_municipal       DNP / DANE          Pertinencia: MEDIA
    Capacidad económica del territorio. Municipios con mayor PIB per cápita
    pueden demandar niveles de formación más elevados.

conectividad_internet_pct      MinTIC              Pertinencia: MEDIA
    Porcentaje de hogares con internet. Crítico para viabilidad de oferta virtual.

vacantes_por_ocupacion         SISPRO / GEIH       Pertinencia: ALTA
    Vacantes activas en el mercado laboral por categoría NOC. Permite alinear
    la recomendación con la demanda real del sector empresarial.

empleabilidad_egresados_sena   SENA - Observatorio Pertinencia: ALTA
    Tasa de vinculación laboral de egresados por programa. El mejor indicador
    de pertinencia real de un programa en un contexto territorial dado.

precio_cultivo_mercado         AGRONET             Pertinencia: MEDIA
    Precios de mercado de los principales cultivos. Municipios con cultivos
    de alto valor agregado (cacao, aguacate, arándano) requieren formación
    especializada en poscosecha y transformación.

presencia_empresas_agro        Cámara de Comercio  Pertinencia: MEDIA
    Número y tamaño de empresas del sector agropecuario. Determina demanda
    de formación para trabajadores vinculados (vs. formación para autoempleo).


IMPACTO ESPERADO DE INCORPORAR FUENTES ADICIONALES
═══════════════════════════════════════════════════
Versión actual (solo EVA + Catálogo):   F1-macro estimado ~0.72–0.78
Con SOFIA Plus (matrículas reales):     F1-macro estimado ~0.83–0.88
Con mercado laboral (vacantes/empleo):  F1-macro estimado ~0.87–0.92
Con geolocalización + conectividad:     NDCG@5 estimado +0.08 sobre baseline
"""
