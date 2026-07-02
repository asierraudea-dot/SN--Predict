import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime

st.set_page_config(
    page_title="SN Predict",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 SN Predict")
st.subheader("Predicción Inteligente de Demanda de Aprendices")

# ==================== CARGA DE DATOS CON FALLBACK ====================
@st.cache_data
def load_data():
    csv_path = 'data_historico.csv'
    excel_path = 'PE04_HISTÓRICO_PREVIOS.xlsx'
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.success("✅ Datos cargados desde CSV")
    elif os.path.exists(excel_path):
        df = pd.read_excel(excel_path, sheet_name='Hoja1')
        df.to_csv(csv_path, index=False)
        st.success("✅ Datos cargados desde Excel y convertidos a CSV")
    else:
        st.error("❌ No se encontró el archivo de datos. Por favor sube 'data_historico.csv' o 'PE04_HISTÓRICO_PREVIOS.xlsx'")
        st.stop()
    
    # Limpieza
    df['FECHA_INICIO_FICHA'] = pd.to_datetime(df['FECHA_INICIO_FICHA'], errors='coerce')
    df['AÑO'] = df['FECHA_INICIO_FICHA'].dt.year.fillna(df.get('AÑO', 2024))
    df['MES'] = df['FECHA_INICIO_FICHA'].dt.month
    return df

df = load_data()

# ==================== RESTO DEL CÓDIGO (igual que antes) ====================
def prepare_features(df):
    df = df.copy()
    df['MUNICIPIO'] = df['NOMBRE_MUNICIPIO_CURSO']
    df['PROGRAMA'] = df['NOMBRE_PROGRAMA_FORMACION'].fillna('').str.strip().str.lower()
    df['NIVEL'] = df['NIVEL_FORMACION']
    df['SECTOR'] = df['NOMBRE_NUEVO_SECTOR'].fillna('OTRO')
    df['MES_INICIO'] = df['MES'].fillna(6)
    df['DURACION'] = df['DURACION_PROGRAMA']
    return df

df_prep = prepare_features(df)

@st.cache_resource
def train_model():
    features = ['MUNICIPIO', 'PROGRAMA', 'NIVEL', 'SECTOR', 'MES_INICIO', 'DURACION']
    X = pd.get_dummies(df_prep[features], drop_first=True)
    y = df_prep['TOTAL_APRENDICES']
    
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = train_model()

# Sidebar y tabs (mismo código anterior)...
st.sidebar.header("📊 Rendimiento del Modelo")
st.sidebar.success("Modelo entrenado con éxito")

municipio_filtro = st.sidebar.selectbox("Filtrar por Municipio", ['Todos'] + sorted(df['NOMBRE_MUNICIPIO_CURSO'].unique()))

# ... (Mantén el resto de tabs igual que en la versión anterior)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictor", "📈 Histórico", "🌍 Dashboard", "💡 Recomendaciones"])

# (Copia aquí el contenido de los tabs de la versión anterior que te di)

st.caption("SN Predict © 2026 | Basado en datos históricos de formación")
