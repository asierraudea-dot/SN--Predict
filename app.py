import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="SN Predict",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.8rem; color: #1E88E5; font-weight: bold; margin-bottom: 0.5rem;}
    .stButton>button {width: 100%; height: 3rem; font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 SN Predict")
st.subheader("Predicción Inteligente de Demanda de Aprendices")

# Carga de datos
@st.cache_data
def load_data():
    df = pd.read_csv('data_historico.csv')
    df['FECHA_INICIO_FICHA'] = pd.to_datetime(df['FECHA_INICIO_FICHA'], errors='coerce')
    df['AÑO'] = df['FECHA_INICIO_FICHA'].dt.year.fillna(df.get('AÑO', 2024))
    df['MES'] = df['FECHA_INICIO_FICHA'].dt.month
    return df

df = load_data()

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

# Modelo
@st.cache_resource
def train_model():
    features = ['MUNICIPIO', 'PROGRAMA', 'NIVEL', 'SECTOR', 'MES_INICIO', 'DURACION']
    X = pd.get_dummies(df_prep[features], drop_first=True)
    y = df_prep['TOTAL_APRENDICES']
    
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, X.columns.tolist()

model, feature_names = train_model()

# Sidebar
st.sidebar.header("📊 Rendimiento del Modelo")
st.sidebar.success("Modelo entrenado con éxito")

municipio_filtro = st.sidebar.selectbox("Filtrar por Municipio", ['Todos'] + sorted(df['NOMBRE_MUNICIPIO_CURSO'].unique()))

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictor", "📈 Histórico", "🌍 Dashboard", "💡 Recomendaciones"])

with tab1:
    st.header("🔮 Predicción de Inscripciones")
    st.markdown("Ingresa los datos del programa que deseas ofrecer:")
    
    col1, col2 = st.columns(2)
    with col1:
        mun_sel = st.selectbox("📍 Municipio", sorted(df['NOMBRE_MUNICIPIO_CURSO'].unique()))
        prog_sel = st.text_input("📘 Nombre del Programa", "MANIPULACION HIGIENICA DE ALIMENTOS")
        nivel_sel = st.selectbox("📚 Nivel de Formación", sorted(df['NIVEL_FORMACION'].unique()))
    
    with col2:
        sector_sel = st.selectbox("🏭 Sector", sorted(df['NOMBRE_NUEVO_SECTOR'].dropna().unique()))
        duracion_sel = st.number_input("⏱️ Duración (horas)", min_value=8, value=48)
        mes_sel = st.slider("📅 Mes de Inicio", 1, 12, datetime.now().month)
    
    if st.button("🚀 Predecir Número de Aprendices", type="primary"):
        input_df = pd.DataFrame([{
            'MUNICIPIO': mun_sel,
            'PROGRAMA': prog_sel.lower(),
            'NIVEL': nivel_sel,
            'SECTOR': sector_sel,
            'MES_INICIO': mes_sel,
            'DURACION': duracion_sel
        }])
        
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
        
        pred = int(round(model.predict(input_encoded)[0]))
        st.success(f"**{pred} aprendices esperados**", icon="🎯")

with tab2:
    st.header("📈 Datos Históricos")
    df_view = df if municipio_filtro == 'Todos' else df[df['NOMBRE_MUNICIPIO_CURSO'] == municipio_filtro]
    st.dataframe(df_view, use_container_width=True, height=600)

with tab3:
    st.header("🌍 Dashboard de Tendencias")
    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby('AÑO')['TOTAL_APRENDICES'].sum().reset_index()
        st.plotly_chart(px.line(yearly, x='AÑO', y='TOTAL_APRENDICES', title="Evolución Anual de Aprendices"), use_container_width=True)
    
    with col2:
        sector_data = df.groupby('NOMBRE_NUEVO_SECTOR')['TOTAL_APRENDICES'].sum().reset_index()
        st.plotly_chart(px.pie(sector_data, names='NOMBRE_NUEVO_SECTOR', values='TOTAL_APRENDICES', title="Distribución por Sector"), use_container_width=True)

with tab4:
    st.header("💡 Recomendaciones")
    st.info("Programas con mayor potencial según datos históricos")
    top = df.groupby(['NOMBRE_MUNICIPIO_CURSO', 'NOMBRE_PROGRAMA_FORMACION'])['TOTAL_APRENDICES'].sum().reset_index()
    top = top.sort_values('TOTAL_APRENDICES', ascending=False).head(15)
    st.dataframe(top, use_container_width=True)

st.caption("SN Predict © 2026 | Basado en datos históricos de formación")
