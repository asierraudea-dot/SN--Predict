import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime

st.set_page_config(page_title="SN Predict", page_icon="🚀", layout="wide")

st.title("🚀 SN Predict")
st.subheader("Predicción Inteligente de Demanda de Aprendices")

# Carga de datos
@st.cache_data
def load_data():
    if os.path.exists('data_historico.csv'):
        df = pd.read_csv('data_historico.csv')
    elif os.path.exists('PE04_HISTÓRICO_PREVIOS.xlsx'):
        df = pd.read_excel('PE04_HISTÓRICO_PREVIOS.xlsx', sheet_name='Hoja1')
        df.to_csv('data_historico.csv', index=False)
    else:
        st.error("No se encontró el archivo de datos.")
        st.stop()
    
    df['FECHA_INICIO_FICHA'] = pd.to_datetime(df['FECHA_INICIO_FICHA'], errors='coerce')
    df['AÑO'] = df['FECHA_INICIO_FICHA'].dt.year.fillna(df.get('AÑO', 2024))
    return df

df = load_data()

# Filtro global
st.sidebar.header("Filtros")
municipio_filtro = st.sidebar.selectbox("Filtrar por Municipio", ['Todos'] + sorted(df['NOMBRE_MUNICIPIO_CURSO'].unique()))

# Preparación
def prepare_features(df):
    df = df.copy()
    df['MUNICIPIO'] = df['NOMBRE_MUNICIPIO_CURSO']
    df['PROGRAMA'] = df['NOMBRE_PROGRAMA_FORMACION'].fillna('').str.strip().str.lower()
    df['NIVEL'] = df['NIVEL_FORMACION']
    df['SECTOR'] = df['NOMBRE_NUEVO_SECTOR'].fillna('OTRO')
    df['MES_INICIO'] = df['FECHA_INICIO_FICHA'].dt.month.fillna(6)
    df['DURACION'] = df['DURACION_PROGRAMA']
    return df

df_prep = prepare_features(df)

# Modelo
@st.cache_resource
def train_model():
    features = ['MUNICIPIO', 'PROGRAMA', 'NIVEL', 'SECTOR', 'MES_INICIO', 'DURACION']
    X = pd.get_dummies(df_prep[features], drop_first=True)
    y = df_prep['TOTAL_APRENDICES']
    model = RandomForestRegressor(n_estimators=180, max_depth=14, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = train_model()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictor", "📊 Estadísticas", "📈 Histórico", "🌍 Dashboard"])

with tab1:
    st.header("🔮 Predicción de Inscripciones")
    col1, col2 = st.columns(2)
    
    with col1:
        mun_sel = st.selectbox("📍 Municipio", sorted(df['NOMBRE_MUNICIPIO_CURSO'].unique()), key="mun_pred")
        prog_sel = st.text_input("📘 Nombre del Programa", "MANIPULACION HIGIENICA DE ALIMENTOS")
        nivel_sel = st.selectbox("📚 Nivel", sorted(df['NIVEL_FORMACION'].unique()))
    
    with col2:
        sector_sel = st.selectbox("🏭 Sector", sorted(df['NOMBRE_NUEVO_SECTOR'].dropna().unique()))
        duracion_sel = st.number_input("⏱️ Duración (horas)", min_value=8, value=48)
        mes_sel = st.slider("📅 Mes de Inicio", 1, 12, datetime.now().month)
    
    if st.button("🚀 Predecir", type="primary"):
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
        st.success(f"**{pred} aprendices esperados en {mun_sel}**", icon="🎯")

with tab2:
    st.header("📊 Estadísticas por Municipio")
    if municipio_filtro != 'Todos':
        df_filtrado = df[df['NOMBRE_MUNICIPIO_CURSO'] == municipio_filtro]
    else:
        df_filtrado = df
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Por Programa")
        top_prog = df_filtrado.groupby('NOMBRE_PROGRAMA_FORMACION')['TOTAL_APRENDICES'].sum().nlargest(10)
        st.bar_chart(top_prog)
    
    with col2:
        st.subheader("Por Sector Productivo")
        top_sector = df_filtrado.groupby('NOMBRE_NUEVO_SECTOR')['TOTAL_APRENDICES'].sum().nlargest(8)
        st.bar_chart(top_sector)

with tab3:
    st.header("📈 Datos Históricos")
    st.dataframe(df_filtrado, use_container_width=True)

with tab4:
    st.header("🌍 Dashboard General")
    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby('AÑO')['TOTAL_APRENDICES'].sum()
        st.plotly_chart(px.line(yearly, title="Evolución Anual"), use_container_width=True)
    with col2:
        sector_pie = df.groupby('NOMBRE_NUEVO_SECTOR')['TOTAL_APRENDICES'].sum()
        st.plotly_chart(px.pie(names=sector_pie.index, values=sector_pie.values, title="Distribución por Sector"), use_container_width=True)

st.caption("SN Predict © 2026 | Basado en datos históricos de formación")
