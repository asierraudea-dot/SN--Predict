import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime

st.set_page_config(page_title="SN Predict", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1E3A8A; font-weight: 700;}
    .stButton>button {width: 100%; height: 3.2rem; font-weight: 600;}
    .metric {font-size: 1.8rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 SN Predict")
st.subheader("Sistema Inteligente de Planificación y Predicción Formativa")

# Carga de datos
@st.cache_data
def load_data():
    if os.path.exists('data_historico.csv'):
        return pd.read_csv('data_historico.csv')
    elif os.path.exists('PE04_HISTÓRICO_PREVIOS.xlsx'):
        df = pd.read_excel('PE04_HISTÓRICO_PREVIOS.xlsx', sheet_name='Hoja1')
        df.to_csv('data_historico.csv', index=False)
        return df
    else:
        st.error("❌ Archivo de datos no encontrado.")
        st.stop()

df = load_data()

# Preprocesamiento
@st.cache_data
def preprocess(df):
    df = df.copy()
    df['MUNICIPIO'] = df['NOMBRE_MUNICIPIO_CURSO']
    df['PROGRAMA'] = df['NOMBRE_PROGRAMA_FORMACION'].fillna('').str.strip().str.lower()
    df['NIVEL'] = df['NIVEL_FORMACION']
    df['SECTOR'] = df['NOMBRE_NUEVO_SECTOR'].fillna('OTRO')
    df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO_FICHA'], errors='coerce')
    df['AÑO'] = df['FECHA_INICIO'].dt.year.fillna(2024)
    df['MES_INICIO'] = df['FECHA_INICIO'].dt.month.fillna(6)
    df['DURACION'] = df['DURACION_PROGRAMA']
    return df

df_prep = preprocess(df)

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

# Filtro
st.sidebar.header("🔍 Filtros")
municipio_filtro = st.sidebar.selectbox("Municipio", ['Todos'] + sorted(df_prep['MUNICIPIO'].unique()))

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictor", "📊 Demanda por Nivel", "📈 Tendencias", "💡 Recomendaciones"])

with tab1:
    st.header("🎯 Predictor Inteligente")
    st.markdown("**Predice demanda y evalúa viabilidad de apertura**")
    
    col1, col2 = st.columns(2)
    with col1:
        mun_sel = st.selectbox("📍 Municipio", sorted(df_prep['MUNICIPIO'].unique()), key="pred_mun")
        prog_sel = st.text_input("📘 Programa", "manipulacion higienica de alimentos")
        nivel_sel = st.selectbox("📚 Nivel", sorted(df_prep['NIVEL'].unique()))
    with col2:
        sector_sel = st.selectbox("🏭 Sector", sorted(df_prep['SECTOR'].unique()))
        duracion_sel = st.number_input("⏱️ Duración (horas)", min_value=8, value=48)
        mes_sel = st.slider("📅 Mes", 1, 12, datetime.now().month)
    
    if st.button("🚀 Generar Predicción y Evaluación", type="primary"):
        input_df = pd.DataFrame([{
            'MUNICIPIO': mun_sel, 'PROGRAMA': prog_sel.lower(),
            'NIVEL': nivel_sel, 'SECTOR': sector_sel,
            'MES_INICIO': mes_sel, 'DURACION': duracion_sel
        }])
        
        input_encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=feature_names, fill_value=0)
        pred = int(round(model.predict(input_encoded)[0]))
        
        if pred >= 30: viab = "🟢 Alta"
        elif pred >= 18: viab = "🟡 Media"
        else: viab = "🔴 Baja"
        
        st.success(f"**{pred} aprendices esperados** | Viabilidad: {viab}", icon="🎯")

with tab2:
    st.header("📊 Demanda por Nivel de Formación")
    df_f = df_prep if municipio_filtro == 'Todos' else df_prep[df_prep['MUNICIPIO'] == municipio_filtro]
    for nivel in df_f['NIVEL'].unique():
        st.subheader(f"**{nivel}**")
        data = df_f[df_f['NIVEL'] == nivel].groupby('PROGRAMA')['TOTAL_APRENDICES'].sum().nlargest(10)
        fig = px.bar(x=data.values, y=data.index, orientation='h', title=f"Top programas - {nivel}")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📈 Tendencias Históricas")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(df_prep.groupby('AÑO')['TOTAL_APRENDICES'].sum(), title="Evolución Anual"), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(df_prep.groupby('SECTOR')['TOTAL_APRENDICES'].sum(), title="Distribución por Sector"), use_container_width=True)

with tab4:
    st.header("💡 Recomendaciones de Apertura")
    df_rec = df_prep if municipio_filtro == 'Todos' else df_prep[df_prep['MUNICIPIO'] == municipio_filtro]
    rec = df_rec.groupby(['PROGRAMA', 'NIVEL', 'SECTOR'])['TOTAL_APRENDICES'].sum().reset_index()
    rec = rec.sort_values('TOTAL_APRENDICES', ascending=False).head(15)
    rec['Viabilidad'] = rec['TOTAL_APRENDICES'].apply(lambda x: "Alta" if x>35 else "Media" if x>20 else "Baja")
    st.dataframe(rec, use_container_width=True)

st.caption("SN Predict © 2026 | Herramienta de Inteligencia para la Formación Profesional")
