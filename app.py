import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime

st.set_page_config(page_title="SN Predict", page_icon="🚀", layout="wide")

# Estilos profesionales
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .header {font-size: 2.8rem; color: #1E3A8A; font-weight: 700;}
    .card {background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.title("🚀 SN Predict")
st.markdown('<p class="header">Sistema Inteligente de Planificación Formativa</p>', unsafe_allow_html=True)

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
        st.error("Archivo de datos no encontrado.")
        st.stop()

df = load_data()

# Filtros
st.sidebar.header("🔍 Filtros")
municipio_filtro = st.sidebar.selectbox("Municipio", ['Todos'] + sorted(df['NOMBRE_MUNICIPIO_CURSO'].unique()))

# Preparación
df['MUNICIPIO'] = df['NOMBRE_MUNICIPIO_CURSO']
df['PROGRAMA'] = df['NOMBRE_PROGRAMA_FORMACION'].fillna('').str.strip()
df['NIVEL'] = df['NIVEL_FORMACION']
df['SECTOR'] = df['NOMBRE_NUEVO_SECTOR'].fillna('OTRO')
df['AÑO'] = pd.to_datetime(df['FECHA_INICIO_FICHA'], errors='coerce').dt.year.fillna(2024)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictor Inteligente", "📊 Demanda por Nivel", "📈 Tendencias", "💡 Recomendaciones"])

with tab1:
    st.header("🎯 Predictor Inteligente")
    st.markdown("**Predice la demanda y evalúa viabilidad de apertura**")
    
    col1, col2 = st.columns([1,1])
    with col1:
        mun_sel = st.selectbox("📍 Municipio", sorted(df['MUNICIPIO'].unique()), key="mun")
        prog_sel = st.text_input("📘 Nombre del Programa", "MANIPULACION HIGIENICA DE ALIMENTOS")
        nivel_sel = st.selectbox("📚 Nivel", sorted(df['NIVEL'].unique()))
    
    with col2:
        sector_sel = st.selectbox("🏭 Sector", sorted(df['SECTOR'].unique()))
        duracion_sel = st.number_input("⏱️ Duración (horas)", min_value=8, value=48)
        mes_sel = st.slider("📅 Mes de Inicio", 1, 12, value=datetime.now().month)
    
    if st.button("🔮 Generar Predicción y Evaluación", type="primary"):
        # Predicción
        input_df = pd.DataFrame([{
            'MUNICIPIO': mun_sel, 'PROGRAMA': prog_sel.lower(),
            'NIVEL': nivel_sel, 'SECTOR': sector_sel,
            'MES_INICIO': mes_sel, 'DURACION': duracion_sel
        }])
        
        # Modelo simple (para demo)
        X = pd.get_dummies(df[['MUNICIPIO','PROGRAMA','NIVEL','SECTOR','MES_INICIO','DURACION']], drop_first=True)
        model = RandomForestRegressor(n_estimators=120, random_state=42)
        model.fit(X, df['TOTAL_APRENDICES'])
        
        input_encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=X.columns, fill_value=0)
        pred = int(round(model.predict(input_encoded)[0]))
        
        # Evaluación
        if pred >= 25:
            viabilidad = "Alta"
            color = "🟢"
        elif pred >= 15:
            viabilidad = "Media"
            color = "🟡"
        else:
            viabilidad = "Baja"
            color = "🔴"
        
        st.success(f"**Predicción: {pred} aprendices** | Viabilidad: {color} {viabilidad}", icon="🎯")

with tab2:
    st.header("📊 Demanda por Nivel de Formación")
    if municipio_filtro != 'Todos':
        df_f = df[df['MUNICIPIO'] == municipio_filtro]
    else:
        df_f = df
    
    niveles = df_f.groupby(['NIVEL', 'NOMBRE_PROGRAMA_FORMACION'])['TOTAL_APRENDICES'].sum().reset_index()
    for nivel in df_f['NIVEL'].unique():
        st.subheader(f"Nivel: **{nivel}**")
        data_nivel = niveles[niveles['NIVEL'] == nivel].nlargest(8, 'TOTAL_APRENDICES')
        st.bar_chart(data_nivel.set_index('NOMBRE_PROGRAMA_FORMACION')['TOTAL_APRENDICES'])

with tab3:
    st.header("📈 Tendencias Históricas")
    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby('AÑO')['TOTAL_APRENDICES'].sum()
        st.plotly_chart(px.line(yearly, title="Evolución General"), use_container_width=True)
    with col2:
        sector_trend = df.groupby(['AÑO', 'SECTOR'])['TOTAL_APRENDICES'].sum().reset_index()
        st.plotly_chart(px.bar(sector_trend, x='AÑO', y='TOTAL_APRENDICES', color='SECTOR'), use_container_width=True)

with tab4:
    st.header("💡 Recomendaciones de Apertura")
    if municipio_filtro != 'Todos':
        df_rec = df[df['MUNICIPIO'] == municipio_filtro]
    else:
        df_rec = df
    
    top = df_rec.groupby(['PROGRAMA', 'NIVEL', 'SECTOR'])['TOTAL_APRENDICES'].agg(['sum', 'mean']).reset_index()
    top = top.sort_values('sum', ascending=False).head(12)
    top['Probabilidad Apertura'] = top['sum'].apply(lambda x: "Alta" if x > 30 else "Media" if x > 15 else "Baja")
    
    st.dataframe(top, use_container_width=True)

st.caption("SN Predict © 2026 | Herramienta de Planificación Formativa Inteligente")
