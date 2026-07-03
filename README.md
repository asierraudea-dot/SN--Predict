[README_sn_predict.md](https://github.com/user-attachments/files/29645036/README_sn_predict.md)

# SN Predict v2 — Sistema Inteligente de Planificación Formativa 

> Occidente de Antioquia · Predicción de demanda con ML y analítica territorial

---

## Descripción

SN Predict v2 es una aplicación Streamlit de inteligencia artificial que analiza **2.551 fichas históricas** de formación del Occidente de Antioquia (2023–2025) para:

- **Predecir** cuántos aprendices participarán en una nueva ficha.
- **Recomendar** los programas más pertinentes por municipio y sector.
- **Identificar brechas** entre oferta formativa y vocación productiva del territorio.
- **Planear** la oferta con base en tendencias y estacionalidad histórica.

---

## Módulos

| Módulo | Descripción |
|---|---|
| 🎯 Predictor inteligente | Filtros dinámicos sector→programa, mapa de calor, historial instructores, predicción ML |
| 📊 Demanda por nivel | Distribución nivel × sector × municipio con tendencias |
| ⚡ Recomendaciones | Ruta formativa Complementarios→Técnico→Tecnólogo por municipio |
| 🔭 Oportunidades | Brechas por rubro productivo (Café, Ganadería, Turismo…) |
| 📖 Manual | Guía completa + tour interactivo |

---

## Instalación

```bash
git clone https://github.com/[usuario]/sn-predict-v2.git
cd sn-predict-v2
pip install -r requirements.txt

# Colocar el dataset en data/
cp PE04_HISTORICO_PREVIOS.xlsx data/

# (Opcional) Entrenar modelo ML
python train.py

# Lanzar la app
streamlit run app/app.py
```

---

## Estructura del repositorio

```
sn-predict-v2/
├── app/
│   ├── app.py           # Aplicación Streamlit principal (5 módulos)
│   ├── data_loader.py   # Carga, limpieza y preprocesamiento de PE04
│   ├── predictor.py     # Predicción heurística y wrapper ML
│   └── components.py    # Componentes de UI reutilizables (CSS/HTML)
├── data/
│   └── PE04_HISTORICO_PREVIOS.xlsx   # Dataset (no subir si es confidencial)
├── models/
│   ├── pipeline_y1_regresion.pkl     # Generado por train.py
│   ├── pipeline_y2_clasificacion.pkl
│   └── feature_names.json
├── notebooks/
│   └── 01_EDA.ipynb
├── train.py             # Pipeline de entrenamiento ML
├── requirements.txt
└── README.md
```

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Frontend | Streamlit 1.35 + Plotly 5.22 |
| ML | Scikit-learn (HistGradientBoosting) |
| Datos | Pandas 2.2 + NumPy 1.26 |
| Mapa | Plotly Mapbox (carto-positron) |
| Despliegue | Streamlit Community Cloud |

---

## Demo

🔗 **App:** [https://[usuario]-sn-predict.streamlit.app](https://streamlit.app)  
🎬 **Video:** *(enlace YouTube/Drive — completar)*

---

## Autor

**[Nombre del estudiante]**  
Inteligencia Artificial · Junio 2026
