"""
app.py — SN Predict v3.2 AUTOCONTENIDO
=======================================
Sistema Inteligente de Planificación y Predicción Formativa
Occidente de Antioquia — Junio 2026

No requiere módulos externos. Datos reales PE04 embebidos.
2.551 fichas · 22 municipios · 12 sectores · 2020-2025
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

try:
    from data_loader import cargar_df, get_sec_prog, get_prog_mun, get_resp_stats
    from data_loader import get_mun_stats, get_rubros, get_niv_sec
    _HL = True
except Exception:
    _HL = False

# =============================================================================
# DATOS REALES EMBEBIDOS — PE04 2020-2025
# =============================================================================
_SP = '{"SERVICIOS":[{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","fichas":123,"prom":27.8,"dur":12,"ap":3419,"meses":4,"ultima":"2024-12-02","estrella":true},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","fichas":93,"prom":22.3,"dur":48,"ap":2076,"meses":4,"ultima":"2024-12-02","estrella":false},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","fichas":87,"prom":25.2,"dur":10,"ap":2195,"meses":0,"ultima":"2025-03-05","estrella":false},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","fichas":82,"prom":22.4,"dur":48,"ap":1839,"meses":1,"ultima":"2025-03-01","estrella":false},{"prog":"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL","nivel":"CURSO ESPECIAL","fichas":62,"prom":19.1,"dur":40,"ap":1187,"meses":1,"ultima":"2025-02-24","estrella":false},{"prog":"ENGLISH DOES WORK - LEVEL 1","nivel":"CURSO ESPECIAL","fichas":53,"prom":58.3,"dur":52,"ap":3091,"meses":0,"ultima":"2025-03-17","estrella":true},{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","fichas":47,"prom":27.6,"dur":2208,"ap":1298,"meses":1,"ultima":"2025-02-25","estrella":true},{"prog":"INGLES BASICO - NIVEL 1","nivel":"CURSO ESPECIAL","fichas":35,"prom":25.1,"dur":48,"ap":877,"meses":1,"ultima":"2025-02-21","estrella":false},{"prog":"HIGIENE Y MANIPULACION DE ALIMENTOS.","nivel":"CURSO ESPECIAL","fichas":33,"prom":17.5,"dur":40,"ap":579,"meses":1,"ultima":"2025-02-27","estrella":false},{"prog":"GESTION DE PROYECTOS COMUNITARIOS","nivel":"CURSO ESPECIAL","fichas":29,"prom":17.5,"dur":48,"ap":507,"meses":1,"ultima":"2025-02-27","estrella":false},{"prog":"INGLES BASICO - NIVEL 2","nivel":"CURSO ESPECIAL","fichas":26,"prom":25.8,"dur":48,"ap":670,"meses":3,"ultima":"2024-12-05","estrella":false},{"prog":"INGLES BASICO - NIVEL 3","nivel":"CURSO ESPECIAL","fichas":24,"prom":27.1,"dur":48,"ap":651,"meses":3,"ultima":"2024-12-05","estrella":true}],"AGROPECUARIO":[{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","fichas":41,"prom":24.8,"dur":2205,"ap":1015,"meses":1,"ultima":"2025-02-18","estrella":true},{"prog":"PRODUCCION AGROPECUARIA ECOLOGICA PARA LA SOBERANIA ALIMENTARIA","nivel":"CURSO ESPECIAL","fichas":41,"prom":22.6,"dur":192,"ap":925,"meses":1,"ultima":"2025-02-21","estrella":false},{"prog":"APLICACION DE LAS BUENAS PRACTICAS AGRICOLAS","nivel":"CURSO ESPECIAL","fichas":30,"prom":18.6,"dur":96,"ap":557,"meses":1,"ultima":"2025-02-27","estrella":false},{"prog":"CRIAR GALLINAS PONEDORAS CON ALIMENTACIÓN ALTERNA PARA PRODUCIR HUEV","nivel":"CURSO ESPECIAL","fichas":27,"prom":21.0,"dur":60,"ap":568,"meses":1,"ultima":"2025-02-27","estrella":false},{"prog":"MULTIPLICACION Y PROPAGACION DE MATERIAL VEGETAL","nivel":"CURSO ESPECIAL","fichas":20,"prom":17.4,"dur":64,"ap":349,"meses":1,"ultima":"2025-02-26","estrella":false},{"prog":"PRACTICAS DE CALIDAD PARA EL BENEFICIO DEL CAFE","nivel":"CURSO ESPECIAL","fichas":19,"prom":19.9,"dur":56,"ap":379,"meses":6,"ultima":"2024-09-12","estrella":false},{"prog":"MANEJO DE LA NUTRICION EN CULTIVOS AGRICOLAS","nivel":"CURSO ESPECIAL","fichas":16,"prom":19.2,"dur":60,"ap":307,"meses":5,"ultima":"2024-10-18","estrella":false},{"prog":"PRODUCCION AGROPECUARIA","nivel":"TÉCNICO","fichas":16,"prom":24.8,"dur":2343,"ap":397,"meses":1,"ultima":"2025-02-19","estrella":true},{"prog":"BUENAS PRACTICAS AGRICOLAS.","nivel":"CURSO ESPECIAL","fichas":13,"prom":18.8,"dur":48,"ap":244,"meses":1,"ultima":"2025-02-26","estrella":false},{"prog":"IMPLEMENTACION DE PRACTICAS DE MANEJO DEL HUEVO","nivel":"CURSO ESPECIAL","fichas":12,"prom":25.9,"dur":48,"ap":311,"meses":6,"ultima":"2024-10-03","estrella":true},{"prog":"INSEMINACION ARTIFICIAL DE HEMBRAS BOVINAS APLICANDO PROTOCOLO CONVE","nivel":"CURSO ESPECIAL","fichas":11,"prom":23.6,"dur":70,"ap":260,"meses":5,"ultima":"2024-10-10","estrella":false},{"prog":"APLICACION DE  BUENAS PRACTICAS GANADERAS EN LA PRODUCCION DE BOVINO","nivel":"CURSO ESPECIAL","fichas":11,"prom":26.8,"dur":48,"ap":295,"meses":4,"ultima":"2024-11-08","estrella":true}],"INDUSTRIA":[{"prog":"ELABORACIÓN DE ARTÍCULOS DECORATIVOS Y UTILITARIOS CON MATERIALES RE","nivel":"CURSO ESPECIAL","fichas":16,"prom":24.6,"dur":40,"ap":393,"meses":10,"ultima":"2024-05-29","estrella":false},{"prog":"TÉCNICAS DE PINTURA EN TELA","nivel":"CURSO ESPECIAL","fichas":15,"prom":25.5,"dur":50,"ap":382,"meses":0,"ultima":"2025-03-04","estrella":false},{"prog":"PATRONAJE DE PRENDAS DE ROPA EXTERIOR FEMENINA","nivel":"CURSO ESPECIAL","fichas":11,"prom":17.5,"dur":40,"ap":193,"meses":20,"ultima":"2023-07-17","estrella":false},{"prog":"ELABORACION DE COMPLEMENTOS EN BISUTERIA CON TECNICA DE ENSARTADO","nivel":"CURSO ESPECIAL","fichas":9,"prom":22.8,"dur":48,"ap":205,"meses":17,"ultima":"2023-10-19","estrella":false},{"prog":"ELABORACIÓN DE ARTÍCULOS DECORATIVOS Y UTILITARIOS APLICANDO TÉCNICA","nivel":"CURSO ESPECIAL","fichas":5,"prom":25.2,"dur":40,"ap":126,"meses":17,"ultima":"2023-10-26","estrella":false},{"prog":"ELABORACION DE ACCESORIOS TEJIDOS EN MOSTACILLAS","nivel":"CURSO ESPECIAL","fichas":5,"prom":29.2,"dur":48,"ap":146,"meses":5,"ultima":"2024-10-10","estrella":true},{"prog":"DISEÑO, LIQUIDACION Y PAGO DE NOMINA.","nivel":"CURSO ESPECIAL","fichas":4,"prom":27.2,"dur":48,"ap":109,"meses":5,"ultima":"2024-10-21","estrella":true},{"prog":"ELABORACION DE ADORNOS Y MUÑECOS NAVIDEÑOS","nivel":"CURSO ESPECIAL","fichas":4,"prom":25.5,"dur":48,"ap":102,"meses":4,"ultima":"2024-11-15","estrella":false},{"prog":"CONFECCIÓN DE ROPA DEPORTIVA","nivel":"CURSO ESPECIAL","fichas":2,"prom":19.5,"dur":90,"ap":39,"meses":8,"ultima":"2024-07-09","estrella":false},{"prog":"CONFECCIÓN DE CAMISETAS DEPORTIVAS","nivel":"CURSO ESPECIAL","fichas":2,"prom":37.5,"dur":41,"ap":75,"meses":3,"ultima":"2024-12-03","estrella":true},{"prog":"CONFECCIÓN DE PANTALÓN","nivel":"CURSO ESPECIAL","fichas":2,"prom":25.0,"dur":42,"ap":50,"meses":7,"ultima":"2024-08-08","estrella":false},{"prog":"OPERACIÓN DE SISTEMAS DE POTABILIZACIÓN DE AGUA","nivel":"TÉCNICO","fichas":2,"prom":35.5,"dur":1945,"ap":71,"meses":25,"ultima":"2023-03-06","estrella":true}],"HOTELERIA Y TURISMO":[{"prog":"OPERACION TURISTICA LOCAL","nivel":"TÉCNICO","fichas":12,"prom":25.2,"dur":2200,"ap":302,"meses":1,"ultima":"2025-02-20","estrella":true},{"prog":"SERVICIO DE RECEPCION HOTELERA","nivel":"TÉCNICO","fichas":11,"prom":43.4,"dur":2409,"ap":477,"meses":10,"ultima":"2024-05-14","estrella":true},{"prog":"RECONOCIMIENTO Y REGISTRO DE ESPECIES PARA EL AVITURISMO","nivel":"CURSO ESPECIAL","fichas":8,"prom":22.6,"dur":60,"ap":181,"meses":3,"ultima":"2024-12-03","estrella":false},{"prog":"COCTELERIA TROPICAL","nivel":"CURSO ESPECIAL","fichas":7,"prom":21.3,"dur":40,"ap":149,"meses":6,"ultima":"2024-10-01","estrella":false},{"prog":"TECNICAS PARA LA PREPARACION DE BEBIDAS A BASE DE CAFE","nivel":"CURSO ESPECIAL","fichas":7,"prom":18.1,"dur":40,"ap":127,"meses":5,"ultima":"2024-10-16","estrella":false},{"prog":"SERVICIO DE RESTAURANTE Y BAR","nivel":"TÉCNICO","fichas":6,"prom":18.5,"dur":2208,"ap":111,"meses":4,"ultima":"2024-11-21","estrella":false},{"prog":"DISEÑO DE  RECORRIDOS TURISTICOS EN ESPACIOS RURALES","nivel":"CURSO ESPECIAL","fichas":5,"prom":24.6,"dur":40,"ap":123,"meses":7,"ultima":"2024-08-05","estrella":true},{"prog":"EMPRENDEDOR EN DESARROLLO DE ACTIVIDADES TURISTICAS EN ESPACIOS NATU","nivel":"CURSO ESPECIAL","fichas":4,"prom":23.0,"dur":288,"ap":92,"meses":18,"ultima":"2023-10-05","estrella":true},{"prog":"OPERACION DE ALOJAMIENTOS RURALES","nivel":"TÉCNICO","fichas":4,"prom":16.5,"dur":1765,"ap":66,"meses":10,"ultima":"2024-05-14","estrella":false},{"prog":"GUIANZA EN RECORRIDOS POR LA NATURALEZA.","nivel":"CURSO ESPECIAL","fichas":4,"prom":15.2,"dur":40,"ap":61,"meses":26,"ultima":"2023-02-07","estrella":false},{"prog":"EJECUCION DE EVENTOS DEPORTIVOS Y RECREATIVOS","nivel":"TÉCNICO","fichas":2,"prom":18.0,"dur":2205,"ap":36,"meses":6,"ultima":"2024-09-23","estrella":false},{"prog":"ACTUALIZACION EN LEGISLACION TURISTICA","nivel":"CURSO ESPECIAL","fichas":2,"prom":22.0,"dur":40,"ap":44,"meses":4,"ultima":"2024-11-05","estrella":false}],"TRANSVERSAL":[{"prog":"INFORMATICA: MICROSOFT WORD, EXCEL E INTERNET","nivel":"CURSO ESPECIAL","fichas":22,"prom":23.0,"dur":40,"ap":505,"meses":1,"ultima":"2025-02-20","estrella":false},{"prog":"SISTEMAS.","nivel":"TÉCNICO","fichas":14,"prom":28.4,"dur":2208,"ap":398,"meses":20,"ultima":"2023-07-17","estrella":true},{"prog":"ANALISIS Y DESARROLLO DE SOFTWARE.","nivel":"TECNÓLOGO","fichas":7,"prom":31.0,"dur":3984,"ap":217,"meses":4,"ultima":"2024-11-25","estrella":true},{"prog":"RECURSOS HUMANOS .","nivel":"TÉCNICO","fichas":6,"prom":26.5,"dur":2208,"ap":159,"meses":7,"ultima":"2024-09-02","estrella":false},{"prog":"CREACIÓN DE BASES DE DATOS CON MICROSOFT ACCESS","nivel":"CURSO ESPECIAL","fichas":4,"prom":23.0,"dur":40,"ap":92,"meses":6,"ultima":"2024-09-09","estrella":false},{"prog":"GESTION DEL DESARROLLO DEL TALENTO HUMANO","nivel":"CURSO ESPECIAL","fichas":4,"prom":17.0,"dur":48,"ap":68,"meses":7,"ultima":"2024-08-15","estrella":false},{"prog":"CREACION DE  FUNCIONES Y GRAFICOS USANDO MICROSOFT EXCEL","nivel":"CURSO ESPECIAL","fichas":3,"prom":27.3,"dur":48,"ap":82,"meses":9,"ultima":"2024-06-19","estrella":false},{"prog":"SISTEMAS TELEINFORMÁTICOS","nivel":"TÉCNICO","fichas":3,"prom":30.0,"dur":2304,"ap":90,"meses":13,"ultima":"2024-02-26","estrella":true},{"prog":"ANALISIS Y DESARROLLO DE SISTEMAS DE INFORMACION","nivel":"TECNÓLOGO","fichas":2,"prom":6.5,"dur":3415,"ap":13,"meses":45,"ultima":"2021-06-28","estrella":false},{"prog":"PROGRAMACION DE APLICACIONES PARA DISPOSITIVOS MOVILES","nivel":"TÉCNICO","fichas":2,"prom":29.0,"dur":2256,"ap":58,"meses":13,"ultima":"2024-02-24","estrella":true},{"prog":"PROCESAMIENTO DE INFORMACION ESTADISTICA","nivel":"CURSO ESPECIAL","fichas":2,"prom":20.0,"dur":60,"ap":40,"meses":18,"ultima":"2023-09-21","estrella":false},{"prog":"DISEÑAR PÁGINAS WEB CON HTML Y JAVASCRIP","nivel":"CURSO ESPECIAL","fichas":1,"prom":20.0,"dur":40,"ap":20,"meses":4,"ultima":"2024-11-18","estrella":false}],"ELECTRICIDAD":[{"prog":"INSTALACION DE SISTEMAS ELECTRICOS RESIDENCIALES Y COMERCIALES","nivel":"TÉCNICO","fichas":10,"prom":20.7,"dur":2208,"ap":207,"meses":4,"ultima":"2024-11-05","estrella":true},{"prog":"ELECTRICIDAD BÁSICA","nivel":"CURSO ESPECIAL","fichas":9,"prom":15.1,"dur":40,"ap":136,"meses":9,"ultima":"2024-06-24","estrella":false},{"prog":"MANTENIMIENTO DE COMPUTADORES NIVEL I","nivel":"CURSO ESPECIAL","fichas":2,"prom":21.0,"dur":40,"ap":42,"meses":4,"ultima":"2024-12-02","estrella":true},{"prog":"CONTROL DE MAQUINAS ELECTRICAS CON LOGICA CABLEADA","nivel":"CURSO ESPECIAL","fichas":1,"prom":16.0,"dur":80,"ap":16,"meses":6,"ultima":"2024-09-16","estrella":false},{"prog":"FUNDAMENTOS DE ELECTRICIDAD","nivel":"CURSO ESPECIAL","fichas":1,"prom":10.0,"dur":60,"ap":10,"meses":25,"ultima":"2023-02-20","estrella":false},{"prog":"INSTALACIONES ELECTRICAS BASADAS EN LA NTC 2050 Y EL REGLAMENTO TECN","nivel":"CURSO ESPECIAL","fichas":1,"prom":16.0,"dur":40,"ap":16,"meses":7,"ultima":"2024-08-12","estrella":false},{"prog":"MANTENIMIENTO Y PRUEBA DE MOTORES ELECTRICOS","nivel":"CURSO ESPECIAL","fichas":1,"prom":15.0,"dur":40,"ap":15,"meses":11,"ultima":"2024-04-08","estrella":false},{"prog":"SEGURIDAD EN RIESGO ELECTRICO","nivel":"CURSO ESPECIAL","fichas":1,"prom":29.0,"dur":40,"ap":29,"meses":9,"ultima":"2024-06-17","estrella":true}],"EDUCACION":[{"prog":"BÁSICO MERCADEO Y SERVICIO AL CLIENTE","nivel":"CURSO ESPECIAL","fichas":8,"prom":27.8,"dur":20,"ap":222,"meses":3,"ultima":"2024-12-05","estrella":true},{"prog":"COMERCIALIZACION DE PRODUCTOS Y SERVICIOS TURISTICOS","nivel":"CURSO ESPECIAL","fichas":5,"prom":19.8,"dur":40,"ap":99,"meses":7,"ultima":"2024-08-20","estrella":false},{"prog":"PLANEACION DE ESTRATEGIAS PEDAGOGICAS Y TECNICAS DIDACTICAS PARA LA ","nivel":"CURSO ESPECIAL","fichas":4,"prom":23.2,"dur":48,"ap":93,"meses":5,"ultima":"2024-10-07","estrella":true},{"prog":"TECNICAS DE PREPARACION DE BEBIDAS A BASE DE CAFE","nivel":"CURSO ESPECIAL","fichas":3,"prom":21.7,"dur":40,"ap":65,"meses":4,"ultima":"2024-11-08","estrella":true},{"prog":"ELABORACION ARTESANAL DE PRODUCTOS DE CHOCOLATERIA FINA","nivel":"CURSO ESPECIAL","fichas":2,"prom":20.0,"dur":40,"ap":40,"meses":10,"ultima":"2024-05-06","estrella":false},{"prog":"APLICACION DE METODOLOGIA SENA PARA DISEÑO CURRICULAR","nivel":"CURSO ESPECIAL","fichas":1,"prom":20.0,"dur":96,"ap":20,"meses":6,"ultima":"2024-09-26","estrella":false},{"prog":"APLICACION DE TECNICAS PARA LA CREACION DE REDES DE VALOR","nivel":"CURSO ESPECIAL","fichas":1,"prom":20.0,"dur":48,"ap":20,"meses":1,"ultima":"2025-02-18","estrella":false},{"prog":"APLICACION DE LA METODOLOGIA DE MARCO LOGICO PARA LA FORMULACION DE ","nivel":"CURSO ESPECIAL","fichas":1,"prom":20.0,"dur":40,"ap":20,"meses":10,"ultima":"2024-05-10","estrella":false},{"prog":"CONTROL DE INVENTARIOS CON HERRAMIENTAS DIGITALES","nivel":"CURSO ESPECIAL","fichas":1,"prom":20.0,"dur":60,"ap":20,"meses":7,"ultima":"2024-08-23","estrella":false},{"prog":"FORMACION LMS TERRITORIUM PARA INSTRUCTORES SENA","nivel":"CURSO ESPECIAL","fichas":1,"prom":21.0,"dur":48,"ap":21,"meses":22,"ultima":"2023-05-23","estrella":false},{"prog":"DISEÑO DE ESTRATEGIAS DIDACTICAS EN MATEMATICAS PARA LA FORMACION PR","nivel":"CURSO ESPECIAL","fichas":1,"prom":22.0,"dur":40,"ap":22,"meses":8,"ultima":"2024-07-29","estrella":true},{"prog":"FORTALECIMIENTO DE LAS COMPETENCIAS BLANDAS PARA LA VIDA Y LA PRODUC","nivel":"CURSO ESPECIAL","fichas":1,"prom":20.0,"dur":48,"ap":20,"meses":1,"ultima":"2025-02-24","estrella":false}],"COMERCIO":[{"prog":"ASESORIA COMERCIAL","nivel":"TÉCNICO","fichas":7,"prom":23.0,"dur":2208,"ap":161,"meses":1,"ultima":"2025-02-24","estrella":true},{"prog":"GESTIÓN DE MERCADOS","nivel":"TECNÓLOGO","fichas":7,"prom":16.6,"dur":3345,"ap":116,"meses":28,"ultima":"2022-12-12","estrella":false},{"prog":"GESTIÓN EMPRESARIAL","nivel":"TECNÓLOGO","fichas":6,"prom":25.5,"dur":3615,"ap":153,"meses":7,"ultima":"2024-09-02","estrella":true},{"prog":"VENTA DE PRODUCTOS EN LINEA","nivel":"TÉCNICO","fichas":5,"prom":43.0,"dur":2208,"ap":215,"meses":6,"ultima":"2024-09-16","estrella":true},{"prog":"MERCADEO Y VENTAS","nivel":"CURSO ESPECIAL","fichas":4,"prom":21.8,"dur":50,"ap":87,"meses":5,"ultima":"2024-10-13","estrella":false},{"prog":"ESTRATEGIAS DE COMUNICACION INTEGRADA DE MARKETING","nivel":"CURSO ESPECIAL","fichas":3,"prom":20.0,"dur":80,"ap":60,"meses":5,"ultima":"2024-10-27","estrella":false},{"prog":"COMERCIALIZACION DE PRODUCTOS AGROINDUSTRIALES","nivel":"CURSO ESPECIAL","fichas":1,"prom":17.0,"dur":48,"ap":17,"meses":1,"ultima":"2025-02-20","estrella":false},{"prog":"EMPRENDEDOR EN COMERCIALIZACIÓN DE PRODUCTOS DEL SECTOR PRIMARIO Y A","nivel":"CURSO ESPECIAL","fichas":1,"prom":22.0,"dur":200,"ap":22,"meses":12,"ultima":"2024-03-20","estrella":true},{"prog":"TÉCNICAS DE VENTAS","nivel":"CURSO ESPECIAL","fichas":1,"prom":20.0,"dur":20,"ap":20,"meses":9,"ultima":"2024-06-24","estrella":false},{"prog":"VITRINISMO COMO ESTRATEGIA DE COMUNICACION Y MERCADEO","nivel":"CURSO ESPECIAL","fichas":1,"prom":20.0,"dur":40,"ap":20,"meses":17,"ultima":"2023-10-17","estrella":false}],"CONSTRUCCION":[{"prog":"ACONDICIONAMIENTO DE ANDAMIOS PARA TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","fichas":12,"prom":7.4,"dur":48,"ap":89,"meses":0,"ultima":"2025-03-06","estrella":false},{"prog":"CONSTRUCCION DE EDIFICACIONES","nivel":"TÉCNICO","fichas":10,"prom":26.7,"dur":2208,"ap":267,"meses":4,"ultima":"2024-11-05","estrella":true},{"prog":"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 1","nivel":"CURSO ESPECIAL","fichas":5,"prom":25.0,"dur":48,"ap":125,"meses":5,"ultima":"2024-10-28","estrella":true},{"prog":"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 2","nivel":"CURSO ESPECIAL","fichas":4,"prom":28.5,"dur":48,"ap":114,"meses":7,"ultima":"2024-09-02","estrella":true},{"prog":"CONSTRUCCION EN EDIFICACIONES.","nivel":"TECNÓLOGO","fichas":4,"prom":19.8,"dur":3984,"ap":79,"meses":7,"ultima":"2024-09-02","estrella":false},{"prog":"INTERPRETACIÓN DE PLANOS ARQUITECTÓNICOS","nivel":"CURSO ESPECIAL","fichas":4,"prom":23.0,"dur":45,"ap":92,"meses":7,"ultima":"2024-09-02","estrella":false},{"prog":"CONSTRUCCIONES LIVIANAS INDUSTRIALIZADAS EN SECO","nivel":"TÉCNICO","fichas":3,"prom":14.3,"dur":2208,"ap":43,"meses":31,"ultima":"2022-08-22","estrella":false},{"prog":"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 3","nivel":"CURSO ESPECIAL","fichas":3,"prom":23.7,"dur":48,"ap":71,"meses":6,"ultima":"2024-09-23","estrella":true},{"prog":"LEVANTAMIENTOS TOPOGRAFICOS Y GEORREFERENCIACION","nivel":"TECNÓLOGO","fichas":3,"prom":21.3,"dur":3984,"ap":64,"meses":16,"ultima":"2023-11-30","estrella":false},{"prog":"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 4","nivel":"CURSO ESPECIAL","fichas":3,"prom":21.3,"dur":48,"ap":64,"meses":6,"ultima":"2024-09-30","estrella":false},{"prog":"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 6: P","nivel":"CURSO ESPECIAL","fichas":2,"prom":21.0,"dur":48,"ap":42,"meses":4,"ultima":"2024-11-15","estrella":false},{"prog":"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 5: M","nivel":"CURSO ESPECIAL","fichas":2,"prom":21.0,"dur":48,"ap":42,"meses":5,"ultima":"2024-10-28","estrella":false}],"SALUD":[{"prog":"JEFES DE AREA PARA TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","fichas":39,"prom":7.9,"dur":8,"ap":309,"meses":0,"ultima":"2025-03-05","estrella":false},{"prog":"TRABAJADOR AUTORIZADO PARA TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","fichas":31,"prom":7.3,"dur":32,"ap":226,"meses":0,"ultima":"2025-03-04","estrella":false},{"prog":"PRIMEROS  AUXILIOS","nivel":"CURSO ESPECIAL","fichas":21,"prom":25.8,"dur":48,"ap":542,"meses":1,"ultima":"2025-02-20","estrella":true},{"prog":"REENTRENAMIENTO EN TRABAJO EN ALTURAS PARA TRABAJADOR AUTORIZADO","nivel":"CURSO ESPECIAL","fichas":17,"prom":7.1,"dur":16,"ap":121,"meses":1,"ultima":"2025-02-20","estrella":false},{"prog":"COORDINADOR DE TRABAJO\xa0EN ALTURAS","nivel":"CURSO ESPECIAL","fichas":5,"prom":8.8,"dur":80,"ap":44,"meses":4,"ultima":"2024-11-18","estrella":false},{"prog":"HABITOS SALUDABLES A PARTIR DE LA ALIMENTACION Y LA ACTIVIDAD FISICA","nivel":"CURSO ESPECIAL","fichas":4,"prom":20.0,"dur":40,"ap":80,"meses":18,"ultima":"2023-09-22","estrella":false},{"prog":"IMPLEMENTACION DEL SISTEMA DE GESTION DE LA SEGURIDAD Y SALUD EN EL ","nivel":"CURSO ESPECIAL","fichas":3,"prom":20.7,"dur":50,"ap":62,"meses":5,"ultima":"2024-10-14","estrella":true},{"prog":"ATENCION INTEGRAL DE URGENCIAS A VICTIMAS DE ATAQUE CON AGENTES QUIM","nivel":"CURSO ESPECIAL","fichas":2,"prom":79.5,"dur":48,"ap":159,"meses":16,"ultima":"2023-11-09","estrella":true},{"prog":"BIOSEGURIDAD","nivel":"CURSO ESPECIAL","fichas":1,"prom":21.0,"dur":48,"ap":21,"meses":12,"ultima":"2024-04-03","estrella":true},{"prog":"RESCATE INDUSTRIAL EN TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","fichas":1,"prom":9.0,"dur":48,"ap":9,"meses":11,"ultima":"2024-04-08","estrella":false}],"MINERIA":[{"prog":"ADITIVOS: ANALISIS Y CONTROL DE CALIDAD EN LA INDUSTRIA ALIMENTARIA","nivel":"CURSO ESPECIAL","fichas":5,"prom":10.2,"dur":40,"ap":51,"meses":17,"ultima":"2023-10-13","estrella":false},{"prog":"ASOCIATIVIDAD","nivel":"EVENTO","fichas":1,"prom":60.0,"dur":4,"ap":60,"meses":22,"ultima":"2023-05-28","estrella":true}],"TEXTILES":[{"prog":"PLANIFICACION DE UN SISTEMA DE GESTION DE LA CALIDAD - NTC ISO 9001","nivel":"CURSO ESPECIAL","fichas":5,"prom":14.4,"dur":40,"ap":72,"meses":18,"ultima":"2023-09-21","estrella":true}]}'
_MS = '{"ABRIAQUÍ":{"ap":746,"fichas":36,"prom":20.7,"sector":"AGROPECUARIO","tend":0.0,"lat":6.6384,"lng":-76.0946},"ANZÁ":{"ap":1552,"fichas":72,"prom":21.6,"sector":"AGROPECUARIO","tend":0.0,"lat":6.3185,"lng":-75.88},"ARMENIA":{"ap":631,"fichas":27,"prom":23.4,"sector":"SERVICIOS","tend":0.0,"lat":5.0893,"lng":-75.6718},"BURITICA":{"ap":1612,"fichas":84,"prom":19.2,"sector":"SERVICIOS","tend":0.0,"lat":6.7297,"lng":-75.9045},"CAICEDO":{"ap":1946,"fichas":80,"prom":24.3,"sector":"SERVICIOS","tend":0.0,"lat":6.4141,"lng":-76.0204},"CAÑASGORDAS":{"ap":3143,"fichas":142,"prom":22.1,"sector":"SERVICIOS","tend":0.0,"lat":6.7393,"lng":-75.9944},"DABEIBA":{"ap":6874,"fichas":320,"prom":21.5,"sector":"SERVICIOS","tend":0.0,"lat":7.0091,"lng":-76.2581},"EBÉJICO":{"ap":2940,"fichas":133,"prom":22.1,"sector":"SERVICIOS","tend":0.0,"lat":6.3328,"lng":-75.7449},"FRONTINO":{"ap":3642,"fichas":158,"prom":23.1,"sector":"SERVICIOS","tend":0.0,"lat":6.7854,"lng":-76.1358},"GIRALDO":{"ap":1900,"fichas":87,"prom":21.8,"sector":"SERVICIOS","tend":0.0,"lat":6.6207,"lng":-75.8819},"HELICONIA":{"ap":1005,"fichas":50,"prom":20.1,"sector":"SERVICIOS","tend":0.0,"lat":6.2066,"lng":-75.7487},"ITAGUÍ":{"ap":10,"fichas":1,"prom":10.0,"sector":"CONSTRUCCION","tend":0.0,"lat":6.1849,"lng":-75.5991},"LA CEJA":{"ap":40,"fichas":1,"prom":40.0,"sector":"SALUD","tend":0.0,"lat":6.0218,"lng":-75.4358},"LIBORINA":{"ap":2213,"fichas":98,"prom":22.6,"sector":"SERVICIOS","tend":0.0,"lat":6.6944,"lng":-75.9336},"MEDELLÍN":{"ap":1389,"fichas":57,"prom":24.4,"sector":"SERVICIOS","tend":0.0,"lat":6.2518,"lng":-75.5636},"OLAYA":{"ap":834,"fichas":35,"prom":23.8,"sector":"SERVICIOS","tend":0.0,"lat":6.592,"lng":-75.8555},"PEQUE":{"ap":973,"fichas":40,"prom":24.3,"sector":"SERVICIOS","tend":0.0,"lat":6.989,"lng":-75.9994},"SABANALARGA":{"ap":1118,"fichas":43,"prom":26.0,"sector":"SERVICIOS","tend":0.0,"lat":6.8878,"lng":-75.7083},"SAN JERÓNIMO":{"ap":3165,"fichas":135,"prom":23.4,"sector":"SERVICIOS","tend":0.0,"lat":6.4827,"lng":-75.7173},"SANTAFÉ DE ANTIOQUIA":{"ap":21072,"fichas":713,"prom":29.6,"sector":"SERVICIOS","tend":0.0,"lat":6.5566,"lng":-75.8238},"SOPETRÁN":{"ap":2865,"fichas":127,"prom":22.6,"sector":"SERVICIOS","tend":0.0,"lat":6.5058,"lng":-75.7383},"URAMITA":{"ap":2544,"fichas":112,"prom":22.7,"sector":"SERVICIOS","tend":0.0,"lat":6.9108,"lng":-76.1642}}'
_RS = '[{"n":"TOL GRANJA","fichas":209,"prom":40.2,"total":8397,"max_ap":320},{"n":"ADRIANA MARIA AGUIRRE LOPEZ","fichas":200,"prom":27.8,"total":5564,"max_ap":100},{"n":"ARLES ASNEY VALOYES OBREGON","fichas":72,"prom":18.6,"total":1339,"max_ap":33},{"n":"GLORIA MILENA FLOREZ MARTINEZ","fichas":55,"prom":23.9,"total":1314,"max_ap":55},{"n":"SEBASTIAN GUARIN FONSECA","fichas":55,"prom":7.9,"total":435,"max_ap":10},{"n":"CARLOS JULIO RUBIO ESCOBAR","fichas":50,"prom":26.5,"total":1327,"max_ap":37},{"n":"JUAN DIEGO MARIN ORTIZ","fichas":49,"prom":26.6,"total":1304,"max_ap":100},{"n":"MARIBEL GALEANO ROJAS","fichas":48,"prom":7.3,"total":352,"max_ap":10},{"n":"JESUS DAVID TRUJILLO BERMEO","fichas":43,"prom":26.4,"total":1135,"max_ap":46},{"n":"JHON ALEXANDER LEZCANO OQUENDO","fichas":36,"prom":21.4,"total":769,"max_ap":46},{"n":"JULIO CESAR LONDOÑO HERNANDEZ","fichas":33,"prom":19.0,"total":626,"max_ap":85},{"n":"LUISA FERNANDA PELAEZ TAMAYO","fichas":33,"prom":23.5,"total":775,"max_ap":31},{"n":"SARA MARIA BORJA CIFUENTES","fichas":32,"prom":18.5,"total":592,"max_ap":66},{"n":"SINDY JULIETH MENA PRIETO","fichas":32,"prom":17.7,"total":567,"max_ap":24},{"n":"EDGARDO MANUEL PERTUZ MENDOZA","fichas":32,"prom":26.1,"total":836,"max_ap":93},{"n":"ALEJANDRO OMAR CONTRERAS AVILA","fichas":31,"prom":31.5,"total":978,"max_ap":415},{"n":"JHONATAN ANGEL VALENCIA","fichas":29,"prom":25.1,"total":728,"max_ap":101},{"n":"JORGE IVAN PINO MACHADO","fichas":29,"prom":24.1,"total":700,"max_ap":31},{"n":"BERTA TULIA PASTRANA MADERA","fichas":28,"prom":30.7,"total":859,"max_ap":419},{"n":"ALEJANDRA MARIA CANO VASQUEZ","fichas":28,"prom":18.0,"total":504,"max_ap":25}]'
_PM = '{"MANIPULACION HIGIENICA DE ALIMENTOS.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":538,"fichas":16,"prom":33.6},{"mun":"SOPETRÁN","ap":328,"fichas":13,"prom":25.2},{"mun":"SAN JERÓNIMO","ap":293,"fichas":11,"prom":26.6},{"mun":"FRONTINO","ap":248,"fichas":9,"prom":27.6},{"mun":"DABEIBA","ap":225,"fichas":8,"prom":28.1},{"mun":"ANZÁ","ap":210,"fichas":8,"prom":26.2},{"mun":"EBÉJICO","ap":202,"fichas":8,"prom":25.2},{"mun":"MEDELLÍN","ap":185,"fichas":8,"prom":23.1}],"ELABORACION DE HELADOS Y POSTRES LACTEOS":[{"mun":"EBÉJICO","ap":121,"fichas":5,"prom":24.2},{"mun":"SAN JERÓNIMO","ap":46,"fichas":2,"prom":23.0},{"mun":"CAICEDO","ap":35,"fichas":1,"prom":35.0},{"mun":"ABRIAQUÍ","ap":32,"fichas":1,"prom":32.0},{"mun":"SABANALARGA","ap":31,"fichas":1,"prom":31.0},{"mun":"CAÑASGORDAS","ap":25,"fichas":1,"prom":25.0},{"mun":"GIRALDO","ap":24,"fichas":1,"prom":24.0},{"mun":"ANZÁ","ap":22,"fichas":1,"prom":22.0}],"COMPORTAMIENTO EMPRENDEDOR":[{"mun":"DABEIBA","ap":393,"fichas":18,"prom":21.8},{"mun":"CAÑASGORDAS","ap":326,"fichas":13,"prom":25.1},{"mun":"URAMITA","ap":177,"fichas":9,"prom":19.7},{"mun":"EBÉJICO","ap":132,"fichas":6,"prom":22.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":114,"fichas":5,"prom":22.8},{"mun":"GIRALDO","ap":104,"fichas":4,"prom":26.0},{"mun":"SOPETRÁN","ap":100,"fichas":5,"prom":20.0},{"mun":"PEQUE","ap":94,"fichas":4,"prom":23.5}],"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR":[{"mun":"DABEIBA","ap":345,"fichas":16,"prom":21.6},{"mun":"CAÑASGORDAS","ap":301,"fichas":11,"prom":27.4},{"mun":"URAMITA","ap":190,"fichas":9,"prom":21.1},{"mun":"EBÉJICO","ap":118,"fichas":6,"prom":19.7},{"mun":"FRONTINO","ap":117,"fichas":6,"prom":19.5},{"mun":"GIRALDO","ap":94,"fichas":4,"prom":23.5},{"mun":"SOPETRÁN","ap":87,"fichas":4,"prom":21.8},{"mun":"SAN JERÓNIMO","ap":86,"fichas":4,"prom":21.5}],"CONSERVACION DE RECURSOS NATURALES":[{"mun":"PEQUE","ap":154,"fichas":4,"prom":38.5},{"mun":"DABEIBA","ap":142,"fichas":6,"prom":23.7},{"mun":"GIRALDO","ap":107,"fichas":5,"prom":21.4},{"mun":"ABRIAQUÍ","ap":98,"fichas":5,"prom":19.6},{"mun":"LIBORINA","ap":98,"fichas":4,"prom":24.5},{"mun":"FRONTINO","ap":95,"fichas":5,"prom":19.0},{"mun":"CAÑASGORDAS","ap":67,"fichas":3,"prom":22.3},{"mun":"SOPETRÁN","ap":62,"fichas":2,"prom":31.0}],"GESTION DE PROYECTOS COMUNITARIOS":[{"mun":"DABEIBA","ap":182,"fichas":10,"prom":18.2},{"mun":"URAMITA","ap":71,"fichas":3,"prom":23.7},{"mun":"SOPETRÁN","ap":56,"fichas":3,"prom":18.7},{"mun":"ANZÁ","ap":48,"fichas":2,"prom":24.0},{"mun":"CAICEDO","ap":38,"fichas":2,"prom":19.0},{"mun":"PEQUE","ap":37,"fichas":2,"prom":18.5},{"mun":"LIBORINA","ap":37,"fichas":2,"prom":18.5},{"mun":"GIRALDO","ap":36,"fichas":2,"prom":18.0}],"INSEMINACION ARTIFICIAL DE HEMBRAS BOVINAS APLICANDO PROTOCOLO CONVE":[{"mun":"MEDELLÍN","ap":122,"fichas":5,"prom":24.4},{"mun":"ABRIAQUÍ","ap":84,"fichas":4,"prom":21.0},{"mun":"LIBORINA","ap":28,"fichas":1,"prom":28.0},{"mun":"BURITICA","ap":26,"fichas":1,"prom":26.0}],"EMPRENDEDOR EN PROPAGACION DE MATERIAL VEGETAL Y ESTABLECIMIENTO DE ":[{"mun":"BURITICA","ap":55,"fichas":2,"prom":27.5},{"mun":"EBÉJICO","ap":54,"fichas":3,"prom":18.0},{"mun":"SAN JERÓNIMO","ap":25,"fichas":1,"prom":25.0},{"mun":"ABRIAQUÍ","ap":21,"fichas":1,"prom":21.0},{"mun":"GIRALDO","ap":16,"fichas":1,"prom":16.0}],"CRÍA Y LEVANTE DE POLLOS DE ENGORDE CON LA IMPLEMENTACIÓN DE MAÍZ Y ":[{"mun":"URAMITA","ap":35,"fichas":2,"prom":17.5},{"mun":"ABRIAQUÍ","ap":16,"fichas":1,"prom":16.0}],"MANEJO RACIONAL DE PLAGUICIDAS":[{"mun":"DABEIBA","ap":56,"fichas":3,"prom":18.7},{"mun":"EBÉJICO","ap":41,"fichas":2,"prom":20.5},{"mun":"SAN JERÓNIMO","ap":36,"fichas":2,"prom":18.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":23,"fichas":1,"prom":23.0},{"mun":"HELICONIA","ap":21,"fichas":1,"prom":21.0},{"mun":"ABRIAQUÍ","ap":15,"fichas":1,"prom":15.0},{"mun":"GIRALDO","ap":9,"fichas":1,"prom":9.0}],"APLICACION DE BUENAS PRACTICAS GANADERAS  EN BOVINOS DE LECHE":[{"mun":"ABRIAQUÍ","ap":65,"fichas":4,"prom":16.2},{"mun":"ANZÁ","ap":29,"fichas":1,"prom":29.0},{"mun":"URAMITA","ap":26,"fichas":1,"prom":26.0},{"mun":"FRONTINO","ap":20,"fichas":1,"prom":20.0},{"mun":"DABEIBA","ap":12,"fichas":1,"prom":12.0}],"MANEJO DE LA NUTRICION EN CULTIVOS AGRICOLAS":[{"mun":"URAMITA","ap":101,"fichas":6,"prom":16.8},{"mun":"EBÉJICO","ap":63,"fichas":3,"prom":21.0},{"mun":"DABEIBA","ap":31,"fichas":2,"prom":15.5},{"mun":"CAICEDO","ap":26,"fichas":1,"prom":26.0},{"mun":"CAÑASGORDAS","ap":26,"fichas":1,"prom":26.0},{"mun":"HELICONIA","ap":23,"fichas":1,"prom":23.0},{"mun":"ANZÁ","ap":20,"fichas":1,"prom":20.0},{"mun":"ABRIAQUÍ","ap":17,"fichas":1,"prom":17.0}],"PRODUCCION AGROPECUARIA ECOLOGICA PARA LA SOBERANIA ALIMENTARIA":[{"mun":"LIBORINA","ap":130,"fichas":7,"prom":18.6},{"mun":"SOPETRÁN","ap":119,"fichas":5,"prom":23.8},{"mun":"CAÑASGORDAS","ap":108,"fichas":3,"prom":36.0},{"mun":"URAMITA","ap":85,"fichas":4,"prom":21.2},{"mun":"MEDELLÍN","ap":77,"fichas":3,"prom":25.7},{"mun":"CAICEDO","ap":54,"fichas":2,"prom":27.0},{"mun":"SAN JERÓNIMO","ap":54,"fichas":3,"prom":18.0},{"mun":"BURITICA","ap":44,"fichas":2,"prom":22.0}],"EMPRENDIMIENTO EN PRODUCCION  DE FRUTALES SEMIPERENNES":[{"mun":"CAICEDO","ap":29,"fichas":1,"prom":29.0},{"mun":"ABRIAQUÍ","ap":26,"fichas":1,"prom":26.0},{"mun":"EBÉJICO","ap":24,"fichas":1,"prom":24.0},{"mun":"DABEIBA","ap":22,"fichas":1,"prom":22.0}],"HIGIENE Y MANIPULACIÓN DE ALIMENTOS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":823,"fichas":33,"prom":24.9},{"mun":"SOPETRÁN","ap":143,"fichas":6,"prom":23.8},{"mun":"EBÉJICO","ap":112,"fichas":5,"prom":22.4},{"mun":"LIBORINA","ap":110,"fichas":4,"prom":27.5},{"mun":"SAN JERÓNIMO","ap":109,"fichas":4,"prom":27.2},{"mun":"ANZÁ","ap":99,"fichas":4,"prom":24.8},{"mun":"GIRALDO","ap":94,"fichas":4,"prom":23.5},{"mun":"DABEIBA","ap":94,"fichas":4,"prom":23.5}],"APLICACION DEL PROGRAMA DE BIOSEGURIDAD EN EMPRESAS AVICOLAS.":[{"mun":"HELICONIA","ap":31,"fichas":2,"prom":15.5},{"mun":"ANZÁ","ap":19,"fichas":1,"prom":19.0}],"CRIAR GALLINAS PONEDORAS CON ALIMENTACIÓN ALTERNA PARA PRODUCIR HUEV":[{"mun":"HELICONIA","ap":152,"fichas":8,"prom":19.0},{"mun":"GIRALDO","ap":72,"fichas":4,"prom":18.0},{"mun":"ANZÁ","ap":65,"fichas":3,"prom":21.7},{"mun":"ARMENIA","ap":62,"fichas":3,"prom":20.7},{"mun":"EBÉJICO","ap":41,"fichas":2,"prom":20.5},{"mun":"SAN JERÓNIMO","ap":30,"fichas":1,"prom":30.0},{"mun":"LIBORINA","ap":28,"fichas":1,"prom":28.0},{"mun":"URAMITA","ap":27,"fichas":1,"prom":27.0}],"EMPRENDEDOR EN PROCESAMIENTO ARTESANAL  DE DERIVADOS  FRUTAS Y HORTA":[{"mun":"MEDELLÍN","ap":25,"fichas":1,"prom":25.0},{"mun":"CAÑASGORDAS","ap":24,"fichas":1,"prom":24.0},{"mun":"OLAYA","ap":24,"fichas":1,"prom":24.0},{"mun":"ANZÁ","ap":20,"fichas":1,"prom":20.0}],"IMPLEMENTACION DE PRACTICAS DE MANEJO DEL HUEVO":[{"mun":"SAN JERÓNIMO","ap":98,"fichas":4,"prom":24.5},{"mun":"SOPETRÁN","ap":54,"fichas":2,"prom":27.0},{"mun":"DABEIBA","ap":30,"fichas":1,"prom":30.0},{"mun":"FRONTINO","ap":28,"fichas":1,"prom":28.0},{"mun":"EBÉJICO","ap":28,"fichas":1,"prom":28.0},{"mun":"CAÑASGORDAS","ap":27,"fichas":1,"prom":27.0},{"mun":"ARMENIA","ap":24,"fichas":1,"prom":24.0},{"mun":"ANZÁ","ap":22,"fichas":1,"prom":22.0}],"PRODUCCION AGROPECUARIA":[{"mun":"FRONTINO","ap":199,"fichas":9,"prom":22.1},{"mun":"DABEIBA","ap":154,"fichas":5,"prom":30.8},{"mun":"ANZÁ","ap":23,"fichas":1,"prom":23.0},{"mun":"CAÑASGORDAS","ap":21,"fichas":1,"prom":21.0}],"PRACTICAS DE CALIDAD PARA EL BENEFICIO DEL CAFE":[{"mun":"ANZÁ","ap":148,"fichas":8,"prom":18.5},{"mun":"BURITICA","ap":69,"fichas":3,"prom":23.0},{"mun":"SABANALARGA","ap":43,"fichas":1,"prom":43.0},{"mun":"SOPETRÁN","ap":37,"fichas":1,"prom":37.0},{"mun":"SAN JERÓNIMO","ap":36,"fichas":2,"prom":18.0},{"mun":"CAICEDO","ap":32,"fichas":2,"prom":16.0},{"mun":"EBÉJICO","ap":30,"fichas":2,"prom":15.0},{"mun":"LIBORINA","ap":21,"fichas":1,"prom":21.0}],"PROGRAMACION DE SOFTWARE .":[{"mun":"EBÉJICO","ap":237,"fichas":10,"prom":23.7},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":154,"fichas":5,"prom":30.8},{"mun":"SOPETRÁN","ap":108,"fichas":4,"prom":27.0},{"mun":"GIRALDO","ap":58,"fichas":2,"prom":29.0},{"mun":"CAÑASGORDAS","ap":42,"fichas":2,"prom":21.0},{"mun":"SAN JERÓNIMO","ap":24,"fichas":1,"prom":24.0},{"mun":"ANZÁ","ap":23,"fichas":1,"prom":23.0},{"mun":"OLAYA","ap":18,"fichas":1,"prom":18.0}],"COCTELERIA TROPICAL":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":67,"fichas":3,"prom":22.3},{"mun":"LIBORINA","ap":42,"fichas":2,"prom":21.0},{"mun":"ANZÁ","ap":40,"fichas":2,"prom":20.0}],"PATRONAJE DE PRENDAS DE ROPA EXTERIOR FEMENINA":[{"mun":"ANZÁ","ap":42,"fichas":2,"prom":21.0},{"mun":"SAN JERÓNIMO","ap":39,"fichas":2,"prom":19.5},{"mun":"FRONTINO","ap":20,"fichas":1,"prom":20.0},{"mun":"OLAYA","ap":19,"fichas":1,"prom":19.0},{"mun":"LIBORINA","ap":19,"fichas":1,"prom":19.0},{"mun":"DABEIBA","ap":16,"fichas":1,"prom":16.0},{"mun":"CAICEDO","ap":14,"fichas":1,"prom":14.0},{"mun":"SABANALARGA","ap":13,"fichas":1,"prom":13.0}],"APLICACION DEL PLAN SANITARIO EN GRANJAS DE PRODUCCION DE HUEVO":[{"mun":"HELICONIA","ap":91,"fichas":5,"prom":18.2},{"mun":"SAN JERÓNIMO","ap":30,"fichas":1,"prom":30.0},{"mun":"ANZÁ","ap":27,"fichas":1,"prom":27.0},{"mun":"SOPETRÁN","ap":26,"fichas":1,"prom":26.0},{"mun":"CAÑASGORDAS","ap":17,"fichas":1,"prom":17.0},{"mun":"MEDELLÍN","ap":17,"fichas":1,"prom":17.0}],"CARACTERIZACION DE ELEMENTOS DEL MARKETING":[{"mun":"DABEIBA","ap":59,"fichas":3,"prom":19.7},{"mun":"CAICEDO","ap":43,"fichas":2,"prom":21.5},{"mun":"ANZÁ","ap":18,"fichas":1,"prom":18.0}],"PRODUCCION DE CAFE":[{"mun":"ANZÁ","ap":120,"fichas":4,"prom":30.0},{"mun":"CAICEDO","ap":34,"fichas":2,"prom":17.0}],"EMPRENDIMIENTO EN LA PRODUCCION DE ESPECIES DULCES ACUICOLAS":[{"mun":"PEQUE","ap":32,"fichas":2,"prom":16.0},{"mun":"ANZÁ","ap":25,"fichas":1,"prom":25.0},{"mun":"BURITICA","ap":22,"fichas":1,"prom":22.0},{"mun":"CAÑASGORDAS","ap":15,"fichas":1,"prom":15.0},{"mun":"DABEIBA","ap":15,"fichas":1,"prom":15.0},{"mun":"EBÉJICO","ap":15,"fichas":1,"prom":15.0}],"CATACION DE CAFE NIVEL 1":[{"mun":"SAN JERÓNIMO","ap":30,"fichas":1,"prom":30.0},{"mun":"CAICEDO","ap":29,"fichas":1,"prom":29.0},{"mun":"ANZÁ","ap":20,"fichas":1,"prom":20.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":10,"fichas":1,"prom":10.0}],"EMPRENDEDOR EN PROCESAMIENTO DE PRODUCTOS DERIVADOS DEL CAFE":[{"mun":"EBÉJICO","ap":57,"fichas":2,"prom":28.5},{"mun":"PEQUE","ap":35,"fichas":1,"prom":35.0},{"mun":"ANZÁ","ap":34,"fichas":1,"prom":34.0},{"mun":"CAICEDO","ap":31,"fichas":1,"prom":31.0},{"mun":"SAN JERÓNIMO","ap":30,"fichas":1,"prom":30.0},{"mun":"DABEIBA","ap":28,"fichas":1,"prom":28.0},{"mun":"SOPETRÁN","ap":26,"fichas":1,"prom":26.0},{"mun":"GIRALDO","ap":19,"fichas":1,"prom":19.0}],"APLICACION DE LAS BUENAS PRACTICAS AGRICOLAS":[{"mun":"DABEIBA","ap":175,"fichas":7,"prom":25.0},{"mun":"URAMITA","ap":152,"fichas":6,"prom":25.3},{"mun":"PEQUE","ap":100,"fichas":2,"prom":50.0},{"mun":"EBÉJICO","ap":88,"fichas":4,"prom":22.0},{"mun":"CAÑASGORDAS","ap":86,"fichas":4,"prom":21.5},{"mun":"SOPETRÁN","ap":82,"fichas":3,"prom":27.3},{"mun":"SAN JERÓNIMO","ap":79,"fichas":4,"prom":19.8},{"mun":"ANZÁ","ap":75,"fichas":5,"prom":15.0}],"INGLES BASICO - NIVEL 1":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":450,"fichas":18,"prom":25.0},{"mun":"SAN JERÓNIMO","ap":80,"fichas":3,"prom":26.7},{"mun":"DABEIBA","ap":63,"fichas":3,"prom":21.0},{"mun":"FRONTINO","ap":45,"fichas":2,"prom":22.5},{"mun":"CAÑASGORDAS","ap":44,"fichas":1,"prom":44.0},{"mun":"SOPETRÁN","ap":42,"fichas":2,"prom":21.0},{"mun":"LIBORINA","ap":36,"fichas":1,"prom":36.0},{"mun":"OLAYA","ap":28,"fichas":1,"prom":28.0}],"EMPRENDIMIENTO EN APICULTURA":[{"mun":"URAMITA","ap":76,"fichas":3,"prom":25.3},{"mun":"FRONTINO","ap":52,"fichas":2,"prom":26.0},{"mun":"CAÑASGORDAS","ap":29,"fichas":1,"prom":29.0},{"mun":"LIBORINA","ap":28,"fichas":1,"prom":28.0},{"mun":"EBÉJICO","ap":26,"fichas":1,"prom":26.0},{"mun":"ANZÁ","ap":26,"fichas":1,"prom":26.0},{"mun":"SOPETRÁN","ap":22,"fichas":1,"prom":22.0},{"mun":"SABANALARGA","ap":18,"fichas":1,"prom":18.0}],"HIGIENE Y MANIPULACION DE ALIMENTOS.":[{"mun":"HELICONIA","ap":183,"fichas":10,"prom":18.3},{"mun":"SOPETRÁN","ap":140,"fichas":8,"prom":17.5},{"mun":"DABEIBA","ap":74,"fichas":4,"prom":18.5},{"mun":"CAÑASGORDAS","ap":66,"fichas":4,"prom":16.5},{"mun":"ANZÁ","ap":66,"fichas":4,"prom":16.5},{"mun":"FRONTINO","ap":32,"fichas":2,"prom":16.0},{"mun":"GIRALDO","ap":18,"fichas":1,"prom":18.0}],"EMPRENDIMIENTO EN PRODUCCION DE CULTIVOS TRANSITORIOS":[{"mun":"LIBORINA","ap":59,"fichas":2,"prom":29.5},{"mun":"BURITICA","ap":35,"fichas":2,"prom":17.5},{"mun":"EBÉJICO","ap":27,"fichas":1,"prom":27.0},{"mun":"CAÑASGORDAS","ap":27,"fichas":1,"prom":27.0},{"mun":"ARMENIA","ap":21,"fichas":1,"prom":21.0},{"mun":"PEQUE","ap":8,"fichas":1,"prom":8.0}],"INFORMATICA: MICROSOFT WORD, EXCEL E INTERNET":[{"mun":"DABEIBA","ap":224,"fichas":10,"prom":22.4},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":161,"fichas":7,"prom":23.0},{"mun":"ARMENIA","ap":58,"fichas":3,"prom":19.3},{"mun":"GIRALDO","ap":34,"fichas":1,"prom":34.0},{"mun":"CAICEDO","ap":28,"fichas":1,"prom":28.0}],"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL":[{"mun":"DABEIBA","ap":341,"fichas":20,"prom":17.1},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":206,"fichas":9,"prom":22.9},{"mun":"MEDELLÍN","ap":172,"fichas":8,"prom":21.5},{"mun":"FRONTINO","ap":125,"fichas":6,"prom":20.8},{"mun":"SAN JERÓNIMO","ap":86,"fichas":4,"prom":21.5},{"mun":"LIBORINA","ap":67,"fichas":3,"prom":22.3},{"mun":"BURITICA","ap":48,"fichas":3,"prom":16.0},{"mun":"SOPETRÁN","ap":43,"fichas":2,"prom":21.5}],"APLICACION DE  BUENAS PRACTICAS GANADERAS EN LA PRODUCCION DE BOVINO":[{"mun":"FRONTINO","ap":81,"fichas":3,"prom":27.0},{"mun":"URAMITA","ap":37,"fichas":1,"prom":37.0},{"mun":"EBÉJICO","ap":37,"fichas":1,"prom":37.0},{"mun":"DABEIBA","ap":31,"fichas":1,"prom":31.0},{"mun":"CAÑASGORDAS","ap":27,"fichas":1,"prom":27.0},{"mun":"SOPETRÁN","ap":22,"fichas":1,"prom":22.0},{"mun":"SAN JERÓNIMO","ap":21,"fichas":1,"prom":21.0},{"mun":"HELICONIA","ap":21,"fichas":1,"prom":21.0}],"ELABORACION DE ACCESORIOS TEJIDOS EN MOSTACILLAS":[{"mun":"ARMENIA","ap":57,"fichas":2,"prom":28.5},{"mun":"MEDELLÍN","ap":35,"fichas":1,"prom":35.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":29,"fichas":1,"prom":29.0},{"mun":"GIRALDO","ap":25,"fichas":1,"prom":25.0}],"ATENCION AL CLIENTE POR MEDIOS TECNOLOGICOS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":120,"fichas":3,"prom":40.0},{"mun":"SAN JERÓNIMO","ap":45,"fichas":2,"prom":22.5},{"mun":"ARMENIA","ap":24,"fichas":1,"prom":24.0},{"mun":"EBÉJICO","ap":22,"fichas":1,"prom":22.0}],"MONITOREO DE PLAGAS EN SISTEMAS PRODUCTIVOS AGRICOLAS":[{"mun":"HELICONIA","ap":45,"fichas":3,"prom":15.0},{"mun":"ARMENIA","ap":30,"fichas":2,"prom":15.0},{"mun":"LIBORINA","ap":24,"fichas":1,"prom":24.0}],"COCINA NAVIDEÑA":[{"mun":"SAN JERÓNIMO","ap":72,"fichas":3,"prom":24.0},{"mun":"EBÉJICO","ap":30,"fichas":1,"prom":30.0},{"mun":"ARMENIA","ap":25,"fichas":1,"prom":25.0},{"mun":"LIBORINA","ap":21,"fichas":1,"prom":21.0},{"mun":"CAICEDO","ap":20,"fichas":1,"prom":20.0},{"mun":"CAÑASGORDAS","ap":20,"fichas":1,"prom":20.0},{"mun":"SOPETRÁN","ap":20,"fichas":1,"prom":20.0}],"PELUQUERIA.":[{"mun":"ARMENIA","ap":62,"fichas":2,"prom":31.0},{"mun":"SAN JERÓNIMO","ap":54,"fichas":2,"prom":27.0},{"mun":"FRONTINO","ap":17,"fichas":1,"prom":17.0},{"mun":"PEQUE","ap":13,"fichas":1,"prom":13.0}],"TENDENCIAS EN DECORACION PARA MANICURA Y PEDICURA":[{"mun":"SAN JERÓNIMO","ap":91,"fichas":4,"prom":22.8},{"mun":"ARMENIA","ap":55,"fichas":2,"prom":27.5},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":22,"fichas":1,"prom":22.0},{"mun":"CAÑASGORDAS","ap":21,"fichas":1,"prom":21.0},{"mun":"SABANALARGA","ap":20,"fichas":1,"prom":20.0},{"mun":"FRONTINO","ap":12,"fichas":1,"prom":12.0}],"ASISTENCIA ADMINISTRATIVA .":[{"mun":"FRONTINO","ap":333,"fichas":10,"prom":33.3},{"mun":"LIBORINA","ap":215,"fichas":8,"prom":26.9},{"mun":"EBÉJICO","ap":145,"fichas":5,"prom":29.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":85,"fichas":3,"prom":28.3},{"mun":"SABANALARGA","ap":83,"fichas":3,"prom":27.7},{"mun":"DABEIBA","ap":81,"fichas":3,"prom":27.0},{"mun":"BURITICA","ap":73,"fichas":3,"prom":24.3},{"mun":"URAMITA","ap":59,"fichas":2,"prom":29.5}],"APLICACION DE FUNCIONES Y COMBINACIONES MULTIPLES EN EXCEL PARA SOLU":[{"mun":"BURITICA","ap":60,"fichas":2,"prom":30.0},{"mun":"DABEIBA","ap":59,"fichas":6,"prom":9.8},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":27,"fichas":1,"prom":27.0},{"mun":"FRONTINO","ap":22,"fichas":1,"prom":22.0},{"mun":"SAN JERÓNIMO","ap":21,"fichas":1,"prom":21.0}],"EXCEL INTERMEDIO":[{"mun":"DABEIBA","ap":64,"fichas":3,"prom":21.3},{"mun":"BURITICA","ap":45,"fichas":2,"prom":22.5},{"mun":"SOPETRÁN","ap":43,"fichas":2,"prom":21.5},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":41,"fichas":2,"prom":20.5},{"mun":"SAN JERÓNIMO","ap":22,"fichas":1,"prom":22.0},{"mun":"LIBORINA","ap":21,"fichas":1,"prom":21.0},{"mun":"FRONTINO","ap":20,"fichas":1,"prom":20.0}],"ELABORACIÓN DE PANES ARTESANALES":[{"mun":"CAÑASGORDAS","ap":72,"fichas":3,"prom":24.0},{"mun":"SAN JERÓNIMO","ap":66,"fichas":2,"prom":33.0},{"mun":"BURITICA","ap":58,"fichas":2,"prom":29.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":49,"fichas":2,"prom":24.5},{"mun":"OLAYA","ap":41,"fichas":1,"prom":41.0},{"mun":"PEQUE","ap":36,"fichas":1,"prom":36.0},{"mun":"DABEIBA","ap":20,"fichas":1,"prom":20.0}],"INSTALACION DE SISTEMAS ELECTRICOS RESIDENCIALES Y COMERCIALES":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":58,"fichas":2,"prom":29.0},{"mun":"FRONTINO","ap":48,"fichas":2,"prom":24.0},{"mun":"DABEIBA","ap":45,"fichas":2,"prom":22.5},{"mun":"LIBORINA","ap":30,"fichas":2,"prom":15.0},{"mun":"SOPETRÁN","ap":18,"fichas":1,"prom":18.0},{"mun":"BURITICA","ap":8,"fichas":1,"prom":8.0}],"DECORACION DE OBJETOS ARTESANALES EN ARTE COUNTRY":[{"mun":"FRONTINO","ap":49,"fichas":2,"prom":24.5},{"mun":"EBÉJICO","ap":46,"fichas":2,"prom":23.0},{"mun":"BURITICA","ap":45,"fichas":2,"prom":22.5},{"mun":"CAÑASGORDAS","ap":25,"fichas":1,"prom":25.0},{"mun":"SOPETRÁN","ap":25,"fichas":1,"prom":25.0},{"mun":"OLAYA","ap":24,"fichas":1,"prom":24.0}],"ELABORACIÓN DE ARTÍCULOS DECORATIVOS Y UTILITARIOS APLICANDO TÉCNICA":[{"mun":"CAÑASGORDAS","ap":31,"fichas":1,"prom":31.0},{"mun":"DABEIBA","ap":28,"fichas":1,"prom":28.0},{"mun":"MEDELLÍN","ap":26,"fichas":1,"prom":26.0},{"mun":"FRONTINO","ap":21,"fichas":1,"prom":21.0},{"mun":"BURITICA","ap":20,"fichas":1,"prom":20.0}],"EMPRENDIMIENTO EN PRODUCCION DE CAFE":[{"mun":"DABEIBA","ap":45,"fichas":2,"prom":22.5},{"mun":"SAN JERÓNIMO","ap":27,"fichas":1,"prom":27.0},{"mun":"BURITICA","ap":17,"fichas":1,"prom":17.0}],"TECNICAS DE PREPARACION DE BEBIDAS A BASE DE CAFE":[{"mun":"EBÉJICO","ap":25,"fichas":1,"prom":25.0},{"mun":"BURITICA","ap":20,"fichas":1,"prom":20.0},{"mun":"SABANALARGA","ap":20,"fichas":1,"prom":20.0}],"ELECTRICIDAD BÁSICA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":49,"fichas":4,"prom":12.2},{"mun":"FRONTINO","ap":31,"fichas":2,"prom":15.5},{"mun":"DABEIBA","ap":22,"fichas":1,"prom":22.0},{"mun":"BURITICA","ap":20,"fichas":1,"prom":20.0},{"mun":"LIBORINA","ap":14,"fichas":1,"prom":14.0}],"RECONOCIMIENTO Y REGISTRO DE ESPECIES PARA EL AVITURISMO":[{"mun":"LIBORINA","ap":46,"fichas":2,"prom":23.0},{"mun":"EBÉJICO","ap":42,"fichas":2,"prom":21.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":35,"fichas":2,"prom":17.5},{"mun":"CAICEDO","ap":30,"fichas":1,"prom":30.0},{"mun":"BURITICA","ap":26,"fichas":1,"prom":26.0},{"mun":"SABANALARGA","ap":9,"fichas":1,"prom":9.0}],"ACONDICIONAMIENTO DE ANDAMIOS PARA TRABAJO EN ALTURAS":[{"mun":"BURITICA","ap":73,"fichas":10,"prom":7.3},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":16,"fichas":2,"prom":8.0}],"JEFES DE AREA PARA TRABAJO EN ALTURAS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":201,"fichas":26,"prom":7.7},{"mun":"BURITICA","ap":68,"fichas":8,"prom":8.5},{"mun":"LIBORINA","ap":31,"fichas":4,"prom":7.8},{"mun":"DABEIBA","ap":9,"fichas":1,"prom":9.0}],"TRABAJADOR AUTORIZADO PARA TRABAJO EN ALTURAS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":162,"fichas":22,"prom":7.4},{"mun":"DABEIBA","ap":35,"fichas":5,"prom":7.0},{"mun":"CAÑASGORDAS","ap":22,"fichas":3,"prom":7.3},{"mun":"BURITICA","ap":7,"fichas":1,"prom":7.0}],"CONTABILIZACION DE OPERACIONES COMERCIALES Y FINANCIERAS.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":136,"fichas":4,"prom":34.0},{"mun":"URAMITA","ap":77,"fichas":3,"prom":25.7},{"mun":"CAÑASGORDAS","ap":66,"fichas":3,"prom":22.0},{"mun":"BURITICA","ap":63,"fichas":2,"prom":31.5},{"mun":"GIRALDO","ap":52,"fichas":2,"prom":26.0},{"mun":"SOPETRÁN","ap":50,"fichas":2,"prom":25.0},{"mun":"EBÉJICO","ap":44,"fichas":2,"prom":22.0},{"mun":"LIBORINA","ap":29,"fichas":1,"prom":29.0}],"PRIMEROS  AUXILIOS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":75,"fichas":3,"prom":25.0},{"mun":"DABEIBA","ap":73,"fichas":3,"prom":24.3},{"mun":"URAMITA","ap":60,"fichas":2,"prom":30.0},{"mun":"SOPETRÁN","ap":58,"fichas":2,"prom":29.0},{"mun":"BURITICA","ap":51,"fichas":2,"prom":25.5},{"mun":"CAICEDO","ap":50,"fichas":2,"prom":25.0},{"mun":"LIBORINA","ap":49,"fichas":2,"prom":24.5},{"mun":"FRONTINO","ap":40,"fichas":2,"prom":20.0}],"INGLES BASICO - NIVEL 2":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":434,"fichas":16,"prom":27.1},{"mun":"SAN JERÓNIMO","ap":57,"fichas":2,"prom":28.5},{"mun":"FRONTINO","ap":51,"fichas":2,"prom":25.5},{"mun":"DABEIBA","ap":47,"fichas":2,"prom":23.5},{"mun":"CAICEDO","ap":39,"fichas":2,"prom":19.5},{"mun":"SOPETRÁN","ap":22,"fichas":1,"prom":22.0},{"mun":"CAÑASGORDAS","ap":20,"fichas":1,"prom":20.0}],"INGLES BASICO - NIVEL 3":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":429,"fichas":15,"prom":28.6},{"mun":"DABEIBA","ap":94,"fichas":3,"prom":31.3},{"mun":"SOPETRÁN","ap":55,"fichas":2,"prom":27.5},{"mun":"CAICEDO","ap":52,"fichas":3,"prom":17.3},{"mun":"FRONTINO","ap":21,"fichas":1,"prom":21.0}],"FORMULACION Y EVALUACION DE PROYECTOS":[{"mun":"CAICEDO","ap":121,"fichas":1,"prom":121.0},{"mun":"DABEIBA","ap":86,"fichas":4,"prom":21.5},{"mun":"EBÉJICO","ap":41,"fichas":2,"prom":20.5},{"mun":"CAÑASGORDAS","ap":26,"fichas":1,"prom":26.0}],"INTERPRETACIÓN DE PLANOS ARQUITECTÓNICOS":[{"mun":"CAÑASGORDAS","ap":43,"fichas":2,"prom":21.5},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":32,"fichas":1,"prom":32.0},{"mun":"CAICEDO","ap":17,"fichas":1,"prom":17.0}],"CREACION DE  FUNCIONES Y GRAFICOS USANDO MICROSOFT EXCEL":[{"mun":"CAICEDO","ap":31,"fichas":1,"prom":31.0},{"mun":"LIBORINA","ap":27,"fichas":1,"prom":27.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":24,"fichas":1,"prom":24.0}],"TÉCNICAS DE PINTURA EN TELA":[{"mun":"LIBORINA","ap":100,"fichas":4,"prom":25.0},{"mun":"MEDELLÍN","ap":74,"fichas":3,"prom":24.7},{"mun":"SAN JERÓNIMO","ap":52,"fichas":2,"prom":26.0},{"mun":"EBÉJICO","ap":33,"fichas":1,"prom":33.0},{"mun":"SABANALARGA","ap":30,"fichas":1,"prom":30.0},{"mun":"FRONTINO","ap":27,"fichas":1,"prom":27.0},{"mun":"CAICEDO","ap":24,"fichas":1,"prom":24.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":22,"fichas":1,"prom":22.0}],"EMPRENDEDOR EN ESTABLECIMIENTO Y COMERCIALIZACION DE CULTIVOS PERENN":[{"mun":"SOPETRÁN","ap":43,"fichas":2,"prom":21.5},{"mun":"GIRALDO","ap":26,"fichas":1,"prom":26.0},{"mun":"CAICEDO","ap":25,"fichas":1,"prom":25.0},{"mun":"EBÉJICO","ap":23,"fichas":1,"prom":23.0}],"FORTALECIMIENTO ORGANIZACIONAL DE UNIDADES PRODUCTIVAS":[{"mun":"DABEIBA","ap":36,"fichas":2,"prom":18.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":32,"fichas":2,"prom":16.0},{"mun":"LIBORINA","ap":20,"fichas":1,"prom":20.0},{"mun":"CAICEDO","ap":15,"fichas":1,"prom":15.0},{"mun":"GIRALDO","ap":15,"fichas":1,"prom":15.0}],"COCINA INTERNACIONAL":[{"mun":"URAMITA","ap":83,"fichas":3,"prom":27.7},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":80,"fichas":4,"prom":20.0},{"mun":"SAN JERÓNIMO","ap":73,"fichas":2,"prom":36.5},{"mun":"CAICEDO","ap":54,"fichas":2,"prom":27.0},{"mun":"GIRALDO","ap":32,"fichas":1,"prom":32.0},{"mun":"CAÑASGORDAS","ap":26,"fichas":1,"prom":26.0},{"mun":"EBÉJICO","ap":26,"fichas":1,"prom":26.0},{"mun":"SABANALARGA","ap":21,"fichas":1,"prom":21.0}],"ELABORACION  DE CONSERVAS DE  FRUTAS Y HORTALIZAS":[{"mun":"SAN JERÓNIMO","ap":55,"fichas":2,"prom":27.5},{"mun":"SOPETRÁN","ap":29,"fichas":1,"prom":29.0},{"mun":"EBÉJICO","ap":24,"fichas":1,"prom":24.0},{"mun":"CAICEDO","ap":20,"fichas":1,"prom":20.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":12,"fichas":1,"prom":12.0}],"ELABORACION ARTESANAL DE PRODUCTOS DE PANIFICACION.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":50,"fichas":2,"prom":25.0},{"mun":"CAICEDO","ap":31,"fichas":1,"prom":31.0},{"mun":"EBÉJICO","ap":20,"fichas":1,"prom":20.0}],"COCINA BÁSICA NIVEL 1":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":43,"fichas":2,"prom":21.5},{"mun":"LIBORINA","ap":26,"fichas":1,"prom":26.0},{"mun":"CAICEDO","ap":25,"fichas":1,"prom":25.0},{"mun":"URAMITA","ap":15,"fichas":1,"prom":15.0}],"COCINA COLOMBIANA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":124,"fichas":5,"prom":24.8},{"mun":"CAICEDO","ap":75,"fichas":3,"prom":25.0},{"mun":"URAMITA","ap":67,"fichas":3,"prom":22.3},{"mun":"SAN JERÓNIMO","ap":66,"fichas":3,"prom":22.0},{"mun":"SABANALARGA","ap":20,"fichas":1,"prom":20.0},{"mun":"SOPETRÁN","ap":15,"fichas":1,"prom":15.0}],"CUALIDADES FISICAS EN EL ENTRENAMIENTO DEPORTIVO":[{"mun":"CAÑASGORDAS","ap":20,"fichas":1,"prom":20.0},{"mun":"SAN JERÓNIMO","ap":19,"fichas":1,"prom":19.0},{"mun":"CAICEDO","ap":17,"fichas":1,"prom":17.0}],"BÁSICO DE CONTABILIDAD COSTOS Y PRESUPUESTOS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":59,"fichas":2,"prom":29.5},{"mun":"CAICEDO","ap":40,"fichas":1,"prom":40.0},{"mun":"SOPETRÁN","ap":38,"fichas":1,"prom":38.0},{"mun":"SAN JERÓNIMO","ap":35,"fichas":1,"prom":35.0},{"mun":"FRONTINO","ap":27,"fichas":1,"prom":27.0},{"mun":"CAÑASGORDAS","ap":20,"fichas":1,"prom":20.0}],"BÁSICO EN AGRICULTURA ECOLOGICA":[{"mun":"DABEIBA","ap":54,"fichas":3,"prom":18.0},{"mun":"GIRALDO","ap":31,"fichas":2,"prom":15.5},{"mun":"CAICEDO","ap":16,"fichas":1,"prom":16.0}],"BUENAS PRACTICAS AGRICOLAS.":[{"mun":"DABEIBA","ap":71,"fichas":4,"prom":17.8},{"mun":"FRONTINO","ap":40,"fichas":2,"prom":20.0},{"mun":"URAMITA","ap":37,"fichas":2,"prom":18.5},{"mun":"BURITICA","ap":30,"fichas":2,"prom":15.0},{"mun":"CAICEDO","ap":30,"fichas":1,"prom":30.0},{"mun":"CAÑASGORDAS","ap":21,"fichas":1,"prom":21.0},{"mun":"SOPETRÁN","ap":15,"fichas":1,"prom":15.0}],"CULTIVOS AGRICOLAS":[{"mun":"CAÑASGORDAS","ap":32,"fichas":2,"prom":16.0},{"mun":"PEQUE","ap":32,"fichas":2,"prom":16.0},{"mun":"CAICEDO","ap":30,"fichas":2,"prom":15.0},{"mun":"SOPETRÁN","ap":29,"fichas":2,"prom":14.5}],"SERVICIOS DE BARISMO":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":56,"fichas":2,"prom":28.0},{"mun":"CAICEDO","ap":26,"fichas":1,"prom":26.0}],"ACCIONES PARA LA CONSERVACION, PROTECCION Y RESTAURACION DE LOS SIST":[{"mun":"CAICEDO","ap":16,"fichas":1,"prom":16.0},{"mun":"PEQUE","ap":15,"fichas":1,"prom":15.0},{"mun":"OLAYA","ap":14,"fichas":1,"prom":14.0}],"IMPLEMENTACION DE PROCESOS PARA LA TRANSICION AGROECOLOGICA.":[{"mun":"CAÑASGORDAS","ap":21,"fichas":1,"prom":21.0},{"mun":"PEQUE","ap":15,"fichas":1,"prom":15.0},{"mun":"SOPETRÁN","ap":15,"fichas":1,"prom":15.0}],"TECNICAS PARA EL ESTABLECIMIENTO Y MANEJO DE LA AGRICULTURA URBANA":[{"mun":"DABEIBA","ap":43,"fichas":2,"prom":21.5},{"mun":"GIRALDO","ap":22,"fichas":1,"prom":22.0},{"mun":"CAÑASGORDAS","ap":16,"fichas":1,"prom":16.0}],"IMPLEMENTACION DE SISTEMAS SOSTENIBLES DE PRODUCCION FORRAJERA PARA ":[{"mun":"CAÑASGORDAS","ap":45,"fichas":3,"prom":15.0}],"FORTALECIMIENTO EN GESTION ORGANIZACIONAL":[{"mun":"DABEIBA","ap":66,"fichas":4,"prom":16.5},{"mun":"CAÑASGORDAS","ap":31,"fichas":2,"prom":15.5},{"mun":"SAN JERÓNIMO","ap":29,"fichas":1,"prom":29.0},{"mun":"HELICONIA","ap":25,"fichas":1,"prom":25.0}],"FORMALIZACION EMPRESARIAL PARA UNIDADES PRODUCTIVAS":[{"mun":"DABEIBA","ap":225,"fichas":14,"prom":16.1},{"mun":"CAÑASGORDAS","ap":47,"fichas":2,"prom":23.5}],"ELABORACIÓN DE ARTÍCULOS DECORATIVOS Y UTILITARIOS CON MATERIALES RE":[{"mun":"EBÉJICO","ap":94,"fichas":4,"prom":23.5},{"mun":"CAÑASGORDAS","ap":74,"fichas":3,"prom":24.7},{"mun":"MEDELLÍN","ap":55,"fichas":2,"prom":27.5},{"mun":"SABANALARGA","ap":47,"fichas":2,"prom":23.5},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":45,"fichas":2,"prom":22.5},{"mun":"FRONTINO","ap":34,"fichas":1,"prom":34.0},{"mun":"DABEIBA","ap":26,"fichas":1,"prom":26.0},{"mun":"SOPETRÁN","ap":18,"fichas":1,"prom":18.0}],"TECNICAS PARA LA PREPARACION DE BEBIDAS A BASE DE CAFE":[{"mun":"EBÉJICO","ap":45,"fichas":3,"prom":15.0},{"mun":"CAÑASGORDAS","ap":23,"fichas":1,"prom":23.0},{"mun":"LIBORINA","ap":20,"fichas":1,"prom":20.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":20,"fichas":1,"prom":20.0},{"mun":"SOPETRÁN","ap":19,"fichas":1,"prom":19.0}],"MANEJAR PLAN SANITARIO EN ESPECIES MENORES CUMPLIENDO CON MANUAL DE ":[{"mun":"FRONTINO","ap":50,"fichas":2,"prom":25.0},{"mun":"CAÑASGORDAS","ap":22,"fichas":1,"prom":22.0}],"CONSTRUCCION DE EDIFICACIONES":[{"mun":"FRONTINO","ap":65,"fichas":2,"prom":32.5},{"mun":"OLAYA","ap":62,"fichas":2,"prom":31.0},{"mun":"DABEIBA","ap":59,"fichas":2,"prom":29.5},{"mun":"URAMITA","ap":43,"fichas":2,"prom":21.5},{"mun":"CAÑASGORDAS","ap":38,"fichas":2,"prom":19.0}],"EMPRENDIMIENTO EN PRODUCCION DE CACAO":[{"mun":"URAMITA","ap":57,"fichas":2,"prom":28.5},{"mun":"FRONTINO","ap":29,"fichas":1,"prom":29.0},{"mun":"CAÑASGORDAS","ap":26,"fichas":1,"prom":26.0},{"mun":"EBÉJICO","ap":26,"fichas":1,"prom":26.0},{"mun":"DABEIBA","ap":22,"fichas":1,"prom":22.0}],"MULTIPLICACION Y PROPAGACION DE MATERIAL VEGETAL":[{"mun":"DABEIBA","ap":118,"fichas":7,"prom":16.9},{"mun":"CAÑASGORDAS","ap":55,"fichas":3,"prom":18.3},{"mun":"EBÉJICO","ap":50,"fichas":3,"prom":16.7},{"mun":"GIRALDO","ap":35,"fichas":2,"prom":17.5},{"mun":"ANZÁ","ap":30,"fichas":2,"prom":15.0},{"mun":"HELICONIA","ap":27,"fichas":1,"prom":27.0},{"mun":"PEQUE","ap":19,"fichas":1,"prom":19.0},{"mun":"SOPETRÁN","ap":15,"fichas":1,"prom":15.0}],"PODA DE ESPECIES VEGETALES":[{"mun":"CAÑASGORDAS","ap":62,"fichas":4,"prom":15.5},{"mun":"URAMITA","ap":51,"fichas":3,"prom":17.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":15,"fichas":1,"prom":15.0}],"COSTOS POR ORDENES DE PRODUCCION EN LAS MIPYMES":[{"mun":"PEQUE","ap":23,"fichas":1,"prom":23.0},{"mun":"EBÉJICO","ap":19,"fichas":1,"prom":19.0},{"mun":"CAÑASGORDAS","ap":16,"fichas":1,"prom":16.0}],"FORTALECIMIENTO EN LA APLICACION DE  PRACTICAS SANITARIAS EN BOVINOS":[{"mun":"SOPETRÁN","ap":65,"fichas":3,"prom":21.7},{"mun":"LIBORINA","ap":52,"fichas":2,"prom":26.0},{"mun":"CAÑASGORDAS","ap":43,"fichas":3,"prom":14.3},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":34,"fichas":1,"prom":34.0}],"ESTRATEGIAS DE COMUNICACION INTEGRADA DE MARKETING":[{"mun":"CAÑASGORDAS","ap":41,"fichas":2,"prom":20.5},{"mun":"GIRALDO","ap":19,"fichas":1,"prom":19.0}],"ASADOS DE CARNE":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":46,"fichas":2,"prom":23.0},{"mun":"CAÑASGORDAS","ap":20,"fichas":1,"prom":20.0},{"mun":"SOPETRÁN","ap":20,"fichas":1,"prom":20.0}],"OPERACION TURISTICA LOCAL":[{"mun":"CAÑASGORDAS","ap":181,"fichas":6,"prom":30.2},{"mun":"LIBORINA","ap":87,"fichas":4,"prom":21.8},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":19,"fichas":1,"prom":19.0},{"mun":"URAMITA","ap":15,"fichas":1,"prom":15.0}],"ELABORACION DE BIOABONOS SOLIDOS":[{"mun":"DABEIBA","ap":53,"fichas":3,"prom":17.7},{"mun":"CAÑASGORDAS","ap":18,"fichas":1,"prom":18.0}],"BÁSICO MERCADEO Y SERVICIO AL CLIENTE":[{"mun":"URAMITA","ap":49,"fichas":1,"prom":49.0},{"mun":"CAÑASGORDAS","ap":44,"fichas":1,"prom":44.0},{"mun":"GIRALDO","ap":27,"fichas":1,"prom":27.0},{"mun":"OLAYA","ap":23,"fichas":1,"prom":23.0},{"mun":"PEQUE","ap":21,"fichas":1,"prom":21.0},{"mun":"FRONTINO","ap":21,"fichas":1,"prom":21.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":20,"fichas":1,"prom":20.0},{"mun":"EBÉJICO","ap":17,"fichas":1,"prom":17.0}],"HIGIENE Y MANIPULACION DE ALIMENTOS":[{"mun":"PEQUE","ap":26,"fichas":1,"prom":26.0},{"mun":"DABEIBA","ap":22,"fichas":1,"prom":22.0},{"mun":"CAÑASGORDAS","ap":15,"fichas":1,"prom":15.0}],"AGROINDUSTRIA ALIMENTARIA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":255,"fichas":10,"prom":25.5},{"mun":"FRONTINO","ap":184,"fichas":7,"prom":26.3},{"mun":"DABEIBA","ap":77,"fichas":3,"prom":25.7}],"ESTRUCTURACION DE MODELOS DE NEGOCIOS VERDES":[{"mun":"SAN JERÓNIMO","ap":39,"fichas":2,"prom":19.5},{"mun":"DABEIBA","ap":34,"fichas":1,"prom":34.0},{"mun":"SOPETRÁN","ap":26,"fichas":1,"prom":26.0},{"mun":"GIRALDO","ap":20,"fichas":1,"prom":20.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":20,"fichas":1,"prom":20.0}],"EMPRENDEDOR EN DESARROLLO DE ACTIVIDADES TURISTICAS EN ESPACIOS NATU":[{"mun":"DABEIBA","ap":31,"fichas":1,"prom":31.0},{"mun":"FRONTINO","ap":25,"fichas":1,"prom":25.0},{"mun":"SAN JERÓNIMO","ap":20,"fichas":1,"prom":20.0},{"mun":"LIBORINA","ap":16,"fichas":1,"prom":16.0}],"EMPRENDIMIENTO DE UNIDADES PRODUCTIVAS":[{"mun":"DABEIBA","ap":164,"fichas":8,"prom":20.5},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":58,"fichas":2,"prom":29.0},{"mun":"HELICONIA","ap":25,"fichas":1,"prom":25.0}],"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 1":[{"mun":"DABEIBA","ap":79,"fichas":3,"prom":26.3},{"mun":"GIRALDO","ap":46,"fichas":2,"prom":23.0}],"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 2":[{"mun":"DABEIBA","ap":85,"fichas":3,"prom":28.3},{"mun":"GIRALDO","ap":29,"fichas":1,"prom":29.0}],"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 3":[{"mun":"DABEIBA","ap":50,"fichas":2,"prom":25.0},{"mun":"GIRALDO","ap":21,"fichas":1,"prom":21.0}],"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 4":[{"mun":"DABEIBA","ap":43,"fichas":2,"prom":21.5},{"mun":"GIRALDO","ap":21,"fichas":1,"prom":21.0}],"APLICACION DE PRACTICAS DE ALIMENTACION EN BOVINOS":[{"mun":"DABEIBA","ap":73,"fichas":3,"prom":24.3},{"mun":"OLAYA","ap":32,"fichas":1,"prom":32.0},{"mun":"FRONTINO","ap":26,"fichas":1,"prom":26.0}],"ORGANIZACION COMUNITARIA SOSTENIBLE":[{"mun":"DABEIBA","ap":61,"fichas":3,"prom":20.3}],"PROMOTORIA EN MANEJO AMBIENTAL":[{"mun":"MEDELLÍN","ap":266,"fichas":10,"prom":26.6},{"mun":"DABEIBA","ap":142,"fichas":6,"prom":23.7},{"mun":"GIRALDO","ap":86,"fichas":4,"prom":21.5},{"mun":"URAMITA","ap":42,"fichas":2,"prom":21.0},{"mun":"ABRIAQUÍ","ap":23,"fichas":1,"prom":23.0}],"SERVICIOS COMERCIALES Y FINANCIEROS":[{"mun":"LIBORINA","ap":59,"fichas":2,"prom":29.5},{"mun":"DABEIBA","ap":49,"fichas":2,"prom":24.5},{"mun":"SOPETRÁN","ap":48,"fichas":2,"prom":24.0}],"CONSTRUCCIONES LIVIANAS INDUSTRIALIZADAS EN SECO":[{"mun":"FRONTINO","ap":38,"fichas":2,"prom":19.0},{"mun":"DABEIBA","ap":5,"fichas":1,"prom":5.0}],"CONTROL FITOSANITARIO EN EL CULTIVO DE CACAO":[{"mun":"FRONTINO","ap":25,"fichas":1,"prom":25.0},{"mun":"DABEIBA","ap":22,"fichas":1,"prom":22.0},{"mun":"URAMITA","ap":22,"fichas":1,"prom":22.0}],"DISEÑO, LIQUIDACION Y PAGO DE NOMINA.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":61,"fichas":3,"prom":20.3},{"mun":"FRONTINO","ap":45,"fichas":2,"prom":22.5},{"mun":"DABEIBA","ap":44,"fichas":1,"prom":44.0},{"mun":"URAMITA","ap":21,"fichas":1,"prom":21.0}],"ASESORIA COMERCIAL":[{"mun":"SOPETRÁN","ap":61,"fichas":2,"prom":30.5},{"mun":"DABEIBA","ap":44,"fichas":2,"prom":22.0},{"mun":"EBÉJICO","ap":34,"fichas":1,"prom":34.0},{"mun":"SAN JERÓNIMO","ap":22,"fichas":2,"prom":11.0}],"GESTION DE MERCADO, COMERCIALIZACION Y VENTAS - BASADO EN EL MODELO ":[{"mun":"DABEIBA","ap":42,"fichas":2,"prom":21.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":23,"fichas":1,"prom":23.0},{"mun":"GIRALDO","ap":20,"fichas":1,"prom":20.0},{"mun":"SABANALARGA","ap":20,"fichas":1,"prom":20.0}],"MANEJO SANITARIO EN GRANJAS PORCINAS":[{"mun":"SOPETRÁN","ap":46,"fichas":2,"prom":23.0},{"mun":"DABEIBA","ap":17,"fichas":1,"prom":17.0}],"EMPRENDIMIENTO EN PRODUCCION DE HUEVO":[{"mun":"DABEIBA","ap":21,"fichas":1,"prom":21.0},{"mun":"LIBORINA","ap":16,"fichas":1,"prom":16.0},{"mun":"SAN JERÓNIMO","ap":16,"fichas":1,"prom":16.0}],"FORTALECIMIENTO EN LA APLICACION DE PRACTICAS  SANITARIAS  EN PRODUC":[{"mun":"DABEIBA","ap":61,"fichas":3,"prom":20.3}],"EMPRENDIMIENTO EN PRODUCCION DE FRUTALES PERENNES":[{"mun":"LIBORINA","ap":26,"fichas":1,"prom":26.0},{"mun":"EBÉJICO","ap":25,"fichas":1,"prom":25.0},{"mun":"DABEIBA","ap":22,"fichas":1,"prom":22.0}],"SERVICIO AL CLIENTE":[{"mun":"DABEIBA","ap":137,"fichas":5,"prom":27.4},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":104,"fichas":5,"prom":20.8},{"mun":"FRONTINO","ap":36,"fichas":2,"prom":18.0}],"COMERCIALIZACION DE PRODUCTOS Y SERVICIOS TURISTICOS":[{"mun":"DABEIBA","ap":35,"fichas":1,"prom":35.0},{"mun":"SAN JERÓNIMO","ap":23,"fichas":1,"prom":23.0},{"mun":"FRONTINO","ap":21,"fichas":2,"prom":10.5},{"mun":"GIRALDO","ap":20,"fichas":1,"prom":20.0}],"DISEÑO DE  RECORRIDOS TURISTICOS EN ESPACIOS RURALES":[{"mun":"FRONTINO","ap":43,"fichas":2,"prom":21.5},{"mun":"DABEIBA","ap":32,"fichas":1,"prom":32.0},{"mun":"SAN JERÓNIMO","ap":28,"fichas":1,"prom":28.0},{"mun":"GIRALDO","ap":20,"fichas":1,"prom":20.0}],"OPERACION DE ALOJAMIENTOS RURALES":[{"mun":"FRONTINO","ap":40,"fichas":2,"prom":20.0},{"mun":"DABEIBA","ap":26,"fichas":2,"prom":13.0}],"ESTABLECIMIENTO DE CULTIVOS EN AGRICULTURA URBANA":[{"mun":"DABEIBA","ap":38,"fichas":2,"prom":19.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":28,"fichas":1,"prom":28.0},{"mun":"SOPETRÁN","ap":23,"fichas":1,"prom":23.0},{"mun":"URAMITA","ap":8,"fichas":1,"prom":8.0}],"SISTEMAS.":[{"mun":"FRONTINO","ap":79,"fichas":3,"prom":26.3},{"mun":"EBÉJICO","ap":68,"fichas":2,"prom":34.0},{"mun":"DABEIBA","ap":63,"fichas":2,"prom":31.5},{"mun":"OLAYA","ap":56,"fichas":2,"prom":28.0},{"mun":"URAMITA","ap":52,"fichas":2,"prom":26.0},{"mun":"SABANALARGA","ap":34,"fichas":1,"prom":34.0},{"mun":"SOPETRÁN","ap":28,"fichas":1,"prom":28.0},{"mun":"GIRALDO","ap":18,"fichas":1,"prom":18.0}],"MANEJO DE SISTEMAS DE INFORMACION EN LA PRODUCCION PECUARIA.":[{"mun":"URAMITA","ap":27,"fichas":1,"prom":27.0},{"mun":"SAN JERÓNIMO","ap":25,"fichas":1,"prom":25.0},{"mun":"DABEIBA","ap":21,"fichas":1,"prom":21.0}],"IMPLEMENTACION DEL SISTEMA DE GESTION DE LA SEGURIDAD Y SALUD EN EL ":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":22,"fichas":1,"prom":22.0},{"mun":"DABEIBA","ap":20,"fichas":1,"prom":20.0},{"mun":"SAN JERÓNIMO","ap":20,"fichas":1,"prom":20.0}],"ASISTENCIA EN ORGANIZACION DE ARCHIVOS .":[{"mun":"DABEIBA","ap":64,"fichas":2,"prom":32.0},{"mun":"FRONTINO","ap":14,"fichas":1,"prom":14.0}],"MUESTREO DE SUELOS AGRICOLAS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":61,"fichas":2,"prom":30.5},{"mun":"FRONTINO","ap":25,"fichas":1,"prom":25.0},{"mun":"EBÉJICO","ap":21,"fichas":1,"prom":21.0}],"COCINA.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":145,"fichas":6,"prom":24.2},{"mun":"EBÉJICO","ap":38,"fichas":2,"prom":19.0}],"FUNDAMENTOS EN HIDROTERAPIA APLICADA A LA ESTETICA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":67,"fichas":3,"prom":22.3},{"mun":"SAN JERÓNIMO","ap":35,"fichas":2,"prom":17.5},{"mun":"PEQUE","ap":13,"fichas":1,"prom":13.0},{"mun":"EBÉJICO","ap":12,"fichas":1,"prom":12.0}],"CREACIÓN DE BASES DE DATOS CON MICROSOFT ACCESS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":51,"fichas":2,"prom":25.5},{"mun":"FRONTINO","ap":41,"fichas":2,"prom":20.5}],"PRACTICAS DE MANEJO SANITARIO EN BOVINOS":[{"mun":"OLAYA","ap":55,"fichas":2,"prom":27.5},{"mun":"FRONTINO","ap":48,"fichas":2,"prom":24.0}],"INSEMINACIÓN A CELO DETECTADO":[{"mun":"MEDELLÍN","ap":122,"fichas":4,"prom":30.5},{"mun":"URAMITA","ap":46,"fichas":2,"prom":23.0},{"mun":"FRONTINO","ap":22,"fichas":1,"prom":22.0},{"mun":"SAN JERÓNIMO","ap":20,"fichas":1,"prom":20.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":14,"fichas":1,"prom":14.0}],"SISTEMAS TELEINFORMÁTICOS":[{"mun":"FRONTINO","ap":68,"fichas":2,"prom":34.0},{"mun":"SABANALARGA","ap":68,"fichas":2,"prom":34.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":34,"fichas":2,"prom":17.0}],"ELABORACION DE COMPLEMENTOS EN BISUTERIA CON TECNICA DE ENSARTADO":[{"mun":"FRONTINO","ap":68,"fichas":3,"prom":22.7},{"mun":"LIBORINA","ap":47,"fichas":2,"prom":23.5},{"mun":"SABANALARGA","ap":26,"fichas":1,"prom":26.0},{"mun":"PEQUE","ap":26,"fichas":1,"prom":26.0},{"mun":"EBÉJICO","ap":25,"fichas":1,"prom":25.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":23,"fichas":1,"prom":23.0},{"mun":"GIRALDO","ap":22,"fichas":1,"prom":22.0},{"mun":"SOPETRÁN","ap":15,"fichas":1,"prom":15.0}],"MERCADEO Y VENTAS":[{"mun":"HELICONIA","ap":25,"fichas":1,"prom":25.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":22,"fichas":1,"prom":22.0},{"mun":"GIRALDO","ap":20,"fichas":1,"prom":20.0},{"mun":"FRONTINO","ap":20,"fichas":1,"prom":20.0}],"RECURSOS HUMANOS .":[{"mun":"GIRALDO","ap":61,"fichas":2,"prom":30.5},{"mun":"LIBORINA","ap":54,"fichas":2,"prom":27.0},{"mun":"FRONTINO","ap":44,"fichas":2,"prom":22.0}],"INSEMINACIÓN A TIEMPO FIJO":[{"mun":"FRONTINO","ap":66,"fichas":4,"prom":16.5},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":44,"fichas":2,"prom":22.0},{"mun":"URAMITA","ap":15,"fichas":1,"prom":15.0}],"COCINA SALUDABLE":[{"mun":"SAN JERÓNIMO","ap":25,"fichas":1,"prom":25.0},{"mun":"GIRALDO","ap":20,"fichas":1,"prom":20.0},{"mun":"SOPETRÁN","ap":16,"fichas":1,"prom":16.0}],"ELABORACION DE ADORNOS Y MUÑECOS NAVIDEÑOS":[{"mun":"SAN JERÓNIMO","ap":29,"fichas":1,"prom":29.0},{"mun":"GIRALDO","ap":28,"fichas":1,"prom":28.0},{"mun":"URAMITA","ap":24,"fichas":1,"prom":24.0},{"mun":"LIBORINA","ap":21,"fichas":1,"prom":21.0}],"IMPLEMENTACION DE CULTIVOS EN LA AGRICULTURA URBANA":[{"mun":"LIBORINA","ap":40,"fichas":2,"prom":20.0},{"mun":"OLAYA","ap":20,"fichas":1,"prom":20.0},{"mun":"EBÉJICO","ap":15,"fichas":1,"prom":15.0},{"mun":"URAMITA","ap":13,"fichas":1,"prom":13.0}],"GESTION DEL DESARROLLO DEL TALENTO HUMANO":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":48,"fichas":3,"prom":16.0},{"mun":"SAN JERÓNIMO","ap":20,"fichas":1,"prom":20.0}],"SISTEMAS AGROPECUARIOS ECOLOGICOS.":[{"mun":"SAN JERÓNIMO","ap":225,"fichas":9,"prom":25.0}],"COCINA PERUANA":[{"mun":"SAN JERÓNIMO","ap":63,"fichas":3,"prom":21.0},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":35,"fichas":1,"prom":35.0},{"mun":"URAMITA","ap":24,"fichas":1,"prom":24.0},{"mun":"SOPETRÁN","ap":20,"fichas":1,"prom":20.0}],"BIOSEGURIDAD APLICADA A LA ESTETICA Y BELLEZA":[{"mun":"SAN JERÓNIMO","ap":90,"fichas":4,"prom":22.5}],"MANIPULACIÓN DE ALIMENTOS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":128,"fichas":5,"prom":25.6}],"GESTION CONTABLE Y DE INFORMACION FINANCIERA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":130,"fichas":6,"prom":21.7}],"VENTA DE PRODUCTOS EN LINEA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":215,"fichas":5,"prom":43.0}],"ANALISIS Y DESARROLLO DE SOFTWARE.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":217,"fichas":7,"prom":31.0}],"GESTIÓN DE MERCADOS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":116,"fichas":7,"prom":16.6}],"SERVICIO DE RECEPCION HOTELERA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":477,"fichas":11,"prom":43.4}],"GESTION AGROEMPRESARIAL":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":303,"fichas":9,"prom":33.7}],"LEVANTAMIENTOS TOPOGRAFICOS Y GEORREFERENCIACION":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":64,"fichas":3,"prom":21.3}],"GESTIÓN ADMINISTRATIVA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":202,"fichas":9,"prom":22.4}],"INGLES PRE INTERMEDIO- NIVEL 2":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":197,"fichas":8,"prom":24.6}],"INGLES BASICO - NIVEL 4":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":201,"fichas":8,"prom":25.1}],"INGLES PRE INTERMEDIO - NIVEL 1":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":157,"fichas":7,"prom":22.4}],"ELABORACION DE PRODUCTOS DE REPOSTERIA.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":89,"fichas":4,"prom":22.2}],"GESTIÓN EMPRESARIAL":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":153,"fichas":6,"prom":25.5}],"DESARROLLO DE HABILIDADES SENSORIALES EN CAFE":[{"mun":"CAICEDO","ap":47,"fichas":3,"prom":15.7},{"mun":"SANTAFÉ DE ANTIOQUIA","ap":30,"fichas":1,"prom":30.0},{"mun":"EBÉJICO","ap":20,"fichas":1,"prom":20.0}],"SERVICIO DE RESTAURANTE Y BAR":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":72,"fichas":4,"prom":18.0},{"mun":"SOPETRÁN","ap":39,"fichas":2,"prom":19.5}],"COCINA ITALIANA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":43,"fichas":2,"prom":21.5},{"mun":"SOPETRÁN","ap":40,"fichas":2,"prom":20.0}],"COORDINADOR DE TRABAJO\xa0EN ALTURAS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":44,"fichas":5,"prom":8.8}],"REENTRENAMIENTO EN TRABAJO EN ALTURAS PARA TRABAJADOR AUTORIZADO":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":121,"fichas":17,"prom":7.1}],"DESARROLLO PUBLICITARIO":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":155,"fichas":7,"prom":22.1}],"CONSTRUCCION EN EDIFICACIONES.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":79,"fichas":4,"prom":19.8}],"PLANEACION DE ESTRATEGIAS PEDAGOGICAS Y TECNICAS DIDACTICAS PARA LA ":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":93,"fichas":4,"prom":23.2}],"ENGLISH DOES WORK - LEVEL 1":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":3091,"fichas":53,"prom":58.3}],"ENGLISH DOES WORK - LEVEL 2":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":454,"fichas":11,"prom":41.3}],"ENGLISH DOES WORK - LEVEL 3":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":425,"fichas":10,"prom":42.5}],"ENGLISH DOES WORK - LEVEL 4":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":182,"fichas":4,"prom":45.5}],"ENGLISH DOES WORK - LEVEL 5":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":152,"fichas":3,"prom":50.7}],"ENGLISH DOES WORK - LEVEL 6":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":197,"fichas":3,"prom":65.7}],"ACEITES ESENCIALES: EXTRACCION, USOS Y APLICACIONES":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":575,"fichas":11,"prom":52.3}],"ADITIVOS: ANALISIS Y CONTROL DE CALIDAD EN LA INDUSTRIA ALIMENTARIA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":51,"fichas":5,"prom":10.2}],"AUDITORIA INTERNA DE CALIDAD - NTC ISO 9001":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":240,"fichas":15,"prom":16.0}],"BUENAS PRACTICAS EN LA MANIPULACION DE LA CARNE":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":473,"fichas":13,"prom":36.4}],"CALIDAD: GESTION DEL MEJORAMIENTO CONTINUO..":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":414,"fichas":9,"prom":46.0}],"COCINA INTERNACIONAL.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":515,"fichas":14,"prom":36.8}],"CONSERVACION DE FRUTAS Y VERDURAS":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":33,"fichas":4,"prom":8.2}],"ESTRATEGIAS PARA EL MEJORAMIENTO DE LA COMPRENSION LECTORA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":36,"fichas":4,"prom":9.0}],"ESTRUCTURA DE LA INFORMACION DOCUMENTADA -NTCISO 9001:2015":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":258,"fichas":4,"prom":64.5}],"GASTRONOMIA COLOMBIANA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":336,"fichas":13,"prom":25.8}],"GUIANZA EN RECORRIDOS POR LA NATURALEZA.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":61,"fichas":4,"prom":15.2}],"HABITOS SALUDABLES A PARTIR DE LA ALIMENTACION Y LA ACTIVIDAD FISICA":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":80,"fichas":4,"prom":20.0}],"PASTELERIA.":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":176,"fichas":6,"prom":29.3}],"PLANIFICACION DE UN SISTEMA DE GESTION DE LA CALIDAD - NTC ISO 9001":[{"mun":"SANTAFÉ DE ANTIOQUIA","ap":72,"fichas":5,"prom":14.4}],"COMUNICACION ASERTIVA Y EFECTIVA EN EQUIPOS DE TRABAJO":[{"mun":"SOPETRÁN","ap":42,"fichas":2,"prom":21.0},{"mun":"URAMITA","ap":19,"fichas":1,"prom":19.0},{"mun":"DABEIBA","ap":16,"fichas":1,"prom":16.0}],"IMPLEMENTACION DE PRACTICAS AMBIENTALES EN GRANJAS AVICOLAS":[{"mun":"EBÉJICO","ap":45,"fichas":3,"prom":15.0},{"mun":"ANZÁ","ap":30,"fichas":2,"prom":15.0}],"EMPRENDEDOR EN LA PRODUCCION DE PLATANO Y BANANO":[{"mun":"CAÑASGORDAS","ap":16,"fichas":1,"prom":16.0},{"mun":"OLAYA","ap":16,"fichas":1,"prom":16.0},{"mun":"EBÉJICO","ap":15,"fichas":1,"prom":15.0}],"GENERACION DE IDEAS INNOVADORAS CON DESIGN THINKING":[{"mun":"DABEIBA","ap":17,"fichas":1,"prom":17.0},{"mun":"URAMITA","ap":17,"fichas":1,"prom":17.0},{"mun":"CAÑASGORDAS","ap":16,"fichas":1,"prom":16.0}],"ATENCION AL CLIENTE EN LOS PROCESOS ADMINISTRATIVOS":[{"mun":"DABEIBA","ap":45,"fichas":3,"prom":15.0}],"EMPRENDEDOR EN MANTENIMIENTO Y COMERCIALIZACION DE CULTIVOS PERENNES":[{"mun":"EBÉJICO","ap":15,"fichas":1,"prom":15.0},{"mun":"GIRALDO","ap":15,"fichas":1,"prom":15.0},{"mun":"SOPETRÁN","ap":15,"fichas":1,"prom":15.0}]}'
_MP = '{"ABRIAQUÍ":[{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":5,"prom":19.6,"ap":98,"meses":12,"ultima":"2024-03-07"},{"prog":"APLICACION DE BUENAS PRACTICAS GANADERAS  EN BOVINOS DE LECHE","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":16.2,"ap":65,"meses":1,"ultima":"2025-02-20"},{"prog":"INSEMINACION ARTIFICIAL DE HEMBRAS BOVINAS APLICANDO PROTOCOLO CO","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":21.0,"ap":84,"meses":5,"ultima":"2024-10-10"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":23.0,"ap":69,"meses":10,"ultima":"2024-05-14"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":21.7,"ap":65,"meses":8,"ultima":"2024-07-05"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":32.0,"ap":96,"meses":5,"ultima":"2024-10-17"},{"prog":"GESTION DE PROYECTOS COMUNITARIOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":9.5,"ap":19,"meses":19,"ultima":"2023-09-05"},{"prog":"CRÍA Y LEVANTE DE POLLOS DE ENGORDE CON LA IMPLEMENTACIÓN DE MAÍZ","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":16.0,"ap":16,"meses":12,"ultima":"2024-03-20"},{"prog":"EMPRENDIMIENTO EN PRODUCCION  DE FRUTALES SEMIPERENNES","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":26.0,"ap":26,"meses":9,"ultima":"2024-06-21"},{"prog":"EMPRENDEDOR EN PROPAGACION DE MATERIAL VEGETAL Y ESTABLECIMIENTO ","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":21.0,"ap":21,"meses":18,"ultima":"2023-09-14"},{"prog":"ELABORACION DE TEJIDO ARTESANAL EN CROCHET CON HILOS METALICOS","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":1,"prom":28.0,"ap":28,"meses":18,"ultima":"2023-09-20"},{"prog":"ELABORACION DE HELADOS Y POSTRES LACTEOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":32.0,"ap":32,"meses":20,"ultima":"2023-08-02"}],"ANZÁ":[{"prog":"PRACTICAS DE CALIDAD PARA EL BENEFICIO DEL CAFE","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":8,"prom":18.5,"ap":148,"meses":6,"ultima":"2024-09-12"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":8,"prom":26.2,"ap":210,"meses":9,"ultima":"2024-06-21"},{"prog":"APLICACION DE LAS BUENAS PRACTICAS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":5,"prom":15.0,"ap":75,"meses":1,"ultima":"2025-02-27"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":24.8,"ap":99,"meses":1,"ultima":"2025-02-19"},{"prog":"HIGIENE Y MANIPULACION DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":16.5,"ap":66,"meses":8,"ultima":"2024-07-11"},{"prog":"PRODUCCION DE CAFE","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":4,"prom":30.0,"ap":120,"meses":12,"ultima":"2024-03-07"},{"prog":"CRIAR GALLINAS PONEDORAS CON ALIMENTACIÓN ALTERNA PARA PRODUCIR H","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":21.7,"ap":65,"meses":11,"ultima":"2024-04-12"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":24.0,"ap":48,"meses":10,"ultima":"2024-05-16"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":24.0,"ap":48,"meses":9,"ultima":"2024-06-27"},{"prog":"MULTIPLICACION Y PROPAGACION DE MATERIAL VEGETAL","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":15.0,"ap":30,"meses":1,"ultima":"2025-02-26"},{"prog":"COCTELERIA TROPICAL","nivel":"CURSO ESPECIAL","sector":"HOTELERIA Y TURISMO","fichas":2,"prom":20.0,"ap":40,"meses":6,"ultima":"2024-10-01"},{"prog":"PATRONAJE DE PRENDAS DE ROPA EXTERIOR FEMENINA","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":2,"prom":21.0,"ap":42,"meses":25,"ultima":"2023-03-08"}],"ARMENIA":[{"prog":"CRIAR GALLINAS PONEDORAS CON ALIMENTACIÓN ALTERNA PARA PRODUCIR H","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":20.7,"ap":62,"meses":6,"ultima":"2024-09-09"},{"prog":"INFORMATICA: MICROSOFT WORD, EXCEL E INTERNET","nivel":"CURSO ESPECIAL","sector":"TRANSVERSAL","fichas":3,"prom":19.3,"ap":58,"meses":6,"ultima":"2024-09-30"},{"prog":"MONITOREO DE PLAGAS EN SISTEMAS PRODUCTIVOS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":15.0,"ap":30,"meses":12,"ultima":"2024-03-26"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":26.5,"ap":53,"meses":9,"ultima":"2024-06-20"},{"prog":"ELABORACION DE ACCESORIOS TEJIDOS EN MOSTACILLAS","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":2,"prom":28.5,"ap":57,"meses":5,"ultima":"2024-10-10"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":24.0,"ap":48,"meses":21,"ultima":"2023-06-21"},{"prog":"PELUQUERIA.","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":2,"prom":31.0,"ap":62,"meses":4,"ultima":"2024-11-25"},{"prog":"TENDENCIAS EN DECORACION PARA MANICURA Y PEDICURA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":27.5,"ap":55,"meses":5,"ultima":"2024-10-21"},{"prog":"APLICACION DE  BUENAS PRACTICAS GANADERAS EN LA PRODUCCION DE BOV","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":18.0,"ap":18,"meses":19,"ultima":"2023-09-06"},{"prog":"GESTION DE PROYECTOS COMUNITARIOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":23.0,"ap":23,"meses":18,"ultima":"2023-09-18"},{"prog":"ETIQUETA Y PROTOCOLO EMPRESARIAL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":25.0,"ap":25,"meses":6,"ultima":"2024-09-09"},{"prog":"EMPRENDIMIENTO EN PRODUCCION DE CULTIVOS TRANSITORIOS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":21.0,"ap":21,"meses":20,"ultima":"2023-08-08"}],"BURITICA":[{"prog":"ACONDICIONAMIENTO DE ANDAMIOS PARA TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","sector":"CONSTRUCCION","fichas":10,"prom":7.3,"ap":73,"meses":0,"ultima":"2025-03-06"},{"prog":"JEFES DE AREA PARA TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","sector":"SALUD","fichas":8,"prom":8.5,"ap":68,"meses":1,"ultima":"2025-02-26"},{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":3,"prom":24.3,"ap":73,"meses":25,"ultima":"2023-02-21"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":23.0,"ap":69,"meses":4,"ultima":"2024-12-02"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":24.0,"ap":72,"meses":1,"ultima":"2025-02-21"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":25.3,"ap":76,"meses":11,"ultima":"2024-04-25"},{"prog":"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":16.0,"ap":48,"meses":6,"ultima":"2024-09-17"},{"prog":"PRACTICAS DE CALIDAD PARA EL BENEFICIO DEL CAFE","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":23.0,"ap":69,"meses":7,"ultima":"2024-08-12"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":21.3,"ap":64,"meses":1,"ultima":"2025-03-01"},{"prog":"EMPRENDIMIENTO EN PRODUCCION DE CULTIVOS TRANSITORIOS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":17.5,"ap":35,"meses":24,"ultima":"2023-04-04"},{"prog":"BUENAS PRACTICAS AGRICOLAS.","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":15.0,"ap":30,"meses":1,"ultima":"2025-02-24"},{"prog":"APLICACION DE FUNCIONES Y COMBINACIONES MULTIPLES EN EXCEL PARA S","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":30.0,"ap":60,"meses":3,"ultima":"2024-12-05"}],"CAICEDO":[{"prog":"COCINA COLOMBIANA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":25.0,"ap":75,"meses":25,"ultima":"2023-03-07"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":23.3,"ap":70,"meses":11,"ultima":"2024-04-23"},{"prog":"DESARROLLO DE HABILIDADES SENSORIALES EN CAFE","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":15.7,"ap":47,"meses":1,"ultima":"2025-02-26"},{"prog":"INGLES BASICO - NIVEL 3","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":17.3,"ap":52,"meses":4,"ultima":"2024-11-26"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":26.3,"ap":79,"meses":1,"ultima":"2025-02-26"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":26.7,"ap":80,"meses":9,"ultima":"2024-06-22"},{"prog":"APLICACION DE LAS BUENAS PRACTICAS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":18.5,"ap":37,"meses":10,"ultima":"2024-05-17"},{"prog":"ELABORACION DE PRODUCTOS DE PANIFICACION A BASE DE QUESO","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":26.5,"ap":53,"meses":10,"ultima":"2024-05-15"},{"prog":"INGLES BASICO - NIVEL 2","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":19.5,"ap":39,"meses":10,"ultima":"2024-05-20"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":25.0,"ap":50,"meses":8,"ultima":"2024-07-26"},{"prog":"PRACTICAS DE CALIDAD PARA EL BENEFICIO DEL CAFE","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":16.0,"ap":32,"meses":6,"ultima":"2024-09-06"},{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":2,"prom":30.0,"ap":60,"meses":15,"ultima":"2023-12-11"}],"CAÑASGORDAS":[{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":13,"prom":25.1,"ap":326,"meses":5,"ultima":"2024-10-24"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":11,"prom":27.4,"ap":301,"meses":6,"ultima":"2024-10-03"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":6,"prom":24.2,"ap":145,"meses":12,"ultima":"2024-03-09"},{"prog":"OPERACION TURISTICA LOCAL","nivel":"TÉCNICO","sector":"HOTELERIA Y TURISMO","fichas":6,"prom":30.2,"ap":181,"meses":1,"ultima":"2025-02-20"},{"prog":"HIGIENE Y MANIPULACION DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":16.5,"ap":66,"meses":1,"ultima":"2025-02-20"},{"prog":"PODA DE ESPECIES VEGETALES","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":15.5,"ap":62,"meses":1,"ultima":"2025-02-20"},{"prog":"MULTIPLICACION Y PROPAGACION DE MATERIAL VEGETAL","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":18.3,"ap":55,"meses":11,"ultima":"2024-04-18"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":30.7,"ap":92,"meses":8,"ultima":"2024-07-16"},{"prog":"ELABORACIÓN DE ARTÍCULOS DECORATIVOS Y UTILITARIOS CON MATERIALES","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":3,"prom":24.7,"ap":74,"meses":22,"ultima":"2023-06-08"},{"prog":"ELABORACIÓN DE PANES ARTESANALES","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":24.0,"ap":72,"meses":4,"ultima":"2024-11-28"},{"prog":"CONTABILIZACION DE OPERACIONES COMERCIALES Y FINANCIERAS.","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":3,"prom":22.0,"ap":66,"meses":13,"ultima":"2024-02-15"},{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":3,"prom":22.3,"ap":67,"meses":25,"ultima":"2023-02-22"}],"DABEIBA":[{"prog":"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":20,"prom":17.1,"ap":341,"meses":5,"ultima":"2024-11-01"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":18,"prom":21.8,"ap":393,"meses":8,"ultima":"2024-07-22"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":16,"prom":21.6,"ap":345,"meses":1,"ultima":"2025-02-26"},{"prog":"FORMALIZACION EMPRESARIAL PARA UNIDADES PRODUCTIVAS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":14,"prom":16.1,"ap":225,"meses":4,"ultima":"2024-11-12"},{"prog":"INFORMATICA: MICROSOFT WORD, EXCEL E INTERNET","nivel":"CURSO ESPECIAL","sector":"TRANSVERSAL","fichas":10,"prom":22.4,"ap":224,"meses":5,"ultima":"2024-10-28"},{"prog":"GESTION DE PROYECTOS COMUNITARIOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":9,"prom":17.7,"ap":159,"meses":4,"ultima":"2024-11-06"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":8,"prom":28.1,"ap":225,"meses":20,"ultima":"2023-08-10"},{"prog":"EMPRENDIMIENTO DE UNIDADES PRODUCTIVAS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":8,"prom":20.5,"ap":164,"meses":4,"ultima":"2024-11-15"},{"prog":"MULTIPLICACION Y PROPAGACION DE MATERIAL VEGETAL","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":7,"prom":16.9,"ap":118,"meses":1,"ultima":"2025-02-26"},{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":6,"prom":23.7,"ap":142,"meses":13,"ultima":"2024-03-05"},{"prog":"APLICACION DE FUNCIONES Y COMBINACIONES MULTIPLES EN EXCEL PARA S","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":6,"prom":9.8,"ap":59,"meses":4,"ultima":"2024-11-12"},{"prog":"SERVICIO AL CLIENTE","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":5,"prom":27.4,"ap":137,"meses":4,"ultima":"2024-11-13"}],"EBÉJICO":[{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":8,"prom":25.2,"ap":202,"meses":12,"ultima":"2024-03-12"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":6,"prom":22.0,"ap":132,"meses":4,"ultima":"2024-11-14"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":6,"prom":19.7,"ap":118,"meses":1,"ultima":"2025-02-18"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":5,"prom":22.4,"ap":112,"meses":1,"ultima":"2025-02-13"},{"prog":"ELABORACION DE HELADOS Y POSTRES LACTEOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":5,"prom":24.2,"ap":121,"meses":5,"ultima":"2024-10-23"},{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":5,"prom":29.0,"ap":145,"meses":13,"ultima":"2024-02-26"},{"prog":"APLICACION DE LAS BUENAS PRACTICAS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":22.0,"ap":88,"meses":1,"ultima":"2025-02-20"},{"prog":"ELABORACIÓN DE ARTÍCULOS DECORATIVOS Y UTILITARIOS CON MATERIALES","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":4,"prom":23.5,"ap":94,"meses":23,"ultima":"2023-04-24"},{"prog":"IMPLEMENTACION DE PRACTICAS AMBIENTALES EN GRANJAS AVICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":15.0,"ap":45,"meses":1,"ultima":"2025-02-20"},{"prog":"TECNICAS PARA LA PREPARACION DE BEBIDAS A BASE DE CAFE","nivel":"CURSO ESPECIAL","sector":"HOTELERIA Y TURISMO","fichas":3,"prom":15.0,"ap":45,"meses":21,"ultima":"2023-06-30"},{"prog":"MULTIPLICACION Y PROPAGACION DE MATERIAL VEGETAL","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":16.7,"ap":50,"meses":1,"ultima":"2025-02-19"},{"prog":"MANEJO DE LA NUTRICION EN CULTIVOS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":21.0,"ap":63,"meses":5,"ultima":"2024-10-09"}],"FRONTINO":[{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":10,"prom":33.3,"ap":333,"meses":1,"ultima":"2025-02-24"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":9,"prom":27.6,"ap":248,"meses":7,"ultima":"2024-08-09"},{"prog":"PRODUCCION AGROPECUARIA","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":9,"prom":22.1,"ap":199,"meses":1,"ultima":"2025-02-19"},{"prog":"AGROINDUSTRIA ALIMENTARIA","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":7,"prom":26.3,"ap":184,"meses":1,"ultima":"2025-02-24"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":6,"prom":19.5,"ap":117,"meses":1,"ultima":"2025-02-25"},{"prog":"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":6,"prom":20.8,"ap":125,"meses":7,"ultima":"2024-08-20"},{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":5,"prom":19.0,"ap":95,"meses":13,"ultima":"2024-03-05"},{"prog":"INSEMINACIÓN A TIEMPO FIJO","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":16.5,"ap":66,"meses":9,"ultima":"2024-06-11"},{"prog":"ELABORACION DE COMPLEMENTOS EN BISUTERIA CON TECNICA DE ENSARTADO","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":3,"prom":22.7,"ap":68,"meses":19,"ultima":"2023-08-18"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":23.3,"ap":70,"meses":12,"ultima":"2024-03-12"},{"prog":"APLICACION DE  BUENAS PRACTICAS GANADERAS EN LA PRODUCCION DE BOV","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":27.0,"ap":81,"meses":12,"ultima":"2024-04-01"},{"prog":"SISTEMAS.","nivel":"TÉCNICO","sector":"TRANSVERSAL","fichas":3,"prom":26.3,"ap":79,"meses":25,"ultima":"2023-02-14"}],"GIRALDO":[{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":5,"prom":21.4,"ap":107,"meses":13,"ultima":"2024-03-04"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":26.0,"ap":104,"meses":10,"ultima":"2024-05-20"},{"prog":"CRIAR GALLINAS PONEDORAS CON ALIMENTACIÓN ALTERNA PARA PRODUCIR H","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":18.0,"ap":72,"meses":4,"ultima":"2024-11-06"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":23.5,"ap":94,"meses":1,"ultima":"2025-02-20"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":23.5,"ap":94,"meses":0,"ultima":"2025-03-05"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":22.7,"ap":68,"meses":11,"ultima":"2024-04-24"},{"prog":"MULTIPLICACION Y PROPAGACION DE MATERIAL VEGETAL","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":17.5,"ap":35,"meses":9,"ultima":"2024-06-12"},{"prog":"BÁSICO EN AGRICULTURA ECOLOGICA","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":15.5,"ap":31,"meses":9,"ultima":"2024-06-07"},{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":2,"prom":19.5,"ap":39,"meses":38,"ultima":"2022-02-14"},{"prog":"CONSTRUCCION DE PAVIMENTO CON PLACA HUELLA PARA VIAS TERCIARIAS 1","nivel":"CURSO ESPECIAL","sector":"CONSTRUCCION","fichas":2,"prom":23.0,"ap":46,"meses":5,"ultima":"2024-10-28"},{"prog":"APLICACION DE LAS BUENAS PRACTICAS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":16.5,"ap":33,"meses":1,"ultima":"2025-02-19"},{"prog":"CONTABILIZACION DE OPERACIONES COMERCIALES Y FINANCIERAS.","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":2,"prom":26.0,"ap":52,"meses":25,"ultima":"2023-02-22"}],"HELICONIA":[{"prog":"HIGIENE Y MANIPULACION DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":10,"prom":18.3,"ap":183,"meses":4,"ultima":"2024-11-13"},{"prog":"CRIAR GALLINAS PONEDORAS CON ALIMENTACIÓN ALTERNA PARA PRODUCIR H","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":8,"prom":19.0,"ap":152,"meses":5,"ultima":"2024-10-15"},{"prog":"APLICACION DEL PLAN SANITARIO EN GRANJAS DE PRODUCCION DE HUEVO","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":5,"prom":18.2,"ap":91,"meses":6,"ultima":"2024-09-13"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":22.3,"ap":67,"meses":9,"ultima":"2024-06-06"},{"prog":"MONITOREO DE PLAGAS EN SISTEMAS PRODUCTIVOS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":15.0,"ap":45,"meses":9,"ultima":"2024-07-01"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":25.0,"ap":75,"meses":18,"ultima":"2023-10-06"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":21.3,"ap":64,"meses":11,"ultima":"2024-04-26"},{"prog":"APLICACION DEL PROGRAMA DE BIOSEGURIDAD EN EMPRESAS AVICOLAS.","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":15.5,"ap":31,"meses":10,"ultima":"2024-05-16"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":28.0,"ap":56,"meses":9,"ultima":"2024-06-07"},{"prog":"APLICACION DE  BUENAS PRACTICAS GANADERAS EN LA PRODUCCION DE BOV","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":21.0,"ap":21,"meses":4,"ultima":"2024-11-08"},{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":1,"prom":17.0,"ap":17,"meses":35,"ultima":"2022-05-09"},{"prog":"EMPRENDIMIENTO DE UNIDADES PRODUCTIVAS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":25.0,"ap":25,"meses":5,"ultima":"2024-10-14"}],"ITAGUÍ":[{"prog":"SISTEMAS DE SEGURIDAD PARA TRABAJOS EN ACCESO POR CUERDAS","nivel":"CURSO ESPECIAL","sector":"CONSTRUCCION","fichas":1,"prom":10.0,"ap":10,"meses":11,"ultima":"2024-04-15"}],"LA CEJA":[{"prog":"PRIMEROS  AUXILIOS","nivel":"CURSO ESPECIAL","sector":"SALUD","fichas":1,"prom":40.0,"ap":40,"meses":18,"ultima":"2023-10-02"}],"LIBORINA":[{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":8,"prom":26.9,"ap":215,"meses":1,"ultima":"2025-02-17"},{"prog":"PRODUCCION AGROPECUARIA ECOLOGICA PARA LA SOBERANIA ALIMENTARIA","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":7,"prom":18.6,"ap":130,"meses":1,"ultima":"2025-02-20"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":6,"prom":23.5,"ap":141,"meses":8,"ultima":"2024-07-15"},{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":4,"prom":24.5,"ap":98,"meses":25,"ultima":"2023-03-02"},{"prog":"JEFES DE AREA PARA TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","sector":"SALUD","fichas":4,"prom":7.8,"ap":31,"meses":4,"ultima":"2024-11-13"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":27.5,"ap":110,"meses":1,"ultima":"2025-02-26"},{"prog":"OPERACION TURISTICA LOCAL","nivel":"TÉCNICO","sector":"HOTELERIA Y TURISMO","fichas":4,"prom":21.8,"ap":87,"meses":13,"ultima":"2024-02-15"},{"prog":"TÉCNICAS DE PINTURA EN TELA","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":4,"prom":25.0,"ap":100,"meses":7,"ultima":"2024-08-30"},{"prog":"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":22.3,"ap":67,"meses":9,"ultima":"2024-06-26"},{"prog":"ELABORACION DE COMPLEMENTOS EN BISUTERIA CON TECNICA DE ENSARTADO","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":2,"prom":23.5,"ap":47,"meses":18,"ultima":"2023-09-11"},{"prog":"COCTELERIA TROPICAL","nivel":"CURSO ESPECIAL","sector":"HOTELERIA Y TURISMO","fichas":2,"prom":21.0,"ap":42,"meses":6,"ultima":"2024-09-16"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":23.5,"ap":47,"meses":9,"ultima":"2024-06-17"}],"MEDELLÍN":[{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":8,"prom":23.1,"ap":185,"meses":8,"ultima":"2024-07-19"},{"prog":"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":8,"prom":21.5,"ap":172,"meses":1,"ultima":"2025-02-24"},{"prog":"INSEMINACION ARTIFICIAL DE HEMBRAS BOVINAS APLICANDO PROTOCOLO CO","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":5,"prom":24.4,"ap":122,"meses":21,"ultima":"2023-07-04"},{"prog":"INSEMINACIÓN A CELO DETECTADO","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":30.5,"ap":122,"meses":12,"ultima":"2024-04-02"},{"prog":"PRODUCCION AGROPECUARIA ECOLOGICA PARA LA SOBERANIA ALIMENTARIA","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":25.7,"ap":77,"meses":8,"ultima":"2024-08-02"},{"prog":"TÉCNICAS DE PINTURA EN TELA","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":3,"prom":24.7,"ap":74,"meses":10,"ultima":"2024-05-21"},{"prog":"ELABORACIÓN DE ARTÍCULOS DECORATIVOS Y UTILITARIOS CON MATERIALES","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":2,"prom":27.5,"ap":55,"meses":20,"ultima":"2023-08-01"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":21.0,"ap":42,"meses":9,"ultima":"2024-06-07"},{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":2,"prom":26.0,"ap":52,"meses":38,"ultima":"2022-02-04"},{"prog":"ECONOMIA SOLIDARIA UN EJE ESTRATEGICO DE SUPERVIVENCIA SOCIAL.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":18.5,"ap":37,"meses":12,"ultima":"2024-03-23"},{"prog":"APLICACION DEL PLAN SANITARIO EN GRANJAS DE PRODUCCION DE HUEVO","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":17.0,"ap":17,"meses":10,"ultima":"2024-05-09"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":24.0,"ap":24,"meses":10,"ultima":"2024-05-18"}],"OLAYA":[{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":5,"prom":28.6,"ap":143,"meses":12,"ultima":"2024-03-18"},{"prog":"PROGRAMACION DE APLICACIONES PARA DISPOSITIVOS MOVILES","nivel":"TÉCNICO","sector":"TRANSVERSAL","fichas":2,"prom":29.0,"ap":58,"meses":13,"ultima":"2024-02-24"},{"prog":"SISTEMAS.","nivel":"TÉCNICO","sector":"TRANSVERSAL","fichas":2,"prom":28.0,"ap":56,"meses":25,"ultima":"2023-02-18"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":22.0,"ap":44,"meses":11,"ultima":"2024-04-16"},{"prog":"CONSTRUCCION DE EDIFICACIONES","nivel":"TÉCNICO","sector":"CONSTRUCCION","fichas":2,"prom":31.0,"ap":62,"meses":4,"ultima":"2024-11-05"},{"prog":"PRACTICAS DE MANEJO SANITARIO EN BOVINOS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":27.5,"ap":55,"meses":22,"ultima":"2023-06-07"},{"prog":"BÁSICO MERCADEO Y SERVICIO AL CLIENTE","nivel":"CURSO ESPECIAL","sector":"EDUCACION","fichas":1,"prom":23.0,"ap":23,"meses":4,"ultima":"2024-11-16"},{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":1,"prom":32.0,"ap":32,"meses":1,"ultima":"2025-02-22"},{"prog":"APLICACION DE PRACTICAS DE ALIMENTACION EN BOVINOS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":32.0,"ap":32,"meses":13,"ultima":"2024-02-16"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":16.0,"ap":16,"meses":8,"ultima":"2024-07-09"},{"prog":"ELABORACIÓN DE PANES ARTESANALES","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":41.0,"ap":41,"meses":19,"ultima":"2023-09-07"},{"prog":"EMPRENDEDOR EN PROCESAMIENTO ARTESANAL  DE DERIVADOS  FRUTAS Y HO","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":24.0,"ap":24,"meses":24,"ultima":"2023-04-04"}],"PEQUE":[{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":4,"prom":38.5,"ap":154,"meses":13,"ultima":"2024-03-04"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":23.5,"ap":94,"meses":8,"ultima":"2024-07-08"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":25.0,"ap":75,"meses":20,"ultima":"2023-07-21"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":24.3,"ap":73,"meses":1,"ultima":"2025-02-26"},{"prog":"CULTIVOS AGRICOLAS","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":2,"prom":16.0,"ap":32,"meses":37,"ultima":"2022-03-02"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":24.5,"ap":49,"meses":10,"ultima":"2024-05-16"},{"prog":"BUENAS PRACTICAS DE MANUFACTURA EN EL PROCESAMIENTO DE CARNE Y DE","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":22.0,"ap":22,"meses":4,"ultima":"2024-11-18"},{"prog":"BÁSICO MERCADEO Y SERVICIO AL CLIENTE","nivel":"CURSO ESPECIAL","sector":"EDUCACION","fichas":1,"prom":21.0,"ap":21,"meses":3,"ultima":"2024-12-05"},{"prog":"FUNDAMENTOS EN HIDROTERAPIA APLICADA A LA ESTETICA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":13.0,"ap":13,"meses":19,"ultima":"2023-08-11"},{"prog":"ELABORACIÓN DE PANES ARTESANALES","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":36.0,"ap":36,"meses":18,"ultima":"2023-10-09"},{"prog":"ELABORACION DE COMPLEMENTOS EN BISUTERIA CON TECNICA DE ENSARTADO","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":1,"prom":26.0,"ap":26,"meses":17,"ultima":"2023-10-19"},{"prog":"GESTION DE PROYECTOS COMUNITARIOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":13.0,"ap":13,"meses":23,"ultima":"2023-05-02"}],"SABANALARGA":[{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":45.0,"ap":180,"meses":5,"ultima":"2024-11-01"},{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":3,"prom":27.7,"ap":83,"meses":25,"ultima":"2023-02-23"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":21.0,"ap":63,"meses":5,"ultima":"2024-10-09"},{"prog":"ELABORACIÓN DE ARTÍCULOS DECORATIVOS Y UTILITARIOS CON MATERIALES","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":2,"prom":23.5,"ap":47,"meses":23,"ultima":"2023-05-09"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":25.0,"ap":50,"meses":11,"ultima":"2024-04-12"},{"prog":"CONFECCIÓN DE CAMISETAS DEPORTIVAS","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":1,"prom":20.0,"ap":20,"meses":3,"ultima":"2024-12-03"},{"prog":"COCINA INTERNACIONAL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":21.0,"ap":21,"meses":7,"ultima":"2024-08-21"},{"prog":"CONSERVACION DE RECURSOS NATURALES","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":1,"prom":28.0,"ap":28,"meses":1,"ultima":"2025-02-18"},{"prog":"ELABORACION DE COMPLEMENTOS EN BISUTERIA CON TECNICA DE ENSARTADO","nivel":"CURSO ESPECIAL","sector":"INDUSTRIA","fichas":1,"prom":26.0,"ap":26,"meses":18,"ultima":"2023-10-03"},{"prog":"ELABORACION DE HELADOS Y POSTRES LACTEOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":31.0,"ap":31,"meses":6,"ultima":"2024-09-16"},{"prog":"CRIAR GALLINAS PONEDORAS CON ALIMENTACIÓN ALTERNA PARA PRODUCIR H","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":1,"prom":23.0,"ap":23,"meses":1,"ultima":"2025-02-27"},{"prog":"COCINA COLOMBIANA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":1,"prom":20.0,"ap":20,"meses":0,"ultima":"2025-03-03"}],"SAN JERÓNIMO":[{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":11,"prom":26.6,"ap":293,"meses":6,"ultima":"2024-09-10"},{"prog":"SISTEMAS AGROPECUARIOS ECOLOGICOS.","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":9,"prom":25.0,"ap":225,"meses":13,"ultima":"2024-03-06"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":27.2,"ap":109,"meses":1,"ultima":"2025-02-25"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":19.5,"ap":78,"meses":8,"ultima":"2024-07-27"},{"prog":"BIOSEGURIDAD APLICADA A LA ESTETICA Y BELLEZA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":22.5,"ap":90,"meses":6,"ultima":"2024-09-09"},{"prog":"IMPLEMENTACION DE PRACTICAS DE MANEJO DEL HUEVO","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":24.5,"ap":98,"meses":8,"ultima":"2024-08-02"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":21.5,"ap":86,"meses":7,"ultima":"2024-08-30"},{"prog":"TENDENCIAS EN DECORACION PARA MANICURA Y PEDICURA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":22.8,"ap":91,"meses":8,"ultima":"2024-07-22"},{"prog":"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":21.5,"ap":86,"meses":10,"ultima":"2024-06-05"},{"prog":"COCINA PERUANA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":21.0,"ap":63,"meses":18,"ultima":"2023-10-06"},{"prog":"COCINA COLOMBIANA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":22.0,"ap":66,"meses":5,"ultima":"2024-10-25"},{"prog":"COCINA NAVIDEÑA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":24.0,"ap":72,"meses":4,"ultima":"2024-12-02"}],"SANTAFÉ DE ANTIOQUIA":[{"prog":"ENGLISH DOES WORK - LEVEL 1","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":53,"prom":58.3,"ap":3091,"meses":0,"ultima":"2025-03-17"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":33,"prom":24.9,"ap":823,"meses":0,"ultima":"2025-03-04"},{"prog":"JEFES DE AREA PARA TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","sector":"SALUD","fichas":26,"prom":7.7,"ap":201,"meses":0,"ultima":"2025-03-05"},{"prog":"TRABAJADOR AUTORIZADO PARA TRABAJO EN ALTURAS","nivel":"CURSO ESPECIAL","sector":"SALUD","fichas":22,"prom":7.4,"ap":162,"meses":0,"ultima":"2025-03-04"},{"prog":"INGLES BASICO - NIVEL 1","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":18,"prom":25.0,"ap":450,"meses":1,"ultima":"2025-02-14"},{"prog":"REENTRENAMIENTO EN TRABAJO EN ALTURAS PARA TRABAJADOR AUTORIZADO","nivel":"CURSO ESPECIAL","sector":"SALUD","fichas":17,"prom":7.1,"ap":121,"meses":1,"ultima":"2025-02-20"},{"prog":"INGLES BASICO - NIVEL 2","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":16,"prom":27.1,"ap":434,"meses":3,"ultima":"2024-12-05"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":15,"prom":34.4,"ap":516,"meses":4,"ultima":"2024-12-02"},{"prog":"AUDITORIA INTERNA DE CALIDAD - NTC ISO 9001","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":15,"prom":16.0,"ap":240,"meses":18,"ultima":"2023-09-21"},{"prog":"INGLES BASICO - NIVEL 3","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":15,"prom":28.6,"ap":429,"meses":3,"ultima":"2024-12-05"},{"prog":"COCINA INTERNACIONAL.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":14,"prom":36.8,"ap":515,"meses":1,"ultima":"2025-02-11"},{"prog":"BUENAS PRACTICAS EN LA MANIPULACION DE LA CARNE","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":13,"prom":36.4,"ap":473,"meses":0,"ultima":"2025-03-11"}],"SOPETRÁN":[{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":13,"prom":25.2,"ap":328,"meses":8,"ultima":"2024-07-17"},{"prog":"HIGIENE Y MANIPULACION DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":8,"prom":17.5,"ap":140,"meses":3,"ultima":"2024-12-04"},{"prog":"HIGIENE Y MANIPULACIÓN DE ALIMENTOS","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":6,"prom":23.8,"ap":143,"meses":1,"ultima":"2025-02-28"},{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":5,"prom":20.0,"ap":100,"meses":4,"ultima":"2024-11-07"},{"prog":"PRODUCCION AGROPECUARIA ECOLOGICA PARA LA SOBERANIA ALIMENTARIA","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":5,"prom":23.8,"ap":119,"meses":6,"ultima":"2024-09-04"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":4,"prom":21.8,"ap":87,"meses":7,"ultima":"2024-08-06"},{"prog":"FORTALECIMIENTO EN LA APLICACION DE  PRACTICAS SANITARIAS EN BOVI","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":21.7,"ap":65,"meses":7,"ultima":"2024-08-27"},{"prog":"APLICACION DE LAS BUENAS PRACTICAS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":27.3,"ap":82,"meses":21,"ultima":"2023-07-04"},{"prog":"CULTIVOS AGRICOLAS","nivel":"TÉCNICO","sector":"AGROPECUARIO","fichas":2,"prom":14.5,"ap":29,"meses":37,"ultima":"2022-03-07"},{"prog":"EXCEL INTERMEDIO","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":21.5,"ap":43,"meses":6,"ultima":"2024-09-25"},{"prog":"MANEJO SANITARIO EN GRANJAS PORCINAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":2,"prom":23.0,"ap":46,"meses":1,"ultima":"2025-02-26"},{"prog":"INGLES BASICO - NIVEL 3","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":2,"prom":27.5,"ap":55,"meses":12,"ultima":"2024-03-18"}],"URAMITA":[{"prog":"COMPORTAMIENTO EMPRENDEDOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":9,"prom":19.7,"ap":177,"meses":4,"ultima":"2024-11-15"},{"prog":"GENERACION DE IDEAS PARA UN NEGOCIO INNOVADOR","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":9,"prom":21.1,"ap":190,"meses":1,"ultima":"2025-02-24"},{"prog":"MANEJO DE LA NUTRICION EN CULTIVOS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":6,"prom":16.8,"ap":101,"meses":6,"ultima":"2024-10-03"},{"prog":"APLICACION DE LAS BUENAS PRACTICAS AGRICOLAS","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":15.5,"ap":62,"meses":1,"ultima":"2025-02-21"},{"prog":"PRODUCCION AGROPECUARIA ECOLOGICA PARA LA SOBERANIA ALIMENTARIA","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":4,"prom":21.2,"ap":85,"meses":13,"ultima":"2024-02-26"},{"prog":"CONTABILIZACION DE OPERACIONES COMERCIALES Y FINANCIERAS.","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":3,"prom":25.7,"ap":77,"meses":13,"ultima":"2024-02-21"},{"prog":"MANIPULACION HIGIENICA DE ALIMENTOS.","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":38.3,"ap":115,"meses":5,"ultima":"2024-10-22"},{"prog":"MANEJO BASICO DE LA HERRAMIENTA DE HOJAS DE CALCULO EXCEL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":9.3,"ap":28,"meses":18,"ultima":"2023-09-18"},{"prog":"COCINA COLOMBIANA","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":22.3,"ap":67,"meses":5,"ultima":"2024-10-26"},{"prog":"PODA DE ESPECIES VEGETALES","nivel":"CURSO ESPECIAL","sector":"AGROPECUARIO","fichas":3,"prom":17.0,"ap":51,"meses":1,"ultima":"2025-02-21"},{"prog":"COCINA INTERNACIONAL","nivel":"CURSO ESPECIAL","sector":"SERVICIOS","fichas":3,"prom":27.7,"ap":83,"meses":18,"ultima":"2023-09-11"},{"prog":"ASISTENCIA ADMINISTRATIVA .","nivel":"TÉCNICO","sector":"SERVICIOS","fichas":2,"prom":29.5,"ap":59,"meses":12,"ultima":"2024-03-13"}]}'
_RB = '{"Café":{"fichas":63,"ap":1391,"muns":{"ANZÁ":322,"EBÉJICO":177,"CAICEDO":173,"SAN JERÓNIMO":123,"BURITICA":106,"SABANALARGA":105,"SOPETRÁN":82,"SANTAFÉ DE ANTIOQUIA":81}},"Cacao":{"fichas":12,"ap":291,"muns":{"URAMITA":123,"FRONTINO":72,"DABEIBA":44,"CAÑASGORDAS":26,"EBÉJICO":26}},"Aguacate":{"fichas":1,"ap":23,"muns":{"SOPETRÁN":23}},"Ganadería":{"fichas":129,"ap":3021,"muns":{"DABEIBA":466,"FRONTINO":421,"SAN JERÓNIMO":325,"LIBORINA":263,"URAMITA":239,"SOPETRÁN":206,"CAÑASGORDAS":199,"SANTAFÉ DE ANTIOQUIA":162}},"Avicultura":{"fichas":38,"ap":777,"muns":{"HELICONIA":152,"EBÉJICO":90,"URAMITA":88,"ANZÁ":87,"DABEIBA":83,"GIRALDO":72,"ARMENIA":62,"SAN JERÓNIMO":30}},"Turismo":{"fichas":10,"ap":203,"muns":{"LIBORINA":46,"EBÉJICO":42,"SANTAFÉ DE ANTIOQUIA":35,"CAICEDO":30,"BURITICA":26,"CAÑASGORDAS":15,"SABANALARGA":9}},"Agricultura":{"fichas":223,"ap":4823,"muns":{"DABEIBA":652,"URAMITA":459,"SAN JERÓNIMO":408,"FRONTINO":363,"EBÉJICO":359,"CAÑASGORDAS":337,"SOPETRÁN":326,"LIBORINA":278}},"Piscicultura":{"fichas":2,"ap":52,"muns":{"CAÑASGORDAS":26,"DABEIBA":26}},"Alturas":{"fichas":105,"ap":798,"muns":{"SANTAFÉ DE ANTIOQUIA":553,"BURITICA":148,"DABEIBA":44,"LIBORINA":31,"CAÑASGORDAS":22}},"Inglés":{"fichas":195,"ap":7355,"muns":{"SANTAFÉ DE ANTIOQUIA":6470,"DABEIBA":204,"SAN JERÓNIMO":137,"SOPETRÁN":119,"FRONTINO":117,"CAICEDO":113,"CAÑASGORDAS":64,"LIBORINA":36}},"Emprendimiento":{"fichas":308,"ap":6963,"muns":{"DABEIBA":1270,"CAÑASGORDAS":808,"URAMITA":655,"EBÉJICO":639,"SAN JERÓNIMO":396,"SOPETRÁN":367,"FRONTINO":343,"BURITICA":327}},"Alimentos":{"fichas":374,"ap":9919,"muns":{"SANTAFÉ DE ANTIOQUIA":3702,"SOPETRÁN":764,"SAN JERÓNIMO":727,"DABEIBA":415,"EBÉJICO":408,"URAMITA":394,"ANZÁ":390,"CAÑASGORDAS":364}},"TIC/Excel":{"fichas":226,"ap":5104,"muns":{"SANTAFÉ DE ANTIOQUIA":1453,"DABEIBA":914,"SAN JERÓNIMO":403,"FRONTINO":386,"EBÉJICO":326,"SOPETRÁN":270,"MEDELLÍN":172,"LIBORINA":169}},"Medio Ambiente":{"fichas":146,"ap":3265,"muns":{"MEDELLÍN":395,"DABEIBA":362,"SAN JERÓNIMO":334,"GIRALDO":244,"LIBORINA":228,"SOPETRÁN":225,"CAÑASGORDAS":196,"CAICEDO":189}}}'

# =============================================================================
# PÁGINA
# =============================================================================
st.set_page_config(
    page_title="SN Predict — SENA Occidente",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── BASE ────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #F4F6FA;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.block-container {
    padding: 1.5rem 2.2rem 1rem !important;
    max-width: 1280px;
}

/* ── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2044 0%, #0D1B2A 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.25) !important;
}
/* Texto genérico del sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #CBD5E1 !important;
}
/* Radio buttons — texto visible */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio span,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #E2E8F0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
/* Radio seleccionado */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div,
[data-testid="stSidebar"] .stRadio [aria-checked="true"] span {
    color: #60A5FA !important;
    font-weight: 600 !important;
}
/* Selectbox en sidebar */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
    color: #94A3B8 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 7px !important;
}
/* Botón del sidebar */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1E40AF, #2563EB) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
    box-shadow: 0 2px 8px rgba(30,64,175,0.35) !important;
    transition: all .18s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    box-shadow: 0 4px 14px rgba(30,64,175,0.5) !important;
    transform: translateY(-1px) !important;
}
/* Divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10) !important;
    margin: 10px 0 !important;
}
/* Icono radio */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 6px 8px !important;
    border-radius: 7px !important;
    transition: background .15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.07) !important;
}

/* ── MÉTRICAS ────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #fff !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.06) !important;
    transition: box-shadow .2s !important;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(15,23,42,0.10) !important;
}
div[data-testid="metric-container"] label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: .04em !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}
div[data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 11px !important;
}

/* ── TABS ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #E8EDF5;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: #475569;
    padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #1E40AF !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.10) !important;
}

/* ── EXPANDER ────────────────────────────────────────────── */
.stExpander {
    background: #fff !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    margin-bottom: .5rem !important;
}
.stExpander summary {
    font-weight: 600 !important;
    color: #0F172A !important;
}

/* ── CARDS ───────────────────────────────────────────────── */
.card {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 2px 10px rgba(15,23,42,0.05);
    margin-bottom: .9rem;
}
.card-h {
    font-size: 12px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 7px;
    letter-spacing: .01em;
}

/* ── BOTONES PRINCIPALES ─────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1E40AF, #2563EB) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    box-shadow: 0 3px 10px rgba(30,64,175,0.30) !important;
    transition: all .18s !important;
    letter-spacing: .02em !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    box-shadow: 0 5px 18px rgba(30,64,175,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #fff !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    transition: all .15s !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #1E40AF !important;
    color: #1E40AF !important;
    background: #EFF6FF !important;
}

/* ── SELECTBOX / INPUTS ──────────────────────────────────── */
div[data-baseweb="select"] {
    border-radius: 8px !important;
}
div[data-baseweb="select"] div {
    border-radius: 8px !important;
    font-size: 13px !important;
}
.stTextInput input {
    border-radius: 8px !important;
    border: 1.5px solid #E2E8F0 !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    transition: border-color .15s !important;
}
.stTextInput input:focus {
    border-color: #1E40AF !important;
    box-shadow: 0 0 0 3px rgba(30,64,175,0.12) !important;
}
.stNumberInput input {
    border-radius: 8px !important;
    font-size: 13px !important;
}

/* ── SLIDER ──────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #1E40AF !important;
    border-color: #1E40AF !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
    background: #1E40AF !important;
    color: #fff !important;
    font-size: 11px !important;
}

/* ── MULTISELECT ─────────────────────────────────────────── */
.stMultiSelect span[data-baseweb="tag"] {
    background: #DBEAFE !important;
    color: #1E40AF !important;
    border-radius: 5px !important;
    font-size: 11px !important;
}

/* ── SUCCESS / INFO / WARNING / ERROR ────────────────────── */
.stSuccess {
    background: #F0FDF4 !important;
    border: 1px solid #BBF7D0 !important;
    border-radius: 8px !important;
    color: #166534 !important;
}
.stInfo {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 8px !important;
    color: #1E40AF !important;
}
.stWarning {
    background: #FFFBEB !important;
    border: 1px solid #FDE68A !important;
    border-radius: 8px !important;
    color: #92400E !important;
}

/* ── ENCABEZADOS ─────────────────────────────────────────── */
.sh {
    font-size: 20px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 4px;
    letter-spacing: -.01em;
}
.sd {
    font-size: 12px;
    color: #64748B;
    margin-bottom: 1.1rem;
    font-weight: 400;
}

/* ── RESULT BOX ──────────────────────────────────────────── */
.res-box {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
    border: 1px solid #BFDBFE;
    border-radius: 14px;
    padding: 1.15rem;
    box-shadow: 0 4px 16px rgba(30,64,175,0.09);
}
.res-n {
    font-size: 48px;
    font-weight: 800;
    color: #1E40AF;
    line-height: 1;
    letter-spacing: -.02em;
}
.res-l { font-size: 11px; color: #3B82F6; margin-top: 3px; font-weight: 500; }
.vt { height: 8px; background: #E2E8F0; border-radius: 4px; overflow: hidden; margin: 6px 0; }
.vf { height: 100%; border-radius: 4px; transition: width .5s cubic-bezier(.4,0,.2,1); }

/* ── BADGES ──────────────────────────────────────────────── */
.bdg {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    letter-spacing: .01em;
}
.bs { background: #DCFCE7; color: #166534; }
.bw { background: #FEF3C7; color: #92400E; }
.bd { background: #FEE2E2; color: #991B1B; }
.bi { background: #DBEAFE; color: #1E40AF; }
.bg { background: #F1F5F9; color: #475569; }

/* ── ALERTAS ─────────────────────────────────────────────── */
.al-s {
    background: #F0FDF4; border: 1px solid #BBF7D0;
    border-left: 4px solid #22C55E; border-radius: 8px;
    padding: 9px 13px; font-size: 11px; color: #166534;
    margin-bottom: 7px; line-height: 1.5;
}
.al-w {
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B; border-radius: 8px;
    padding: 9px 13px; font-size: 11px; color: #92400E;
    margin-bottom: 7px; line-height: 1.5;
}
.al-r {
    background: #FEF2F2; border: 1px solid #FECACA;
    border-left: 4px solid #EF4444; border-radius: 8px;
    padding: 9px 13px; font-size: 11px; color: #991B1B;
    margin-bottom: 7px; line-height: 1.5;
}
.al-i {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-left: 4px solid #3B82F6; border-radius: 8px;
    padding: 9px 13px; font-size: 11px; color: #1E40AF;
    margin-bottom: 7px; line-height: 1.5;
}

/* ── RUTA FORMATIVA ──────────────────────────────────────── */
.rt {
    border-radius: 9px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 11px;
    line-height: 1.55;
    font-weight: 500;
}

/* ── FOOTER ──────────────────────────────────────────────── */
.foot {
    text-align: center;
    font-size: 11px;
    color: #94A3B8;
    padding: .85rem 0;
    border-top: 1px solid #E2E8F0;
    margin-top: 2rem;
    font-weight: 400;
}

/* ── TABLA ───────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── SCROLLBAR ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARGA
# =============================================================================
@st.cache_data(show_spinner="Cargando datos PE04…")
def _load():
    if not _HL:
        return None, {}, {}, {}, {}, {}
    try:
        df = cargar_df()
        return (df, get_sec_prog(df), get_prog_mun(df),
                get_resp_stats(df), get_mun_stats(df), get_rubros(df))
    except Exception:
        return None, {}, {}, {}, {}, {}

_df, _sp, _pm, _rs, _ms, _rb = _load()
TIENE = _df is not None

SEC_PROG  = _sp  or json.loads(_SP)
PROG_MUN  = _pm  or json.loads(_PM)
RESP      = _rs  or json.loads(_RS)
MUN_STATS = _ms  or json.loads(_MS)
RUBROS    = _rb  or json.loads(_RB)
MUN_PROGS = json.loads(_MP)

SECTORES   = sorted(SEC_PROG.keys())
MUNICIPIOS = sorted(MUN_STATS.keys())
ML = [1, 12]; MM = [2, 3]; MA = [9, 10]
MESES_L = ["","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
CN = {"CURSO ESPECIAL":"#1E40AF","TÉCNICO":"#0F6E56","TECNÓLOGO":"#B45309","EVENTO":"#64748B"}
LN = {"CURSO ESPECIAL":"Complementario","TÉCNICO":"Técnico","TECNÓLOGO":"Tecnólogo","EVENTO":"Evento"}
MF = {
    "SANTAFÉ DE ANTIOQUIA":1.30,"DABEIBA":1.12,"FRONTINO":1.05,"CAÑASGORDAS":1.01,
    "SOPETRÁN":1.03,"SAN JERÓNIMO":1.04,"EBÉJICO":1.01,"URAMITA":1.02,
    "LIBORINA":1.00,"BURITICA":0.93,"ABRIAQUÍ":0.90,"ANZÁ":0.94,
    "CAICEDO":1.00,"HELICONIA":0.92,"GIRALDO":0.96,"OLAYA":0.97,
    "PEQUE":0.98,"SABANALARGA":1.06,"MEDELLÍN":1.02,"ARMENIA":0.98,
    "ITAGUÍ":0.95,"LA CEJA":0.97,
}

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        "<div style='padding:16px 4px 20px;border-bottom:1px solid rgba(255,255,255,.10)'>"
        "<div style='display:flex;align-items:center;gap:12px'>"
        "<div style='width:44px;height:44px;"
        "background:linear-gradient(135deg,#1E40AF,#2563EB);"
        "border-radius:12px;display:flex;align-items:center;"
        "justify-content:center;font-size:22px;"
        "box-shadow:0 4px 12px rgba(30,64,175,0.4);flex-shrink:0'>🎓</div>"
        "<div>"
        "<div style='font-size:17px;font-weight:800;color:#F8FAFC;letter-spacing:-.01em'>SN Predict</div>"
        "<div style='font-size:11px;color:#93C5FD;font-weight:500'>SENA Occidente · v3.2</div>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    modulo = st.radio("nav", [
        "🎯 Predictor inteligente",
        "📊 Demanda por nivel",
        "⚡ Recomendaciones",
        "🔭 Oportunidades de mejora",
        "📖 Manual de usuario",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown(
        "<p style='font-size:10px;color:#94A3B8;font-weight:600;letter-spacing:.06em'>FILTRO GLOBAL</p>",
        unsafe_allow_html=True,
    )
    g_mun = st.selectbox("Municipio", ["Todos"] + MUNICIPIOS, key="g_mun", label_visibility="collapsed")
    g_sec = st.selectbox("Sector",    ["Todos"] + SECTORES,   key="g_sec", label_visibility="collapsed")
    st.divider()
    if st.button("▶  Tour rápido"):
        st.session_state["tour"] = 0
        st.session_state["show_tour"] = True
    _ml = (
        (_HERE / "models" / "pipeline_y1_regresion.pkl").exists() or
        (_HERE.parent / "models" / "pipeline_y1_regresion.pkl").exists()
    )
    _ml_txt  = "Modelo ML activo ✅" if _ml else "Modo heurístico ℹ️"
    _ml_clr  = "#DCFCE7" if _ml else "#DBEAFE"
    _ml_tc   = "#166534" if _ml else "#1E40AF"
    _dat_txt = f"{len(_df):,} fichas cargadas 📊" if TIENE else "Datos embebidos ⚠️"
    _dat_clr = "#DCFCE7" if TIENE else "#FEF3C7"
    _dat_tc  = "#166534" if TIENE else "#92400E"
    st.markdown(
        f"<div style='background:{_ml_clr};color:{_ml_tc};border-radius:6px;"
        f"padding:5px 10px;font-size:11px;font-weight:600;margin-bottom:5px'>"
        f"{_ml_txt}</div>"
        f"<div style='background:{_dat_clr};color:{_dat_tc};border-radius:6px;"
        f"padding:5px 10px;font-size:11px;font-weight:600'>"
        f"{_dat_txt}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:10px;color:#475569;text-align:center;margin-top:12px'>"
        "SN Predict © 2026 · SENA Centro Occidente</p>",
        unsafe_allow_html=True,
    )

# =============================================================================
# TOUR
# =============================================================================
TOUR = [
    ("🎓 Bienvenido a SN Predict",
     "Analiza 2.551 fichas históricas de 22 municipios del Centro SENA Occidente (2020-2025). "
     "Predice demanda, identifica oportunidades y optimiza la oferta formativa con datos reales."),
    ("🎯 Filtros dinámicos del Predictor",
     "1) Escribe palabra clave para buscar programas en todos los sectores. "
     "2) Selecciona Sector → lista de programas se filtra sola. "
     "3) Elige programa → duración y nivel se auto-rellenan. ⭐ = mayor demanda histórica."),
    ("🗺️ Mapa de calor e instructores",
     "El mapa muestra intensidad de demanda por municipio (Total / Fichas / Prom). "
     "El historial de instructores revela quién convoca más — factor clave en la predicción."),
    ("📊 Demanda y ⚡ Recomendaciones",
     "En Demanda filtra por nivel para comparar sectores. "
     "En Recomendaciones verás ruta formativa: Complementarios → Técnico → Tecnólogo, "
     "con fechas de última oferta y badge de demanda para cada programa."),
    ("🔭 Oportunidades de mejora",
     "Barras apiladas por nivel (azul=Complementario, verde=Técnico, ámbar=Tecnólogo). "
     "Filtra por año y rubro productivo. Carga un Excel nuevo para actualizar datos."),
]

if st.session_state.get("show_tour"):
    si = st.session_state.get("tour", 0)
    ttl, txt = TOUR[si]
    dots_html = ""
    for _i in range(len(TOUR)):
        _w  = "18" if _i == si else "7"
        _bg = "#fff" if _i == si else "rgba(255,255,255,0.35)"
        dots_html += (
            f'<div style="width:{_w}px;height:6px;border-radius:3px;'
            f'background:{_bg};display:inline-block;margin-right:4px"></div>'
        )
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1E3A8A,#065F46);'
        f'border-radius:12px;padding:1.1rem 1.25rem;margin-bottom:1rem">'
        f'<div style="font-size:15px;font-weight:700;color:#fff;margin-bottom:5px">'
        f'{ttl}<span style="float:right;font-size:11px;opacity:.65">Paso {si+1}/{len(TOUR)}</span></div>'
        f'<div style="font-size:12px;color:rgba(255,255,255,.88);line-height:1.7">{txt}</div>'
        f'<div style="margin-top:10px">{dots_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    cp, cn, _ = st.columns([1, 1, 3])
    with cp:
        if si > 0 and st.button("← Anterior", key="tp"):
            st.session_state["tour"] -= 1
            st.rerun()
    with cn:
        lbl = "Siguiente →" if si < len(TOUR) - 1 else "✓ Cerrar"
        if st.button(lbl, key="tn", type="primary"):
            if si < len(TOUR) - 1:
                st.session_state["tour"] += 1
                st.rerun()
            else:
                st.session_state["show_tour"] = False
                st.rerun()

# =============================================================================
# KPIs
# =============================================================================
_s  = {k: v for k, v in MUN_STATS.items() if g_mun == "Todos" or k == g_mun}
_tf = sum(v["fichas"] for v in _s.values()) if _s else 2551
_ta = sum(v["ap"]     for v in _s.values()) if _s else 62472
_tm = len(_s) if _s else 22
c1k, c2k, c3k, c4k = st.columns(4)
c1k.metric("Fichas históricas",    f"{_tf:,}",  "2020-2025")
c2k.metric("Aprendices formados",  f"{_ta:,}",  f"prom {round(_ta/max(_tf,1),1)}/ficha")
c3k.metric("Municipios cubiertos", str(_tm),     "Occidente Ant.")
c4k.metric("Tasa alto impacto",    "28.5%",      "▲ >25 ap/ficha")
st.write("")

# =============================================================================
# HELPERS
# =============================================================================
def vig_badge(m: int) -> str:
    if m > 12:
        return f'<span class="bdg bd">⚠️ {m}m sin ofertar</span>'
    if m > 6:
        return f'<span class="bdg bw">🕐 {m}m</span>'
    return f'<span class="bdg bs">✅ {m}m</span>'

def niv_badge(niv: str) -> str:
    cls = {"CURSO ESPECIAL":"bi","TÉCNICO":"bs","TECNÓLOGO":"bw","EVENTO":"bg"}.get(niv,"bg")
    lbl = LN.get(niv, niv)
    return f'<span class="bdg {cls}">{lbl}</span>'

def dem_badge(prom: float) -> str:
    if prom >= 26:
        return '<span class="bdg bs">⭐ Alta demanda</span>'
    if prom >= 21:
        return '<span class="bdg bi">Normal</span>'
    return '<span class="bdg bg">Baja</span>'

def barra(nombre: str, val: float, max_v: float, color: str, ancho: int = 160) -> str:
    pct = round(val / max(max_v, 1) * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">'
        f'<span style="font-size:11px;color:#475569;width:{ancho}px;overflow:hidden;'
        f'text-overflow:ellipsis;white-space:nowrap;flex-shrink:0">{nombre}</span>'
        f'<div style="flex:1;height:5px;background:#E2E8F0;border-radius:2px;overflow:hidden">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:2px"></div></div>'
        f'<span style="font-size:10px;font-weight:600;color:{color};width:32px;text-align:right">{val}</span>'
        f'</div>'
    )

def tabla_html(headers: list, rows: list) -> str:
    th = "".join(
        f'<th style="text-align:left;padding:7px 9px;font-size:10px;font-weight:600;'
        f'color:#64748B;border-bottom:.5px solid #E2E8F0;background:#F8FAFC">{h}</th>'
        for h in headers
    )
    trs = ""
    for row in rows:
        tds = "".join(
            f'<td style="padding:7px 9px;font-size:11px;color:#475569;'
            f'border-bottom:.5px solid #E2E8F0">{c}</td>'
            for c in row
        )
        trs += f"<tr>{tds}</tr>"
    return (
        f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;'
        f'background:#fff;border-radius:10px;overflow:hidden">'
        f'<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'
    )

# =============================================================================
# MÓDULO 1 — PREDICTOR INTELIGENTE
# =============================================================================
if modulo == "🎯 Predictor inteligente":

    st.markdown('<div class="sh" style="display:flex;align-items:center;gap:10px">🎯 Predictor inteligente</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sd">Sector → filtra programas · ⭐ = mayor demanda · '
        'Programa → auto-rellena datos · Mapa de calor + Historial instructores</div>',
        unsafe_allow_html=True,
    )

    kw = st.text_input(
        "🔍 Búsqueda por palabra clave",
        placeholder="café, ganadería, alturas, inglés, emprendimiento, excel…",
        key="kw_v3",
    )
    if kw and len(kw) >= 2:
        todos_p = []
        for sec, progs in SEC_PROG.items():
            for p in progs:
                if not isinstance(p, dict):
                    continue
                prog_n = p.get("prog", p.get("NOMBRE_PROGRAMA_FORMACION", ""))
                nivel  = p.get("nivel", p.get("NIVEL_FORMACION", "CURSO ESPECIAL"))
                dur    = p.get("dur",   p.get("DURACION_PROGRAMA", 48))
                prom   = p.get("prom",  p.get("TOTAL_APRENDICES", 0))
                meses  = p.get("meses", 0)
                estr   = p.get("estrella", False)
                if prog_n:
                    todos_p.append((prog_n, sec, nivel, dur, prom, meses, estr))
        mts = [(p,s,n,du,pm,m,e) for p,s,n,du,pm,m,e in todos_p
               if kw.lower() in p.lower()][:6]
        if mts:
            st.markdown(f"**{len(mts)} resultado(s) para «{kw}»:**")
            cks = st.columns(min(len(mts), 3))
            for idx_b, (prog, sec, niv, dur, prom, mes, estr) in enumerate(mts):
                with cks[idx_b % 3]:
                    ico  = "⭐" if estr else ("⚠️" if mes > 12 else "📌")
                    lbl_k = f"{ico} {prog[:28]} | {niv} · {prom} ap/f"
                    if st.button(lbl_k, key=f"kw{idx_b}", use_container_width=True):
                        st.session_state["v3s"] = sec
                        st.session_state["v3p"] = prog
                        st.rerun()
        else:
            st.caption(f"Sin resultados para «{kw}». Prueba: café, ganadería, inglés…")

    col_f, col_r = st.columns([1.05, 0.95], gap="large")

    with col_f:
        st.markdown('<div class="card"><div class="card-h">⚙️ Parámetros de la ficha</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            idx_s = 0
            if st.session_state.get("v3s") in SECTORES:
                idx_s = SECTORES.index(st.session_state["v3s"])
            p_sec = st.selectbox("📊 Sector productivo", SECTORES, index=idx_s, key="ps_v3")
            if p_sec != st.session_state.get("v3s"):
                st.session_state["v3s"] = p_sec
                st.session_state.pop("v3p", None)
        progs_s = SEC_PROG.get(p_sec, [])
        pnames  = [p["prog"] for p in progs_s]
        with cb:
            p_mun = st.selectbox("📍 Municipio", MUNICIPIOS, key="pm_v3")

        idx_p = 0
        if st.session_state.get("v3p") in pnames:
            idx_p = pnames.index(st.session_state["v3p"])
        if pnames:
            p_prog = st.selectbox(
                f"📚 Programa ({len(pnames)} en {p_sec})",
                pnames, index=idx_p, key="pp_v3",
                help="Lista filtrada por el sector seleccionado",
            )
        else:
            p_prog = None
            st.info("Selecciona un sector para ver los programas.", icon="ℹ️")

        pm_meta = next((p for p in progs_s if p.get("prog") == p_prog), {})
        if pm_meta:
            m  = pm_meta.get("meses", 0)
            ul = pm_meta.get("ultima", "—")
            es = pm_meta.get("estrella", False)
            if es:
                st.markdown(
                    f'<div class="al-w">⭐ <b>Programa estrella</b> en {p_sec} — '
                    f'{pm_meta.get("fichas",0)} fichas · prom. {pm_meta.get("prom",0)} ap/f · '
                    f'última oferta: {ul}</div>', unsafe_allow_html=True)
            elif m > 12:
                st.markdown(
                    f'<div class="al-r">⚠️ Sin ofertar hace <b>{m} meses</b> (última: {ul}). '
                    f'Verificar demanda antes de abrir.</div>', unsafe_allow_html=True)
            elif m > 6:
                st.markdown(
                    f'<div class="al-w">🕐 Sin ofertar hace {m} meses (última: {ul}). '
                    f'Reactivación recomendada.</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="al-s">✅ Programa activo · última oferta: {ul} · '
                    f'{pm_meta.get("fichas",0)} fichas · prom. {pm_meta.get("prom",0)} ap/f</div>',
                    unsafe_allow_html=True)

        niveles  = ["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO","EVENTO"]
        auto_niv = pm_meta.get("nivel","CURSO ESPECIAL")
        auto_dur = pm_meta.get("dur", 48)
        idx_niv  = niveles.index(auto_niv) if auto_niv in niveles else 0
        cc, cd = st.columns(2)
        with cc:
            p_niv = st.selectbox("🎓 Nivel de formación", niveles, index=idx_niv, key="pn_v3")
        with cd:
            p_dur = st.number_input(
                "⏱ Duración (horas)", min_value=4, max_value=5000,
                value=auto_dur, step=4, key="pd_v3")
        p_mes = st.slider("📅 Mes de inicio", 1, 12, 7, key="pmes_v3",
                          help="Feb–Mar y Sep–Oct: mayor demanda histórica")
        mes_lbl = ("🔥 Mes pico (+18%)" if p_mes in MM else
                   "📈 Segundo pico (+12%)" if p_mes in MA else
                   "❄️ Mes bajo (-28%)" if p_mes in ML else "")
        st.caption(f"📅 Mes seleccionado: **{MESES_L[p_mes]}**  {mes_lbl}")

        inst_opts = ["— Sin preferencia (prom. 24.5) —"] + [
            f"{r['n']} — {r['prom']} ap/f ({r['fichas']} fichas)" for r in (RESP or [])
        ]
        p_inst = st.selectbox("👤 Instructor", range(len(inst_opts)),
                               format_func=lambda i: inst_opts[i], key="pi_v3")
        hist_i = RESP[p_inst - 1]["prom"] if p_inst > 0 and RESP else 24.5
        st.markdown('</div>', unsafe_allow_html=True)
        btn = st.button("🚀 Generar predicción y evaluación",
                        type="primary", use_container_width=True, key="btn_v3")

        if p_prog:
            muns_p = PROG_MUN.get(p_prog[:68], [])
            if muns_p:
                with st.expander(f"🗺️ Municipios con oferta histórica ({len(muns_p)})", expanded=False):
                    mx = muns_p[0]["ap"]
                    for mp in muns_p:
                        pct = round(mp["ap"] / max(mx, 1) * 100)
                        ico = "⭐ " if mp["prom"] >= 26 else ""
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                            f'<span style="font-size:11px;font-weight:500;color:#0F172A;width:190px">'
                            f'{ico}{mp["mun"]}</span>'
                            f'<div style="flex:1;height:5px;background:#E2E8F0;border-radius:2px;overflow:hidden">'
                            f'<div style="width:{pct}%;height:100%;background:#1E40AF;border-radius:2px"></div></div>'
                            f'<span style="font-size:10px;color:#64748B;width:95px;text-align:right">'
                            f'{mp["ap"]:,} ap · {mp["prom"]}/f</span></div>',
                            unsafe_allow_html=True)

    with col_r:
        if btn and p_prog:
            f_n  = {"CURSO ESPECIAL":1.0,"TÉCNICO":0.95,"TECNÓLOGO":0.85}.get(p_niv, 1.0)
            f_m  = 1.18 if p_mes in MM else (1.12 if p_mes in MA else (0.72 if p_mes in ML else 1.0))
            f_d  = 0.82 if p_dur > 1000 else (0.90 if p_dur > 400 else 1.0)
            f_u  = MF.get(p_mun, 0.95)
            f_i  = hist_i / 24.5
            pred = max(4, min(round(24.5 * f_n * f_m * f_d * f_u * f_i), 90))
            viab = min(97, round(28 + pred * 2.6))
            alto = pred > 25
            vc   = "#0F6E56" if viab >= 70 else ("#B45309" if viab >= 45 else "#991B1B")
            st.session_state["pred_v3"] = dict(
                pred=pred, viab=viab, alto=alto, vc=vc,
                mun=p_mun, mes=p_mes, niv=p_niv, prog=p_prog, hist=hist_i)

        res = st.session_state.get("pred_v3")
        if res:
            bg  = "#DCFCE7" if res["alto"] else "#FEF3C7"
            tc  = "#166534" if res["alto"] else "#92400E"
            txt = "▲ Alto impacto" if res["alto"] else "▼ Impacto normal"
            mes_msg = (
                "<span style='color:#0F6E56;font-weight:500'>🔥 Feb–Mar: mayor demanda (+18%).</span>"
                if res["mes"] in MM else
                "<span style='color:#0F6E56'>📈 Sep–Oct: segundo pico (+12%).</span>"
                if res["mes"] in MA else
                "<span style='color:#B45309'>❄️ Dic–Ene: demanda baja (-28%).</span>"
                if res["mes"] in ML else
                "Condiciones estándar de demanda histórica."
            )
            st.markdown(
                f'<div class="res-box">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:9px">'
                f'<div><div class="res-n">{res["pred"]}</div><div class="res-l">Aprendices predichos</div></div>'
                f'<span style="background:{bg};color:{tc};font-size:11px;font-weight:700;'
                f'padding:4px 12px;border-radius:20px">{txt}</span></div>'
                f'<div style="font-size:10px;color:#3B82F6;display:flex;'
                f'justify-content:space-between;margin-bottom:2px">'
                f'<span>Viabilidad de apertura</span>'
                f'<span style="font-weight:700">{res["viab"]}%</span></div>'
                f'<div class="vt">'
                f'<div class="vf" style="width:{res["viab"]}%;background:{res["vc"]}"></div></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:9px;color:#93C5FD">'
                f'<span>Baja</span><span>Media</span><span>Alta</span></div>'
                f'<div style="margin-top:9px;font-size:11px;color:#334155;line-height:1.6">'
                f'<b>{res["mun"]}</b> · {MESES_L[res["mes"]]} · {res["niv"]}<br>{mes_msg}</div>'
                f'</div>',
                unsafe_allow_html=True)
            sugs = []
            if res["mes"] not in MM + MA:
                sugs.append("📅 Mover inicio a **feb–mar o sep–oct** puede aumentar asistencia +15%.")
            if res["hist"] < 22:
                sugs.append("👤 Instructor con **historial >26 ap/ficha** mejora la predicción ~20%.")
            if res["niv"] == "CURSO ESPECIAL" and p_dur > 400:
                sugs.append("⏱ Para Curso Especial, duración **<80 h** tiene mejor convocatoria.")
            if pm_meta.get("meses", 0) > 12:
                sugs.append("⚠️ Programa **>12 meses** sin ofertar — difusión previa necesaria.")
            if not sugs:
                sugs.append("✅ Parámetros óptimos para este contexto.")
            with st.expander("💡 Sugerencias de optimización", expanded=True):
                for s in sugs:
                    st.markdown(s)

        if progs_s:
            st.markdown("")
            st.markdown(f"**⭐ Programas de {p_sec} por demanda histórica**")
            mx_p = max((p.get("prom", 0) for p in progs_s), default=1)
            for p in progs_s[:8]:
                pct = round(p.get("prom",0) / max(mx_p,1) * 100)
                m_p = p.get("meses", 0)
                es_p= p.get("estrella", False)
                c_b = "#F59E0B" if es_p else ("#991B1B" if m_p>12 else ("#B45309" if m_p>6 else "#1E40AF"))
                ico = "⭐" if es_p else ("⚠️" if m_p>12 else ("🕐" if m_p>6 else "✅"))
                bdr = "border-left:3px solid #F59E0B;padding-left:8px;" if es_p else ""
                st.markdown(
                    f'<div style="margin-bottom:5px;{bdr}">'
                    f'<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">'
                    f'<span style="color:#334155;overflow:hidden;text-overflow:ellipsis;'
                    f'white-space:nowrap;max-width:75%">{ico} {p["prog"][:54]}</span>'
                    f'<span style="font-weight:700;color:{c_b}">{p["prom"]}</span></div>'
                    f'<div style="height:4px;background:#E2E8F0;border-radius:2px;overflow:hidden">'
                    f'<div style="width:{pct}%;height:100%;background:{c_b};border-radius:2px"></div></div>'
                    f'</div>', unsafe_allow_html=True)

        if RESP:
            st.markdown("")
            st.markdown("**👥 Historial de instructores**")
            mx_r = max(r["prom"] for r in RESP)
            ci1, ci2 = st.columns(2)
            for i, r in enumerate(RESP[:14]):
                col_i = ci1 if i < 7 else ci2
                cr = "#0F6E56" if r["prom"]>=26 else ("#1E40AF" if r["prom"]>=20 else "#B45309")
                with col_i:
                    st.markdown(barra(r["n"][:24], r["prom"], mx_r, cr, 155), unsafe_allow_html=True)
            st.caption("🟢 ≥26 alto · 🔵 20–25 normal · 🟡 <20 bajo")

    st.markdown("---")
    col_mp, col_tp = st.columns([2.2, 0.8])
    with col_mp:
        modo_mp = st.radio("Mapa de calor por:",
                           ["Total aprendices","N.º fichas","Prom/ficha"],
                           horizontal=True, key="mmp_v3")
        sc_mp = {"Total aprendices":"ap","N.º fichas":"fichas","Prom/ficha":"prom"}[modo_mp]
        df_mp = pd.DataFrame([{
            "Municipio":k,"Lat":v["lat"],"Lon":v["lng"],
            "ap":v["ap"],"fichas":v["fichas"],"prom":v["prom"],
            "Sector":v["sector"],"tend":v.get("tend",0),
        } for k,v in MUN_STATS.items()])
        fig_mp = px.scatter_mapbox(
            df_mp, lat="Lat", lon="Lon", size=sc_mp, color=sc_mp,
            hover_name="Municipio",
            hover_data={"ap":":.0f","fichas":":.0f","prom":":.1f","Sector":True,"Lat":False,"Lon":False},
            color_continuous_scale="YlOrRd", size_max=48, zoom=7.6,
            mapbox_style="carto-positron")
        fig_mp.update_layout(height=380, margin=dict(t=0,b=0,l=0,r=0),
            coloraxis_colorbar=dict(thickness=10,len=0.6,
                title=dict(text=modo_mp[:14],side="right")))
        st.plotly_chart(fig_mp, use_container_width=True)
    with col_tp:
        st.markdown("**Top municipios**")
        tops = sorted(MUN_STATS.items(), key=lambda x: -x[1]["ap"])[:10]
        mx_t = tops[0][1]["ap"]
        for mn, v in tops:
            t  = v.get("tend",0)
            tb = f"▲{t}%" if t>0 else (f"▼{abs(t)}%" if t<0 else "→")
            tc = "#0F6E56" if t>0 else ("#991B1B" if t<0 else "#94A3B8")
            pct= round(v["ap"]/mx_t*100)
            st.markdown(
                f'<div style="margin-bottom:7px">'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">'
                f'<span style="color:#0F172A;font-weight:500">{mn[:20]}</span>'
                f'<span style="color:{tc};font-size:10px;font-weight:600">{tb}</span></div>'
                f'<div style="height:5px;background:#E2E8F0;border-radius:2px;overflow:hidden">'
                f'<div style="width:{pct}%;height:100%;background:#1E40AF;border-radius:2px"></div></div>'
                f'<div style="font-size:9px;color:#94A3B8;margin-top:1px">'
                f'{v["ap"]:,} ap · {v["prom"]}/f</div></div>',
                unsafe_allow_html=True)

# =============================================================================
# MÓDULO 2 — DEMANDA POR NIVEL
# =============================================================================
elif modulo == "📊 Demanda por nivel":
    st.markdown('<div class="sh" style="display:flex;align-items:center;gap:10px">📊 Demanda por nivel de formación</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd">Distribución histórica por nivel, sector y municipio</div>', unsafe_allow_html=True)

    nf = st.radio("Filtrar por nivel:",
                  ["Todos","CURSO ESPECIAL","TÉCNICO","TECNÓLOGO"],
                  horizontal=True, key="nf_v3")
    c1d, c2d = st.columns(2)
    with c1d:
        yd = {"Todos":[51557,8713,1682,252],"CURSO ESPECIAL":[51557,0,0,252],
              "TÉCNICO":[0,8713,0,0],"TECNÓLOGO":[0,0,1682,0]}[nf]
        fig_n = go.Figure(go.Bar(
            x=["Complementario","Técnico","Tecnólogo","Evento"], y=yd,
            marker_color=["#1E40AF","#0F6E56","#B45309","#64748B"],
            marker_cornerradius=4, width=0.55))
        fig_n.update_layout(
            title=dict(text="Aprendices por nivel",font=dict(size=13)),
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,height=260,margin=dict(t=38,b=10,l=10,r=10),
            yaxis=dict(gridcolor="#F1F5F9",tickfont=dict(size=10)),
            xaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_n, use_container_width=True)
    with c2d:
        SD=[("Servicios",1232),("Agropecuario",518),("Salud",124),("Industria",93),
            ("Hotelería",84),("Comercio",71),("Transversal",67),("Construcción",52),
            ("Educación",48),("Electricidad",23),("Minería",6),("Textiles",5)]
        fig_s = px.pie(values=[x[1] for x in SD],names=[x[0] for x in SD],
            hole=0.55,title="Fichas por sector",
            color_discrete_sequence=["#1E40AF","#0F6E56","#F59E0B","#7C3AED",
                "#DC2626","#64748B","#0891B2","#BE123C","#059669","#D97706","#374151","#9333EA"])
        fig_s.update_layout(height=260,margin=dict(t=38,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)",title=dict(font=dict(size=13)),
            legend=dict(font=dict(size=10),x=0.8))
        st.plotly_chart(fig_s, use_container_width=True)

    NSD = {
        "Todos":        [("Servicios",58000),("Agropecuario",12300),("Salud",3200),("Transversal",5400),("Industria",4200),("Hotelería",4500)],
        "CURSO ESPECIAL":[("Servicios",19284),("Agropecuario",9401),("Salud",2473),("Transversal",4280),("Industria",3267),("Hotelería",3277)],
        "TÉCNICO":      [("Servicios",3041),("Agropecuario",1986),("Hotelería",1061),("Construcción",418),("Transversal",735),("Comercio",376)],
        "TECNÓLOGO":    [("Servicios",410),("Agropecuario",321),("Transversal",230),("Comercio",269),("Construcción",143)],
    }
    datos = NSD[nf]; mx_d = datos[0][1]
    nf_lbl = LN.get(nf, nf) if nf != "Todos" else "todos los niveles"
    st.markdown(f"**Aprendices por sector — {nf_lbl}**")
    cd1,cd2 = st.columns(2)
    for i,(sec,val) in enumerate(datos):
        with (cd1 if i%2==0 else cd2):
            pct = round(val/mx_d*100)
            st.markdown(
                f'<div style="margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">'
                f'<span style="color:#334155">{sec}</span>'
                f'<span style="font-weight:600;color:#0F172A">{val:,}</span></div>'
                f'<div style="height:6px;background:#E2E8F0;border-radius:3px;overflow:hidden">'
                f'<div style="width:{pct}%;height:100%;background:#1E40AF;border-radius:3px"></div></div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("**Avances por municipio**")
    filas = []
    for mn,v in sorted(MUN_STATS.items(),key=lambda x:-x[1]["ap"]):
        if g_mun!="Todos" and mn!=g_mun: continue
        if g_sec!="Todos" and v["sector"]!=g_sec: continue
        t=v.get("tend",0); tb=f"▲ {t}%" if t>0 else (f"▼ {abs(t)}%" if t<0 else "→")
        es="🟢 Alto" if v["prom"]>=27 else ("🔵 Normal" if v["prom"]>=22 else "🟡 Bajo")
        filas.append([mn,str(v["fichas"]),f'{v["ap"]:,}',str(v["prom"]),v["sector"],tb,es])
    if filas:
        st.markdown(tabla_html(["Municipio","Fichas","Aprendices","Prom/f","Sector","Tendencia","Estado"],filas), unsafe_allow_html=True)
    else:
        st.info("Ajusta los filtros globales.", icon="ℹ️")

# =============================================================================
# MÓDULO 3 — RECOMENDACIONES
# =============================================================================
elif modulo == "⚡ Recomendaciones":
    st.markdown('<div class="sh" style="display:flex;align-items:center;gap:10px">⚡ Recomendaciones por municipio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd">Programas históricos con última oferta · Ruta formativa estratégica · ⭐ de mayor demanda</div>', unsafe_allow_html=True)

    cr1,cr2 = st.columns([1.0,1.6])
    with cr1:
        r_mun = st.selectbox("Municipio", MUNICIPIOS, key="rmun_v3")
        r_tipo= st.radio("Tipo",["Todos","Complementarios","Técnico","Tecnólogo"],
                         horizontal=False, key="rtp_v3")
        st.markdown("""
<div class="rt" style="background:#DCFCE7;border:.5px solid #86EFAC">
  <b style="color:#166534">1. Cursos Complementarios — base estratégica</b><br>
  <span style="color:#166534;font-size:10px">Bajo costo · Alta convocatoria · Forma capital humano primero.</span>
</div>
<div style="text-align:center;font-size:15px;color:#94A3B8;margin:2px 0">↓</div>
<div class="rt" style="background:#DBEAFE;border:.5px solid #93C5FD">
  <b style="color:#1E40AF">2. Programas Técnicos</b><br>
  <span style="color:#1E40AF;font-size:10px">Consolida competencias · Mayor empleabilidad.</span>
</div>
<div style="text-align:center;font-size:15px;color:#94A3B8;margin:2px 0">↓</div>
<div class="rt" style="background:#FEF3C7;border:.5px solid #FCD34D">
  <b style="color:#92400E">3. Tecnólogos</b><br>
  <span style="color:#92400E;font-size:10px">Nivel superior · Requiere base técnica y demanda laboral local.</span>
</div>""", unsafe_allow_html=True)
        if r_mun in MUN_STATS:
            v=MUN_STATS[r_mun]; t=v.get("tend",0)
            tc="#0F6E56" if t>0 else ("#991B1B" if t<0 else "#94A3B8")
            st.markdown(
                f'<div style="background:#F8FAFC;border:.5px solid #E2E8F0;border-radius:8px;'
                f'padding:10px 12px;margin-top:10px;font-size:11px;line-height:1.8">'
                f'<b style="color:#0F172A">{r_mun}</b><br>{v["fichas"]} fichas · {v["ap"]:,} ap<br>'
                f'Sector: <b>{v["sector"]}</b><br>'
                f'<span style="color:{tc};font-weight:600">{"▲ +" if t>0 else "▼ "}{t}% tendencia</span></div>',
                unsafe_allow_html=True)
    with cr2:
        st.markdown(f"**Programas históricos en {r_mun}**")
        pm_list=MUN_PROGS.get(r_mun,[])
        nf_map={"Todos":None,"Complementarios":"CURSO ESPECIAL","Técnico":"TÉCNICO","Tecnólogo":"TECNÓLOGO"}
        nf_fil=nf_map[r_tipo]
        if nf_fil: pm_list=[p for p in pm_list if p["nivel"]==nf_fil]
        if pm_list:
            for i,p in enumerate(pm_list[:10],1):
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:8px;padding:9px 12px;'
                    f'background:#fff;border:.5px solid #E2E8F0;border-radius:9px;margin-bottom:5px;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,.03)">'
                    f'<span style="font-size:12px;font-weight:700;color:#CBD5E1;min-width:20px">#{i}</span>'
                    f'<div style="flex:1">'
                    f'<div style="font-size:12px;font-weight:600;color:#0F172A;margin-bottom:4px;line-height:1.3">{p["prog"]}</div>'
                    f'<div style="display:flex;gap:4px;flex-wrap:wrap">'
                    f'{niv_badge(p["nivel"])}{vig_badge(p["meses"])}{dem_badge(p["prom"])}'
                    f'<span class="bdg bg">{p["fichas"]} fichas · {p["prom"]} ap/f</span>'
                    f'</div></div></div>', unsafe_allow_html=True)
        else:
            st.info(f"Sin registros para {r_mun} con este filtro.", icon="ℹ️")

    st.markdown("---")
    st.markdown("**⭐ Programas estrella — mayor promedio histórico**")
    ests=[("Servicio de recepción hotelera","TÉCNICO","HOTELERIA Y TURISMO",43.4,11),
          ("Venta de productos en línea","CURSO ESPECIAL","COMERCIO",43.0,5),
          ("Análisis y desarrollo software","TECNÓLOGO","TRANSVERSAL",31.0,7),
          ("Manipulación higiénica alimentos","CURSO ESPECIAL","SERVICIOS",27.8,123),
          ("Sistemas","TÉCNICO","TRANSVERSAL",28.4,14),
          ("Construcción de edificaciones","TÉCNICO","CONSTRUCCION",26.7,10)]
    ce=st.columns(3)
    for i,(prog,niv,sec,prom,fichas) in enumerate(ests):
        nc=CN.get(niv,"#64748B")
        with ce[i%3]:
            st.markdown(
                f'<div style="background:#fff;border:.5px solid #E2E8F0;border-radius:10px;'
                f'padding:12px;box-shadow:0 1px 4px rgba(0,0,0,.04);margin-bottom:8px">'
                f'<div style="font-size:12px;font-weight:600;color:#0F172A;margin-bottom:4px">⭐ {prog}</div>'
                f'<div style="font-size:10px;color:#64748B;margin-bottom:6px">{fichas} fichas · {sec}</div>'
                f'<div style="font-size:20px;font-weight:700;color:{nc}">{prom}</div>'
                f'<div style="font-size:9px;color:{nc};opacity:.75">ap/ficha promedio</div></div>',
                unsafe_allow_html=True)

# =============================================================================
# MÓDULO 4 — OPORTUNIDADES DE MEJORA
# =============================================================================
elif modulo == "🔭 Oportunidades de mejora":
    st.markdown('<div class="sh" style="display:flex;align-items:center;gap:10px">🔭 Oportunidades de mejora</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd">Barras apiladas por nivel · Filtros de año y rubro · Carga de datos actualizados</div>', unsafe_allow_html=True)

    RKWS={"Café":["cafe","café","cafeto"],"Cacao":["cacao"],"Aguacate":["aguacate"],
          "Ganadería":["ganader","bovino","vacuno","pecuari"],
          "Avicultura":["avicul","gallina","pollo","ponedora"],
          "Turismo":["turismo","ecoturismo"],"Agricultura":["agric","cultivo","agropecuar"],
          "Piscicultura":["piscic","tilapia","trucha"],"Alturas":["altura"],
          "Inglés":["ingles","english"],"Emprendimiento":["emprend","negocio"],
          "Alimentos":["alimento","higien","manipul","cocina","gastro"],
          "TIC/Excel":["excel","ofimati","sistemas","software","inform"],
          "Medio Ambiente":["ambient","ecolog","conserv"]}

    with st.expander("📂 Cargar datos actualizados", expanded=False):
        cu1,cu2=st.columns([1.2,0.8])
        with cu1:
            arch=st.file_uploader("Nuevo PE04 (.xlsx)",type=["xlsx","xls"],key="up_v3")
            if arch:
                try:
                    dn=pd.read_excel(arch); dn.columns=[c.strip() for c in dn.columns]
                    dn["TOTAL_APRENDICES"]=pd.to_numeric(dn.get("TOTAL_APRENDICES",0),errors="coerce").fillna(0)
                    dn["AÑO"]=pd.to_numeric(dn.get("AÑO",0),errors="coerce")
                    st.session_state["df_ex_v3"]=dn
                    st.success(f"✅ {arch.name}: {len(dn):,} fichas · Años: {sorted(dn['AÑO'].dropna().unique().astype(int).tolist())}")
                except Exception as e:
                    st.error(f"Error: {e}")
        with cu2:
            st.markdown("**Para carga permanente:**")
            st.code("git add data/PE04_NUEVO.xlsx\ngit commit -m nuevo-PE04\ngit push", language="bash")
            st.caption("Streamlit Cloud recarga en ~60 s.")

    dex=st.session_state.get("df_ex_v3")
    if TIENE and dex is not None:
        df_opp=pd.concat([_df,dex],ignore_index=True)
        if "IDENTIFICADOR_FICHA" in df_opp.columns:
            df_opp=df_opp.drop_duplicates(subset=["IDENTIFICADOR_FICHA"])
    elif TIENE:
        df_opp=_df.copy()
    else:
        st.warning("Dataset no disponible. Carga un archivo.",icon="⚠️"); st.stop()

    st.markdown("---")
    cf1,cf2,cf3=st.columns(3)
    with cf1:
        st.markdown("**📅 Años**")
        años_d=sorted(df_opp["AÑO"].dropna().unique().astype(int).tolist())
        años_s=st.multiselect("Años",años_d,default=años_d,key="años_v3",label_visibility="collapsed")
    with cf2:
        st.markdown("**🌿 Rubro**")
        rub_s=st.selectbox("Rubro",["Todos los rubros"]+list(RKWS.keys()),key="rub_v3",label_visibility="collapsed")
    with cf3:
        st.markdown("**🎓 Nivel**")
        niv_s=st.multiselect("Nivel",["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO","EVENTO"],
            default=["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO"],key="niv_v3",label_visibility="collapsed")

    if not años_s:
        st.warning("Selecciona al menos un año.",icon="⚠️"); st.stop()
    df_f=df_opp[df_opp["AÑO"].isin(años_s)].copy()
    if rub_s!="Todos los rubros":
        kws=RKWS.get(rub_s,[])
        if kws: df_f=df_f[df_f["NOMBRE_PROGRAMA_FORMACION"].str.lower().str.contains("|".join(kws),na=False)]
    if niv_s: df_f=df_f[df_f["NIVEL_FORMACION"].isin(niv_s)]
    if df_f.empty:
        st.info("Sin datos para este filtro.",icon="ℹ️"); st.stop()

    at=" · ".join(str(a) for a in sorted(años_s))
    ko1,ko2,ko3,ko4=st.columns(4)
    ko1.metric("Fichas",f"{len(df_f):,}")
    ko2.metric("Aprendices",f"{int(df_f['TOTAL_APRENDICES'].sum()):,}")
    ko3.metric("Municipios",df_f["NOMBRE_MUNICIPIO_CURSO"].nunique())
    ko4.metric("Años",f"{len(años_s)} ({at})")

    st.markdown("---")
    cb1,cb2=st.columns([1.5,0.5])
    with cb1:
        st.markdown(f"**Municipios líderes — aprendices por nivel · {rub_s} · {at}**")
        pvt=df_f.groupby(["NOMBRE_MUNICIPIO_CURSO","NIVEL_FORMACION"])["TOTAL_APRENDICES"].sum().reset_index()
        pw=pvt.pivot_table(index="NOMBRE_MUNICIPIO_CURSO",columns="NIVEL_FORMACION",
                           values="TOTAL_APRENDICES",fill_value=0)
        pw.columns=[str(c) for c in pw.columns]; pw.columns.name=None
        pw=pw.reset_index(); pw["TOTAL"]=pw.drop(columns=["NOMBRE_MUNICIPIO_CURSO"]).sum(axis=1)
        pw=pw.sort_values("TOTAL",ascending=True).tail(18)
        fig_st=go.Figure()
        for nv in ["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO","EVENTO"]:
            if nv in pw.columns and pw[nv].sum()>0:
                fig_st.add_trace(go.Bar(
                    name=LN.get(nv,nv),y=pw["NOMBRE_MUNICIPIO_CURSO"],x=pw[nv],
                    orientation="h",marker_color=CN[nv],marker_cornerradius=3,
                    hovertemplate=f"<b>%{{y}}</b><br>{LN.get(nv,nv)}: %{{x:,}} ap<extra></extra>"))
        fig_st.update_layout(barmode="stack",height=max(280,len(pw)*24+80),
            margin=dict(t=10,b=30,l=10,r=10),
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h",y=1.02,x=0,font=dict(size=11),bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#F1F5F9",title="Aprendices",tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig_st,use_container_width=True)
    with cb2:
        st.markdown("**Por nivel**")
        tn=df_f.groupby("NIVEL_FORMACION")["TOTAL_APRENDICES"].sum(); gr=tn.sum() or 1
        for nv in ["CURSO ESPECIAL","TÉCNICO","TECNÓLOGO","EVENTO"]:
            if nv in tn.index and tn[nv]>0:
                val=int(tn[nv]); pct=round(val/gr*100); clr=CN[nv]
                st.markdown(
                    f'<div style="margin-bottom:9px">'
                    f'<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">'
                    f'<span style="display:flex;align-items:center;gap:5px">'
                    f'<span style="width:9px;height:9px;border-radius:2px;background:{clr};display:inline-block"></span>'
                    f'{LN[nv]}</span>'
                    f'<span style="font-weight:700">{val:,} <span style="color:#94A3B8;font-weight:400">({pct}%)</span></span>'
                    f'</div><div style="height:7px;background:#E2E8F0;border-radius:3px;overflow:hidden">'
                    f'<div style="width:{pct}%;height:100%;background:{clr};border-radius:3px"></div></div></div>',
                    unsafe_allow_html=True)
        st.markdown("**Evolución anual**")
        ev=df_f.groupby(["AÑO","NIVEL_FORMACION"])["TOTAL_APRENDICES"].sum().reset_index()
        if not ev.empty:
            fig_ev=px.bar(ev,x="AÑO",y="TOTAL_APRENDICES",color="NIVEL_FORMACION",
                color_discrete_map=CN,barmode="stack",
                labels={"AÑO":"Año","TOTAL_APRENDICES":"Aprendices","NIVEL_FORMACION":"Nivel"})
            fig_ev.update_layout(height=200,margin=dict(t=10,b=20,l=10,r=10),
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=9),orientation="h",y=1.1),
                xaxis=dict(tickfont=dict(size=10),dtick=1),yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig_ev,use_container_width=True)

    st.markdown("---")
    cd1,cd2=st.columns(2)
    with cd1:
        st.markdown("**Brechas detectadas**")
        tdm=sorted(df_f["NOMBRE_MUNICIPIO_CURSO"].dropna().unique().tolist())
        ct2=df_f[df_f["NIVEL_FORMACION"]=="TECNÓLOGO"]["NOMBRE_MUNICIPIO_CURSO"].dropna().unique().tolist()
        st2=[m for m in tdm if m not in ct2]
        if ct2:
            muns_t = ', '.join(ct2[:5]) + ('…' if len(ct2) > 5 else '')
            st.success(f"Con Tecnólogo ({len(ct2)}): {muns_t}.")
        if st2:
            muns_s = ', '.join(st2[:6]) + ('…' if len(st2) > 6 else '')
            st.warning(f"Sin Tecnólogo ({len(st2)}): {muns_s}. Candidatos a escalar.")
        if rub_s!="Todos los rubros":
            sr=[m for m in MUNICIPIOS if m not in tdm][:5]
            if sr:
                st.error(f"Sin cobertura en {rub_s}: {', '.join(sr)}. Oportunidad para nuevas fichas.")
    with cd2:
        st.markdown("**Resumen por municipio**")
        tab=(df_f.groupby("NOMBRE_MUNICIPIO_CURSO")["TOTAL_APRENDICES"]
             .agg(Total="sum",Fichas="count").reset_index()
             .sort_values("Total",ascending=False).head(22))
        tab.columns=["Municipio","Aprendices","Fichas"]
        tab["Aprendices"]=tab["Aprendices"].astype(int)
        tab["Fichas"]=tab["Fichas"].astype(int)
        tab["Prom"]=(tab["Aprendices"]/tab["Fichas"].replace(0,1)).round(1)
        st.table(tab.reset_index(drop=True))

# =============================================================================
# MÓDULO 5 — MANUAL DE USUARIO
# =============================================================================
elif modulo == "📖 Manual de usuario":
    st.markdown('<div class="sh" style="display:flex;align-items:center;gap:10px">📖 Manual de usuario</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd">Guía completa SN Predict v3.2 — SENA Occidente de Antioquia</div>', unsafe_allow_html=True)

    with st.expander("🎯 Módulo 1 — Predictor inteligente", expanded=True):
        st.markdown("""
**Cómo usar los filtros dinámicos:**
1. **Búsqueda por palabra clave**: escribe "café", "ganadería" o "alturas" para buscar en todos los sectores.
2. **Sector productivo** → la lista de programas se actualiza automáticamente con los de mayor demanda.
3. **Programa** → duración, nivel y alerta de vigencia se auto-rellenan.
4. **⭐ Programas estrella** = mayor demanda histórica (percentil 65+ en su sector).
5. Completa municipio, mes, instructor → **Generar predicción**.

| Ícono | Significado |
|---|---|
| ✅ | Activo — ofertado en los últimos 6 meses |
| 🕐 | Sin ofertar 6–12 meses — reactivación recomendada |
| ⚠️ | Sin ofertar >12 meses — verificar demanda |
| ⭐ | Programa estrella — mayor demanda en el sector |

**Mapa de calor:** cambia el modo entre Total aprendices / N.º fichas / Prom/ficha.

**Historial instructores:** 🟢 ≥26 alto · 🔵 20–25 normal · 🟡 <20 bajo.
""")

    with st.expander("📊 Módulo 2 — Demanda por nivel"):
        st.markdown("""
- Filtra por **Complementario / Técnico / Tecnólogo** para comparar sectores dentro del nivel.
- La tabla de municipios responde al **Filtro Global** de la barra lateral (Municipio y Sector).
- La gráfica circular muestra distribución de fichas por sector en todo el histórico.
""")

    with st.expander("⚡ Módulo 3 — Recomendaciones"):
        st.markdown("""
- Selecciona un municipio → verás sus programas históricos con **fecha de última oferta**.
- **Cursos Complementarios = base estratégica**: baja inversión, alta convocatoria, forman capital humano antes de abrir técnicos.
- Ruta recomendada: **Complementarios → Técnico → Tecnólogo** para desarrollo territorial progresivo.
- Filtra por tipo para ver solo el nivel de interés.
""")

    with st.expander("🔭 Módulo 4 — Oportunidades de mejora"):
        st.markdown("""
- **Barras apiladas**: azul=Complementario · verde=Técnico · ámbar=Tecnólogo. Un vistazo al balance de niveles por municipio.
- Filtra por **año** para ver evolución; por **rubro** (Café, Ganadería, Turismo…) para ver formación por cadena productiva.
- **Carga nuevo Excel**: sube un PE04 actualizado y los gráficos se actualizan en la sesión.
- Para carga permanente: copia el archivo a `data/` → `git push` → Streamlit recarga en ~60 s.
""")

    with st.expander("⚙️ Consideraciones técnicas"):
        st.markdown("""
- **Datos:** PE04 Histórico SENA Occidente (2020-2025) · 2.551 fichas · 22 municipios · 12 sectores.
- **Predicción heurística:** calibrada con datos reales. Error típico ±6–10 aprendices.
- **Modelo ML:** ejecuta `python train.py` para activar Gradient Boosting. La app lo detecta automáticamente.
- **Versión autocontenida:** funciona como archivo único sin módulos externos. Si `data_loader.py` está presente, usa datos en tiempo real.
- Las predicciones son **apoyo a la decisión** — el juicio del coordinador siempre prevalece.
""")

# Footer
st.markdown(
    '<div class="foot">SN Predict v3.2 © 2026 · Centro SENA Occidente de Antioquia · '
    'Herramienta de Inteligencia para la Formación Profesional</div>',
    unsafe_allow_html=True,
)
