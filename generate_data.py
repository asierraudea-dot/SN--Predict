import pandas as pd

def generar_csv_desde_excel():
    """
    Convierte el archivo Excel original en data_historico.csv listo para la app
    """
    print("🔄 Procesando archivo Excel...")
    
    # Leer el Excel
    df = pd.read_excel('/home/workdir/attachments/PE04_HISTÓRICO_PREVIOS.xlsx', sheet_name='Hoja1')
    
    # Limpieza básica
    df = df.copy()
    
    # Convertir fechas
    if 'FECHA_INICIO_FICHA' in df.columns:
        df['FECHA_INICIO_FICHA'] = pd.to_datetime(df['FECHA_INICIO_FICHA'], errors='coerce')
    
    if 'FECHA_TERMINACION_FICHA' in df.columns:
        df['FECHA_TERMINACION_FICHA'] = pd.to_datetime(df['FECHA_TERMINACION_FICHA'], errors='coerce')
    
    # Guardar como CSV
    df.to_csv('data_historico.csv', index=False, encoding='utf-8')
    
    print(f"✅ Archivo generado exitosamente: data_historico.csv")
    print(f"   Total de registros: {len(df)}")
    print(f"   Columnas: {df.columns.tolist()}")
    
    # Mostrar vista previa
    print("\nVista previa de las primeras filas:")
    print(df.head(3)[['NOMBRE_MUNICIPIO_CURSO', 'NOMBRE_PROGRAMA_FORMACION', 'TOTAL_APRENDICES', 'AÑO']].to_string())

if __name__ == "__main__":
    generar_csv_desde_excel()
