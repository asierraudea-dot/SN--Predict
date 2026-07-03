"""
integracion_en_app.py
=====================
INSTRUCCIONES: cómo integrar oportunidades_mejora.py en tu app.py existente.

Paso 1 — Agrega el import al inicio de app.py (después de los imports actuales):
    from oportunidades_mejora import render_oportunidades

Paso 2 — En el bloque elif del módulo Oportunidades, reemplaza TODO
    el contenido por una sola línea:

    elif modulo == "🔭 Oportunidades de mejora":
        render_oportunidades(df, MUN_STATS)

Paso 3 — En data_loader.py, reemplaza la sección DATA_RAW por el bloque
    de búsqueda automática de múltiples archivos (ya en el data_loader corregido).

Así quedará el bloque de módulos en app.py:
"""

BLOQUE_MODULOS = '''
# ─────────────────────────────────────────────────────────────────────────────
# MÓDULOS — en app.py, este bloque reemplaza los elif de módulos
# ─────────────────────────────────────────────────────────────────────────────

from oportunidades_mejora import render_oportunidades  # ← agregar al inicio

# ... (código existente de KPIs y sidebar sin cambios) ...

if modulo == "🎯 Predictor inteligente":
    # ... código existente del predictor ...
    pass

elif modulo == "📊 Demanda por nivel":
    # ... código existente de demanda ...
    pass

elif modulo == "⚡ Recomendaciones":
    # ... código existente de recomendaciones ...
    pass

elif modulo == "🔭 Oportunidades de mejora":
    # UNA SOLA LÍNEA — todo el módulo está en oportunidades_mejora.py
    render_oportunidades(df, MUN_STATS)

elif modulo == "📖 Manual de usuario":
    # ... código existente del manual ...
    pass
'''

print(BLOQUE_MODULOS)
print("=" * 60)
print("Archivos a subir a GitHub:")
print("  1. oportunidades_mejora.py  → raíz del repo (junto a app.py)")
print("  2. data_loader.py           → versión corregida (ya entregada)")
print("  3. app.py                   → agregar import + reemplazar elif")
