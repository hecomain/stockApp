import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

# Carpeta con los archivos descargados anteriormente
carpeta = "/Users/hcm/stockDB/datos/source"

st.subheader("🔄 Actualizar datos de un símbolo existente")

# Listar los archivos XLSX existentes
archivos_disponibles = [f for f in os.listdir(carpeta) if f.endswith(".xlsx")]

if archivos_disponibles:
    archivo_sel = st.selectbox("Selecciona el archivo a actualizar:", archivos_disponibles)
    if st.button("📈 Actualizar datos"):
        try:
            # Extraer datos del nombre del archivo
            partes = archivo_sel.replace(".xlsx", "").split("_")
            simbolo = partes[0]
            fecha_inicio = partes[1]
            intervalo = partes[2]

            st.info(f"📊 Actualizando {simbolo} desde {fecha_inicio} con intervalo {intervalo}")

            # Leer el archivo existente
            path_archivo = os.path.join(carpeta, archivo_sel)
            df_existente = pd.read_excel(path_archivo)
            df_existente['Date'] = pd.to_datetime(df_existente['Date'])

            # Calcular desde qué fecha continuar (última fecha + 1 día)
            ultima_fecha = df_existente['Date'].max()
            desde = (ultima_fecha + timedelta(days=1)).date()
            hasta = datetime.today().date()

            if desde >= hasta:
                st.warning("✅ Ya tienes los datos hasta hoy. No hay nada que actualizar.")
            else:
                # Descargar nuevos datos desde Yahoo Finance
                nuevos_datos = yf.download(simbolo, start=str(desde), end=str(hasta), interval=intervalo)

                if nuevos_datos.empty:
                    st.warning("⚠️ No se encontraron datos nuevos.")
                else:
                    nuevos_datos.reset_index(inplace=True)
                    nuevos_datos['Date'] = pd.to_datetime(nuevos_datos['Date']).dt.tz_localize(None)

                    # Unir y guardar
                    df_actualizado = pd.concat([df_existente, nuevos_datos], ignore_index=True)
                    df_actualizado.drop_duplicates(subset='Date', inplace=True)

                    df_actualizado.to_excel(path_archivo, index=False)
                    st.success(f"✅ Datos de {simbolo} actualizados exitosamente.")
        except Exception as e:
            st.error(f"❌ Error al actualizar: {e}")
else:
    st.warning("No se encontraron archivos XLSX en la carpeta.")
