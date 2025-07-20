import streamlit as st
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import dropbox
import pytz
import plotly.graph_objects as go
import tempfile

from plotly.subplots import make_subplots
from prophet import Prophet
from fpdf import FPDF
from datetime import date, timedelta, datetime, timezone
from textblob import TextBlob

from libs.modulo_senales_avanzadas import generar_senales_avanzadas
from libs.modulo_backtesting import backtest_senales
from libs.utils import (
    validar_ticker,
    calcular_indicadores,
    preparar_dataframe_para_guardar,
    generar_senal_ultima_fila,
    interpretar_tecnicamente,
    generar_interpretacion_tecnica,
    generar_senales,
    interpretar_senales,
    subir_a_dropbox,
    get_simbolos_disponibles,
    obtener_datos_yfinance_history,
    convertir_a_zona_horaria_local
)


# === Configuración inicial ===
st.set_page_config(page_title="📈 Descargador y Analizador de Acciones")
st.title("📈 Descargador y Analizador de Acciones")

# Inicializar session_state
if "form_guardado" not in st.session_state:
    st.session_state["form_guardado"] = True
    st.session_state["simbolos"] = "AAPL, TSLA"
    st.session_state["fecha"] = datetime(2022, 1, 1)
    st.session_state["intervalo"] = "1d"
    st.session_state["carpeta"] = "./datos/source"
    st.session_state["descargado"] = False
    st.session_state["zona_horaria"] = "America/New_York"
    st.session_state["dropbox_token"] = ""
    st.session_state["subir_dropbox"] = False
    st.session_state["volumen_usuario"] = 1000000
    st.session_state["incluir_sma_largo_plazo"] = True



    
with st.form("formulario_descarga"):
    st.session_state.simbolos = st.text_input("Símbolos (separados por coma):", value=st.session_state.simbolos)
    st.session_state.fecha = st.date_input("Fecha de inicio", value=st.session_state.fecha)
    st.session_state.intervalo = st.selectbox("Intervalo de datos", ["1d", "1h", "15m", "1m"], index=["1d", "1h", "15m", "1m"].index(st.session_state.intervalo))
    st.session_state.carpeta = st.text_input("Carpeta donde guardar archivos", value=st.session_state.carpeta)

    zona_opciones = [
        "UTC", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
        "America/Bogota", "America/Mexico_City", "America/Argentina/Buenos_Aires",
        "Europe/London", "Europe/Madrid", "Asia/Tokyo", "Asia/Shanghai"
    ]
    st.session_state.zona_horaria = st.selectbox("Zona Horaria", zona_opciones,
                                                 index=zona_opciones.index(st.session_state.zona_horaria))

    st.session_state.dropbox_token = st.text_input("🔐 Token Dropbox", st.session_state.dropbox_token, type="password")
    st.session_state.subir_dropbox = st.checkbox("¿Subir ZIP a Dropbox?", st.session_state.subir_dropbox)
    submitted = st.form_submit_button("📥 Procesar y exportar")

if submitted:
    st.session_state.descargado = True
    
    if st.session_state.descargado:
        simbolos_list = [s.strip().upper() for s in st.session_state.simbolos.split(',') if s.strip()]
        fecha = st.session_state.fecha
        intervalo = st.session_state.intervalo
        carpeta = st.session_state.carpeta
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        for simbolo in simbolos_list:
            st.write(f"🔄 Procesando {simbolo} desde= {fecha} int={intervalo}...")
            try:
                tz_ny = pytz.timezone("America/New_York")
                ahora_ny = datetime.now(tz_ny)
                #st.write(ahora_ny)

                # == DESCARGA DE DATOS 

                data = obtener_datos_yfinance_history(simbolo,fecha, intervalo)
                
                if data.empty:
                    st.warning(f"⚠️ No se encontraron datos para {simbolo}")
                    continue
            
                # Asegurarse de que las columnas estén en nivel único
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                # Verifica si 'Close' está presente
                if 'Close' not in data.columns:
                    st.error(f"❌ No se encontró columna 'Close' en los datos de {simbolo}.")
                    continue
                            
                data.columns = [col.split("_")[0] if "_" in col else col for col in data.columns]

                # CALCULO DE INDICADORES 
                        
                data = calcular_indicadores(data)

                # CALCULO DE INDICADORES 
                
                data = generar_senales(data)
                data = generar_senales_avanzadas(data, volumen_minimo=1000000)

                # AJUSTE DE ZONA HORARIA
                #data = convertir_azona_horaria_local(data, zona_horaria="America/New_York")
                data = convertir_a_zona_horaria_local(data, 
                    intervalo=st.session_state.intervalo, 
                    zona_origen="America/New_York", 
                    zona_destino="America/New_York", 
                    quitar_tz=True)
                
                                      
                data = data[pd.to_numeric(data["Close"], errors="coerce").notnull()]

                # PREPARAR DATOS PARA GUARDAR 
                
                data = preparar_dataframe_para_guardar(data)      
    
                # Guardar Excel
                                        
                filename = f"{simbolo}_{fecha}_{intervalo}.xlsx".replace(":", "-")                  
                full_path = os.path.join(carpeta, filename)            
                data.to_excel(full_path, sheet_name="Datos Técnicos")       

                # Guardar CVS
                filename_cvs = f"{simbolo}_{fecha}_{intervalo}.csv".replace(":", "-") 
                full_path_cvs = os.path.join(carpeta, filename_cvs)
                data.to_csv(full_path_cvs, index_label='Date')   
                
                st.success(f"✅ Datos de {simbolo} guardados en: {filename_cvs}")
                        
            except Exception as e:
                st.error(f"❌ Error al procesar {simbolo}: {e}")
    
    # Crear ZIP
                        
    zip_path = os.path.join(carpeta, "datos_acciones_excel.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for archivo in os.listdir(carpeta):
            # if archivo.endswith(".csv"): -- CSV Option
            if archivo.endswith(".xlsx"):
                zipf.write(os.path.join(carpeta, archivo), arcname=archivo)
            
    # Descargar desde UI
    with open(zip_path, "rb") as f:
        st.download_button("⬇️ Descargar ZIP", f, "datos_acciones_excel.zip", mime="application/zip")
        
    # Subir a Dropbox si aplica
    if st.session_state.subir_dropbox and st.session_state.dropbox_token:
        try:
            dbx = dropbox.Dropbox(st.session_state.dropbox_token)
            with open(zip_path, "rb") as f:
                dbx.files_upload(f.read(), "/StockApp/datos_acciones.zip", mode=dropbox.files.WriteMode("overwrite"))
            st.success("✅ ZIP subido a Dropbox")
        except Exception as e:
            st.error(f"❌ Error al subir a Dropbox: {e}")
        
                                   
# Visualización
st.subheader("📊 Visualizar un símbolo")
                                    

simbolos_disponibles = get_simbolos_disponibles(st.session_state.carpeta)
archivos_xlsx = [f for f in os.listdir(st.session_state.carpeta) if f.endswith(".xlsx")]
    
seleccion = st.selectbox("Selecciona símbolo:", simbolos_disponibles)

simbolo_elegido = seleccion.split(" ")[0]  # extrae "AAPL" de "AAPL (desde 2022-01-01)"
#archivo_sel = [f for f in archivos_xlsx if f.startswith(simbolo_elegido)][0]
archivo_sel = next((f for f in archivos_xlsx if f.startswith(simbolo_elegido)), None)

df = pd.DataFrame()
                
if archivo_sel:
    try:
        df = pd.read_excel(os.path.join(st.session_state.carpeta, archivo_sel), engine="openpyxl", index_col=0, parse_dates=True)
        #df_filtrado = df[(df.index.date >= fecha_inicio) & (df.index.date <= fecha_fin)]

        # Si deseas aplicar un filtro de fechas aquí:
        #df = df[df.index >= st.session_state["fecha"]]

        
        if 'df' in globals():
            st.subheader("📅 Filtro de Fechas")
            fecha_min = df.index.min().date()
            fecha_max = df.index.max().date()

            fecha_inicio = st.date_input("Desde", value=df.index.min().date(), key="filtro_inicio")
            fecha_fin = st.date_input("Hasta", value=df.index.max().date(), key="filtro_fin")
            df = df[(df.index.date >= fecha_inicio) & (df.index.date <= fecha_fin)]

        
            st.subheader("📌 Interpretación Técnica")
            st.text(generar_interpretacion_tecnica(df))
    
            # Asegurar que el volumen esté definido
            if "volumen_usuario" not in st.session_state:
                st.session_state.volumen_usuario = 1000000  # valor por defect
    
            # Asegurar que la variable de SMA largo plazo esté definida
            if "incluir_sma_largo_plazo" not in st.session_state:
                st.session_state.incluir_sma_largo_plazo = True

                                               
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            specs=[[{"type": "candlestick"}], [{"type": "scatter"}], [{"type": "scatter"}]]
        )
                            
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
             name="Precio"
        ), row=1, col=1)
            
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode='lines', name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_40"], mode='lines', name="SMA 40"), row=1, col=1)
                            
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode='lines', name="MACD", line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], mode='lines', name="Signal Line", line=dict(color='red')), row=2, col=1)
                            
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], mode='lines', name="RSI", line=dict(color='brown')), row=3, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=3, col=1)


         # === Señales SMA ===
        if 'Signal' in df.columns:
            df_signals = df.dropna(subset=["Signal"])
        
            for i, row in df_signals.iterrows():
                texto = row["Signal"]
                if any(x in texto for x in ["SMA20", "SMA40", "SMA100", "SMA200"]):
                    color = "green" if "Compra" in texto else "red"
                    simbolo = "triangle-up" if "Compra" in texto else "triangle-down"
                    precio = row["Close"] if pd.notnull(row["Close"]) else 0
        
                    fig.add_trace(go.Scatter(
                        x=[i],
                        y=[precio],
                        mode="markers",
                        marker=dict(symbol=simbolo, color=color, size=12),
                        name=texto,
                        showlegend=False,
                        hovertemplate=f"{texto}<br>Precio: {precio:.2f}"
                    ), row=1, col=1)
                            
        fig.update_layout(
            height=900,
            title=f"Gráfico Combinado: {seleccion}",
            xaxis_title="Fecha",
            xaxis_rangeslider_visible=False,
            showlegend=True
        )

        
                            
        st.plotly_chart(fig, use_container_width=True)
                
                
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df.index, df['Close'], label='Close', color='blue')
        ax1.plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--', color='orange')
        ax1.plot(df.index, df['SMA_40'], label='SMA 40', linestyle='--', color='green')
        ax1.set_title(f"{seleccion} - Precio y Medias Móviles")
        ax1.legend()
        ax1.grid()
        st.pyplot(fig1)



        st.subheader("📈 Precio con SMA 20 / 40 / 100 / 200")
        
        # Validar que existan las columnas necesarias
        requeridas = ['Close', 'SMA_20', 'SMA_40', 'SMA_100', 'SMA_200']
        if all(col in df.columns for col in requeridas):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df['Close'], label='Precio de Cierre', color='black', linewidth=1.5)
            ax.plot(df.index, df['SMA_20'], label='SMA 20', color='orange', linestyle='--')
            ax.plot(df.index, df['SMA_40'], label='SMA 40', color='blue', linestyle='--')
            ax.plot(df.index, df['SMA_100'], label='SMA 100', color='green', linestyle='--')
            ax.plot(df.index, df['SMA_200'], label='SMA 200', color='red', linestyle='--')
        
            ax.set_title("Precio vs. Medias Móviles")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Precio")
            ax.legend()
            ax.grid(True)
        
            st.pyplot(fig)
        else:
            st.warning("No se encontraron todas las columnas SMA necesarias en los datos.")
    
                                
        fig2, ax2 = plt.subplots(figsize=(10, 2))
        ax2.plot(df.index, df['RSI_14'], label='RSI', color='brown')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_title(f"{seleccion} - RSI 14")
        ax2.legend()
        ax2.grid()
        st.pyplot(fig2)
                                
        fig3, ax3 = plt.subplots(figsize=(10, 2))
        ax3.plot(df.index, df['MACD'], label='MACD', color='purple')
        ax3.plot(df.index, df['Signal_Line'], label='Signal', color='red', linestyle='--')
        ax3.set_title(f"{seleccion} - MACD")
        ax3.legend()
        ax3.grid()
        st.pyplot(fig3)

        # === Opciones de análisis ===
        st.subheader("⚙️ Opciones de Análisis Técnico")
        
        incluir_sma_largo_plazo = st.checkbox("Incluir SMA100 y SMA200 en el análisis", value=st.session_state.get("incluir_sma_largo_plazo", True))
        st.session_state.incluir_sma_largo_plazo = incluir_sma_largo_plazo

        # Inicializar variables por defecto si no existen
        volumen_usuario = st.session_state.get("volumen_usuario", 1000000)
        incluir_sma_largo = st.session_state.get("incluir_sma_largo_plazo", True)

        try:
            df = generar_senales_avanzadas(df, volumen_minimo=volumen_usuario, incluir_sma_largo_plazo=incluir_sma_largo)
        except Exception as e:
            st.error(f"❌ Error generando señales avanzadas: {e}")
            df = pd.DataFrame()

        
        #df = generar_senales_avanzadas(df, volumen_minimo=volumen_usuario, incluir_sma_largo_plazo=incluir_sma_largo)

                                
        st.subheader("📌 Señales técnicas")
        senales = generar_senal_ultima_fila(df)
        for senal in senales:
            st.write("- ", senal)
                                
        st.subheader("🧾 Panel resumen técnico")
        resumen = interpretar_tecnicamente(df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📉 RSI", resumen["RSI"])
        with col2:
            st.metric("📈 SMA", resumen["SMA"])
        with col3:
            st.metric("📊 MACD", resumen["MACD"])
                                
                                
        # Interpretación técnica
        st.subheader("📌 Interpretación Técnica")
        st.text(generar_interpretacion_tecnica(df))                      
        st.markdown(f"**🧠 Interpretación general:** {resumen['Global']}")
                
    except Exception as e:
        st.error(f"❌ Error al leer el archivo de {seleccion}: {e}")
        df = pd.DataFrame()
else:
    st.warning("No se encontró el archivo seleccionado.")

# === Ejemplo de integración en app Streamlit ===
st.subheader("🧠 Señales técnicas avanzadas")

activar = st.checkbox("Activar señales avanzadas de trading", value=True)

if activar:
    volumen_usuario = st.number_input("Volumen mínimo (opcional)", min_value=0, value=st.session_state.get("volumen_usuario", 1000000), step=100000)
    st.session_state.volumen_usuario = volumen_usuario
    incluir_sma_largo = st.session_state.get("incluir_sma_largo_plazo", True)

    st.info("Se aplicarán señales de SMA, MACD y RSI combinadas.")
    if not df.empty:
        try:
            df = generar_senales_avanzadas(df, volumen_minimo=volumen_usuario, incluir_sma_largo_plazo=incluir_sma_largo)
            st.write("📋 Señales detectadas:")
            st.dataframe(df[['Close', 'SMA_20', 'SMA_40', 'SMA_100', 'SMA_200','MACD', 'Signal_Line', 'RSI_14', 'Volume', 'Signal']].tail(30))
        except Exception as e:
            st.error(f"❌ Error en señales avanzadas: {e}")
    else:
        st.warning("No hay datos cargados aún.")

if activar and not df.empty and "Signal" in df.columns:
    st.markdown("### 🧾 Interpretación técnica automática")
    st.success(interpretar_senales(df))



# ===============================
# 🎯 Módulo de Backtesting
# ===============================
st.subheader("📊 Backtesting de Estrategia")

activar_bt = st.checkbox("Activar simulación de estrategia (backtesting)", value=False)

if activar_bt:
    capital_inicial = st.number_input("Capital inicial ($)", min_value=1000, value=10000, step=500)
    comision = st.number_input("Comisión por operación (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05) / 100

    if 'df' in globals() and "Signal" in df.columns:

        resumen_bt, valor_final, retorno_pct = backtest_senales(df, capital_inicial=capital_inicial, comision_por_trade=comision)

        st.markdown("### 📋 Historial de Operaciones")
        st.dataframe(resumen_bt)

        st.markdown("### 💰 Resultados del Simulador")
        st.success(f"Valor final: ${valor_final:,.2f}")
        st.info(f"Retorno: {retorno_pct:.2f}%")

    else:
        st.warning("No hay señales disponibles para simular. Activa las señales primero.")


# ===============================
# 🎯 Incorporacion de iA para predeiccion de tendencias 
# ===============================


st.subheader("🔮 Predicción de Tendencia con Prophet")


simbolos_disponibles = get_simbolos_disponibles(st.session_state.carpeta)
archivos_excel = [f for f in os.listdir(st.session_state.carpeta) if f.endswith(".xlsx")]

seleccion = st.selectbox("Selecciona un símbolo para predecir", simbolos_disponibles, key="pred_simbolo")
simbolo_elegido_pro = seleccion.split(" ")[0]  # extrae "AAPL" de "AAPL (desde 2022-01-01)"  
archivo_sel_pro = [f for f in archivos_xlsx if f.startswith(simbolo_elegido_pro)][0]

horizonte_dias = st.selectbox("¿Cuántos días deseas predecir?", [7, 14, 30], key="horizonte")

if st.button("Predecir tendencia"):
    st.session_state.prediccion_hecha = True

    if st.session_state.get("prediccion_hecha"):
    # --- BLOQUE DE PREDICCIÓN COMPLETA ---
    
        archivo = [f for f in archivos_excel if f.startswith(archivo_sel_pro)][0]
        df = pd.read_excel(os.path.join(st.session_state.carpeta, archivo), engine="openpyxl", parse_dates=["Date"])
        
        if "Close" not in df.columns:
            st.error("⚠️ No se encontró la columna 'Close'.")
        else:

            # --- BLOQUE DE MEJORA DE PROPHET ---
            from prophet import Prophet
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # 1. Preparamos el DataFrame
            # Asegúrate de haber calculado antes: SMA_20, SMA_40, Volume, RSI_14, MACD, Signal_Line
            df_prophet = df.copy().reset_index().rename(columns={"Date": "ds", "Close": "y"})
            
            # Remover valores nulos para evitar errores
            regresores = ['Volume', 'SMA_20', 'SMA_40', 'RSI_14', 'MACD', 'Signal_Line']
            df_prophet = df_prophet.dropna(subset=['y'] + regresores)
            
            # 2. Crear el modelo Prophet con regresores
            m = Prophet(
                changepoint_prior_scale=0.1,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            
            # Agregar regresores al modelo
            for reg in regresores:
                if reg in df_prophet.columns:
                    m.add_regressor(reg)
            
            # 3. Entrenar el modelo
            m.fit(df_prophet)
            
            # 4. Hacer predicción a futuro
            future = m.make_future_dataframe(periods=30)
            
            # Agregamos los regresores al futuro
            for reg in regresores:
                future[reg] = df_prophet[reg].iloc[-1]  # valor actual (último)
            
            # Predecir
            forecast = m.predict(future)
            
            # 5. Validación cruzada


            dias_totales = (df_prophet['ds'].max() - df_prophet['ds'].min()).days
            
            if dias_totales >= 365 + 90:
                cv_results = cross_validation(m, initial='365 days', period='90 days', horizon='90 days')
            elif dias_totales >= 180 + 30:
                cv_results = cross_validation(m, initial='180 days', period='30 days', horizon='30 days')
            else:
                st.warning("No hay suficientes datos para realizar validación cruzada significativa.")
                cv_results = None


            # 6. Mostrar métricas en Streamlit
            if cv_results is not None:
                metrics = performance_metrics(cv_results)
                st.subheader("⬆️ Métricas de Precisión de Prophet")
                st.dataframe(metrics)
            
            # 7. Gráfico de predicción
            fig_pred = m.plot(forecast)
            st.pyplot(fig_pred)
            
            # 8. Componentes de la predicción
            fig_comp = m.plot_components(forecast)
            st.pyplot(fig_comp)
            
            # --- FIN DEL BLOQUE ---

        
            # Interpretación simple
            y_pred_final = forecast["yhat"].iloc[-1]
            y_actual = df["Close"].iloc[-1]
            direccion = "📈 ALZA probable" if y_pred_final > y_actual else "📉 BAJA probable"
            st.success(f"Resultado: {direccion} en los próximos {horizonte_dias} días.")
        
        # --- Comparación visual con indicadores técnicos ---
        st.subheader("📊 Comparación con Indicadores Técnicos")
            
        # Recortamos la predicción al mismo periodo
        pred_rango = forecast[forecast["ds"] >= df["Date"].iloc[-60]]
            
        # Recalculamos indicadores
        df_ind = df.copy()
        df_ind["SMA_20"] = df_ind["Close"].rolling(20).mean()
        df_ind["SMA_40"] = df_ind["Close"].rolling(40).mean()
            
        ema_12 = df_ind["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df_ind["Close"].ewm(span=26, adjust=False).mean()
        df_ind["MACD"] = ema_12 - ema_26
        df_ind["Signal"] = df_ind["MACD"].ewm(span=9, adjust=False).mean()
            
        # Gráfico combinado
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_ind["Date"], df_ind["Close"], label="Precio Real", color="black")
        ax.plot(df_ind["Date"], df_ind["SMA_20"], label="SMA 20", color="blue", linestyle="--")
        ax.plot(df_ind["Date"], df_ind["SMA_40"], label="SMA 40", color="green", linestyle="--")
        ax.plot(pred_rango["ds"], pred_rango["yhat"], label="Predicción Prophet", color="orange")
            
        ax.set_title(f"{seleccion} - Comparación: Prophet vs Indicadores")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
            
        # MACD
        fig_macd, ax_macd = plt.subplots(figsize=(12, 2.5))
        ax_macd.plot(df_ind["Date"], df_ind["MACD"], label="MACD", color="purple")
        ax_macd.plot(df_ind["Date"], df_ind["Signal"], label="Signal", color="red", linestyle="--")
        ax_macd.set_title("MACD")
        ax_macd.legend()
        ax_macd.grid()
        st.pyplot(fig_macd)
            
        # Interpretación básica
        interpretacion = []
        if df_ind["SMA_20"].iloc[-1] > df_ind["SMA_40"].iloc[-1]:
            interpretacion.append("SMA: Tendencia alcista 📈")
        else:
            interpretacion.append("SMA: Tendencia bajista 📉")
            
        if df_ind["MACD"].iloc[-1] > df_ind["Signal"].iloc[-1]:
            interpretacion.append("MACD: Señal de compra 💚")
        else:
            interpretacion.append("MACD: Señal de venta ❤️")
            
        if y_pred_final > y_actual:
            interpretacion.append("Prophet: predice alza 📈")
        else:
            interpretacion.append("Prophet: predice baja 📉")
            
        st.markdown("### 🧠 Interpretación combinada:")
        for i in interpretacion:
            st.write(f"- {i}")



# ===============================
# 🎯 GENERADOR DE REPORTE PDF 
# ===============================


# Solo mostrar si ya se hizo una predicción
if "prediccion_hecha" in st.session_state and st.session_state.prediccion_hecha:
        st.subheader("📄 Generar reporte en PDF")
    
        if st.button("📤 Exportar reporte PDF"):
            try:
                import tempfile
                from fpdf import FPDF
                import matplotlib.pyplot as plt
    
                # Gráfico combinado (fig) — si no está guardado, debes regenerarlo
                if "fig" not in globals():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Precio"))
                    fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['yhat'], name="Predicción"))
                    fig.update_layout(title="Precio + Predicción")
    
                # Gráfico de MACD con matplotlib
                fig_macd, ax_macd = plt.subplots(figsize=(10, 3))
                ax_macd.plot(df.index, df['MACD'], label='MACD', color='purple')
                ax_macd.plot(df.index, df['Signal_Line'], label='Signal', linestyle='--', color='red')
                ax_macd.set_title("MACD")
                ax_macd.legend()
                ax_macd.grid()
    
                # Guardar los gráficos como imágenes temporales
                img_paths = []
                for grafico in [fig, fig_macd]:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        if isinstance(grafico, go.Figure):
                            grafico.write_image(tmpfile.name)
                        else:
                            grafico.savefig(tmpfile.name, bbox_inches='tight')
                        img_paths.append(tmpfile.name)
    
                # Crear PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Reporte de Análisis Técnico - {seleccion}", ln=True, align='C')
                pdf.ln(10)
    
                for i in interpretacion:
                    pdf.multi_cell(0, 10, f"- {i}")
                pdf.ln(5)
    
                for path in img_paths:
                    pdf.image(path, w=180)
                    pdf.ln(5)
    
                # Descargar
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    pdf.output(tmp_pdf.name)
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button(
                            label="⬇️ Descargar reporte PDF",
                            data=f,
                            file_name=f"reporte_{seleccion}.pdf",
                            mime="application/pdf"
                        )
    
            except Exception as e:
                st.error(f"❌ Error al generar el PDF: {e}")


# ===============================
# 🎯 NOTICIAS 
# ===============================  



# SideBar Ticker Global

if "ticker_global" not in st.session_state:
    st.session_state.ticker_global = "AAPL"  # valor por defecto

st.sidebar.subheader("🔎 Análisis por símbolo")
st.session_state.ticker_global = st.sidebar.text_input(
    "Ticker (ej: AAPL)", value=st.session_state.ticker_global, key="input_ticker_global"
)


from polygon import RESTClient


st.sidebar.subheader("📰 Noticias financieras")

# Inputs del usuario
#ticker_noticias = st.sidebar.text_input("Ticker (ej: AAPL)", value="AAPL")
ticker_noticias = st.session_state.ticker_global
dias = st.sidebar.number_input("Días recientes", min_value=1, max_value=30, value=5)

if st.sidebar.button("🔍 Ver noticias"):

    if ticker_noticias:
        if not validar_ticker(ticker_noticias):
            st.sidebar.warning(f"⚠️ '{ticker_noticias}' no es un símbolo válido.")
            st.stop()

        else:
            st.sidebar.success(f"✅ '{ticker_noticias}' válido ({yf.Ticker(ticker_noticias).info.get('shortName', '')})")   
            try:
                client = RESTClient(api_key="vaxEVjyp8mwMHpQ3XB5E3gxMqtYx3XL5")  # Reemplaza con tu clave real
        
                desde = (pd.Timestamp.now() - pd.Timedelta(days=dias)).strftime("%Y-%m-%d")
        
                noticias = client.list_ticker_news(
                    ticker_noticias,
                    published_utc_gte=desde,
                    limit=10,
                    order="desc"
                )
        
                registros = []
           
                for n in noticias:
                    fecha = pd.to_datetime(n.published_utc).strftime("%Y-%m-%d %H:%M")
                    st.markdown(f"- **{fecha}**: [{n.title}]({n.article_url})", unsafe_allow_html=True)
        
                df_noticias = pd.DataFrame(registros)
                st.markdown(f"### 🗞 Noticias recientes para **{ticker_noticias}**")
                st.dataframe(df_noticias)
        
            except Exception as e:
                st.error(f"❌ Error al obtener noticias: {e}")


# ===============================
# 🎯 REPORTES TRIMESTRALES 
# =============================== 


import requests


# Asegúrate de tener tu API key de Finnhub
finnhub_api_key = "d1dmg71r01qpp0b2ora0d1dmg71r01qpp0b2orag"

st.sidebar.subheader("📢 Reportes Trimestrales (Earnings)")

# Usamos el mismo ticker actualmente seleccionado en la app
ticker_actual = st.session_state.ticker_global

if st.sidebar.button(f"📊 Obtener Earnings de {ticker_actual}"):
    url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker_actual}&token={finnhub_api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data:
            st.warning(f"⚠️ No se encontraron earnings para {ticker_actual}")
        else:
            st.success(f"✅ Últimos reportes de {ticker_actual}:")
            for report in data:
                fecha = report.get('period', 'N/A')
                eps_estimado = report.get('estimate', 'N/A')
                eps_real = report.get('actual', 'N/A')
                sorpresa = report.get('surprisePercent', 0.0)
            
                # Interpretación visual
                if sorpresa > 0:
                    icono = "🟢 📈"
                    interpretacion = "¡Mejor de lo esperado!"
                elif sorpresa < 0:
                    icono = "🔴 📉"
                    interpretacion = "Peor de lo esperado"
                else:
                    icono = "🟡 ➖"
                    interpretacion = "En línea con expectativas"

                st.markdown(f"""
                <div style='font-size: 18px; line-height: 1.4'>
                    <b>📅 Fecha del reporte:</b> {fecha}  <br>
                    <b>📈 EPS Estimado:</b> {eps_estimado}  <br>
                    <b>✅ EPS Real:</b> {eps_real}  <br>
                    <b>🎯 Sorpresa:</b> {sorpresa:.2f}% {icono}  <br>
                    <b>🧠 Interpretación:</b> {interpretacion}
                </div>
                <hr style="margin: 8px 0">
                """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"❌ Error al obtener earnings: {e}")

# ===============================
# 🎯 QUE SE DICE EN REDES
# =============================== 



from libs.reddit_sentiment import (
    obtener_menciones_en_reddit,
    obtener_posts_reddit,
    obtener_data_sentimientos_reddit
)


simbolo_reddit = st.session_state.ticker_global
tabs = st.tabs(["📊 Panel Técnico", "📰 Noticias", "📢 Reddit", "📢 Reddit Emociones", "📊 Reddit"])

with tabs[2]:  # 📢 Reddit 1
    st.subheader("📢 Menciones en Reddit")
    #simbolo_reddit = st.selectbox("Selecciona un símbolo", st.session_state.simbolos_list, key="simbolo_reddit")
    
    with st.spinner("Buscando menciones..."):
        menciones = obtener_menciones_en_reddit(simbolo_reddit, limite=50)

    if menciones:
        for m in menciones:
            st.markdown(f"🔹 **[{m['subreddit']}]** [{m['titulo']}]({m['link']}) — 👍 {m['upvotes']} votos")
    else:
        st.info("No se encontraron menciones recientes en Reddit.")



with tabs[3]:  # 📢 Reddit 2
    st.subheader("📢 Reddit - Análisis de Sentimiento")

    try:
        
        with st.spinner("Determinando emociones..."):
            posts = obtener_posts_reddit(simbolo_reddit)

        if posts:
            for post in posts:
                st.markdown(f"### 📌 r/{post['subreddit']} | 👍 {post['score']} | 💬 {post['comentarios']}")
                st.markdown(f"**{post['titulo']}**")
                st.markdown(f"🧠 Sentimiento: `{post['sentimiento']}`")
                st.markdown(f"[🔗 Ver publicación]({post['url']})")
                st.markdown("---")
        else:
            st.info("No se encontraron publicaciones relevantes aún.")

    except Exception as e:
        st.error(f"❌ Error al obtener publicaciones de Reddit: {e}")



with tabs[4]:  # 📊 Distribución de Sentimiento por Acción
    st.subheader("📊 Distribución de Sentimiento por Acción (Reddit)")

    simbolo_reddit = st.session_state.ticker_global
    sentimientos = obtener_data_sentimientos_reddit(simbolo_reddit)

    df_sentimiento = pd.DataFrame(sentimientos)

    if df_sentimiento.empty:
        st.warning("No se encontraron publicaciones recientes sobre este símbolo.")
    else:
       
        # --------------------- Filtro por fecha ---------------------
        st.subheader("📅 Filtro por Fecha y Análisis de Sentimiento")

        df_sentimiento["Fecha"] = pd.to_datetime(df_sentimiento["Fecha"])
        fecha_min = df_sentimiento["Fecha"].min().date()
        fecha_max = df_sentimiento["Fecha"].max().date()

        fecha_inicio = st.date_input("Desde", fecha_min)
        fecha_fin = st.date_input("Hasta", fecha_max)
        simbolo_filtrado = st.selectbox("Símbolo", sorted(df_sentimiento["Símbolo"].unique()))

        filtro = (
            (df_sentimiento["Fecha"] >= pd.to_datetime(fecha_inicio)) &
            (df_sentimiento["Fecha"] <= pd.to_datetime(fecha_fin)) &
            (df_sentimiento["Símbolo"] == simbolo_filtrado)
        )
        df_filtrado = df_sentimiento[filtro]

#        if not df_filtrado.empty:
#            st.markdown(f"### 🔍 Sentimiento para **{simbolo_filtrado}** del {fecha_inicio} al {fecha_fin}")
#            pivot = df_filtrado.pivot_table(index="Fecha", columns="Sentimiento", values="Cantidad", aggfunc="sum").fillna(0)

#            fig, ax = plt.subplots(figsize=(8, 4))
#            pivot.plot(kind="bar", stacked=True, ax=ax)
#            ax.set_title(f"Tendencia de Sentimiento - {simbolo_filtrado}")
#            ax.set_ylabel("Cantidad de menciones")
#            ax.set_xlabel("Fecha")
#            ax.legend(title="Sentimiento")
#            st.pyplot(fig)
#        else:
#            st.warning("No hay datos disponibles para los filtros seleccionados.")


            # 🔁 Gráfico de Tendencia de Sentimiento
#     st.subheader("📈 Tendencia de Sentimiento en el Tiempo")
    
#     if not df_filtrado.empty:
#         pivot_lineas = df_filtrado.pivot_table(index="Fecha", columns="Sentimiento", values="Cantidad", aggfunc="sum").fillna(0)
    
#         fig_linea, ax_linea = plt.subplots(figsize=(8, 4))
#         pivot_lineas.plot(kind="line", marker='o', ax=ax_linea)
#         ax_linea.set_title(f"Tendencia de Sentimiento Diario - {simbolo_filtrado}")
#         ax_linea.set_ylabel("Cantidad de menciones")
#         ax_linea.set_xlabel("Fecha")
# #         ax_linea.grid(True)
#         ax_linea.legend(title="Sentimiento")
#         st.pyplot(fig_linea)
#     else:
#         st.warning("No hay datos suficientes para mostrar la tendencia.")



     # Agrupar cantidad por Fecha, Símbolo y Sentimiento
        df_sentimiento = df_sentimiento.groupby(["Fecha", "Símbolo", "Sentimiento"], as_index=False)["Cantidad"].sum()

        # Gráfico de barras agrupado por símbolo
        resumen = df_sentimiento.groupby(["Símbolo", "Sentimiento"])["Cantidad"].sum().unstack().fillna(0)
        fig_sent, ax_sent = plt.subplots(figsize=(8, 5))
        resumen.plot(kind="bar", stacked=True, ax=ax_sent)
        ax_sent.set_title("📊 Distribución de Sentimiento por Acción")
        ax_sent.set_xlabel("Símbolo")
        ax_sent.set_ylabel("Cantidad de publicaciones")
        ax_sent.legend(title="Sentimiento")
        st.pyplot(fig_sent)


    # 🔁 Gráfico de Tendencia de Sentimiento Estilizado
    st.subheader(f"📈 Tendencia de Sentimiento para {simbolo_filtrado}")
    
    if not df_filtrado.empty:
        pivot_lineas = df_filtrado.pivot_table(
            index="Fecha",
            columns="Sentimiento",
            values="Cantidad",
            aggfunc="sum"
        ).fillna(0)
    
        fig_linea, ax_linea = plt.subplots(figsize=(10, 4))
    
        if "Positivo" in pivot_lineas.columns:
            ax_linea.plot(
                pivot_lineas.index,
                pivot_lineas["Positivo"],
                label="Positivo",
                color="orange",
                linestyle="-",
                marker="o",
                linewidth=2
            )
    
        if "Negativo" in pivot_lineas.columns:
            ax_linea.plot(
                pivot_lineas.index,
                pivot_lineas["Negativo"],
                label="Negativo",
                color="orangered",
                linestyle="--",
                linewidth=2
            )
    
        if "Neutral" in pivot_lineas.columns:
            ax_linea.plot(
                pivot_lineas.index,
                pivot_lineas["Neutral"],
                label="Neutral",
                color="gray",
                linestyle=":",
                linewidth=2
            )
    
        ax_linea.set_title(f"Tendencia de Sentimiento para {simbolo_filtrado}")
        ax_linea.set_xlabel("Fecha")
        ax_linea.set_ylabel("Cantidad de Publicaciones")
        ax_linea.grid(True, linestyle="--", alpha=0.5)
        ax_linea.legend()
        fig_linea.tight_layout()
    
        st.pyplot(fig_linea)
    else:
        st.warning("No hay datos suficientes para mostrar la tendencia.")





# ===============================
# 🎯 REINICIAR FORM 
# ===============================                

if st.button("🔁 Nueva descarga"):
    st.session_state.clear()
    st.write("✅ Estado reiniciado. Modifica cualquier campo para volver a empezar.")

#    st.experimental_rerun()

