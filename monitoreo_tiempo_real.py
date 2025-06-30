# monitoreo_tiempo_real.py

import os
import streamlit as st
import pandas as pd
import yfinance as yf
import pytz
import time
import plotly.graph_objects as go


from datetime import datetime, timedelta
from modulo_senales_avanzadas import generar_senales_avanzadas, generar_senales_frecuencia_alta, interpretar_senales, calcular_indicadores
from utils import esta_en_horario_mercado, graficar_con_tecnica, obtener_datos_yfinance_today, obtener_datos_yfinance_live, filtrar_datos_horario_mercado
from alertas import reproducir_alerta_sonora, enviar_alerta_email

# === Configuración inicial ===
st.set_page_config(page_title="📡 Monitoreo en Tiempo Real", layout="wide")
st.title("📡 Monitoreo de Acciones en Tiempo Real")

# === Parámetros de usuario ===
carpeta_datos = "./datos/source"
carpeta_logs = "./datos/livemonitor"
zona_ny = pytz.timezone("America/New_York")

# === Selección de símbolo ===
archivos = [f for f in os.listdir(carpeta_datos) if f.endswith(".xlsx")]
simbolos = [f.replace(".xlsx", "") for f in archivos]
simbolo_file = st.sidebar.selectbox("Selecciona un símbolo para monitoreo:", simbolos)


# === Configuración ===
intervalo_usr = st.sidebar.selectbox("Intervalo de monitoreo", ["15m", "1h", "1d"])
semanas_entrenamiento = st.sidebar.slider("¿Cuántas semanas de datos usar para tendencia?", 2, 12, 4)
comparar_modelos = st.sidebar.checkbox("Comparar señales estándar vs alta frecuencia", value=True)
guardar_log = st.sidebar.checkbox("Guardar archivo CSV con señales detectadas", value=True)
actualizar_cada = st.sidebar.number_input("Intervalo de actualización (segundos)", min_value=30, value=300, step=30)


# === Paso 2: Botón de pausa y reanudación ===
if "monitoreo_activo" not in st.session_state:
    st.session_state.monitoreo_activo = False

col1, col2 = st.sidebar.columns(2)

if col1.button("⏸️ Pausar"):
    st.session_state.monitoreo_activo = False
    st.info("⛔ Monitoreo pausado")

if col2.button("▶️ Reanudar"):
    st.session_state.monitoreo_activo = True
    st.success("🟢 Monitoreo activo")


ticker = simbolo_file.split("_")[0].upper()

modo_desarrollo = st.sidebar.checkbox("🛠 Modo desarrollo (ignorar horario mercado)", value=False)


# === Botón de inicio ===
iniciar = st.sidebar.button("🚀 Iniciar Monitoreo")

if iniciar:
    st.success(f"⏱️ Monitoreando {ticker} cada {actualizar_cada}s con velas de {intervalo_usr}")
    log = []


    #st.write("Símbolo:", ticker)
    #st.write("Intervalo:", intervalo_usr)
    #st.write("Periodo:", semanas_entrenamiento)

    #df_live = yf.download("AAPL", period="1d", interval="15m")
    #st.write(df_live)


    df_hist = pd.read_excel(f"{carpeta_datos}/{simbolo_file}.xlsx", index_col=0, parse_dates=True)
    df_hist = df_hist.last(f"{semanas_entrenamiento * 7}D")

    sesgo_base = interpretar_senales(generar_senales_avanzadas(df_hist))
    st.info(f"📉 Sesgo técnico histórico ({semanas_entrenamiento} semanas): {sesgo_base}")

    while True:

        #Verifica si el mercado esta en operacion  
        
        now_ny = datetime.now()

        if not modo_desarrollo and not esta_en_horario_mercado():
            st.warning("⚠️ Fuera del horario de mercado (lunes a viernes, 9:30 a.m. - 4:00 p.m. hora NY, excluyendo feriados). El monitoreo en tiempo real está desactivado.")
            st.stop()
        elif modo_desarrollo:
            st.info("🛠 Modo desarrollo activo: Ignorando validación de horario de mercado.")

        #df_live = obtener_datos_yfinance_live(ticker, intervalo=intervalo_usr)
        df_live = obtener_datos_yfinance_today(ticker, intervalo=intervalo_usr)

        st.subheader("Datos")
        st.write(df_live)

        if isinstance(df_live.columns, pd.MultiIndex):
            df_live.columns = df_live.columns.get_level_values(0)
    
        if df_live.empty:
            st.error("❌ No se pudo obtener datos en vivo.")
            time.sleep(actualizar_cada)
            continue

        df_live = df_live.rename(columns=str.title)
        df_live = df_live.rename(columns={"Adj Close": "Adj_Close"})
        df_live = df_live[~df_live.index.duplicated(keep='last')]

        # === Cálculo de indicadores técnicos necesarios ===
        df_live["SMA_5"] = df_live["Close"].rolling(window=5).mean()
        df_live["SMA_10"] = df_live["Close"].rolling(window=10).mean()
        df_live["SMA_20"] = df_live["Close"].rolling(window=20).mean()
        df_live["SMA_40"] = df_live["Close"].rolling(window=40).mean()
        #df_live["SMA_100"] = df_live["Close"].rolling(window=100).mean()
        #df_live["SMA_200"] = df_live["Close"].rolling(window=200).mean()
        
        # MACD y Línea de Señal
        exp1 = df_live["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df_live["Close"].ewm(span=26, adjust=False).mean()
        df_live["MACD"] = exp1 - exp2
        df_live["Signal_Line"] = df_live["MACD"].ewm(span=9, adjust=False).mean()
        
        # RSI (14)
        delta = df_live["Close"].diff()
        ganancia = delta.where(delta > 0, 0)
        perdida = -delta.where(delta < 0, 0)
        media_ganancia = ganancia.rolling(window=14).mean()
        media_perdida = perdida.rolling(window=14).mean()
        rs = media_ganancia / media_perdida
        df_live["RSI_14"] = 100 - (100 / (1 + rs))


        # Señales
        #df_live = calcular_indicadores(df_live)
        señales_normales = generar_senales_avanzadas(df_live.copy(), volumen_minimo=0, incluir_sma_largo_plazo=False)
        señales_altafreq = generar_senales_frecuencia_alta(df_live.copy()) if comparar_modelos else None

        ultima = señales_normales.iloc[-1]
        #ultima = señales_normales.tail(1).squeeze()
        #st.write("Última fila:", señales_normales.iloc[-1])



        if not señales_normales.empty and "Close" in señales_normales.columns:

            #st.write("Columnas disponibles:", señales_normales.columns.tolist())

            # Obtenemos la última fila con datos válidos
            ultima = señales_normales.dropna(subset=["Close", "Volume","Open","High","Low"]).tail(1).squeeze()

            # Obtenemos la última fila con datos válidos
            #ultima = df_live.dropna(subset=["Close", "Volume"]).tail(1).squeeze()
            
            # Extraemos la señal generada en esta última observación
            nueva_senal = ultima.get("Signal", "")

            # Activamos alertas solo si hay una señal de compra o venta
            if isinstance(nueva_senal, str) and (
                "Compra" in nueva_senal or "Venta" in nueva_senal
            ):
                reproducir_alerta_sonora()
            
                #enviar_alerta_email(
                #    f"🚨 Señal detectada en {ticker} a las {ultima.name.strftime('%Y-%m-%d %H:%M')}:\n\n{nueva_senal}"
                #)

            
            if isinstance(ultima, pd.Series):
                st.subheader("📈 Última vela")
                # Interpretaciones
                col1UV, col2UV = st.columns(2)
                with col1UV:
                    st.metric("Precio", f"{ultima['Close']:.2f}")
                    st.metric("Volumen", f"{int(ultima['Volume'])}")    
                with col2UV:
                    st.metric("Open:", f"{ultima['Open']:.2f}")
                    st.metric("High:", f"{ultima['High']:.2f}")
                    st.metric("Low:", f"{ultima['Low']:.2f}")
                        
                
            else:
                st.warning("No se pudo extraer la última vela correctamente.")
        else:
            st.warning("No hay datos disponibles para mostrar la última vela.")



        # Interpretaciones
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Señales estándar")
            st.success(interpretar_senales(señales_normales))
        if señales_altafreq is not None:
            with col2:
                st.markdown("### Alta frecuencia")
                #st.success(interpretar_senales(señales_altafreq))
                st.success(interpretar_senales(señales_altafreq, columna="Signal_HF"))

        df_live.index = pd.to_datetime(df_live.index)

        # Convertir índice UTC a hora de Nueva York
        #df = df.tz_convert("America/New_York")

        # Gráfico
        fig = go.Figure(data=[
            go.Candlestick(
                x=señales_normales.index,
                open=señales_normales['Open'],
                high=señales_normales['High'],
                low=señales_normales['Low'],
                close=señales_normales['Close'],
                name="Velas"
            )
        ])
        #fig.update_layout(title=f"Gráfico de {ticker} ({intervalo_usr})", xaxis_rangeslider_visible=False)


        fig.update_layout(
            title=f"📉 Velas de {ticker} en tiempo real",
            xaxis_title="Fecha y Hora",
            yaxis_title="Precio",
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(
            type="date",
            rangebreaks=[
                # Oculta fines de semana
                dict(bounds=["sat", "mon"]),
                # Oculta horas fuera de mercado
                dict(bounds=[16, 9.5], pattern="hour")
            ]
        )

        st.plotly_chart(fig, use_container_width=True,  key=f"grafico_monitoreo_{ticker}")
       

        if not df_live.empty:
            with st.expander("📉 Gráfico técnico interactivo"):

                # Antes de graficar
                df_live = filtrar_datos_horario_mercado(df_live, exchange="NYSE", zona_horaria="America/New_York")                           
                fig = graficar_con_tecnica(df_live, titulo=f"{ticker} - Análisis Técnico")
                st.plotly_chart(fig, use_container_width=True,  key=f"grafico_ineractivo_{ticker}")


        #st.write(df_live["Signal"].dropna().unique())
        st.write(df_live.info())
        st.dataframe(df_live.tail(10))

        # Guardar log
        if guardar_log:
            log.append({
                "timestamp": now_ny.strftime("%Y-%m-%d %H:%M:%S"),
                "precio": ultima['Close'],
                "volumen": ultima['Volume'],
                "señales": ultima['Signal'] if 'Signal' in ultima else ""
            })

            filename = f"log_{ticker}_{now_ny.strftime("%Y-%m-%d")}.csv"
            full_path = os.path.join(carpeta_logs, filename)
            
            pd.DataFrame(log).to_csv(full_path, index=False)



        #time.sleep(actualizar_cada)
