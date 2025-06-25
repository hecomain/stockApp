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

#from datetime import date
#from datetime import timedelta
from prophet import Prophet
from fpdf import FPDF
from datetime import date, timedelta, datetime, timezone
#from datetime import datetime
from textblob import TextBlob

from modulo_senales_avanzadas import generar_senales_avanzadas
from modulo_backtesting import backtest_senales




#from pmdarima import auto_arima
#from statsmodels.tsa.statespace.sarimax import SARIMAX

def validar_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        return bool(info and "shortName" in info)
    except Exception:
        return False

def calcular_indicadores(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def preparar_dataframe_para_guardar(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            st.warning(f"No se pudo convertir el índice a fechas: {e}")
    df.index.name = "Date"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    df = df[~df.index.astype(str).str.contains('Ticker|Date', na=False)]
    if 'Close' in df.columns:
        df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]
    return df

def generar_senal_ultima_fila(df):
    señales = []
    if 'RSI_14' in df.columns:
        rsi = df['RSI_14'].iloc[-1]
        if rsi < 30:
            señales.append("🟢 RSI < 30: Posible sobreventa")
        elif rsi > 70:
            señales.append("🔴 RSI > 70: Posible sobrecompra")
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        sma_20 = df['SMA_20'].iloc[-2:]
        sma_50 = df['SMA_50'].iloc[-2:]
        if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
            señales.append("🟢 Cruce alcista de SMA 20 sobre SMA 50")
        elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
            señales.append("🔴 Cruce bajista de SMA 20 bajo SMA 50")
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        macd = df['MACD'].iloc[-2:]
        signal = df['Signal_Line'].iloc[-2:]
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            señales.append("🟢 MACD cruzó por encima del Signal Line")
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            señales.append("🔴 MACD cruzó por debajo del Signal Line")
    return señales if señales else ["⚪ Sin señales claras"]

def interpretar_tecnicamente(df):
    resumen = {"RSI": "⚪ Neutral", "SMA": "⚪ Neutral", "MACD": "⚪ Neutral", "Global": "⚪ Sin señal clara"}
    rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else None
    if rsi is not None:
        if rsi < 30:
            resumen["RSI"] = "🟢 Sobreventa"
        elif rsi > 70:
            resumen["RSI"] = "🔴 Sobrecompra"
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        sma20 = df['SMA_20'].iloc[-1]
        sma50 = df['SMA_50'].iloc[-1]
        if sma20 > sma50:
            resumen["SMA"] = "🟢 Tendencia alcista"
        elif sma20 < sma50:
            resumen["SMA"] = "🔴 Tendencia bajista"
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        macd = df['MACD'].iloc[-1]
        signal = df['Signal_Line'].iloc[-1]
        if macd > signal:
            resumen["MACD"] = "🟢 Momentum positivo"
        elif macd < signal:
            resumen["MACD"] = "🔴 Momentum negativo"
    puntaje = sum(1 if v.startswith("🟢") else -1 if v.startswith("🔴") else 0 for v in resumen.values())
    if puntaje >= 2:
        resumen["Global"] = "🟢 Sesgo positivo"
    elif puntaje <= -2:
        resumen["Global"] = "🔴 Sesgo negativo"
    else:
        resumen["Global"] = "⚪ Mixto/Indefinido"
    return resumen



def generar_interpretacion_tecnica(df):
    # Tomamos la última fila (último dato disponible)
    ult = df.iloc[-1]
    interpretacion = []
    
    if ult['RSI_14'] > 70:
        interpretacion.append("🔴 RSI indica sobrecompra. Podría haber una corrección bajista.")
    elif ult['RSI_14'] < 30:
        interpretacion.append("🟢 RSI indica sobreventa. Podría haber un rebote alcista.")
    else:
        interpretacion.append("⚪ RSI en zona neutral.")
    if ult['MACD'] > ult['Signal_Line']:
        interpretacion.append("🟢 MACD cruzó por encima de la señal: posible impulso alcista.")
    elif ult['MACD'] < ult['Signal_Line']:
        interpretacion.append("🔴 MACD cruzó por debajo de la señal: posible impulso bajista.")
    else:
        interpretacion.append("⚪ MACD está neutral.")
    if ult['SMA_20'] > ult['SMA_50']:
        interpretacion.append("🟢 SMA 20 está por encima de SMA 50: tendencia alcista de corto plazo.")
    elif ult['SMA_20'] < ult['SMA_50']:
        interpretacion.append("🔴 SMA 20 está por debajo de SMA 50: tendencia bajista de corto plazo.")
    else:
        interpretacion.append("⚪ Las medias móviles están convergiendo.")
    return "\n".join(interpretacion)


def generar_senales(df):
    senales = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        señal = ""
        if prev["MACD"] < prev["Signal_Line"] and curr["MACD"] > curr["Signal_Line"]:
            señal += "Buy (MACD) "
        elif prev["MACD"] > prev["Signal_Line"] and curr["MACD"] < curr["Signal_Line"]:
            señal += "Sell (MACD) "
        if prev["RSI_14"] < 30 and curr["RSI_14"] >= 30:
            señal += "Buy (RSI) "
        elif prev["RSI_14"] > 70 and curr["RSI_14"] <= 70:
            señal += "Sell (RSI) "
        if prev["SMA_20"] < prev["SMA_50"] and curr["SMA_20"] > curr["SMA_50"]:
            señal += "Buy (SMA) "
        elif prev["SMA_20"] > prev["SMA_50"] and curr["SMA_20"] < curr["SMA_50"]:
            señal += "Sell (SMA) "
        senales.append(señal.strip())
    senales.insert(0, "")
    df["Signal"] = senales
    return df


def interpretar_senales(df):
    """
    Devuelve una interpretación general basada en las señales recientes.
    """

    ultimas_senales = df["Signal"].dropna().tail(10).tolist()
    resumen = {"Compra": 0, "Venta": 0}

    for señal in ultimas_senales:
        if "Compra" in señal:
            resumen["Compra"] += 1
        if "Venta" in señal:
            resumen["Venta"] += 1

    # Generar interpretación textual
    if resumen["Compra"] > resumen["Venta"]:
        return f"🟢 Tendencia alcista detectada. {resumen['Compra']} señales de compra vs {resumen['Venta']} de venta."
    elif resumen["Venta"] > resumen["Compra"]:
        return f"🔴 Tendencia bajista detectada. {resumen['Venta']} señales de venta vs {resumen['Compra']} de compra."
    elif resumen["Compra"] == 0 and resumen["Venta"] == 0:
        return "⚪ No se detectaron señales claras de compra o venta recientemente."
    else:
        return f"🟡 Mercado lateral o indeciso: {resumen['Compra']} compras, {resumen['Venta']} ventas."


def subir_a_dropbox(token, archivo_local, ruta_destino_dropbox):
    try:
        dbx = dropbox.Dropbox(token)
        with open(archivo_local, "rb") as f:
            dbx.files_upload(f.read(), ruta_destino_dropbox, mode=dropbox.files.WriteMode("overwrite"))
        return True
    except Exception as e:
        st.error(f"❌ Error al subir a Dropbox: {e}")
        return False


def get_simbolos_disponibles(carpeta):
    try:
        simbolos_disponibles = []
        archivos_xlsx = [f for f in os.listdir(carpeta) if f.endswith(".xlsx")]
            
        for archivo in archivos_xlsx:
            try:
                partes = archivo.replace(".xlsx", "").split("_")
                simbolo = partes[0]
                fecha = partes[1]  # Esto da "2022-01-01"
                simbolos_disponibles.append(f"{simbolo} (desde {fecha})")
            except Exception as e:
                st.warning(f"No se pudo extraer fecha de {archivo}: {e}")

        return simbolos_disponibles
    except Exception as e:
                st.warning(f"No se encontraron datos en {carpeta}: {e}")


# Interfaz de usuario
st.title("📈 Descargador y Analizador de Acciones")

# Inicializar session_state
if "form_guardado" not in st.session_state:
    st.session_state["form_guardado"] = True
    st.session_state["simbolos"] = "AAPL, TSLA"
    st.session_state["fecha"] = datetime(2022, 1, 1)
    st.session_state["intervalo"] = "1d"
    st.session_state["carpeta"] = "./datos"
    st.session_state["descargado"] = False
    st.session_state["zona_horaria"] = "America/New_York"
    st.session_state["dropbox_token"] = ""
    st.session_state["subir_dropbox"] = False

    
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
            st.write(f"🔄 Procesando {simbolo}...")
            try:
                data = yf.download(simbolo, start=fecha, interval=intervalo)
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
                        
                data = calcular_indicadores(data)
                data = preparar_dataframe_para_guardar(data)      
                data = generar_senales(data)
                data = generar_senales_avanzadas(data, volumen_minimo=1000000)
            
            
                # 1. Convertir la zona horaria del índice a tu hora local
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')  
                #data.index = data.index.tz_convert('America/New_York')
                data.index = data.index.tz_convert(st.session_state.zona_horaria)
                # 🔧 Convertir índice con zona horaria a sin zona horaria
                data.index = data.index.tz_localize(None)                            
                data.index.name = "Date"
               
                data = data[pd.to_numeric(data["Close"], errors="coerce").notnull()]
    
    
                # Guardar Excel
                        
                #filename = f"{simbolo}_{fecha}_{intervalo}.csv".replace(":", "-")  -- CSV Option
                filename = f"{simbolo}_{fecha}_{intervalo}.xlsx".replace(":", "-")                  
                full_path = os.path.join(carpeta, filename)
                #data.to_csv(full_path, index_label='Date')  -- CSV Option                  
                data.to_excel(full_path, sheet_name="Datos Técnicos")                   
                st.success(f"✅ Datos de {simbolo} guardados en: {filename}")
                        
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
                            
#archivos_csv = [f for f in os.listdir(carpeta) if f.endswith(".csv")]  -- CSV Option       
#simbolos_disponibles = [f.split("_")[0] for f in archivos_xlxs]
#seleccion = st.selectbox("Selecciona un símbolo:", simbolos_disponibles, key="selector_simbolo")
#archivo_sel = [f for f in archivos_xlxs if f.startswith(seleccion)][0]
#df = pd.read_csv(os.path.join(carpeta, archivo_sel), index_col='Date', parse_dates=True)
                
                            
#archivos_xlxs = [f for f in os.listdir(st.session_state.carpeta) if f.endswith(".xlsx")]
#simbolos_disponibles = [f.split("_")[0] for f in archivos_xlxs]
#seleccion = st.selectbox("Selecciona un símbolo para ver sus gráficos:", simbolos_disponibles)
#archivo_sel = next((f for f in archivos_xlxs if f.startswith(seleccion)), None)


    
#simbolos_disponibles = []
#archivos_xlsx = [f for f in os.listdir(st.session_state.carpeta) if f.endswith(".xlsx")]
    
#for archivo in archivos_xlsx:
#    try:
#        partes = archivo.replace(".xlsx", "").split("_")
#        simbolo = partes[0]
#        fecha = partes[1]  # Esto da "2022-01-01"
#        simbolos_disponibles.append(f"{simbolo} (desde {fecha})")
#    except Exception as e:
#        st.warning(f"No se pudo extraer fecha de {archivo}: {e}")

        

simbolos_disponibles = get_simbolos_disponibles(st.session_state.carpeta)
archivos_xlsx = [f for f in os.listdir(st.session_state.carpeta) if f.endswith(".xlsx")]
    
seleccion = st.selectbox("Selecciona símbolo:", simbolos_disponibles)
simbolo_elegido = seleccion.split(" ")[0]  # extrae "AAPL" de "AAPL (desde 2022-01-01)"  
archivo_sel = [f for f in archivos_xlsx if f.startswith(simbolo_elegido)][0]
    
    
                
if archivo_sel:
    try:
        df = pd.read_excel(os.path.join(st.session_state.carpeta, archivo_sel), engine="openpyxl", index_col=0, parse_dates=True)
        st.subheader("📌 Interpretación Técnica")
        st.text(generar_interpretacion_tecnica(df))
                                               
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
             name="Velas"
        ), row=1, col=1)
            
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode='lines', name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode='lines', name="SMA 50"), row=1, col=1)
                            
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode='lines', name="MACD", line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], mode='lines', name="Signal Line", line=dict(color='red')), row=2, col=1)
                            
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], mode='lines', name="RSI", line=dict(color='brown')), row=3, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=3, col=1)
                            
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
        ax1.plot(df.index, df['SMA_50'], label='SMA 50', linestyle='--', color='green')
        ax1.set_title(f"{seleccion} - Precio y Medias Móviles")
        ax1.legend()
        ax1.grid()
        st.pyplot(fig1)
                                
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

# === Ejemplo de integración en app Streamlit ===
st.subheader("🧠 Señales técnicas avanzadas")

activar = st.checkbox("Activar señales avanzadas de trading", value=True)
volumen_usuario = st.number_input("Volumen mínimo (opcional)", min_value=0, value=1000000, step=100000)

if activar:
    st.info("Se aplicarán señales de SMA, MACD y RSI combinadas.")
    # Supongamos que `df` ya existe y tiene los indicadores calculados
    if 'df' in globals():
        df = generar_senales_avanzadas(df, volumen_minimo=volumen_usuario)
        st.write("📋 Señales detectadas:")
        st.dataframe(df[['Close', 'SMA_20', 'SMA_50', 'MACD', 'Signal_Line', 'RSI_14', 'Volume', 'Signal']].tail(30))
    else:
        st.warning("No hay datos cargados aún.")

# Mostrar interpretación en Streamlit si hay señales
if activar and 'df' in globals() and "Signal" in df.columns:
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
# 🎯 Módulo de almacenamiento de transacciones
# ===============================

# Inicializar almacenamiento de transacciones

# Ruta del archivo CSV

carpeta_op = st.session_state.carpeta           
opciones_csv = "opciones/transacciones_opciones.csv"
full_path_op = os.path.join(carpeta_op, opciones_csv)

# Inicializar o cargar las transacciones existentes
if "transacciones_opciones" not in st.session_state:
    if os.path.exists(opciones_csv):
        st.session_state.transacciones_opciones = pd.read_csv(opciones_csv).to_dict(orient="records")
    else:
        st.session_state.transacciones_opciones = []

st.subheader("📝 Registrar transacción de opción bursátil")

with st.form("form_opciones"):
    col1, col2 = st.columns(2)
    with col1:
        simbolo = st.text_input("Símbolo del subyacente (ej: AAPL)", value="AAPL")
        strike = st.number_input("🎯 Strike Price", min_value=0.0, step=0.5)
        tipo = st.selectbox("📈 Tipo de opción", ["Call", "Put"])
        fecha_expiracion = st.date_input("📆 Fecha de expiración")
    with col2:
        prima = st.number_input("💰 Prima por contrato ($)", min_value=0.0, step=0.1)
        contratos = st.number_input("🔢 Cantidad de contratos", min_value=1, step=1, value=1)
        valor_final = st.number_input("📉 Valor final de la opción ($)", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Registrar transacción")

    if submitted:
        transaccion = {
            "Fecha": date.today(),
            "Símbolo": simbolo.upper(),
            "Tipo": tipo,
            "Strike": strike,
            "Expiración": fecha_expiracion,
            "Prima": prima,
            "Contratos": contratos,
            "Valor final": valor_final,
            "Total pagado": prima * contratos * 100,
            "Total recuperado": valor_final * contratos * 100,
            "Ganancia/Pérdida": (valor_final - prima) * contratos * 100
        }

        # Cargar archivo existente si existe
        if os.path.exists(full_path_op):
            df_existente = pd.read_csv(full_path_op)
        else:
            df_existente = pd.DataFrame()

        # Agregar la nueva fila
        df_nuevo = pd.concat([df_existente, pd.DataFrame([transaccion])], ignore_index=True)

        # Guardar todo en CSV
        df_nuevo.to_csv(full_path_op, index=False)

        # Actualizar session_state también
        st.session_state.transacciones_opciones = df_nuevo.to_dict(orient="records")    
        st.success("✅ Transacción registrada correctamente.")


st.subheader("📊 Backtesting de Opciones con Filtros y Evolución Temporal")

# Validar existencia del archivo
if not os.path.exists(full_path_op):
    st.warning("❗ No se encontró el archivo de transacciones. Registra al menos una opción para comenzar.")
else:
    df = pd.read_csv(full_path_op, parse_dates=["Fecha", "Expiración"], dayfirst=True)

    # --- Filtros ---
    st.sidebar.header("🔎 Filtros de Opciones")
    simbolos = st.sidebar.multiselect("Filtrar por símbolo", options=df["Símbolo"].unique(), default=list(df["Símbolo"].unique()))
    tipos = st.sidebar.multiselect("Tipo de opción", options=df["Tipo"].unique(), default=list(df["Tipo"].unique()))
    fechas = st.sidebar.date_input("Rango de fechas", value=(df["Fecha"].min(), df["Fecha"].max()))
    
    df_filtrado = df[
        df["Símbolo"].isin(simbolos) &
        df["Tipo"].isin(tipos) &
        (df["Fecha"] >= pd.to_datetime(fechas[0])) &
        (df["Fecha"] <= pd.to_datetime(fechas[1]))
    ]
    st.markdown("### 📋 Transacciones filtradas")
    st.dataframe(df_filtrado)

    # --- Métricas principales ---
    total_invertido = (df_filtrado["Prima"] * df_filtrado["Contratos"] * 100).sum()
    total_recuperado = (df_filtrado["Valor final"] * df_filtrado["Contratos"] * 100).sum()
    ganancia_total = df_filtrado["Ganancia/Pérdida"].sum()
    pct_ganancia = (ganancia_total / total_invertido * 100) if total_invertido > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("💸 Total Invertido", f"${total_invertido:,.2f}")
    col2.metric("💰 Recuperado", f"${total_recuperado:,.2f}")
    col3.metric("📈 Ganancia", f"${ganancia_total:,.2f} ({pct_ganancia:.2f}%)")

    # --- Evolución temporal acumulada ---
    st.markdown("### 📉 Evolución acumulada de Ganancia/Pérdida")

    df_linea = df_filtrado.sort_values("Fecha")
    df_linea["Resultado Acumulado"] = df_linea["Ganancia/Pérdida"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_linea["Fecha"], df_linea["Resultado Acumulado"], marker='o')
    ax.set_title("Ganancia/Pérdida acumulada a lo largo del tiempo")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Resultado Acumulado ($)")
    ax.grid()
    st.pyplot(fig)

    # --- Gráfico por símbolo ---
    st.markdown("### 📊 Ganancia total por símbolo")

    resumen = df_filtrado.groupby("Símbolo")["Ganancia/Pérdida"].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(resumen["Símbolo"], resumen["Ganancia/Pérdida"], color="skyblue")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_ylabel("Ganancia / Pérdida ($)")
    ax2.set_title("Ganancia por símbolo (filtrados)")
    st.pyplot(fig2)

    # --- Porcentaje de operaciones ganadoras / perdedoras ---
    st.markdown("### 🧮 Análisis de Resultados de Transacciones")
    
    ganadoras = df_filtrado[df_filtrado["Ganancia/Pérdida"] > 0].shape[0]
    perdedoras = df_filtrado[df_filtrado["Ganancia/Pérdida"] < 0].shape[0]
    neutras = df_filtrado[df_filtrado["Ganancia/Pérdida"] == 0].shape[0]
    total_ops = len(df_filtrado)
    
    pct_ganadoras = (ganadoras / total_ops) * 100 if total_ops > 0 else 0
    pct_perdedoras = (perdedoras / total_ops) * 100 if total_ops > 0 else 0
    pct_neutras = (neutras / total_ops) * 100 if total_ops > 0 else 0
    
    col4, col5, col6 = st.columns(3)
    col4.metric("✅ Ganadoras", f"{ganadoras} ({pct_ganadoras:.1f}%)")
    col5.metric("❌ Perdedoras", f"{perdedoras} ({pct_perdedoras:.1f}%)")
    col6.metric("➖ Neutras", f"{neutras} ({pct_neutras:.1f}%)")
    
    # --- Pie chart ---
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(
        [ganadoras, perdedoras, neutras],
        labels=["Ganadoras", "Perdedoras", "Neutras"],
        autopct="%1.1f%%",
        colors=["green", "red", "gray"],
        startangle=90
    )
    ax_pie.axis("equal")
    ax_pie.set_title("Distribución de Resultados")
    st.pyplot(fig_pie)


    # Descargar resultados
    st.download_button(
        label="⬇️ Descargar resultados en Excel",
        data=df.to_csv(index=False),
        file_name="resultados_backtesting.csv",
        mime="text/csv"
    )


 # --- Top 3 operaciones con más ganancia ---
    st.markdown("### 🏆 Mejores y Peores Operaciones")
    
    top_ganadoras = df_filtrado.sort_values("Ganancia/Pérdida", ascending=False).head(5)
    top_perdedoras = df_filtrado.sort_values("Ganancia/Pérdida", ascending=True).head(5)
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown("#### 🟢 Top 5 Ganadoras")
        st.dataframe(top_ganadoras[["Símbolo", "Fecha", "Tipo", "Strike", "Prima", "Valor final", "Ganancia/Pérdida"]])
    
    with col8:
        st.markdown("#### 🔴 Top 5 Perdedoras")
        st.dataframe(top_perdedoras[["Símbolo", "Fecha", "Tipo", "Strike", "Prima", "Valor final", "Ganancia/Pérdida"]])
 # --- Top 3 operaciones con más ganancia ---



# Mostrar historial y permitir edición
if st.session_state.transacciones_opciones:
    st.markdown("### 📋 Historial de transacciones registradas")
    df_transacciones = pd.DataFrame(st.session_state.transacciones_opciones)

    # Asegurar columnas necesarias
    columnas_necesarias = ["Valor final", "Total recuperado", "Ganancia/Pérdida"]
    for col in columnas_necesarias:
        if col not in df_transacciones.columns:
            df_transacciones[col] = 0.0

    for i, row in df_transacciones.iterrows():
        with st.expander(f"🗂 {row['Símbolo']} - {row['Tipo']} - Strike {row['Strike']}"):
            cols = st.columns(3)
            with cols[0]:
                nuevo_valor_final = st.number_input(
                    f"📉 Nuevo valor final ({row['Símbolo']})", 
                    value=float(row['Valor final']), 
                    key=f"valor_final_{i}"
                )
            with cols[1]:
                actualizar = st.button(f"💾 Actualizar {row['Símbolo']} #{i}", key=f"actualizar_{i}")
            if actualizar:
                st.session_state.transacciones_opciones[i]["Valor final"] = nuevo_valor_final
                st.session_state.transacciones_opciones[i]["Total recuperado"] = nuevo_valor_final * row['Contratos'] * 100
                st.session_state.transacciones_opciones[i]["Ganancia/Pérdida"] = (nuevo_valor_final - row["Prima"]) * row['Contratos'] * 100
                pd.DataFrame(st.session_state.transacciones_opciones).to_csv(full_path_op, index=False)
                st.success("✅ Transacción actualizada.")

    # Mostrar tabla general
    st.markdown("### 📋 Resumen de Resultados")
    st.dataframe(pd.DataFrame(st.session_state.transacciones_opciones))



# ===============================
# 🎯 Incorporacion de iA para predeiccion de tendencias 
# ===============================

# Arima - necesita dependencias asi que pasamos s prophet



##st.subheader("📉 Predicción de Tendencia con ARIMA")

# Cargar archivos históricos existentes
##archivos_historicos = [f for f in os.listdir(st.session_state.carpeta) if f.endswith(".xlsx")]
##simbolos_disponibles = [f.split("_")[0] for f in archivos_historicos]
##seleccion = st.selectbox("Selecciona un símbolo para predecir", simbolos_disponibles)

# Selección de horizonte de predicción
##horizonte = st.selectbox("¿Cuántos días quieres predecir?", [7, 14, 30])

##if st.button("🔮 Predecir con ARIMA"):
##    archivo = [f for f in archivos_historicos if f.startswith(seleccion)][0]
##    df = pd.read_excel(os.path.join(st.session_state.carpeta, archivo), engine="openpyxl", index_col="Date", parse_dates=True)

##    if "Close" not in df.columns:
##        st.error("No se encontró columna 'Close' en el archivo.")
##    else:
##        series = df["Close"].dropna()

##        try:
##            modelo = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
##            modelo_fit = modelo.fit(series)

##            pred = modelo_fit.predict(n_periods=horizonte, return_conf_int=True)
##            pred_values, conf_int = pred

##           fechas_futuras = [series.index[-1] + timedelta(days=i+1) for i in range(horizonte)]

            # Visualización
##            fig, ax = plt.subplots(figsize=(10, 4))
##            ax.plot(series.index, series, label="Histórico")
##            ax.plot(fechas_futuras, pred_values, label="Predicción", color="orange")
##            ax.fill_between(fechas_futuras, conf_int[:, 0], conf_int[:, 1], color="orange", alpha=0.2, label="Confianza 95%")
##            ax.set_title(f"Predicción con ARIMA para {seleccion}")
##            ax.legend()
##            st.pyplot(fig)

            # Interpretación
##            direccion = "📈 ALZA probable" if pred_values[-1] > series.iloc[-1] else "📉 BAJA probable"
##            st.success(f"Resultado: {direccion} en los próximos {horizonte} días.")

##        except Exception as e:
##            st.error(f"Error al generar el modelo ARIMA: {e}")


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
            # Asegúrate de haber calculado antes: SMA_20, SMA_50, Volume, RSI_14, MACD, Signal_Line
            df_prophet = df.copy().reset_index().rename(columns={"Date": "ds", "Close": "y"})
            
            # Remover valores nulos para evitar errores
            regresores = ['Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'Signal_Line']
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
        df_ind["SMA_50"] = df_ind["Close"].rolling(50).mean()
            
        ema_12 = df_ind["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df_ind["Close"].ewm(span=26, adjust=False).mean()
        df_ind["MACD"] = ema_12 - ema_26
        df_ind["Signal"] = df_ind["MACD"].ewm(span=9, adjust=False).mean()
            
        # Gráfico combinado
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_ind["Date"], df_ind["Close"], label="Precio Real", color="black")
        ax.plot(df_ind["Date"], df_ind["SMA_20"], label="SMA 20", color="blue", linestyle="--")
        ax.plot(df_ind["Date"], df_ind["SMA_50"], label="SMA 50", color="green", linestyle="--")
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
        if df_ind["SMA_20"].iloc[-1] > df_ind["SMA_50"].iloc[-1]:
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
import pandas as pd
import streamlit as st

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



from reddit_sentiment import obtener_menciones_en_reddit
from reddit_sentiment import obtener_posts_reddit


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

    # Este diccionario se debe construir dinámicamente en tu lógica de scraping Reddit
    #sentimientos_por_simbolo = {
    #    "AAPL": {"Positivo": 12, "Neutral": 5, "Negativo": 3},
    #    "TSLA": {"Positivo": 18, "Neutral": 7, "Negativo": 4},
    #    "META": {"Positivo": 7, "Neutral": 3, "Negativo": 6}
    #}




    # Crear lista para almacenar datos
    sentimientos = []
    
    # Procesar publicaciones de Reddit
    for post in reddit.subreddit("wallstreetbets").search(simbolo, limit=100):
        fecha = datetime.fromtimestamp(post.created_utc).date()
        texto = post.title + " " + post.selftext
        analisis = TextBlob(texto)
        sentimiento = "Positivo" if analisis.sentiment.polarity > 0.1 else "Negativo" if analisis.sentiment.polarity < -0.1 else "Neutral"
    
        sentimientos.append({
            "Fecha": fecha,
            "Símbolo": simbolo.upper(),
            "Sentimiento": sentimiento,
            "Cantidad": 1
        })
    
    # Crear DataFrame
    df_sentimiento = pd.DataFrame(sentimientos)
    
    # Agrupar por fecha, símbolo y sentimiento
    df_sentimiento = df_sentimiento.groupby(["Fecha", "Símbolo", "Sentimiento"], as_index=False).sum()

    
    
    # Convertir a DataFrame
    df_sentimientos = pd.DataFrame(sentimientos).T
    
    # Crear gráfico de barras apiladas
    fig_sent, ax_sent = plt.subplots(figsize=(8, 5))
    df_sentimientos.plot(kind="bar", stacked=True, ax=ax_sent)
    ax.set_title("Sentimiento en Reddit por Acción")
    ax.set_xlabel("Símbolo")
    ax.set_ylabel("Cantidad de publicaciones")
    ax.legend(title="Sentimiento")
    ax.grid(True)
    
    st.pyplot(fig_sent)

    # Asegúrate de que `df_sentimiento` ya esté creado e incluya las columnas: Fecha, Símbolo, Sentimiento, Cantidad

    st.subheader("📅 Filtro por Fecha y Análisis de Sentimiento")
    
    # Conversión de fechas si es necesario
    df_sentimiento["Fecha"] = pd.to_datetime(df_sentimiento["Fecha"])
    
    # Rango de fechas
    fecha_min = df_sentimiento["Fecha"].min().date()
    fecha_max = df_sentimiento["Fecha"].max().date()
    
    fecha_inicio = st.date_input("Desde", fecha_min)
    fecha_fin = st.date_input("Hasta", fecha_max)
    simbolo_filtrado = st.selectbox("Símbolo", sorted(df_sentimiento["Símbolo"].unique()))
    
    # Aplicar filtros
    filtro = (
        (df_sentimiento["Fecha"] >= pd.to_datetime(fecha_inicio)) &
        (df_sentimiento["Fecha"] <= pd.to_datetime(fecha_fin)) &
        (df_sentimiento["Símbolo"] == simbolo_filtrado)
    )
    df_filtrado = df_sentimiento[filtro]
    
    # Gráfico
    if not df_filtrado.empty:
        st.markdown(f"### 🔍 Sentimiento para **{simbolo_filtrado}** del {fecha_inicio} al {fecha_fin}")
        pivot = df_filtrado.pivot_table(index="Fecha", columns="Sentimiento", values="Cantidad", aggfunc="sum").fillna(0)
    
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"Tendencia de Sentimiento - {simbolo_filtrado}")
        ax.set_ylabel("Cantidad de menciones")
        ax.set_xlabel("Fecha")
        ax.legend(title="Sentimiento")
        st.pyplot(fig)
    else:
        st.warning("No hay datos disponibles para los filtros seleccionados.")


# ===============================
# 🎯 REINICIAR FORM 
# ===============================                

if st.button("🔁 Nueva descarga"):
    st.session_state.clear()
    st.write("✅ Estado reiniciado. Modifica cualquier campo para volver a empezar.")

#    st.experimental_rerun()

