import os
import pandas as pd
import yfinance as yf
import dropbox
import streamlit as st
from datetime import datetime, timedelta
import pytz
import pandas_market_calendars as mcal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import yfinance as yf
import pandas_market_calendars as mcal




def validar_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        return bool(info and "shortName" in info)
    except Exception:
        return False


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

    # RSI
    if 'RSI_14' in df.columns:
        rsi = df['RSI_14'].iloc[-1]
        if rsi < 30:
            señales.append("🟢 RSI < 30: Posible sobreventa")
        elif rsi > 70:
            señales.append("🔴 RSI > 70: Posible sobrecompra")

    # SMA Cruces: 20 / 50
    if 'SMA_20' in df.columns and 'SMA_40' in df.columns:
        sma_20 = df['SMA_20'].iloc[-2:]
        sma_40 = df['SMA_40'].iloc[-2:]
        if sma_20.iloc[-1] > sma_40.iloc[-1] and sma_20.iloc[-2] <= sma_40.iloc[-2]:
            señales.append("🟢 Cruce alcista: SMA 20 sobre SMA 40")
        elif sma_20.iloc[-1] < sma_40.iloc[-1] and sma_20.iloc[-2] >= sma_40.iloc[-2]:
            señales.append("🔴 Cruce bajista: SMA 20 bajo SMA 40")

    # SMA Cruces: 100 / 200
    if 'SMA_100' in df.columns and 'SMA_200' in df.columns:
        sma_100 = df['SMA_100'].iloc[-2:]
        sma_200 = df['SMA_200'].iloc[-2:]
        if sma_100.iloc[-1] > sma_200.iloc[-1] and sma_100.iloc[-2] <= sma_200.iloc[-2]:
            señales.append("🟢 Cruce dorado: SMA 100 sobre SMA 200 (señal de largo plazo)")
        elif sma_100.iloc[-1] < sma_200.iloc[-1] and sma_100.iloc[-2] >= sma_200.iloc[-2]:
            señales.append("🔴 Cruce de la muerte: SMA 100 bajo SMA 200 (alerta bajista)")

    # MACD
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

    if 'SMA_20' in df.columns and 'SMA_40' in df.columns:
        sma20 = df['SMA_20'].iloc[-1]
        sma50 = df['SMA_40'].iloc[-1]
        if sma20 > sma50:
            resumen["SMA"] = "🟢 Tendencia alcista"
        elif sma20 < sma50:
            resumen["SMA"] = "🔴 Tendencia bajista"

    if st.session_state.get("incluir_sma_largo_plazo", True):
        if 'SMA_100' in df.columns and 'SMA_200' in df.columns:
            sma100 = df['SMA_100'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1]
            if sma100 > sma200:
                resumen["SMA"] += " | 🟢 Largo plazo alcista"
            elif sma100 < sma200:
                resumen["SMA"] += " | 🔴 Largo plazo bajista"

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

    if ult['SMA_20'] > ult['SMA_40']:
        interpretacion.append("🟢 SMA 20 está por encima de SMA 40: tendencia alcista de corto plazo.")
    elif ult['SMA_20'] < ult['SMA_40']:
        interpretacion.append("🔴 SMA 20 está por debajo de SMA 40: tendencia bajista de corto plazo.")
    else:
        interpretacion.append("⚪ Las medias móviles están convergiendo (20/50).")

    if 'SMA_100' in df.columns and 'SMA_200' in df.columns:
        if ult['SMA_100'] > ult['SMA_200']:
            interpretacion.append("🟢 SMA 100 por encima de SMA 200: tendencia alcista de largo plazo.")
        elif ult['SMA_100'] < ult['SMA_200']:
            interpretacion.append("🔴 SMA 100 por debajo de SMA 200: tendencia bajista de largo plazo.")
        else:
            interpretacion.append("⚪ Cruce neutral entre SMA 100 y 200.")

    return "\n".join(interpretacion)


def generar_senales(df):
    senales = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        señal = ""

        # MACD
        if prev["MACD"] < prev["Signal_Line"] and curr["MACD"] > curr["Signal_Line"]:
            señal += "Buy (MACD) "
        elif prev["MACD"] > prev["Signal_Line"] and curr["MACD"] < curr["Signal_Line"]:
            señal += "Sell (MACD) "

        # RSI
        if prev["RSI_14"] < 30 and curr["RSI_14"] >= 30:
            señal += "Buy (RSI) "
        elif prev["RSI_14"] > 70 and curr["RSI_14"] <= 70:
            señal += "Sell (RSI) "

        # SMA 20 / 50
        if prev["SMA_20"] < prev["SMA_40"] and curr["SMA_20"] > curr["SMA_40"]:
            señal += "Buy (SMA corto) "
        elif prev["SMA_20"] > prev["SMA_40"] and curr["SMA_20"] < curr["SMA_40"]:
            señal += "Sell (SMA corto) "

        # SMA 100 / 200
        if "SMA_100" in df.columns and "SMA_200" in df.columns:
            if prev["SMA_100"] < prev["SMA_200"] and curr["SMA_100"] > curr["SMA_200"]:
                señal += "Buy (SMA largo) "
            elif prev["SMA_100"] > prev["SMA_200"] and curr["SMA_100"] < curr["SMA_200"]:
                señal += "Sell (SMA largo) "

        senales.append(señal.strip())
    senales.insert(0, "")
    df["Signal"] = senales
    return df


def interpretar_senales(df):
    ultimas_senales = df["Signal"].dropna().tail(10).tolist()
    resumen = {"Compra": 0, "Venta": 0}
    for señal in ultimas_senales:
        if "Compra" in señal:
            resumen["Compra"] += 1
        if "Venta" in señal:
            resumen["Venta"] += 1
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
                fecha = partes[1]
                simbolos_disponibles.append(f"{simbolo} (desde {fecha})")
            except Exception as e:
                st.warning(f"No se pudo extraer fecha de {archivo}: {e}")
        return simbolos_disponibles
    except Exception as e:
        st.warning(f"No se encontraron datos en {carpeta}: {e}")
        return []


def calcular_indicadores(df):
    """
    Calcula indicadores técnicos estándar para análisis:
    - Medias móviles (SMA)
    - RSI (14)
    - MACD y Signal Line
    """
    df = df.sort_index()

    # === Medias móviles simples ===
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_40"] = df["Close"].rolling(window=40).mean()
    df["SMA_100"] = df["Close"].rolling(window=100).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # === RSI (14) ===
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # === MACD y línea de señal ===
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def esta_en_horario_mercado():
    ny = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny)

    # Verifica si es fin de semana
    if now_ny.weekday() >= 5:
        return False

    # Verifica si es feriado
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=now_ny.date(), end_date=now_ny.date())
    if schedule.empty:
        return False

    # Verifica si está dentro del horario 9:30 a 16:00
    start_time = schedule.iloc[0]['market_open'].tz_convert(ny).time()
    end_time = schedule.iloc[0]['market_close'].tz_convert(ny).time()
    
    return start_time <= now_ny.time() <= end_time


def graficar_con_tecnicaOLD(df, titulo="Gráfico Técnico", mostrar_rsi=True, mostrar_volumen=True, zona_horaria="America/New_York"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df.empty:
        return None

    # Asegurar index con zona horaria
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(zona_horaria)
    else:
        df.index = df.index.tz_convert(zona_horaria)

    fig = make_subplots(
        rows=3 if mostrar_rsi else 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.3] if mostrar_rsi else [0.7, 0.3]
    )

    # === Velas ===
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="Precio"
    ), row=1, col=1)

    # === Señales SMA ===
    if 'Signal' in df.columns:
        df_signals = df.dropna(subset=["Signal"])
    
        for i, row in df_signals.iterrows():
            texto = row["Signal"]
            if any(x in texto for x in ["SMA5", "SMA10", "SMA20", "SMA40"]):
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

    # === Volumen ===
    if mostrar_volumen and 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name="Volumen",
            marker_color='lightblue'
        ), row=2, col=1)

    # === RSI ===
    if mostrar_rsi and 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI_14'],
            name="RSI 14", line=dict(color='purple')
        ), row=3, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash="dot"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash="dot"), row=3, col=1)

    fig.update_layout(
        height=800,
        title=titulo,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    return fig


def graficar_con_tecnica(df, titulo="Gráfico Técnico", mostrar_rsi=True, mostrar_volumen=True, zona_horaria="America/New_York"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df.empty:
        return None

    # Asegurar index con zona horaria
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(zona_horaria)
    else:
        df.index = df.index.tz_convert(zona_horaria)  

    fig = make_subplots(
        rows=3 if mostrar_rsi else 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.3] if mostrar_rsi else [0.7, 0.3]
    )

    # === Velas ===
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="Precio"
    ), row=1, col=1)

    # === Líneas de SMA ===
    smas = {
        "SMA_5": "orange",
        "SMA_10": "blue",
        "SMA_20": "green",
        "SMA_40": "red"
    }
    for sma, color in smas.items():
        if sma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[sma],
                mode="lines",
                name=sma,
                line=dict(color=color, width=1.5)
            ), row=1, col=1)

    # === Señales SMA ===
    if 'Signal' in df.columns:
        df_signals = df.dropna(subset=["Signal"])
    
        for i, row in df_signals.iterrows():
            texto = row["Signal"]
            if any(x in texto for x in ["SMA5", "SMA10", "SMA20", "SMA40"]):
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

    # === Volumen ===
    if mostrar_volumen and 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name="Volumen",
            marker_color='lightblue'
        ), row=2, col=1)

    # === RSI ===
    if mostrar_rsi and 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI_14'],
            name="RSI 14", line=dict(color='purple')
        ), row=3, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash="dot"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash="dot"), row=3, col=1)

    fig.update_layout(
        height=800,
        title=titulo,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )


    # Obtén el calendario de NYSE
    nyse = mcal.get_calendar('NYSE')
    
    # Supón que el rango de tu gráfico es el último año
    schedule = nyse.schedule(
        start_date=df.index.min().date(), 
        end_date=df.index.max().date()
    )
    
    # Lista de fechas de mercado abierto
    open_days = schedule.index
    
    # Lista de feriados (días entre tu rango menos los open_days)
    all_days = pd.date_range(start=df.index.min().date(), end=df.index.max().date(), freq="B")
    holidays = sorted(set(all_days.date) - set(open_days.date))  
    #st.write(holidays)

    fig.update_xaxes(
            type="date",
            rangebreaks=[
                # Oculta fines de semana
                dict(bounds=["sat", "mon"]), 
                # Oculta horas fuera de mercado
                dict(bounds=[16, 9.5], pattern="hour"),
                # Salta feriados
                #dict(values=pd.to_datetime(holidays))
            ]
        )

    #fig.update_xaxes(rangebreaks=[
    #    dict(bounds=["sat", "mon"]),  # Salta fines de semana
    #    dict(bounds=[16, 9.5], pattern="hour"),  # Salta horas fuera de mercado (16:00-9:30 NY)
    #    dict(values=pd.to_datetime(holidays))  # Salta feriados
    #])

    return fig


def filtrar_datos_horario_mercado(df, exchange="NYSE", zona_horaria="America/New_York"):
    """
    Filtra un DataFrame para dejar solo las filas dentro del horario de mercado válido (sin fines de semana, feriados, fuera de horario).
    
    :param df: DataFrame con un DatetimeIndex en UTC o zona horaria compatible.
    :param exchange: Bolsa a usar para el calendario (por defecto NYSE).
    :param zona_horaria: Zona horaria para la comparación.
    :return: DataFrame filtrado.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df  # Nada que hacer

    # Asegurarse que el índice esté en la zona horaria correcta
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(zona_horaria)
    else:
        df.index = df.index.tz_convert(zona_horaria)

    # Obtener el calendario del exchange
    cal = mcal.get_calendar(exchange)

    # Generar el horario del mercado para el rango de fechas del DataFrame
    schedule = cal.schedule(start_date=df.index.min().date(), end_date=df.index.max().date())

    # Expandir el horario a un rango válido por minuto
    valid_times = mcal.date_range(schedule, frequency="1min")

    # Filtrar el DataFrame dejando solo los índices válidos
    df_filtrado = df[df.index.floor("min").isin(valid_times)]

    return df_filtrado



def obtener_datos_yfinance_live(ticker, intervalo="15m", lookback_horas=2):
    tz_ny = pytz.timezone("America/New_York")
    ahora_ny = datetime.now(tz_ny)
    #ahora_ny = datetime.now()
    inicio = ahora_ny - timedelta(hours=lookback_horas)
    
    df = yf.download(
        tickers=ticker,
        start=inicio,
        end=ahora_ny + timedelta(minutes=15),  # Forzar que incluya última vela
        interval=intervalo,
        progress=False
    )

    df.index = df.index.tz_convert("America/New_York")  # Convertir a hora NY
    return df


def obtener_datos_yfinance_today(ticker, intervalo="15m"):
    today_ny = datetime.today()

    tz_ny = pytz.timezone("America/New_York")
    ahora_ny = datetime.now(tz_ny)
    
    startDate="2025-06-01"
   
    #df = yf.download(ticker, period="14d", interval=intervalo)

    #df = yf.download(ticker, start=startDate, interval=intervalo)


    df = yf.download(
        tickers=ticker,
        start=startDate,
        end=ahora_ny + timedelta(minutes=15),  # Forzar que incluya última vela
        interval=intervalo,
        progress=False
    )
    
    df.index = df.index.tz_convert("America/New_York")  # Convertir a hora NY
    return df


