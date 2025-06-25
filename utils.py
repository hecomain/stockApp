import os
import pandas as pd
import yfinance as yf
import dropbox
import streamlit as st


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