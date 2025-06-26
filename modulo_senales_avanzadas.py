import pandas as pd


def generar_senales_avanzadas(df, volumen_minimo=None, incluir_sma_largo_plazo=True):
    """
    Genera señales de compra/venta usando lógica combinada:
    - Cruce de SMAs (SMA20 y SMA50)
    - MACD cruzando la línea de señal
    - RSI fuera de los umbrales
    - Filtro opcional por volumen mínimo
    """

    # Asegurar SMAs de largo plazo si se solicitan
    if incluir_sma_largo_plazo:
        if 'SMA_100' not in df.columns:
            df['SMA_100'] = df['Close'].rolling(window=100).mean()
        if 'SMA_200' not in df.columns:
            df['SMA_200'] = df['Close'].rolling(window=200).mean()


    condiciones = []

    for i in range(1, len(df)):
        señal = []

        # Filtro por volumen si está activado
        if volumen_minimo and df["Volume"].iloc[i] < volumen_minimo:
            condiciones.append("")  # No generar señal
            continue

        # --- Cruce de SMA ---
        if df["SMA_20"].iloc[i] > df["SMA_50"].iloc[i] and df["SMA_20"].iloc[i-1] <= df["SMA_50"].iloc[i-1]:
            señal.append("Compra (SMA20 ↑ SMA50)")
        elif df["SMA_20"].iloc[i] < df["SMA_50"].iloc[i] and df["SMA_20"].iloc[i-1] >= df["SMA_50"].iloc[i-1]:
            señal.append("Venta (SMA20 ↓ SMA50)")

        # --- Cruce SMA largo plazo ---
        if incluir_sma_largo_plazo and 'SMA_100' in df.columns and 'SMA_200' in df.columns:
            if df['SMA_100'].iloc[i] > df['SMA_200'].iloc[i] and df['SMA_100'].iloc[i-1] <= df['SMA_200'].iloc[i-1]:
                señal.append('Compra (SMA100 ↑ SMA200)')
            elif df['SMA_100'].iloc[i] < df['SMA_200'].iloc[i] and df['SMA_100'].iloc[i-1] >= df['SMA_200'].iloc[i-1]:
                señal.append('Venta (SMA100 ↓ SMA200)')


        # --- Cruce MACD / Signal Line ---
        if df["MACD"].iloc[i] > df["Signal_Line"].iloc[i] and df["MACD"].iloc[i-1] <= df["Signal_Line"].iloc[i-1]:
            señal.append("Compra (MACD ↑ Signal)")
        elif df["MACD"].iloc[i] < df["Signal_Line"].iloc[i] and df["MACD"].iloc[i-1] >= df["Signal_Line"].iloc[i-1]:
            señal.append("Venta (MACD ↓ Signal)")

        # --- RSI extremos ---
        if df["RSI_14"].iloc[i] < 30:
            señal.append("Compra (RSI < 30)")
        elif df["RSI_14"].iloc[i] > 70:
            señal.append("Venta (RSI > 70)")

        condiciones.append(" | ".join(señal))

    condiciones.insert(0, "")  # Para alinear con el DataFrame original
    df["Signal"] = condiciones
    return df

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
        interpretacion.append("⚪ Las medias móviles 20 y 40 están convergiendo.")

    if st.session_state.get("incluir_sma_largo_plazo", True):
        if ult['SMA_100'] > ult['SMA_200']:
            interpretacion.append("🟢 SMA 100 > SMA 200: tendencia alcista de largo plazo.")
        elif ult['SMA_100'] < ult['SMA_200']:
            interpretacion.append("🔴 SMA 100 < SMA 200: tendencia bajista de largo plazo.")
        else:
            interpretacion.append("⚪ SMA 100 y 200 están convergiendo.")

    return "\n".join(interpretacion)

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
