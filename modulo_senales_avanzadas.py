import pandas as pd


def generar_senales_avanzadas(df, volumen_minimo=None):
    """
    Genera señales de compra/venta usando lógica combinada:
    - Cruce de SMAs (SMA20 y SMA50)
    - MACD cruzando la línea de señal
    - RSI fuera de los umbrales
    - Filtro opcional por volumen mínimo
    """

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
