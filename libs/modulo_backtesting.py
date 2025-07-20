
import pandas as pd

def backtest_senales(df, capital_inicial=10000, comision_por_trade=0.001):
    """
    Simula una estrategia basada en señales de compra/venta.
    Supuestos:
    - Compra 100% del capital disponible cuando hay señal de compra
    - Vende todo cuando hay señal de venta
    - No apalancamiento
    """

    capital = capital_inicial
    acciones = 0
    historial = []

    for i in range(1, len(df)):
        fila = df.iloc[i]
        signal = fila.get("Signal", "")
        precio = fila.get("Close")

        if "Compra" in signal and capital > 0:
            acciones = capital / precio * (1 - comision_por_trade)
            capital = 0
            historial.append((fila.name, "BUY", precio, acciones, capital))

        elif "Venta" in signal and acciones > 0:
            capital = acciones * precio * (1 - comision_por_trade)
            acciones = 0
            historial.append((fila.name, "SELL", precio, acciones, capital))

    # Valor final
    valor_final = capital + acciones * df.iloc[-1]["Close"] if acciones > 0 else capital
    retorno_pct = (valor_final - capital_inicial) / capital_inicial * 100

    resumen = pd.DataFrame(historial, columns=["Fecha", "Acción", "Precio", "Acciones", "Capital"])
    return resumen, valor_final, retorno_pct
