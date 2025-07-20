# Crear script completo en Streamlit que permita comparar múltiples strikes (PUTs y CALLs)
# usando el último precio de una acción con yfinance y mostrar los Greeks + interpretación

import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd

# Función para calcular Greeks y precio para Call y Put
def calcular_greeks(S, K, T_dias, r, sigma, tipo_opcion):
    T = T_dias / 252
    if T <= 0:
        return {"Precio": 0, "Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo_opcion == "Call":
        precio = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        precio = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        "Precio": round(precio, 4),
        "Delta": round(delta, 4),
        "Gamma": round(gamma, 6),
        "Vega": round(vega, 4),
        "Theta": round(theta, 4)
    }

# Streamlit UI
st.title("Análisis de Opciones: Greeks y Comparación por Strike")

symbol = st.text_input("Símbolo de la acción (ej: BAC, QQQ, AAPL):", value="BAC").upper()
tipo_opcion = st.selectbox("Tipo de opción", ["Call", "Put"])
iv = st.slider("Volatilidad implícita (%)", 10, 100, 30) / 100
dte = st.slider("Días hasta vencimiento", 1, 60, 7)
r = st.number_input("Tasa libre de riesgo (%)", value=5.0) / 100

# Obtener el último precio
ticker = yf.Ticker(symbol)
ultimo_precio = ticker.history(period="1d")["Close"].iloc[-1]
st.markdown(f"**Último precio de {symbol}: ${ultimo_precio:.2f}**")

# Definir strikes relativos al precio actual
desviacion = st.slider("Rango de strikes +/- (%)", 5, 50, 20)
paso = st.selectbox("Paso entre strikes", [0.5, 1, 2.5, 5], index=1)

strike_min = int((1 - desviacion / 100) * ultimo_precio)
strike_max = int((1 + desviacion / 100) * ultimo_precio)
strikes = np.arange(strike_min, strike_max + paso, paso)

# Calcular y mostrar resultados
datos = []
for K in strikes:
    greeks = calcular_greeks(ultimo_precio, K, dte, r, iv, tipo_opcion)
    fila = {
        "Strike": K,
        "Precio Opción": greeks["Precio"],
        "Delta": greeks["Delta"],
        "Gamma": greeks["Gamma"],
        "Vega": greeks["Vega"],
        "Theta": greeks["Theta"]
    }
    datos.append(fila)

df = pd.DataFrame(datos)
st.dataframe(df.style.format({"Precio Opción": "${:.2f}", "Vega": "{:.4f}", "Gamma": "{:.6f}", "Theta": "{:.4f}", "Delta": "{:.4f}"}))

st.markdown("💡 **Tip**: Busca el strike donde Delta se acerque a -0.5 (puts) o +0.5 (calls) para opciones at-the-money, o analiza Gamma si te interesa la sensibilidad acelerada.")
