import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import date

# --- FUNCIONES ---
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


def interpretar_griegos(tipo, delta, gamma, vega, theta):
    interpretacion = {}
    if tipo == "Call":
        interpretacion["Delta"] = f"Delta {delta}: la opción ganaría ${delta:.2f} si la acción sube $1."
    else:
        interpretacion["Delta"] = f"Delta {delta}: la opción ganaría ${abs(delta):.2f} si la acción baja $1."

    interpretacion["Gamma"] = f"Gamma {gamma}: sensibilidad creciente, si el precio se acerca al strike el delta se moverá más rápido."
    interpretacion["Vega"] = f"Vega {vega}: por cada 1% de aumento en IV, el precio cambia ${vega:.4f}."
    interpretacion["Theta"] = f"Theta {theta}: pierde ${abs(theta):.4f} por día si todo se mantiene constante."
    return interpretacion

# --- UI STREAMLIT ---
st.title("Greeks de Opciones e Interpretación Técnica")

symbol = st.text_input("Símbolo de la acción:", value="BAC").upper()
tipo_opcion = st.selectbox("Tipo de opción:", ["Call", "Put"])
fecha_vencimiento = st.date_input("Fecha de vencimiento de la opción")
strike_input = st.number_input("Strike de la opción:", value=50.0)
iv_input = st.number_input("Volatilidad implícita de la opción (%)", value=30.0, min_value=0.0, max_value=500.0) / 100
r = st.number_input("Tasa libre de riesgo (%)", value=5.0) / 100

# Obtener precio actual de la acción
ticker = yf.Ticker(symbol)
data = ticker.history(period="1d")
if data.empty:
    st.error("No se pudo obtener el precio del activo.")
else:
    precio_actual = data["Close"].iloc[-1]
    st.markdown(f"**Último precio de {symbol}: ${precio_actual:.2f}**")

    # Días hasta vencimiento
    hoy = date.today()
    dias_hasta_venc = (fecha_vencimiento - hoy).days

    if dias_hasta_venc <= 0:
        st.warning("La fecha de vencimiento debe ser futura.")
    else:
        # Calcular griegos
        griegos = calcular_greeks(precio_actual, strike_input, dias_hasta_venc, r, iv_input, tipo_opcion)
        interpretacion = interpretar_griegos(tipo_opcion, griegos['Delta'], griegos['Gamma'], griegos['Vega'], griegos['Theta'])

        st.subheader("📊 Resultado de los Greeks")
        st.write(griegos)

        st.subheader("🧠 Interpretación técnica")
        for key, texto in interpretacion.items():
            st.markdown(f"**{key}:** {texto}")

        # Simulación de escenarios futuros
        st.subheader("🔮 Simulador de escenarios")
        nuevo_precio = st.number_input("Nuevo precio estimado del activo:", value=round(precio_actual, 4))
        nueva_iv = st.number_input("Nuevo IV estimado (%)", value=iv_input * 100, min_value=0.0, max_value=500.0) / 100
        num_contratos = st.number_input("Cantidad de contratos: ", min_value=1, step=1, value=1)

        sim_griegos = calcular_greeks(nuevo_precio, strike_input, dias_hasta_venc, r, nueva_iv, tipo_opcion)
        valor_por_contrato = sim_griegos['Precio'] * 100
        valor_total = valor_por_contrato * num_contratos

        st.markdown(f"**Nuevo precio estimado de la opción: ${sim_griegos['Precio']}**")
        st.write({k: v for k, v in sim_griegos.items() if k != 'Precio'})
        st.markdown(f"**💰 Valor por contrato (100 acciones): ${valor_por_contrato:.2f}**")
        st.markdown(f"**📦 Valor total estimado por {num_contratos} contrato(s): ${valor_total:.2f}**")

        # Comparación con precio de entrada
        precio_entrada = st.number_input("Precio de entrada de la opción (por contrato):", min_value=0.0, value=0.0)
        ganancia_total = valor_total - (precio_entrada * 100 * num_contratos)
        st.markdown(f"**📈 Ganancia/Pérdida estimada total: ${ganancia_total:.2f}**")
