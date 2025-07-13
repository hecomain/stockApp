import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="📈 Última Jornada de Trading", layout="wide")

st.title("📊 Últimos Datos del Día de Trading")

# Sidebar - Ingreso del símbolo
symbol = st.sidebar.text_input("Ingrese el símbolo de la acción (ej: AAPL, MSFT, TSLA):", value="AAPL").upper()
st.sidebar.page_link("pages/premiums.py", label="Premiums", icon="📊")

def obtener_datos_ultimo_dia(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1d", interval="1m")

    if df.empty:
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = df.index.tz_convert("America/New_York")
    df["Hora"] = df.index.strftime("%H:%M")

    return df

# Procesar símbolo ingresado
if symbol:
    df_ultimo_dia = obtener_datos_ultimo_dia(symbol)

    if not df_ultimo_dia.empty:
        st.success(f"Datos cargados correctamente para {symbol}")

        # Mostrar últimos valores como métricas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔔 Apertura", f"${df_ultimo_dia['Open'].iloc[0]:.2f}")
        col2.metric("🔺 Máximo", f"${df_ultimo_dia['High'].max():.2f}")
        col3.metric("🔻 Mínimo", f"${df_ultimo_dia['Low'].min():.2f}")
        col4.metric("✅ Cierre", f"${df_ultimo_dia['Close'].iloc[-1]:.2f}")

        # Mostrar gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_ultimo_dia.index,
            y=df_ultimo_dia["Close"],
            mode="lines+markers",
            name="Close Price",
            line=dict(color="blue")
        ))
        fig.update_layout(
            title=f"📈 Evolución intradía de {symbol}",
            xaxis_title="Hora",
            yaxis_title="Precio de Cierre ($)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tabla
        with st.expander("📋 Ver tabla completa de datos"):
            st.dataframe(df_ultimo_dia)

        # Opción para descargar CSV
        csv = df_ultimo_dia.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv,
            file_name=f"{symbol}_ultimo_dia.csv",
            mime="text/csv"
        )

    else:
        st.warning("⚠️ No se encontraron datos para el símbolo ingresado.")
