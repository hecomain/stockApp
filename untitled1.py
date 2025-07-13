import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import time

def obtener_opciones_y_agrupadas(ticker_symbol, fecha_inicio, frecuencia="1H"):
    ticker = yf.Ticker(ticker_symbol)

    df_price = ticker.history(start=fecha_inicio, interval="15m")
    if df_price.empty:
        st.warning("⚠️ No hay datos históricos de precio disponibles.")
        return None, None

    all_options = []
    try:
        options_dates = ticker.options
        for date in options_dates:
            opt = ticker.option_chain(date)
            for tipo, df in zip(["Call", "Put"], [opt.calls, opt.puts]):
                df = df.copy()
                df["Type"] = tipo
                df["Expiry"] = pd.to_datetime(date)
                df["netPremium"] = df["lastPrice"] * df["openInterest"] * 100
                df["Timestamp"] = pd.Timestamp.now()
                all_options.append(df)
    except Exception as e:
        st.error(f"❌ Error obteniendo opciones: {e}")
        return df_price, None

    st.write(all_options)

    df_opts = pd.concat(all_options)
    df_opts["Timestamp"] = df_opts["Timestamp"].dt.floor(frecuencia)
    df_group = df_opts.groupby(["Timestamp", "Type"])["netPremium"].sum().reset_index()
    df_pivot = df_group.pivot(index="Timestamp", columns="Type", values="netPremium").fillna(0)

    return df_price, df_pivot

def graficar_flujo_opciones(df_price, df_pivot, ticker):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_price.index,
        y=df_price["Close"],
        name=f"{ticker} Price",
        yaxis="y2",
        line=dict(color="gold", width=2)
    ))
    
    if "Call" in df_pivot.columns:
        fig.add_trace(go.Scatter(
            x=df_pivot.index,
            y=df_pivot["Call"],
            name="Call Net Premium",
            line=dict(color="green")
        ))
    
    if "Put" in df_pivot.columns:
        fig.add_trace(go.Scatter(
            x=df_pivot.index,
            y=df_pivot["Put"],
            name="Put Net Premium",
            line=dict(color="red")
        ))
    
    fig.update_layout(
        title=f"{ticker} - Flujo de Opciones por Hora",
        xaxis_title="Hora",
        yaxis=dict(title="Net Premium"),
        yaxis2=dict(title="Stock Price", overlaying="y", side="right"),
        template="plotly_dark",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"grafico_flow_{ticker}_{int(time.time()*1000)}")

# === Streamlit App ===
st.sidebar.title("⚙️ Configuración del gráfico")
ticker = st.sidebar.text_input("Símbolo", value="AAPL")
frecuencia = st.sidebar.selectbox("Frecuencia", ["15min", "30min", "1H"], index=2)
fecha_inicio = st.sidebar.date_input("Fecha de inicio", value=pd.Timestamp.today())

if st.sidebar.button("📈 Generar gráfico"):
    with st.spinner(f"Obteniendo datos de {ticker}..."):
        df_price, df_pivot = obtener_opciones_y_agrupadas(ticker, fecha_inicio, frecuencia="1H" if frecuencia=="1H" else frecuencia)

        st.write(df_price)
        st.write(df_pivot)
    if df_price is not None and df_pivot is not None:
        graficar_flujo_opciones(df_price, df_pivot, ticker)


####################################


import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go


def map_contract_size(val):
    if val == 'REGULAR':
        return 100
    elif val == 'MINI':
        return 10
    else:
        return 100  # Por defecto

st.set_page_config(layout="wide")

# === Sidebar inputs ===
st.sidebar.title("⚙️ Configuración del Gráfico")
symbol = st.sidebar.text_input("Símbolo", value="AAPL")
data_type = st.sidebar.radio("Fuente de datos", ["Histórico", "En Vivo"])
interval = st.sidebar.selectbox("Intervalo", ["15m", "30m", "60m"])
start_date = st.sidebar.date_input("Fecha inicial (solo histórico)")
run_button = st.sidebar.button("🚀 Generar Gráfico")

if run_button and symbol:
    ticker = yf.Ticker(symbol)
    
    # === Fechas de expiración ===
    try:
        options_dates = ticker.options
        exp_date = st.sidebar.selectbox("Fecha de expiración", options_dates)
        
        opt = ticker.option_chain(exp_date)
        calls = opt.calls.copy()
        puts = opt.puts.copy()

        calls["contractSizeNum"] = calls["contractSize"].apply(map_contract_size)
        puts["contractSizeNum"] = puts["contractSize"].apply(map_contract_size)

        calls["premium"] = calls["lastPrice"] * calls["contractSizeNum"] * calls["openInterest"]
        puts["premium"] = puts["lastPrice"] * puts["contractSizeNum"] * puts["openInterest"]
        
        # Formatear fecha y hora
        for df in [calls, puts]:
            df["Hora"] = pd.to_datetime(df["lastTradeDate"]).dt.strftime("%H:%M")
        
        # Agrupar por hora
        calls_grouped = calls.groupby("Hora")["premium"].sum().reset_index().assign(tipo="Call")
        puts_grouped = puts.groupby("Hora")["premium"].sum().reset_index().assign(tipo="Put")
        
        df_premium = pd.concat([calls_grouped, puts_grouped])

        # Precio acción en horas coincidentes
        if data_type == "Histórico":
            hist = ticker.history(start=start_date, interval=interval)
        else:
            hist = ticker.history(period="1d", interval=interval)
        
        hist["Hora"] = hist.index.strftime("%H:%M")
        merged = df_premium.merge(hist[["Hora", "Close"]], on="Hora", how="left")
        
        # === Gráfico ===
        fig = go.Figure()

        # Premium Call
        df_call = merged[merged["tipo"] == "Call"]
        fig.add_trace(go.Scatter(x=df_call["Hora"], y=df_call["premium"], name="Call Premium", yaxis="y1", line=dict(color="blue")))

        # Premium Put
        df_put = merged[merged["tipo"] == "Put"]
        fig.add_trace(go.Scatter(x=df_put["Hora"], y=df_put["premium"], name="Put Premium", yaxis="y1", line=dict(color="red")))

        # Precio acción
        fig.add_trace(go.Scatter(x=hist["Hora"], y=hist["Close"], name="Precio Acción", yaxis="y2", line=dict(color="green")))

        fig.update_layout(
            title=f"Premium y Precio Acción - {symbol}",
            xaxis=dict(title="Hora"),
            yaxis=dict(title="Premium", side="left"),
            yaxis2=dict(title="Precio Acción", overlaying="y", side="right"),
            template="plotly_white",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error al procesar: {e}")



        ####################



def mostrar_resumen_mercado_y_opciones(df_price, df_options_filtrado):
    st.subheader("📌 Resumen del Mercado y Opciones")

    # --- PRECIOS DE LA ACCIÓN ---
    df_price["hour"] = df_price.index.strftime("%H:%M")
    df_price["datetime"] = df_price.index

    df_options_filtrado.loc[df_options["Type"] == "Put", "premium"] *= -1

    valor_pre_market = df_price[df_price["hour"] < "09:30"]["Close"].iloc[0] if not df_price[df_price["hour"] < "09:30"].empty else None
    valor_apertura = df_price[df_price["hour"] == "09:30"]["Close"].iloc[0] if not df_price[df_price["hour"] == "09:30"].empty else None
    valor_cierre = df_price[df_price["hour"] == "16:00"]["Close"].iloc[0] if not df_price[df_price["hour"] == "16:00"].empty else None
    valor_post_market = df_price[df_price["hour"] > "16:00"]["Close"].iloc[-1] if not df_price[df_price["hour"] > "16:00"].empty else None
    valor_actual = df_price["Close"].iloc[-1]

    # Cálculo de cambio %
    if valor_apertura and valor_cierre:
        cambio = ((valor_cierre - valor_apertura) / valor_apertura) * 100
        color_cambio = "green" if cambio >= 0 else "red"
        cambio_str = f"{cambio:+.2f}%"
    else:
        cambio_str = "N/D"
        color_cambio = "gray"

    # Cálculo de cambio entre Pre y Open%
    if valor_pre_market and valor_apertura:
        cambio_pre = ((valor_apertura - valor_pre_market) / valor_pre_market) * 100
        color_cambio_pre = "green" if cambio_pre >= 0 else "red"
        cambio_str_pre = f"{cambio_pre:+.2f}%"
    else:
        cambio_str_pre = "N/D"
        color_cambio_pre = "gray"

    # Mostrar precios
    st.markdown("### 📈 Resumen de Precio de la Acción")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pre-Market", f"${valor_pre_market:.2f}" if valor_pre_market else "N/D")
        st.metric("Apertura", f"${valor_apertura:.2f}" if valor_apertura else "N/D", delta=cambio_str_pre)
    with col2:
        st.metric("Actual", f"${valor_actual:.2f}")
        st.metric("Cierre", f"${valor_cierre:.2f}" if valor_cierre else "N/D", delta=cambio_str)
    with col3:
        st.metric("Post-Market", f"${valor_post_market:.2f}" if valor_post_market else "N/D")

    # --- MÁXIMOS EN OPCIONES ---
    def resumen_max(df, tipo):
        df_tipo = df[df["Type"] == tipo]
        if df_tipo.empty:
            return "N/A", "N/A", "N/A", "N/A", "N/A"
        fila_max = df_tipo.loc[df_tipo["premium"].abs().idxmax()]
        return (
            fila_max["strike"],
            fila_max["ask"],
            fila_max["openInterest"],
            f"${fila_max['premium']:,.2f}",
            fila_max["hour"]
        )

    st.markdown("### 💼 Máximos en Opciones")

    col_call, col_put = st.columns(2)
    with col_call:
        s, a, oi, p, h = resumen_max(df_options_filtrado, "Call")
        st.markdown(f"**🟢 CALL**")
        st.markdown(f"**Strike:** {s}  \n**Ask:** {a}  \n**Open Int.:** {oi}  \n**Premium:** {p}  \n**Hora:** {h}")
    with col_put:
        s, a, oi, p, h = resumen_max(df_options_filtrado, "Put")
        st.markdown(f"**🔴 PUT**")
        st.markdown(f"**Strike:** {s}  \n**Ask:** {a}  \n**Open Int.:** {oi}  \n**Premium:** {p}  \n**Hora:** {h}")



mostrar_resumen_mercado_y_opciones(df_price_filtrado, df_options_filtrado)




