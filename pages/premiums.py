import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pytz
import time


from datetime import datetime, timedelta
from libs.utils import (
    obtener_datos_yfinance_history,
    obtener_datos_opciones,
    map_contract_size,
    detectar_operaciones_institucionales,
    guardar_datos_opciones_y_precio
)

tz_ny = pytz.timezone("America/New_York")
today_ny = datetime.today()
ahora_ny = datetime.now(tz_ny)
days_before = today_ny - timedelta(days=5)


def graficar_premiums(df_options, df_price, ticker_symbol, window_smooth=3):
    

    # Agrupar y sumar
    df_grouped = df_options.groupby(["Type", "hour"])["premium"].sum().reset_index()

    # Suvizar Graficos
    df_grouped["premium_smooth"] = df_grouped["premium"].rolling(window=window_smooth, min_periods=1).mean()

    # Crear rango uniforme de horas
    horas = sorted(df_grouped["hour"].unique())

    # Extraer horas presentes
    horas_premium = df_grouped["hour"].unique()
    horas_price = df_price.index.strftime("%H:%M").unique()

    # Generar rango común de horas
    horas_comunes = sorted(set(horas_premium).union(set(horas_price)))

    #st.write(df_options)
    
    # Reindexar premiums
    calls = df_grouped[df_grouped["Type"] == "Call"].set_index("hour").reindex(horas_comunes, fill_value=0).reset_index()
    puts = df_grouped[df_grouped["Type"] == "Put"].set_index("hour").reindex(horas_comunes, fill_value=0).reset_index()

    # Reindexar precio
    df_price_reindexed = df_price.copy()
    df_price_reindexed["hour"] = df_price.index.strftime("%H:%M")
    df_price_reindexed = df_price_reindexed.set_index("hour").reindex(horas_comunes, method="ffill").reset_index()


    # Convertir hora en df_price al mismo formato y zona horaria
    #df_price = df_price.copy()
    #df_price.index = df_price.index.tz_convert("America/New_York")
    #df_price["hour"] = df_price.index.strftime("%H:%M")

    # Filtrar solo las horas que coinciden
    #df_price = df_price[df_price["hour"].isin(horas)]

    #st.write(df_price)

    fig = go.Figure()

    # Premiums CALL
    fig.add_trace(go.Scatter(
        x=calls["hour"], y=calls["premium"],
        name="Call Premium",
        mode="lines+markers",
        yaxis="y1",
        line=dict(color="green", shape="spline", smoothing=0.5)
    ))

    # Premiums PUT
    fig.add_trace(go.Scatter(
        x=puts["hour"], y=puts["premium"],
        name="Put Premium",
        mode="lines+markers",
        yaxis="y1",
        line=dict(color="red", shape="spline", smoothing=0.5)
    ))

    # Precio de la acción
    fig.add_trace(go.Scatter(
        x=df_price_reindexed["hour"],
        y=df_price_reindexed["Close"],
        name=f"{ticker_symbol} Close",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="yellow")
    ))

    fig.update_layout(
        title=f"Premiums y Precio de {ticker_symbol}",
        xaxis_title="Hora",
        yaxis=dict(title="Premiums ($)", side="left"),
        yaxis2=dict(title="Precio Acción ($)", overlaying="y", side="right"),
        template="plotly_white",
        height=600,
        shapes=[
        # Rango pre-market (ejemplo: 04:00 a 09:30 NY)
        dict(
            type="rect",
            xref="x",
            yref="paper",
            x0="09:30",
            x1="16:00",
            y0=0,
            y1=1,
            fillcolor="lightgray",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
        # Rango after-market (ejemplo: 16:00 a 20:00 NY)
        #dict(
        #    type="rect",
        #    xref="x",
        #    yref="paper",
        #    x0="16:00",
        #    x1="20:00",
        #    y0=0,
        #    y1=1,
         #   fillcolor="lightgray",
         #   opacity=0.2,
         #   layer="below",
         #   line_width=0,
        #)
    ]
    )

    return fig


def crear_tabla_opciones(df, tipo_opcion, min_contratos_resaltar=0):
    # Filtrar por tipo
    df_tipo = df[df["Type"] == tipo_opcion].copy()

    # Si está vacío, devuelve None
    if df_tipo.empty:
        return None

    # Convertir impliedVolatility a porcentaje con 2 decimales
    df_tipo["impliedVolatility"] = (df_tipo["impliedVolatility"] * 100).round(2).astype(str) + '%'
    df_tipo["lastTradeDate"] = df_tipo["lastTradeDate"].dt.tz_localize(None)

    # Definir colores por fila
    row_colors = []
    for _, row in df_tipo.iterrows():
        if row["openInterest"] >= min_contratos_resaltar:
            row_colors.append('#F94F00')  # Resaltado: gris oscuro
        elif row["inTheMoney"]:
            row_colors.append('#333333')  # In The Money: gris medio
        else:
            row_colors.append('#000000')  # Normal: negro

    # Define los valores de las columnas que quieres mostrar
    header_vals = ["<b>Last Trade</b>","<b>Strike</b>", "<b>Last Price</b>", "<b>Bid</b>", "<b>Ask</b>", "<b>Open Interest</b>", "<b>Imp Volatility</b>"]
    cell_vals = [
        df_tipo["lastTradeDate"],
        df_tipo["strike"],
        df_tipo["lastPrice"],
        df_tipo["bid"],
        df_tipo["ask"],
        df_tipo["openInterest"],
        df_tipo["impliedVolatility"]
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_vals,
            fill_color='#000000',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=cell_vals,
            fill_color=[row_colors],
            font=dict(color='white'),
            align='center'
        )
    )])

    return fig

# === Resumen del Mercado ===

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

    # --- MÁXIMOS EN OPCIONES: PREMIUMS---
    def resumen_max_premiums(df, tipo):
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

     # --- MÁXIMOS EN OPCIONES: CONTRATOS ---
    def resumen_max_openInt(df, tipo):
        df_tipo = df[df["Type"] == tipo]
        if df_tipo.empty:
            return "N/A", "N/A", "N/A", "N/A", "N/A"
        fila_max = df_tipo.loc[df_tipo["openInterest"].abs().idxmax()]
        return (
            fila_max["strike"],
            fila_max["ask"],
            fila_max["openInterest"],
            f"${fila_max['premium']:,.2f}",
            fila_max["hour"]
        )

    st.markdown("### 💼 Máximos en Opciones")

    col_call_p,  col_call_o, col_put_p, col_put_o = st.columns(4)
    with col_call_p:
        s, a, oi, p, h = resumen_max_premiums(df_options_filtrado, "Call")
        st.markdown(f"**🟢 CALL**")
        st.markdown(f"**Premiums**  \n**Strike:** {s}  \n**Ask:** {a}  \n**Open Int.:** {oi}  \n**Premium:** {p}  \n**Hora:** {h}")
    with col_call_o:
        s, a, oi, p, h = resumen_max_openInt(df_options_filtrado, "Call")
        st.markdown(f"**🟢 CALL**")
        st.markdown(f"**Contracts**  \n**Strike:** {s}  \n**Ask:** {a}  \n**Open Int.:** {oi}  \n**Premium:** {p}  \n**Hora:** {h}")
    with col_put_p:
        s, a, oi, p, h = resumen_max_premiums(df_options_filtrado, "Put")
        st.markdown(f"**🔴 PUT**")
        st.markdown(f"**Premiums**  \n**Strike:** {s}  \n**Ask:** {a}  \n**Open Int.:** {oi}  \n**Premium:** {p}  \n**Hora:** {h}")
    with col_put_o:
        s, a, oi, p, h = resumen_max_openInt(df_options_filtrado, "Put")
        st.markdown(f"**🔴 PUT**")
        st.markdown(f"**Contracts**  \n**Strike:** {s}  \n**Ask:** {a}  \n**Open Int.:** {oi}  \n**Premium:** {p}  \n**Hora:** {h}")




# === Parámetros de usuario ===
carpeta_datos = "./datos/opciones"

# ======== Streamlit App ========
st.set_page_config(page_title="📈 Análisis de Opciones y Precio", layout="wide")
st.title("📈 Análisis de Opciones y Precio")

symbol = st.sidebar.text_input("Símbolo", value="SPY").upper()
int_options = ["5m","10m","15m", "30m", "60m"]
int_default_idx = int_options.index("15m")
intervalo = st.sidebar.selectbox("Frecuencia actualización (Live)", int_options, index=int_default_idx)
usar_historico = st.sidebar.checkbox("Usar datos históricos", value=True)
actualizar_auto = st.sidebar.checkbox("Actualizar automáticamente", value=False)
refrescar_cada = {"5m": 300, "10m": 600, "15m": 900, "30m": 1800, "60m": 3600}[intervalo]

#smooth_window = st.sidebar.slider("Suavizado (ventana)", min_value=1, max_value=10, value=3)

if st.sidebar.button("🔄 Actualizar manualmente"):
    st.rerun()
if actualizar_auto:
    time.sleep(refrescar_cada)
    st.rerun()


if symbol:
    ticker = yf.Ticker(symbol)
    expiries = ticker.options
    expiry = st.sidebar.selectbox("Fecha de expiración", expiries)

    

     # Mostrar los gráficos lado a lado
    col1, col2 = st.columns([7, 3])

    if expiry:
        df_options = obtener_datos_opciones(symbol, expiry)

        if not df_options.empty:
            fechas_disponibles = sorted(df_options["date"].unique())
            fecha_seleccionada = st.sidebar.selectbox("Filtrar por fecha de lastTradeDate", fechas_disponibles)
            min_contratos = st.sidebar.number_input("Resaltar opciones con contratos mayores a:", min_value=0, value=500, step=100)
            umbral_institucional = st.sidebar.number_input("Umbral de Open Interest (institucional)", min_value=100, value=500, step=100)



            df_options_filtrado = df_options[df_options["date"] == fecha_seleccionada]

            df_toShow = df_options_filtrado
            
            # Ordenar por fecha y hora
            df_toShow = df_toShow.sort_values(by=["Type", "strike"]).reset_index(drop=True)
            

            if usar_historico:
                #df_price = ticker.history(period="5d", interval=intervalo)
                df_price = obtener_datos_yfinance_history(symbol,fecha_seleccionada,intervalo)
            else:
                #df_price = ticker.history(period="1d", interval=intervalo)
                df_price = obtener_datos_yfinance_history(symbol,fecha_seleccionada,intervalo)

            # Asegurarnos que el índice es datetime y tiene la zona horaria correcta
            df_price.index = df_price.index.tz_convert("America/New_York")

            # Separar fecha y hora
            df_price["date"] = df_price.index.date
            df_price["hour"] = df_price.index.strftime("%H:%M")

            df_price_filtrado = df_price[df_price["date"] == fecha_seleccionada]
            #st.write(df_price_filtrado)

            if not df_options_filtrado.empty and not df_price.empty:

                with col1:               
                    fig = graficar_premiums(df_options_filtrado, df_price_filtrado, symbol)
                    st.plotly_chart(fig, use_container_width=True, key="grafico_principal")

                with col2:

                    # Agrupar y sumar premium
                    resumen_opciones = df_options_filtrado.groupby("Type")["premium"].sum().reset_index()
                    resumen_opciones.loc[resumen_opciones["Type"] == "Put", "premium"] *= -1
                    
                    # Crear gráfico de barras con Plotly
                    fig_barras = go.Figure()
                    
                    colores = {"Call": "green", "Put": "red"}
                    
                    for _, row in resumen_opciones.iterrows():
                        fig_barras.add_trace(go.Bar(
                            x=[row["Type"]],
                            y=[row["premium"]],
                            marker_color=colores.get(row["Type"], "gray"),
                            name=row["Type"]
                        ))
                    
                    fig_barras.update_layout(
                        title=f"📊 Total de Premiums por Tipo ({fecha_seleccionada})",
                        yaxis_title="Premium acumulado (USD)",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig_barras, use_container_width=True, key="grafico_barras")

                #
                #Resumen de Valor de la Opcion y Maximos en Opciones
                #
            
                mostrar_resumen_mercado_y_opciones(df_price_filtrado, df_options_filtrado)
                
                #Grafico x Strike y Tipo de Opcion 
                
                # Agrupar por strike y tipo (Call / Put) y sumar openInterest
                df_strike = df_options_filtrado.groupby(["strike", "Type"])["openInterest"].sum().reset_index()
                df_strike = df_strike.sort_values("strike")

                fig_strike = go.Figure()

                # Agrega barras para CALLS (verde)
                df_call = df_strike[df_strike["Type"] == "Call"]
                fig_strike.add_trace(go.Bar(
                    x=df_call["strike"],
                    y=df_call["openInterest"],
                    name="Call",
                    marker_color="green"
                ))
                
                # Agrega barras para PUTS (rojo)
                df_put = df_strike[df_strike["Type"] == "Put"]
                fig_strike.add_trace(go.Bar(
                    x=df_put["strike"],
                    y=df_put["openInterest"],
                    name="Put",
                    marker_color="red"
                ))
                
                fig_strike.update_layout(
                    title="Cantidad de opciones por Strike",
                    xaxis_title="Strike",
                    yaxis_title="Open Interest",
                    barmode="group",
                    template="plotly_white"
                )
                
                # Mostrar en Streamlit
                st.plotly_chart(fig_strike, use_container_width=True, key="grafico_strike")

                with st.expander("📋 Opciones por tipo"):
                    # === Streamlit ===

                    col1r1, col2r1 = st.columns(2)
                    
                    with col1r1:
                        st.markdown("### 🟢 Calls")
                        fig_calls = crear_tabla_opciones(df_toShow, "Call", min_contratos_resaltar=min_contratos)
                        if fig_calls:
                            st.plotly_chart(fig_calls, use_container_width=True)
                        else:
                            st.info("No hay opciones Call disponibles.")
                    
                    with col2r1:
                        st.markdown("### 🔴 Puts")
                        fig_puts = crear_tabla_opciones(df_toShow, "Put", min_contratos_resaltar=min_contratos)
                        if fig_puts:
                            st.plotly_chart(fig_puts, use_container_width=True)
                        else:
                            st.info("No hay opciones Put disponibles.")

            else:
                st.warning("No hay datos suficientes para graficar en la fecha seleccionada.")


            if st.sidebar.button("📥 Guardar CSVs"):
                ruta_opts, ruta_price = guardar_datos_opciones_y_precio(
                    df_options, df_price, symbol, expiry, intervalo
                )
                st.success(f"✅ Archivos guardados:\n- {ruta_opts}\n- {ruta_price}")

            #st.write(df_options)
            detectar_operaciones_institucionales(df_options_filtrado, umbral=umbral_institucional)

        else:
            st.warning("No se encontraron datos de opciones.")




