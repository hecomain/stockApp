import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date, timedelta, datetime, timezone

from utils import (
    validar_ticker,
    calcular_indicadores,
    preparar_dataframe_para_guardar,
    generar_senal_ultima_fila,
    interpretar_tecnicamente,
    generar_interpretacion_tecnica,
    generar_senales,
    interpretar_senales,
    subir_a_dropbox,
    get_simbolos_disponibles,
)


# Inicializar session_state
if "form_opciones" not in st.session_state:
    st.session_state["form_guardado"] = True
    st.session_state["simbolos"] = "AAPL, TSLA"


# ===============================
# 🎯 Módulo de almacenamiento de transacciones
# ===============================

# Inicializar almacenamiento de transacciones

# Ruta del archivo CSV

carpeta_op = "/Users/hcm/stockDB/datos/opciones/"         
opciones_csv = "transacciones_opciones.csv"
full_path_op = os.path.join(carpeta_op, opciones_csv)


# Inicializar o cargar las transacciones existentes
if "transacciones_opciones" not in st.session_state:
    if os.path.exists(opciones_csv):
        st.session_state.transacciones_opciones = pd.read_csv(opciones_csv).to_dict(orient="records")
    else:
        st.session_state.transacciones_opciones = []

st.subheader("📝 Registrar transacción de opción bursátil")

with st.form("form_opciones"):
    col1, col2 = st.columns(2)
    with col1:
        simbolo = st.text_input("Símbolo del subyacente (ej: AAPL)", value="AAPL")
        strike = st.number_input("🎯 Strike Price", min_value=0.0, step=0.5)
        tipo = st.selectbox("📈 Tipo de opción", ["Call", "Put"])
        fecha_expiracion = st.date_input("📆 Fecha de expiración")
    with col2:
        prima = st.number_input("💰 Prima por contrato ($)", min_value=0.0, step=0.1)
        contratos = st.number_input("🔢 Cantidad de contratos", min_value=1, step=1, value=1)
        valor_final = st.number_input("📉 Valor final de la opción ($)", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Registrar transacción")

    if submitted:
        transaccion = {
            "Fecha": date.today(),
            "Símbolo": simbolo.upper(),
            "Tipo": tipo,
            "Strike": strike,
            "Expiración": fecha_expiracion,
            "Prima": prima,
            "Contratos": contratos,
            "Valor final": valor_final,
            "Total pagado": prima * contratos * 100,
            "Total recuperado": valor_final * contratos * 100,
            "Ganancia/Pérdida": (valor_final - prima) * contratos * 100
        }

        # Cargar archivo existente si existe
        if os.path.exists(full_path_op):
            df_existente = pd.read_csv(full_path_op)
        else:
            df_existente = pd.DataFrame()

        # Agregar la nueva fila
        df_nuevo = pd.concat([df_existente, pd.DataFrame([transaccion])], ignore_index=True)

        # Guardar todo en CSV
        df_nuevo.to_csv(full_path_op, index=False)

        # Actualizar session_state también
        st.session_state.transacciones_opciones = df_nuevo.to_dict(orient="records")    
        st.success("✅ Transacción registrada correctamente.")



# Validar existencia del archivo
if not os.path.exists(full_path_op):
    st.warning("❗ No se encontró el archivo de transacciones. Registra al menos una opción para comenzar.")
else:
    df = pd.read_csv(full_path_op, parse_dates=["Fecha", "Expiración"], dayfirst=True)

   
    st.markdown("### 📋 Historial de transacciones registradas")
    df_transacciones = pd.DataFrame(df)
    
    # Asegurar columnas necesarias
    columnas_necesarias = ["Valor final", "Total recuperado", "Ganancia/Pérdida"]
    for col in columnas_necesarias:
        if col not in df_transacciones.columns:
            df_transacciones[col] = 0.0
    
    for i, row in df_transacciones.iterrows():
        with st.expander(f"🗂 {row['Símbolo']} - {row['Tipo']} - Strike {row['Strike']}"):
            cols = st.columns(3)
            with cols[0]:
                nuevo_valor_final = st.number_input(
                    f"📉 Nuevo valor final ({row['Símbolo']})", 
                    value=float(row['Valor final']), 
                    key=f"valor_final_{i}"
                )
            with cols[1]:
                actualizar = st.button(f"💾 Actualizar {row['Símbolo']} #{i}", key=f"actualizar_{i}")
            if actualizar:
                #df_transacciones[i]["Valor final"] = nuevo_valor_final
                #df_transacciones[i]["Total recuperado"] = nuevo_valor_final * row['Contratos'] * 100
                #df_transacciones[i]["Ganancia/Pérdida"] = (nuevo_valor_final - row["Prima"]) * row['Contratos'] * 100
                #pd.DataFrame(df_transacciones).to_csv(full_path_op, index=False)

                df_transacciones.at[i, "Valor final"] = nuevo_valor_final
                df_transacciones.at[i, "Total recuperado"] = nuevo_valor_final * row['Contratos'] * 100
                df_transacciones.at[i, "Ganancia/Pérdida"] = (nuevo_valor_final - row["Prima"]) * row['Contratos'] * 100
                df_transacciones.to_csv(full_path_op, index=False)

                
                st.success("✅ Transacción actualizada.")
    
    # Mostrar tabla general
    st.markdown("### 📋 Resumen de Resultados")
    st.dataframe(pd.DataFrame(df))


    st.subheader("📊 Backtesting de Opciones con Filtros y Evolución Temporal")

     # --- Filtros ---
    st.sidebar.header("🔎 Filtros de Opciones")
    simbolos = st.sidebar.multiselect("Filtrar por símbolo", options=df["Símbolo"].unique(), default=list(df["Símbolo"].unique()))
    tipos = st.sidebar.multiselect("Tipo de opción", options=df["Tipo"].unique(), default=list(df["Tipo"].unique()))
    fechas = st.sidebar.date_input("Rango de fechas", value=(df["Fecha"].min(), df["Fecha"].max()))
    
    df_filtrado = df[
        df["Símbolo"].isin(simbolos) &
        df["Tipo"].isin(tipos) &
        (df["Fecha"] >= pd.to_datetime(fechas[0])) &
        (df["Fecha"] <= pd.to_datetime(fechas[1]))
    ]
    st.markdown("### 📋 Transacciones filtradas")
    st.dataframe(df_filtrado)

    

    # --- Métricas principales ---
    total_invertido = (df_filtrado["Prima"] * df_filtrado["Contratos"] * 100).sum()
    total_recuperado = (df_filtrado["Valor final"] * df_filtrado["Contratos"] * 100).sum()
    ganancia_total = df_filtrado["Ganancia/Pérdida"].sum()
    pct_ganancia = (ganancia_total / total_invertido * 100) if total_invertido > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("💸 Total Invertido", f"${total_invertido:,.2f}")
    col2.metric("💰 Recuperado", f"${total_recuperado:,.2f}")
    col3.metric("📈 Ganancia", f"${ganancia_total:,.2f} ({pct_ganancia:.2f}%)")

    # --- Evolución temporal acumulada ---
    st.markdown("### 📉 Evolución acumulada de Ganancia/Pérdida")

    df_linea = df_filtrado.sort_values("Fecha")
    df_linea["Resultado Acumulado"] = df_linea["Ganancia/Pérdida"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_linea["Fecha"], df_linea["Resultado Acumulado"], marker='o')
    ax.set_title("Ganancia/Pérdida acumulada a lo largo del tiempo")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Resultado Acumulado ($)")
    ax.grid()
    st.pyplot(fig)

    # --- Gráfico por símbolo ---
    st.markdown("### 📊 Ganancia total por símbolo")

    resumen = df_filtrado.groupby("Símbolo")["Ganancia/Pérdida"].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(resumen["Símbolo"], resumen["Ganancia/Pérdida"], color="skyblue")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_ylabel("Ganancia / Pérdida ($)")
    ax2.set_title("Ganancia por símbolo (filtrados)")
    st.pyplot(fig2)

    # --- Porcentaje de operaciones ganadoras / perdedoras ---
    st.markdown("### 🧮 Análisis de Resultados de Transacciones")
    
    ganadoras = df_filtrado[df_filtrado["Ganancia/Pérdida"] > 0].shape[0]
    perdedoras = df_filtrado[df_filtrado["Ganancia/Pérdida"] < 0].shape[0]
    neutras = df_filtrado[df_filtrado["Ganancia/Pérdida"] == 0].shape[0]
    total_ops = len(df_filtrado)
    
    pct_ganadoras = (ganadoras / total_ops) * 100 if total_ops > 0 else 0
    pct_perdedoras = (perdedoras / total_ops) * 100 if total_ops > 0 else 0
    pct_neutras = (neutras / total_ops) * 100 if total_ops > 0 else 0
    
    col4, col5, col6 = st.columns(3)
    col4.metric("✅ Ganadoras", f"{ganadoras} ({pct_ganadoras:.1f}%)")
    col5.metric("❌ Perdedoras", f"{perdedoras} ({pct_perdedoras:.1f}%)")
    col6.metric("➖ Neutras", f"{neutras} ({pct_neutras:.1f}%)")
    
    # --- Pie chart ---
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(
        [ganadoras, perdedoras, neutras],
        labels=["Ganadoras", "Perdedoras", "Neutras"],
        autopct="%1.1f%%",
        colors=["green", "red", "gray"],
        startangle=90
    )
    ax_pie.axis("equal")
    ax_pie.set_title("Distribución de Resultados")
    st.pyplot(fig_pie)


    # Descargar resultados
    st.download_button(
        label="⬇️ Descargar resultados en Excel",
        data=df.to_csv(index=False),
        file_name="resultados_backtesting.csv",
        mime="text/csv"
    )


 # --- Top 3 operaciones con más ganancia ---
    st.markdown("### 🏆 Mejores y Peores Operaciones")
    
    top_ganadoras = df_filtrado.sort_values("Ganancia/Pérdida", ascending=False).head(5)
    top_perdedoras = df_filtrado.sort_values("Ganancia/Pérdida", ascending=True).head(5)
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown("#### 🟢 Top 5 Ganadoras")
        st.dataframe(top_ganadoras[["Símbolo", "Fecha", "Tipo", "Strike", "Prima", "Valor final", "Ganancia/Pérdida"]])
    
    with col8:
        st.markdown("#### 🔴 Top 5 Perdedoras")
        st.dataframe(top_perdedoras[["Símbolo", "Fecha", "Tipo", "Strike", "Prima", "Valor final", "Ganancia/Pérdida"]])
 # --- Top 3 operaciones con más ganancia ---



# Mostrar historial y permitir edición



#if st.session_state.transacciones_opciones:
    