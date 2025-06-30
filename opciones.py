import streamlit as st
import pandas as pd
from datetime import date

# Inicializar almacenamiento de transacciones
if "transacciones_opciones" not in st.session_state:
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
            "Total pagado": prima * contratos * 100
            "Total recuperado": valor_final * contratos * 100,
            "Ganancia/Pérdida": (valor_final - prima) * contratos * 100
        }
        st.session_state.transacciones_opciones.append(transaccion)
        st.success("✅ Transacción registrada correctamente.")

# Mostrar todas las transacciones registradas
if st.session_state.transacciones_opciones:
    st.markdown("### 📋 Historial de transacciones registradas")
    df_transacciones = pd.DataFrame(st.session_state.transacciones_opciones)
    st.dataframe(df_transacciones)
