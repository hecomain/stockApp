import os
import smtplib
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import platform
import time



# Configura tus credenciales y destinatarios (solo una vez, o desde variables de entorno)
CORREO_REMITENTE = os.getenv("ALERTA_EMAIL") or "tucorreo@gmail.com"
CORREO_PASSWORD = os.getenv("ALERTA_PASSWORD") or "tu_clave"
CORREO_DESTINATARIO = os.getenv("ALERTA_DESTINO") or "destino@ejemplo.com"

def reproducir_alerta_sonora():
    """Reproduce un sonido de alerta según el sistema operativo."""
    try:
        if platform.system() == "Darwin":  # macOS
            os.system("afplay /System/Library/Sounds/Glass.aiff")
        elif platform.system() == "Windows":
            import winsound
            winsound.MessageBeep()
        else:
            os.system("paplay /usr/share/sounds/freedesktop/stereo/complete.oga")
    except:
        st.warning("⚠️ No se pudo reproducir el sonido.")


def enviar_alerta_email(mensaje, asunto="🚨 Alerta de Señal de Trading"):
    """Envía un correo electrónico con el mensaje de alerta."""
    try:
        msg = MIMEMultipart()
        msg["From"] = CORREO_REMITENTE
        msg["To"] = CORREO_DESTINATARIO
        msg["Subject"] = asunto

        msg.attach(MIMEText(mensaje, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(CORREO_REMITENTE, CORREO_PASSWORD)
        server.send_message(msg)
        server.quit()
        st.success("✉️ Alerta enviada por correo electrónico")
    except Exception as e:
        st.error(f"❌ Error al enviar correo: {e}")


def reproducir_alerta_sonora2():
    """
    Reproduce un sonido de alerta en la interfaz de Streamlit.
    """
    sound_html = """
    <audio autoplay>
        <source src="data:audio/wav;base64,{sound}" type="audio/wav">
    </audio>
    """
    # Pequeño beep en base64 (puedes cambiar el sonido por uno de tu preferencia)
    beep_base64 = (
        "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
    )
    st.markdown(
        sound_html.format(sound=beep_base64),
        unsafe_allow_html=True
    )
