import streamlit as st
from huggingface_hub import InferenceClient

# 1. Configuración de pantalla
st.set_page_config(page_title="Chat Mamá", page_icon="💬", layout="centered")

# 2. Control de Acceso con Contraseña
def verificar_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 Acceso Privado")
        pwd_input = st.text_input("Ingresa la contraseña:", type="password")
        if st.button("Entrar"):
            if pwd_input == st.secrets.get("APP_PASSWORD", "mi_clave_secreta"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta")
        return False
    return True

if not verificar_password():
    st.stop()

# 3. Inicializar Cliente de Inferencia Liviano
@st.cache_resource
def obtener_cliente_hf():
    hf_token = st.secrets.get("HF_TOKEN", None)
    # Conexión directa a la API del modelo
    client = InferenceClient(
        model="josuealbelaez/qwen2.5-1.5b-estilo-mama",
        token=hf_token
    )
    return client

client = obtener_cliente_hf()

# 4. Estilos CSS estilo WhatsApp
st.markdown("""
    <style>
    .stApp { background-color: #efeae2; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #d9fdd3;
        border-radius: 12px;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #ffffff;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Chat con Mamá")

# 5. Historial de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🔄 Reiniciar Conversación"):
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6. Captura de Entrada e Inferencia Vía API
if prompt := st.chat_input("Escribe un mensaje..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # Construcción del contexto de mensajes
        messages_api = [
            {
                "role": "system", 
                "content": "Eres la mamá de Josué. Responde utilizando su vocabulario habitual, tono cariñoso, modismos y brevedad característica de WhatsApp."
            }
        ]
        
        for m in st.session_state.messages:
            messages_api.append({"role": m["role"], "content": m["content"]})

        with st.spinner("Escribiendo..."):
            try:
                # Llamada remota a la API sin consumir RAM local
                response = client.chat_completion(
                    messages=messages_api,
                    max_tokens=80,
                    temperature=0.4,
                    top_p=0.85
                )
                
                respuesta = response.choices[0].message.content.strip()

                # Filtro post-procesamiento de saludos para turnos avanzados
                respuestas_previas = [m for m in st.session_state.messages if m['role'] == 'assistant']
                if len(respuestas_previas) > 0:
                    muletillas = [
                        "Epa maijo. Jehová te bendiga.", "Epa maijo. Jehová te bendiga",
                        "Epa maijo.", "Epa maijo", 
                        "Jehová te bendiga.", "Jehová te bendiga"
                    ]
                    for muletilla in muletillas:
                        if respuesta.startswith(muletilla):
                            respuesta = respuesta[len(muletilla):].strip()

                if not respuesta:
                    respuesta = "Por acá todo bien, mi amor. ¿Y tú cómo vas?"

            except Exception as e:
                respuesta = "Ocurrió un pequeño error de conexión con la API. Intenta nuevamente en un momento."

            st.write(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})