import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
            # Compara con la variable configurada en Secrets
            if pwd_input == st.secrets.get("APP_PASSWORD", "mi_clave_secreta"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta")
        return False
    return True

if not verificar_password():
    st.stop()

# 3. Carga del Modelo en Caché (Para no recargarlo en cada mensaje)
@st.cache_resource
def cargar_modelo():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_repo_id = "josuealbelaez/qwen2.5-1.5b-estilo-mama"

    # Obtener token de Hugging Face desde secrets
    hf_token = st.secrets.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(lora_repo_id, token=hf_token)

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        token=hf_token
    )

    # Cargar adaptador LoRA privado usando el token
    model = PeftModel.from_pretrained(base_model, lora_repo_id, token=hf_token)
    model.eval()

    return tokenizer, model

with st.spinner("Cargando el cerebro de Mamá... esto tomará unos segundos la primera vez."):
    tokenizer, model = cargar_modelo()

# 4. Estilos CSS para simular diseño tipo WhatsApp
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

# 5. Gestión del Historial de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Botón para reiniciar conversación
if st.sidebar.button("🔄 Reiniciar Conversación"):
    st.session_state.messages = []
    st.rerun()

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6. Captura de Entrada e Inferencia
if prompt := st.chat_input("Escribe un mensaje..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generación de la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Escribiendo..."):
            system_prompt = (
                "<|im_start|>system\n"
                "Eres la mamá de Josué. Responde utilizando su vocabulario habitual, "
                "tono cariñoso, modismos y brevedad característica de WhatsApp.\n"
                "<|im_end|>\n"
            )

            prompt_completo = system_prompt
            for m in st.session_state.messages:
                prompt_completo += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            prompt_completo += "<|im_start|>assistant\n"

            device = next(model.parameters()).device
            inputs = tokenizer(prompt_completo, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    min_new_tokens=5,
                    max_new_tokens=80,
                    temperature=0.4,
                    top_p=0.85,
                    top_k=30,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][input_length:]
            respuesta = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Filtro de muletillas repetidas en turnos avanzados
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

            st.write(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})