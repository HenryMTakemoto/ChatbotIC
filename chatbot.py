import streamlit as st
import asyncio
import threading
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# CONFIGURAÇÕES DE SEGURANÇA (SECRETS)

# Defina qual IA usar: "NVIDIA" ou "GROQ"
PROVEDOR_IA = "NVIDIA"

# Tenta pegar as chaves do "Cofre" (Secrets) do Streamlit
# Se não encontrar, para o código e avisa o usuário.
try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
except Exception as e:
    st.error("🚨 ERRO DE SEGURANÇA: Chaves de API não encontradas!")
    st.info("""
    Para corrigir:
    1. Se estiver rodando localmente: Crie um arquivo `.streamlit/secrets.toml`
    2. Se estiver no Streamlit Cloud: Vá em Settings > Secrets e cole as chaves lá.
    """)
    st.stop()

SYSTEM_PROMPT = """
Você é o Assistente Virtual Oficial do projeto 'Cia Agro'.
Sua persona: Um agrônomo especialista e prestativo.
Seu objetivo: Ajudar usuários com dúvidas sobre o projeto e agricultura geral.
Regras: Seja conciso, técnico e use emojis.
"""

# INICIALIZAÇÃO DA IA (Função Compartilhada)
def get_llm():
    try:
        if PROVEDOR_IA == "GROQ":
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.3-70b-versatile",
                temperature=0.5
            )
        elif PROVEDOR_IA == "NVIDIA":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            return ChatNVIDIA(
                nvidia_api_key=NVIDIA_API_KEY,
                model="meta/llama-3.1-405b-instruct",
                temperature=0.5,
                max_completion_tokens=1024
            )
    except Exception as e:
        st.error(f"Erro na IA: {e}")
        return None

# Instância global da IA para usar tanto no Site quanto no Telegram
llm_instance = get_llm()

# Memória do Telegram (Global)
telegram_memory = {}

# LÓGICA DO TELEGRAM (Roda em Segundo Plano)

async def telegram_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_name = update.effective_user.first_name
    telegram_memory[update.effective_user.id] = []
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Olá {user_name}! 🚜\nEstou acordado e usando o modelo Llama 3.1 405B! Pode perguntar."
    )

async def telegram_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    
    # Gerencia memória
    if user_id not in telegram_memory: telegram_memory[user_id] = []
    telegram_memory[user_id].append(HumanMessage(content=text))
    if len(telegram_memory[user_id]) > 10: telegram_memory[user_id] = telegram_memory[user_id][-10:]
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        if llm_instance:
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + telegram_memory[user_id]
            response = llm_instance.invoke(msgs)
            
            telegram_memory[user_id].append(AIMessage(content=response.content))
            
            # Tenta enviar (com fallback de formatação)
            try:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response.content, parse_mode="Markdown")
            except:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response.content)
    except Exception as e:
        print(f"Erro Telegram: {e}")

# Função que inicia o loop do Telegram
def run_telegram_bot():
    # Cria um novo loop de eventos para esta thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start', telegram_start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), telegram_handle_message))
    
    print("🤖 Telegram Bot iniciado em Background...")
    app.run_polling()
    app.run_polling(stop_signals=[], drop_pending_updates=True)


# THREAD DE BACKGROUND
@st.cache_resource
def start_background_bot():
    # Só inicia se tivermos o token válido
    if TELEGRAM_TOKEN:
        t = threading.Thread(target=run_telegram_bot, daemon=True)
        t.start()
        return t
    return None

start_background_bot()

# INTERFACE DO SITE (STREAMLIT)
st.set_page_config(page_title="Cia Agro Chat", page_icon="🚜")
st.title("🚜 Cia Agro: Central de Inteligência")

if not TELEGRAM_TOKEN:
    st.error("Configure as Secrets para iniciar o bot!")
else:
    st.success("✅ Servidor Ativo! O Bot do Telegram também deve estar funcionando agora.")
    st.caption("Backend: NVIDIA NIM | Modelo: Llama 3.1 405B (Instruct)")

st.markdown("---")
st.caption("Este site serve como 'cérebro' do bot.")

# CHAT WEB 
if "web_messages" not in st.session_state: st.session_state["web_messages"] = []

for msg in st.session_state["web_messages"]:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Teste a IA por aqui também..."):
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state["web_messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        if llm_instance:
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=m["content"]) for m in st.session_state["web_messages"] if m["role"]=="user"]
            resp = llm_instance.invoke(msgs)
            st.markdown(resp.content)

            st.session_state["web_messages"].append({"role": "assistant", "content": resp.content})
