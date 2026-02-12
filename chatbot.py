import streamlit as st
import asyncio
import threading
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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

SYSTEM_PROMPT_BASE = """
Você é o Assistente Virtual do projeto 'Cia Agro'.
IDENTIDADE: Inteligência Artificial baseada no Llama 3, configurada pela equipe Cia Agro.

INSTRUÇÕES DE RESPOSTA E CITAÇÃO:
1. AO USAR O CONTEXTO: Se você encontrar a resposta nos trechos abaixo, INICIE sua resposta citando a fonte explicitamente.
   Exemplo: "Segundo o documento [Nome do Arquivo] (Pág. X), a recomendação é..."
   
2. CONHECIMENTO GERAL: Se a resposta NÃO estiver no contexto, responda com seu conhecimento de agronomia, mas avise:
   "Essa informação não consta nos documentos carregados, mas geralmente..."

3. NÃO INVENTE: Não crie dados técnicos que não existam no texto.

Seja técnico, conciso e use emojis 🚜.
"""
# Variável Global para armazenar o Banco de Dados Vetorial
vectorstore_global = None

# FUNÇÕES DE RAG (Processamento de PDF)

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def processar_pdf(uploaded_file):
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp_doc.pdf")
    documents = loader.load()
    
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
    
    # Chunk size ajustado para tabelas
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings_model()
    # Salva no estado global persistente
    global_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return len(chunks)

def buscar_informacao(pergunta):
    # Acessa o estado global persistente
    if global_state.vectorstore is None:
        return None
    
    # k=15 (Aumentamos drasticamente a busca para garantir que o dado venha)
    docs = global_state.vectorstore.similarity_search(pergunta, k=15)
    
    trechos_formatados = []
    for d in docs:
        nome_arquivo = d.metadata.get("source", "Doc")
        pagina = d.metadata.get("page", "?")
        conteudo = d.page_content.replace('\n', ' ') 
        trechos_formatados.append(f"📄 [FONTE: {nome_arquivo} | Pág: {pagina}]\n{conteudo}")
    
    return "\n\n-----------------\n\n".join(trechos_formatados)



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
                temperature=0.4,
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
    msg = "Olá {user_name}! Sou o Cia Agro Bot, estou rodando o modelo Llama 3.1. 🚜"
    if vectorstore_global:
        msg += "\n📚 Estou lendo os manuais técnicos! Ao responder, citarei a fonte."
    else:
        msg += "\n⚠️ Nenhum manual carregado. Usando conhecimento geral."
    
    telegram_memory[update.effective_user.id] = []
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

async def telegram_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    
    if user_id not in telegram_memory: telegram_memory[user_id] = []
    telegram_memory[user_id].append(HumanMessage(content=text))
    if len(telegram_memory[user_id]) > 6: telegram_memory[user_id] = telegram_memory[user_id][-6:] 
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        if llm_instance:
            contexto_pdf = buscar_informacao(text)
            
            if contexto_pdf:
                prompt_final = f"{SYSTEM_PROMPT_BASE}\n\nCONTEXTO RECUPERADO DOS MANUAIS:\n{contexto_pdf}"
                print(f"🔍 RAG: Contexto encontrado no PDF.")
            else:
                prompt_final = SYSTEM_PROMPT_BASE + "\n\n(Nenhum documento PDF carregado. Use seu conhecimento geral.)"
            
            msgs = [SystemMessage(content=prompt_final)] + telegram_memory[user_id]
            
            response = llm_instance.invoke(msgs)
            telegram_memory[user_id].append(AIMessage(content=response.content))
            
            try:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response.content, parse_mode="Markdown")
            except:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response.content)
    except Exception as e:
        print(f"Erro Telegram: {e}")

def run_telegram_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start', telegram_start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), telegram_handle_message))
    print("🤖 Telegram Bot iniciado...")
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
st.set_page_config(page_title="Cia Agro RAG", page_icon="🚜")
st.title("🚜 Cia Agro: Central de Inteligência (RAG)")

if not TELEGRAM_TOKEN:
    st.error("Configure as Secrets para iniciar o bot!")
else:
    st.success("✅ Servidor Ativo! O Bot do Telegram está rodando.")

st.sidebar.header("📂 Base de Conhecimento")
uploaded_file = st.sidebar.file_uploader("Suba o Boletim Técnico (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processando PDF (Chunking & Embedding)..."):
        num_chunks = processar_pdf(uploaded_file)
    st.sidebar.success(f"✅ PDF Indexado! ({num_chunks} trechos)")
    st.info(f"Arquivo carregado: {uploaded_file.name}")
else:
    st.sidebar.warning("Nenhum PDF carregado.")

st.markdown("---")

# CHAT WEB 
if "web_messages" not in st.session_state: st.session_state["web_messages"] = []

for msg in st.session_state["web_messages"]:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Pergunte algo..."):
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state["web_messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        if llm_instance:
            contexto = buscar_informacao(prompt)
            with st.expander("🔍 Debug: Ver o que a IA leu no PDF"):
                    st.text(contexto)
            prompt_final = f"{SYSTEM_PROMPT_BASE}\n\nCONTEXTO DO PDF:\n{contexto}" if contexto else SYSTEM_PROMPT_BASE
            
            msgs = [SystemMessage(content=prompt_final)] + [HumanMessage(content=m["content"]) for m in st.session_state["web_messages"] if m["role"]=="user"]
            resp = llm_instance.invoke(msgs)
            st.markdown(resp.content)
            st.session_state["web_messages"].append({"role": "assistant", "content": resp.content})





