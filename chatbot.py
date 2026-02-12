import streamlit as st
import asyncio
import threading
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ESTADO GLOBAL E COMPARTILHADO 
# No Streamlit Cloud, threads diferentes precisam de uma variável de módulo 
# para compartilhar dados fora do session_state.
if "vectorstore_global" not in st.session_state:
    st.session_state.vectorstore_global = None

# Esta é a "ponte" que o Telegram usará para ler o que tem na Web
vectorstore_ponte = None

# CONFIGURAÇÕES DE SEGURANÇA (SECRETS)
PROVEDOR_IA = "NVIDIA"

try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
except Exception as e:
    st.error("🚨 ERRO DE SEGURANÇA: Chaves de API não encontradas!")
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

# FUNÇÕES DE RAG

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
def processar_pdf(uploaded_file):
    global vectorstore_ponte 
    
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp_doc.pdf")
    documents = loader.load()
    
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = get_embeddings_model()
    
    # Lógica para Múltiplos PDFs
    novo_vstore = FAISS.from_documents(chunks, embeddings)
    
    if st.session_state.vectorstore_global is None:
        st.session_state.vectorstore_global = novo_vstore
    else:
        st.session_state.vectorstore_global.merge_from(novo_vstore)
    
    # Sincroniza a ponte para a thread do Telegram
    vectorstore_ponte = st.session_state.vectorstore_global
    return len(chunks)
    
def buscar_informacao(pergunta):
    global vectorstore_ponte
    # Prioriza a ponte para garantir que o Telegram funcione
    vs = vectorstore_ponte or st.session_state.get("vectorstore_global")
    
    if vs is None:
        return None
    
    docs = vs.similarity_search(pergunta, k=15)
    trechos_formatados = []
    for d in docs:
        nome_arquivo = d.metadata.get("source", "Doc")
        pagina = d.metadata.get("page", "?")
        conteudo = d.page_content.replace('\n', ' ') 
        trechos_formatados.append(f"📄 [FONTE: {nome_arquivo} | Pág: {pagina}]\n{conteudo}")
    
    return "\n\n-----------------\n\n".join(trechos_formatados)

# INICIALIZAÇÃO DA IA

def get_llm():
    try:
        if PROVEDOR_IA == "GROQ":
            from langchain_groq import ChatGroq
            return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.5)
        elif PROVEDOR_IA == "NVIDIA":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            return ChatNVIDIA(nvidia_api_key=NVIDIA_API_KEY, model="meta/llama-3.1-405b-instruct", temperature=0.4)
    except Exception as e:
        print(f"Erro na IA: {e}")
        return None

llm_instance = get_llm()
telegram_memory = {}

# ÓGICA DO TELEGRAM 

async def telegram_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global vectorstore_ponte
    user_name = update.effective_user.first_name 
    msg = f"Olá {user_name}! Sou o Cia Agro Bot, estou pronto para ajudar. 🚜"
    
    if vectorstore_ponte or st.session_state.get("vectorstore_global"):
        msg += "\n📚 Manuais carregados! Citarei as fontes nas respostas."
    else:
        msg += "\n⚠️ Nenhum manual carregado na Web. Usarei conhecimento geral."
    
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
            else:
                prompt_final = SYSTEM_PROMPT_BASE + "\n\n(Use seu conhecimento geral de agronomia.)"
            
            msgs = [SystemMessage(content=prompt_final)] + telegram_memory[user_id]
            response = llm_instance.invoke(msgs)
            telegram_memory[user_id].append(AIMessage(content=response.content))
            
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response.content, parse_mode="Markdown")
    except Exception as e:
        print(f"Erro Telegram: {e}")

def run_telegram_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start', telegram_start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), telegram_handle_message))
    app.run_polling(stop_signals=[], drop_pending_updates=True)

@st.cache_resource
def start_background_bot():
    if TELEGRAM_TOKEN:
        t = threading.Thread(target=run_telegram_bot, daemon=True)
        t.start()
        return t
    return None

start_background_bot()

# INTERFACE STREAMLIT
st.set_page_config(page_title="Cia Agro RAG", page_icon="🚜")
st.title("🚜 Cia Agro: Central de Inteligência (RAG)")

st.sidebar.header("📂 Base de Conhecimento")
uploaded_file = st.sidebar.file_uploader("Suba o Boletim Técnico (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processando e Adicionando ao Banco..."):
        num_chunks = processar_pdf(uploaded_file)
    st.sidebar.success(f"✅ PDF Indexado! ({num_chunks} trechos)")

st.markdown("---")

if "web_messages" not in st.session_state: st.session_state["web_messages"] = []

for msg in st.session_state["web_messages"]:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Pergunte algo..."):
    st.chat_message("user").markdown(prompt)
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
