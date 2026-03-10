import streamlit as st
import asyncio
import threading
import tempfile
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# CONFIGURAÇÕES DE SEGURANÇA (SECRETS)
PROVEDOR_IA = "NVIDIA"

try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
except Exception:
    st.error("🚨 ERRO DE SEGURANÇA: Chaves de API não encontradas!")
    st.info("""
    Para corrigir:
    1. Localmente: Crie `.streamlit/secrets.toml`
    2. Streamlit Cloud: Settings > Secrets
    """)
    st.stop()

SYSTEM_PROMPT = """
Você é o Assistente Virtual Oficial do projeto 'Cia Agro'.
Sua persona: Um agrônomo especialista e prestativo.
Seu objetivo: Ajudar usuários com dúvidas sobre o projeto e agricultura geral.

REGRAS IMPORTANTES:
- Se um CONTEXTO DE DOCUMENTOS for fornecido abaixo, priorize ABSOLUTAMENTE essas informações.
- Ao usar informações do contexto, cite a fonte (nome do arquivo e página).
- Se a informação NÃO estiver no contexto, avise o usuário e responda com seu conhecimento geral.
- Seja conciso, técnico e use emojis relevantes.
"""

# ESTADO GLOBAL COMPARTILHADO ENTRE THREADS
# Usamos um dicionário mutável como "ponte" thread-safe
_shared_state = {
    "vectorstore": None,       # FAISS vectorstore acumulado
    "pdf_names": [],           # Lista de todos os PDFs indexados
    "lock": threading.Lock()   # Lock para acesso seguro entre threads
}


# INICIALIZAÇÃO DA IA
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
        st.error(f"Erro ao inicializar IA: {e}")
        return None

llm_instance = get_llm()


# FUNÇÕES RAG — ACUMULAÇÃO DE PDFs
def get_embeddings_model():
    """Retorna o modelo de embeddings (cached para não recarregar)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def _cached_embeddings():
    """Singleton do modelo de embeddings — carrega uma única vez."""
    return get_embeddings_model()


def add_pdf_to_vectorstore(pdf_file) -> tuple[bool, str]:
    """
    Processa um PDF e ADICIONA ao vectorstore global acumulado.
    Retorna (sucesso: bool, mensagem: str).
    
    FIX PRINCIPAL: Em vez de substituir o vectorstore, usamos
    vectorstore.merge_from() para mesclar novos chunks aos existentes.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    pdf_name = pdf_file.name

    # Verifica se este PDF já foi indexado
    with _shared_state["lock"]:
        if pdf_name in _shared_state["pdf_names"]:
            return False, f"⚠️ '{pdf_name}' já está indexado."

    try:
        # Salva o PDF em arquivo temporário para o PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        # Carrega e divide o PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Adiciona metadado de nome original em cada chunk
        for page in pages:
            page.metadata["source_name"] = pdf_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(pages)

        os.unlink(tmp_path)  # Remove arquivo temporário

        if not chunks:
            return False, f"❌ Nenhum texto extraído de '{pdf_name}'."

        embeddings = _cached_embeddings()

        with _shared_state["lock"]:
            if _shared_state["vectorstore"] is None:
                # Primeiro PDF: cria o vectorstore do zero
                _shared_state["vectorstore"] = FAISS.from_documents(chunks, embeddings)
            else:
                # PDFs seguintes: MESCLA ao vectorstore existente
                new_vs = FAISS.from_documents(chunks, embeddings)
                _shared_state["vectorstore"].merge_from(new_vs)

            _shared_state["pdf_names"].append(pdf_name)

        return True, f"✅ '{pdf_name}' indexado! ({len(chunks)} chunks)"

    except Exception as e:
        return False, f"❌ Erro ao processar '{pdf_name}': {e}"


def query_rag(question: str, k: int = 15) -> str:
    """
    Consulta o vectorstore global e retorna contexto formatado.
    Thread-safe: usa lock para leitura segura.
    """
    with _shared_state["lock"]:
        vs = _shared_state["vectorstore"]

    if vs is None:
        return ""

    try:
        docs = vs.similarity_search(question, k=k)
        if not docs:
            return ""

        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source_name", "Desconhecido")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Fonte: {source} | Página: {page + 1}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        print(f"Erro RAG query: {e}")
        return ""


def build_messages_with_rag(question: str, history: list) -> list:
    """
    Monta a lista de mensagens para o LLM, injetando o contexto RAG
    como parte da mensagem do usuário.
    """
    context = query_rag(question)

    if context:
        augmented_question = (
            f"CONTEXTO DE DOCUMENTOS (use como fonte primária):\n"
            f"{context}\n\n"
            f"---\n\n"
            f"PERGUNTA DO USUÁRIO: {question}"
        )
    else:
        augmented_question = question

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages += history  # histórico de turns anteriores
    messages.append(HumanMessage(content=augmented_question))
    return messages


# TELEGRAM BOT
telegram_memory: dict[int, list] = {}


async def telegram_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_name = update.effective_user.first_name
    telegram_memory[update.effective_user.id] = []

    with _shared_state["lock"]:
        pdf_count = len(_shared_state["pdf_names"])
        pdf_list = _shared_state["pdf_names"]

    if pdf_count > 0:
        pdf_info = f"\n📚 Base de conhecimento ativa: {pdf_count} PDF(s)\n• " + "\n• ".join(pdf_list)
    else:
        pdf_info = "\n📭 Nenhum PDF indexado ainda."

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            f"Olá {user_name}! 🚜\n"
            f"Sou o Assistente Cia Agro com RAG integrado!{pdf_info}\n\n"
            f"Pode perguntar sobre qualquer documento da base!"
        )
    )


async def telegram_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /status — mostra PDFs disponíveis no RAG."""
    with _shared_state["lock"]:
        pdf_names = list(_shared_state["pdf_names"])

    if pdf_names:
        lista = "\n".join(f"• {n}" for n in pdf_names)
        msg = f"📚 *Base de Conhecimento Ativa ({len(pdf_names)} PDFs):*\n{lista}"
    else:
        msg = "📭 Nenhum PDF indexado. Acesse o painel web para fazer upload."

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode="Markdown"
    )


async def telegram_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    # Inicializa memória do usuário se necessário
    if user_id not in telegram_memory:
        telegram_memory[user_id] = []

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        if not llm_instance:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ Erro: LLM não inicializado."
            )
            return

        # Histórico sem a pergunta atual
        history = telegram_memory[user_id][-8:]  # últimas 4 turns (8 msgs)

        # Monta mensagens com contexto RAG
        messages = build_messages_with_rag(text, history)
        response = llm_instance.invoke(messages)
        reply = response.content

        # Atualiza memória (sem o contexto RAG para não inflar o histórico)
        telegram_memory[user_id].append(HumanMessage(content=text))
        telegram_memory[user_id].append(AIMessage(content=reply))

        # Limita histórico a 20 mensagens
        if len(telegram_memory[user_id]) > 20:
            telegram_memory[user_id] = telegram_memory[user_id][-20:]

        # Envia resposta com fallback de formatação
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=reply,
                parse_mode="Markdown"
            )
        except Exception:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=reply
            )

    except Exception as e:
        print(f"Erro Telegram handler: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"⚠️ Erro ao processar sua mensagem: {str(e)[:100]}"
        )


def run_telegram_bot():
    """Roda o bot do Telegram em thread separada com seu próprio event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", telegram_start))
    app.add_handler(CommandHandler("status", telegram_status))
    app.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), telegram_handle_message)
    )

    print("🤖 Telegram Bot iniciado em Background...")
    app.run_polling(stop_signals=None, drop_pending_updates=True)


@st.cache_resource
def start_background_bot():
    if TELEGRAM_TOKEN:
        t = threading.Thread(target=run_telegram_bot, daemon=True)
        t.start()
        return t
    return None

start_background_bot()


# INTERFACE STREAMLIT
st.set_page_config(page_title="Cia Agro — Painel RAG", page_icon="🚜", layout="wide")
st.title("🚜 Cia Agro: Central de Inteligência Agronômica")

col_status, col_model = st.columns(2)
with col_status:
    st.success("✅ Servidor ativo! Telegram Bot rodando em background.")
with col_model:
    st.caption("🧠 Backend: NVIDIA NIM | Modelo: Llama 3.1 405B Instruct")

st.markdown("---")

# SIDEBAR: Upload de PDFs
with st.sidebar:
    st.header("📂 Base de Conhecimento (RAG)")

    uploaded_files = st.file_uploader(
        "Envie um ou mais PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Os PDFs são ACUMULADOS — cada novo upload é adicionado à base existente."
    )

    if uploaded_files:
        for pdf_file in uploaded_files:
            # Evita reprocessar arquivos já carregados nesta sessão
            session_key = f"indexed_{pdf_file.name}_{pdf_file.size}"
            if session_key not in st.session_state:
                with st.spinner(f"Indexando {pdf_file.name}..."):
                    success, msg = add_pdf_to_vectorstore(pdf_file)
                st.write(msg)
                if success:
                    st.session_state[session_key] = True

    st.markdown("---")
    st.subheader("📚 PDFs Indexados")
    with _shared_state["lock"]:
        current_pdfs = list(_shared_state["pdf_names"])

    if current_pdfs:
        for name in current_pdfs:
            st.write(f"✅ {name}")
        st.caption(f"Total: {len(current_pdfs)} documento(s) na base")
    else:
        st.info("Nenhum PDF indexado ainda.")

    if current_pdfs:
        if st.button("🗑️ Limpar Base de Conhecimento", type="secondary"):
            with _shared_state["lock"]:
                _shared_state["vectorstore"] = None
                _shared_state["pdf_names"] = []
            # Limpa flags de sessão
            keys_to_delete = [k for k in st.session_state if k.startswith("indexed_")]
            for k in keys_to_delete:
                del st.session_state[k]
            st.rerun()

# CHAT WEB
st.subheader("💬 Chat Web (Teste e Debug)")

# Mostra indicador de contexto RAG
with _shared_state["lock"]:
    has_rag = _shared_state["vectorstore"] is not None

if has_rag:
    st.info(f"🔍 RAG ativo — respondendo com base em {len(current_pdfs)} PDF(s)")
else:
    st.warning("📭 RAG inativo — respondendo apenas com conhecimento geral da IA")

# Inicializa histórico do chat web
if "web_messages" not in st.session_state:
    st.session_state["web_messages"] = []
if "web_history" not in st.session_state:
    st.session_state["web_history"] = []  # histórico LangChain

# Renderiza mensagens anteriores
for msg in st.session_state["web_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input do usuário
if prompt := st.chat_input("Faça uma pergunta sobre os documentos ou agronomia geral..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["web_messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conhecimento..."):
            if llm_instance:
                messages = build_messages_with_rag(
                    prompt,
                    st.session_state["web_history"]
                )
                resp = llm_instance.invoke(messages)
                reply = resp.content

                # Atualiza histórico LangChain
                st.session_state["web_history"].append(HumanMessage(content=prompt))
                st.session_state["web_history"].append(AIMessage(content=reply))

                # Limita histórico
                if len(st.session_state["web_history"]) > 20:
                    st.session_state["web_history"] = st.session_state["web_history"][-20:]

                st.markdown(reply)
                st.session_state["web_messages"].append(
                    {"role": "assistant", "content": reply}
                )
            else:
                st.error("❌ LLM não inicializado. Verifique as credenciais.")

# DEBUG: Contexto RAG 
with st.expander("🔬 Debug — Ver contexto RAG da última pergunta"):
    if st.session_state["web_messages"]:
        last_user = next(
            (m["content"] for m in reversed(st.session_state["web_messages"])
             if m["role"] == "user"),
            None
        )
        if last_user:
            ctx = query_rag(last_user)
            if ctx:
                st.text_area("Contexto recuperado:", ctx, height=300)
            else:
                st.info("Nenhum contexto RAG recuperado para a última pergunta.")
