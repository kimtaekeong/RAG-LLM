import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from datetime import datetime

# ⭐️ Embedding 설정
USE_BGE_EMBEDDING = True

if not USE_BGE_EMBEDDING:
    os.environ["OPENAI_API_KEY"] = "OPENAI API KEY 입력"

LANGSERVE_ENDPOINT = "http://localhost/chat/"

# 필수 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 프롬프트 템플릿
RAG_PROMPT_TEMPLATE = """당신은 질문에 친절하게 답변하는 AI 입니다. 검색된 다음 문맥을 사용하여 질문에 답하세요. 답을 모른다면 모른다고 답변하세요.
Question: {question}
Context: {context}
Answer:"""

st.set_page_config(page_title="OLLAMA Local 모델 테스트", page_icon="💬")
st.title("OLLAMA Local 모델 테스트")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]

def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner="Embedding files...")
def embed_files(files):
    retrievers = []
    for file in files:
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            length_function=len,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=text_splitter)

        if USE_BGE_EMBEDDING:
            model_name = "BAAI/bge-m3"
            model_kwargs = {"device": "cuda"}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        else:
            embeddings = OpenAIEmbeddings()

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
        retriever = vectorstore.as_retriever()
        retrievers.append(retriever)

    return retrievers

with st.sidebar:
    files = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt" ],
        accept_multiple_files=True
    )


print("test;",files)

if files:
    retrievers = embed_files(files)

print_history()

if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        ollama = RemoteRunnable(LANGSERVE_ENDPOINT)
        chat_container = st.empty()

        if files:

            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            # 모든 retriever를 결합하여 문맥을 만듭니다.
            combined_docs = []
            for retriever in retrievers:
                retrieved_docs = retriever.get_relevant_documents(user_input)
                combined_docs.extend(retrieved_docs)

            context = format_docs(combined_docs)
            print("문서:", context)

            rag_chain = (
                prompt | ollama | StrOutputParser()
            )
            answer = rag_chain.stream({"question": user_input, "context": context})
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))

        

        else:
            prompt = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )
            chain = prompt | ollama | StrOutputParser()
            answer = chain.stream(user_input)
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))

        current_time = datetime.now().strftime("%Y:%m:%d:%H:%M")

        log_file_path = "dat_log.log"
        with open(log_file_path, 'a' if os.path.exists(log_file_path) else 'w') as log_file:
            log_file.write(f"{current_time} - '{user_input}' \n '{''.join(chunks)}'\n")

