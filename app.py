
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enterprise Document Intelligence & RAG Assistant")

st.write(
    "Upload your PDF and ask questions based on its content."
)


# ==========================================
# 2. CHAT HISTORY INITIALIZATION
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed" not in st.session_state:
    st.session_state.processed = False


# ==========================================
# 3. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API key not found. Check your .env file.")
    st.stop()


# ==========================================
# 4. PDF UPLOAD
# ==========================================

st.subheader("Upload PDF Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ==========================================
# 5. PROCESS PDF
# ==========================================

if uploaded_file is not None:

    if st.button("Process PDF"):

        with st.spinner("Processing your PDF..."):

            temp_path = None

            try:
                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                # Load PDF
                loader = PyPDFLoader(temp_path)
                documents = loader.load()

                # Text chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                chunks = text_splitter.split_documents(documents)

                if not chunks:
                    st.error("No readable text found in this PDF.")
                    st.stop()

                # Load embedding model
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Create FAISS vector database
                vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                # Store in session
                st.session_state.vectorstore = vectorstore
                st.session_state.processed = True
                st.session_state.chat_history = []
                st.session_state.pdf_name = uploaded_file.name

                st.success("PDF processed successfully!")

                st.write("**PDF Name:**", uploaded_file.name)
                st.write("**Total Pages:**", len(documents))
                st.write("**Total Chunks:**", len(chunks))
                st.write("**Total Vectors:**", vectorstore.index.ntotal)

            except Exception as e:
                st.error(f"Error processing PDF: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# ==========================================
# 6. INITIALIZE GEMINI LLM
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=google_api_key
)


# ==========================================
# 7. CHAT INTERFACE WITH HISTORY
# ==========================================

if st.session_state.get("processed", False):

    st.divider()

    st.subheader("💬 Chat with Your PDF")

    st.caption(
        f"Currently chatting with: {st.session_state.pdf_name}"
    )

    # Display previous messages
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.write(source)

    # User question input
    query = st.chat_input(
        "Ask something about your PDF..."
    )

    if query:

        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Searching document and generating answer..."):

                try:
                    # Retrieve relevant chunks
                    results = st.session_state.vectorstore.similarity_search(
                        query,
                        k=3
                    )

                    # Prepare context
                    context = "\n\n".join(
                        [doc.page_content for doc in results]
                    )

                    # Create prompt
                    prompt = f"""
                    You are an AI assistant.
                    Answer the user's question using only the provided context.

                    If the answer is not available in the context,
                    say that the information is not available in the document.

                    Context:
                    {context}

                    User Question:
                    {query}

                    Answer:
                    """

                    # Gemini response
                    response = llm.invoke(prompt)

                    answer = response.content

                    st.write(answer)

                    # Prepare source references
                    sources = []

                    for doc in results:
                        page_number = doc.metadata.get("page", 0) + 1
                        sources.append(f"Page {page_number}")

                    with st.expander("View Sources"):
                        for source in sources:
                            st.write(source)

                    # Save assistant response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error generating answer: {e}")


# ==========================================
# 8. CLEAR CHAT HISTORY
# ==========================================

if st.session_state.chat_history:

    if st.button("Clear Chat History"):

        st.session_state.chat_history = []
        st.rerun()
