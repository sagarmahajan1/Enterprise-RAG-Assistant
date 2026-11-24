# Enterprise Document Intelligence & RAG Assistant

An AI-powered PDF question-answering application built using Retrieval-Augmented Generation (RAG). It allows users to upload a PDF document and ask questions based on its content.

## 🚀 Live Demo

**Try the Application:** [Enterprise RAG Assistant](https://enterprise-rag-assistant-ytrqjag8wrb32yqucmad9n.streamlit.app/)

**GitHub Repository:** [Enterprise-RAG-Assistant](https://github.com/sagarmahajan1/Enterprise-RAG-Assistant)

## 📌 Project Overview

This application extracts text from PDF documents, splits the extracted text into smaller chunks, converts the chunks into vector embeddings, and stores them in a FAISS vector database.

When a user asks a question, the application retrieves the most relevant document chunks and passes them to Google's Gemini LLM to generate a context-grounded answer.

## ✨ Features

- Upload and process PDF documents.
- Extract text using PyPDFLoader.
- Split text into chunks using RecursiveCharacterTextSplitter.
- Generate embeddings using Hugging Face.
- Store and retrieve vectors using FAISS.
- Generate answers using Google Gemini.
- Display source page references.
- Interactive chat interface using Streamlit.
- Maintain chat history during the active session.
- Clear chat history when required.

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- Hugging Face Sentence Transformers
- FAISS Vector Database
- PyPDF
- Python Dotenv

## 🧠 RAG Architecture

1. **Document Loading:** Extract text from the uploaded PDF using PyPDFLoader.
2. **Text Chunking:** Divide extracted text into smaller chunks using RecursiveCharacterTextSplitter.
3. **Embedding Generation:** Convert text chunks into numerical vector representations using Hugging Face embeddings.
4. **Vector Storage:** Store embeddings in the FAISS vector database.
5. **Semantic Retrieval:** Retrieve the top 3 relevant chunks based on the user's question.
6. **LLM Response:** Pass the retrieved context and user question to Google Gemini to generate a context-grounded answer.

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sagarmahajan1/Enterprise-RAG-Assistant.git
cd Enterprise-RAG-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv env
```

### 3. Activate the Environment

For Windows:

```bash
env\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Google Gemini API Key

Create a `.env` file in the project root directory and add your API key:

```env
GOOGLE_API_KEY=your_google_api_key
```

Replace `your_google_api_key` with your actual Google Gemini API key.

**Note:** Never upload your actual API key or `.env` file to GitHub.

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 📂 Project Structure

```text
Enterprise-RAG-Assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env                 # Create locally; not uploaded to GitHub
│
