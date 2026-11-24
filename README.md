# Enterprise Document Intelligence & RAG Assistant

An AI-powered PDF question-answering application built using Retrieval-Augmented Generation (RAG). It allows users to upload a PDF document and ask questions based on its content.

## Project Overview

This application extracts text from PDF documents, splits it into smaller chunks, converts the chunks into vector embeddings, and stores them in a FAISS vector database.

When a user asks a question, the application retrieves the most relevant document chunks and passes them to Google's Gemini LLM to generate a context-grounded answer.

## Features

- Upload and process PDF documents.
- Extract text using PyPDFLoader.
- Split text into chunks using RecursiveCharacterTextSplitter.
- Generate embeddings using Hugging Face.
- Store and retrieve vectors using FAISS.
- Generate answers using Google Gemini.
- Display source page references.
- Interactive chat interface using Streamlit.
- Maintain chat history during the active session.

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- Hugging Face Sentence Transformers
- FAISS
- PyPDF

## RAG Architecture

1. **Document Loading:** Extract text from the uploaded PDF.
2. **Text Chunking:** Divide extracted text into smaller chunks.
3. **Embedding Generation:** Convert text chunks into numerical vector representations.
4. **Vector Storage:** Store embeddings in the FAISS vector database.
5. **Semantic Retrieval:** Retrieve the top 3 relevant chunks based on the user's question.
6. **LLM Response:** Pass retrieved context and the question to Gemini to generate an answer.

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/sagarmahajan1/Enterprise-RAG-Assistant.git
cd Enterprise-RAG-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv env
```

### 3. Activate the environment

Windows:

```bash
env\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY=your_google_api_key
```

Replace `your_google_api_key` with your actual Google Gemini API key.

### 6. Run the application

```bash
streamlit run app.py
```

## Project Structure

```text
Enterprise-RAG-Assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env
│
└── documents/
    └── AI_Data_Science_RAG_Test_Document.pdf
```

## Future Improvements

- Support for multiple PDF documents.
- Persistent vector database storage.
- Improved document retrieval and response quality.
- Application deployment.

## Author

**Sagar Mahajan**

GitHub: https://github.com/sagarmahajan1
