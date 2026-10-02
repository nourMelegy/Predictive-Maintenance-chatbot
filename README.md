# Predictive Maintenance Chatbot

A predictive maintenance chatbot built using PyTorch, LangChain, RAG, FAISS, Groq, and Streamlit.

## Project Structure

* `main.py` - Main Streamlit chatbot application.
* `rag.py` - Creates embeddings, builds the FAISS index, and retrieves relevant information.
* `intents.csv` - Dataset used for intent classification.
* `classifier.pt` - Trained PyTorch intent classification model.
* `knowledge_base/` - Text documents used as the chatbot's knowledge base.
* `requirements.txt` - Python packages required to run the project.

## Intents

The chatbot supports four intents:

1. Maintenance Prediction
2. Troubleshooting
3. Maintenance Advice
4. Fault Diagnosis

## Technologies

* Python
* PyTorch
* Streamlit
* LangChain
* Groq
* Sentence Transformers
* FAISS
* Scikit-learn
* Pandas

## Installation

Create and activate a Python virtual environment.

Install the required packages:

```bash
pip install -r requirements.txt
```

## Groq API Key

The chatbot uses a Groq API key.

Set the API key as an environment variable before running the application.

For Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

## Running the Chatbot

From the project folder, run:

```powershell
streamlit run main.py
```

The chatbot will open in the browser.

## Chatbot Pipeline

The chatbot follows this process:

User Message
↓
PyTorch LSTM Intent Classifier
↓
Predicted Intent
↓
LangChain Router
↓
Specialist Agent
↓
RAG Knowledge Retrieval / Tools
↓
Groq LLM
↓
Response

## RAG Knowledge Base

The chatbot uses text documents stored in the `knowledge_base` folder.

The documents are converted into embeddings using the `all-MiniLM-L6-v2` sentence-transformer model and stored in a FAISS index for similarity-based retrieval.

## Conversation Memory

The Streamlit application stores the conversation in session state so previous messages can be included as conversation history.

## Streamlit Interface

The sidebar displays:

* Predicted intent
* Classification confidence
* Selected specialist agent
* Tool calls
* Usage counter
