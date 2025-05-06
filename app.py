import os
import pandas as pd
import csv
import re
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import streamlit as st

# Write FAQ from text files into CSV
with open('./telecom_faq_General_Account_Management.csv', 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Question', 'Answer'])  # CSV header
    for files_txt in os.listdir('txt_files'):
        if files_txt.endswith('.txt'):
            with open(os.path.join('txt_files', files_txt), 'r', encoding='utf-8') as infile:
                for line in infile:
                    match = re.match(r"^(.*?):\s*(.*?)$", line)
                    if match:
                        question, answer = match.groups()
                        writer.writerow([question.strip(), answer.strip()])
                    else:
                        print(f"Skipping malformed line: {line.strip()}")  # skipped line

# --- Configuration ---
FAQ_FILE = r"C:\Users\dhany\Desktop\telecom_project\telecom_faq_General_Account_Management.csv"
LLM_MODEL = "llama3"  # Or the specific Llama model tag you pulled
EMBEDDING_MODEL = "nomic-embed-text"  # Ollama embedding model tag
VECTORSTORE_DIR = "faq_vectorstore"  # Directory to save vector store

# --- Ensure FAQ File Exists ---
db_location = "./chrome_langchain_db"
add_docs = not os.path.exists(db_location)
if not os.path.exists(FAQ_FILE):
    print(f"Error: FAQ file '{FAQ_FILE}' not found. Please create it with your Q&A pairs.")
    exit()

# Initialize models and embeddings
print(f"Initializing Ollama LLM ({LLM_MODEL})...")
llm = Ollama(model=LLM_MODEL)
print(f"Initializing Ollama Embeddings ({EMBEDDING_MODEL})...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Read the FAQ CSV
df = pd.read_csv(FAQ_FILE)

# Prepare documents and vector store
if add_docs:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(page_content=row["Q"] + " " + row["A"])
        ids.append(i)
        documents.append(document)

# Initialize vector store
if os.path.exists(VECTORSTORE_DIR):
    vectorstore = Chroma(collection_name='v_db', persist_directory=db_location, embedding_function=embeddings)
    print("Loaded existing vector store.")
    if add_docs:
        vectorstore.add_documents(documents=documents, ids=ids)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Change `k` to retrieve a fewer set of documents

# Create prompt template
template = """
You are an expert in answering questions about telecommunication customer care.

Here are some relevant answers: {answer}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chain prompt with the LLM
chain = prompt | llm

# Handle conversation and respond
def handle_conversation(user_input):
    print(f"Received user input: {user_input}")  # Debug print
    # Ensure retriever invocation is correct
    answer = retriever.invoke(user_input)  # Retrieve relevant documents
    print(f"Retrieved answer: {answer}")  # Debug print
    result = chain.invoke({"answer": answer, "question": user_input})  # Use the model to get a response
    print(f"Generated response: {result}")  # Debug print
    return result

# Streamlit app configuration
st.set_page_config(page_title="Tele Customer Care Bot", layout="centered")
st.title("Customer Support ChatBot")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input form
with st.form("Chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask Your Question:", placeholder="Type your question here...")
    submitted = st.form_submit_button("Send")

# Handle user input and generate response
if submitted and user_input:
    st.session_state.messages.append(("user", user_input))
    # Get Bot response
    response = handle_conversation(user_input)
    st.session_state.messages.append(("bot", response))

# Display chat history
for sender, message in st.session_state.messages:
    if sender == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
