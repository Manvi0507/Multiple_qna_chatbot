from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain.document_loaders import DirectoryLoader
import matplotlib
import cv2

# Initialize loaders for different document types
pdf_loader = DirectoryLoader("./med_data", glob="**/*.pdf", loader_cls=UnstructuredPDFLoader)
docx_loader = DirectoryLoader("./med_data", glob="**/*.docx")
ppt_loader = DirectoryLoader("./med_data", glob="**/*.pptx")
txt_loader = DirectoryLoader("./med_data", glob="**/*.txt")

# Combine all loaders
loaders = [pdf_loader, docx_loader, ppt_loader, txt_loader]

# Load documents from all loaders
documents = []
for loader in loaders:
    documents.extend(loader.load())


from chromadb import Chroma
from langchain.embeddings import GoogleGenAIEmbeddings

# Initialize the ChromaDB client
vector_store = Chroma()

# Initialize Google GenAI embeddings model
embedding_model = GoogleGenAIEmbeddings()

# Store the documents into ChromaDB with their embeddings
for doc in documents:
    vector_store.add(doc.page_content, embedding_model.embed(doc.page_content))

from langchain.retrievers import EmbeddingRetriever

# Initialize the retriever with the vector store
retriever = EmbeddingRetriever(vector_store=vector_store, embeddings=embedding_model)

def retrieve_documents(query):
    # Retrieve top documents using the retriever
    retrieved_docs = retriever.retrieve(query)
    return retrieved_docs

from langchain.llms import GroqLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize the language model (LLM)
llm = GroqLLM(model="llama-3.1-8b-instant")

# Define a prompt template
prompt_template = PromptTemplate(
    template="Given the following documents: {context}, answer the question: {query}"
)

# Create a RetrievalQA chain
rag_chain = RetrievalQA(llm=llm, retriever=retriever, prompt_template=prompt_template)

from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.langchain import RagasEvaluatorChain

# Define an evaluator chain with chosen metrics
evaluator = RagasEvaluatorChain(metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

def evaluate_response(query, response, context):
    # Evaluate the response using the Ragas evaluator
    return evaluator.evaluate(query=query, response=response, context=context)

from langchain.memory import ConversationBufferMemory

# Initialize memory
memory = ConversationBufferMemory()

# Integrate memory into the RAG chain
rag_chain = RetrievalQA(llm=llm, retriever=retriever, prompt_template=prompt_template, memory=memory)

import streamlit as st

st.title("Multi-Document RAG QnA Chatbot")

# Input field for user query
query = st.text_input("Enter your question:")
if query:
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)
    
    # Generate a response using the RAG chain
    response = rag_chain.run(query=query)
    
    # Display the generated answer
    st.write("Answer:", response)
    
    # Evaluate the response quality
    evaluation = evaluate_response(query, response, retrieved_docs)
    st.write("Evaluation Metrics:", evaluation)


