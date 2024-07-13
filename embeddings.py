import os
import json
from pymongo import MongoClient
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import quote_plus
import sys
import os
# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mongo_db_connection import mongo_connection

# Initialize Hugging Face embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load PDF data
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Load the PDF file

pdf_path = '/home/satheeshbabu/python_projects/EDUCATION_CHATBOT_RAG/Breathing_Exchange_Of_Gases.pdf'
pdf_text = load_pdf(pdf_path)

# Split the text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=32, chunk_overlap=4) 
text_splitter = RecursiveCharacterTextSplitter() 
texts = text_splitter.split_text(pdf_text)

collection = mongo_connection.get_collection("PUC_BIOLOGY")

# Insert each text chunk into MongoDB with its embedding
print("Inserting into MongoDB Atlas.....")
for i, text_chunk in enumerate(texts):
    embedding = embeddings.embed_documents([text_chunk])[0]
    document = {"chunk_id": i, "text": text_chunk, "embedding": embedding}
    collection.insert_one(document)
print(f"Successfully inserted embeddings along with their chunks into {collection} collection")


 