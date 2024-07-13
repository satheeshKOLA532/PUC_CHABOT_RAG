import os
import numpy as np
from pymongo import MongoClient
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import quote_plus
from groq import Groq
import datetime
import pytz,json,re,logging
from langchain.memory import ConversationBufferMemory
import sys
from dotenv import load_dotenv
from dotenv import dotenv_values
# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mongo_db_connection import mongo_connection
def load_dotenv_with_encoding(dotenv_path, encoding='utf-8'):
    with open(dotenv_path, 'r', encoding=encoding) as f:
        config = dotenv_values(stream=f)
    for key, value in config.items():
        os.environ[key] = value

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Set up logging configuration
logging.basicConfig(filename='qa_bot.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Initialize GROQ client with API key
groq_client = Groq(api_key=os.getenv("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"))

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        # Initialize HuggingFace embeddings
        self.model_name = model_name
        self.model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def generate_embeddings(self, text):
        return self.embeddings.embed_query(text)

#instance object for the embedding model class    
embedding_model = EmbeddingModel()

# collection = mongo_connection.get_collection("schemes_test_total_NEW")
collection = mongo_connection.get_collection("PUC_BIOLOGY")

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) #calculates the dot product and  calculate the L2 norm (Euclidean norm) of each vector, which represents its magnitude in the high-dimensional space.
#Dividing the dot product by the product of the norms normalizes the score to fall between -1 and 1.
#A higher cosine similarity value (closer to 1) indicates a smaller angle between the vectors, implying greater similarity between the query and the document embedding.

# Function to retrieve relevant documents
def retrieve_documents(query, top_k=5):
    print(f"generating embeddings...")
    query_embedding = embedding_model.generate_embeddings(query)
    documents = []
    for scheme in collection.find():
        similarity = cosine_similarity(query_embedding, scheme["embedding"])
        documents.append((scheme, similarity))
    documents.sort(key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in documents[:top_k]]

# for RAG
custom_prompt_template = """Consider the following  NCERT 11 th standard biology textbook which contains the information about , “”””human physiology and chapter BREATHING AND EXCHANGE OF GASES””” document:
Consider the following textbook document:
{context}

###Task:You are a powerful knowledge base(RAG) which has complete and clear information about  NCERT 11 th standard biology textbook which contains the information about , “”””human physiology and chapter name BREATHING AND EXCHANGE OF GASES”””. Now your role is to imitate the behaviour of the tutor. The user will ask you information about anything that was related to this chapter “””human Physiology your job is to always retrieve the most relevant information form the context and give that response to the user.””” Follow the below ###Rules, while responding to the user it was mandatory to follow the below ###Rules.
###Rules:
→If the user says hi say hi and how can I help you today? or if user says  how are you then respond to the user similarly in your own way.
→If the user asks a question that is present in the context then always fetch the most relevant information and give that response to the user.
→If the user doesn’t know the exact question to ask, the user will ask the question then understand the question and retrieve the information and respond to the user according to it.
→If the user asks you any question that was not a part of the chapter then respond I don’t have relevant information to the given question.
→And for every question that the user will ask, follow up your response with a relevant question.
→Use chat history to carry the context it means when the user asks the second question which is related to the first question but he doesn’t mention properly in that condition use chat history to respond to the user question.


Question: {question}

Chat history:
{history}


Answer:
"""


prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
# Initialize ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="history", input_key="question")

# Function to run the QA
def run_qa(message):
    relevant_documents = retrieve_documents(message)
    context = ""
    for doc in relevant_documents:
        context += f"Text: {doc['text']}\n\n"
    print("contexualized text from Retrieval:", context)
    history = memory.load_memory_variables({}).get('history', '')
    print("history before sending to the llm:", history)
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": custom_prompt_template.format(context=context, question=message, history=message[1:])}],
            model="Llama3-70b-8192",
            max_tokens=2000,
            temperature=0.5,
            top_p=1,
            stream=False
        )
        # print("Chat Completion:", chat_completion)
        response = chat_completion.choices[0].message.content
        logging.info(f"response from RAG:{response}")
        memory.save_context({"question": message}, {"response": response})
        return response
    except Exception as e:
        print("Exception occurred:", e)
        return f"Error: {e}"
if __name__ == "__main__":
    queries = [
        "What are the documents required for the kcr kit scheme?"
    ]

    for query in queries:
        result = run_qa(query, "gemma-7b-it")
        print(f"Query: {query}\nResult: {result}\n")

