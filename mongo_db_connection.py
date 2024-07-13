from pymongo import MongoClient
from urllib.parse import quote_plus
import streamlit as st

uname = st.secrets['un']
password = st.secrets['pwd']
class MongoDBConnection:
    def __init__(self, username, password, cluster_uri, dbname):
        self.username = quote_plus(username)
        self.password = quote_plus(password)
        self.cluster_uri = cluster_uri
        self.dbname = dbname
        self.uri = f"mongodb+srv://{self.username}:{self.password}@{self.cluster_uri}/?retryWrites=true&w=majority&appName=PUC-1"
        self.client = MongoClient(self.uri)
        self.db = self.client[self.dbname]

    def get_collection(self, collection_name):
        return self.db[collection_name]

# Usage example:
# Create a global instance of the MongoDBConnection class
mongo_connection = MongoDBConnection(
    username=uname,
    password=password,
    cluster_uri="puc-1.pi4o1jt.mongodb.net",
    dbname="NCERT_PUC"
)
