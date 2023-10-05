import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from bs4 import BeautifulSoup
import requests
import pickle
import tiktoken
import os
from langchain.embeddings import OpenAIEmbeddings

urls = [
    "https://www.gov.uk/guidance/student-finance-england-how-to-guide",
    "https://www.gov.uk/guidance/checking-the-status-of-your-student-finance-application",
    "https://www.gov.uk/guidance/updating-your-personal-details",
    "https://www.gov.uk/guidance/guidance-for-students-parents-and-partners-providing-evidence-to-support-a-student-finance-application",
    "https://www.gov.uk/guidance/getting-your-first-student-finance-payment",
    "https://www.gov.uk/guidance/supporting-your-child-or-partners-student-finance-application-in-3-easy-steps",
    "https://studentfinance.campaign.gov.uk/"
]



def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)
pages = []

for i in urls:
    pages.append({'text': extract_text_from(i), 'source': i})


text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs, metadatas = [], []
for page in pages:
    splits = text_splitter.split_text(page['text'])
    docs.extend(splits)
    metadatas.extend([{"source": page['source']}] * len(splits))
    print(f"Split {page['source']} into {len(splits)} chunks")


store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key="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"), metadatas=metadatas)
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)






