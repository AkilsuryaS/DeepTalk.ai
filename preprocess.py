from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  
import os

def load_document():
    pdf_path = "d2l.pdf"
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_and_save():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Load and process the document
    print("Loading document...")
    text = load_document()
    
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)
    
    print("Generating embeddings and creating FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create FAISS vector store
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    
    # Save the FAISS vector store
    print("Saving processed data...")
    vectorstore.save_local("data/faiss_db")  # Directory to save the FAISS database
    
    print("Preprocessing complete! Data saved in the 'data' directory.")

if __name__ == "__main__":
    process_and_save()