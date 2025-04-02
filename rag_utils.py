import os
import logging
import pinecone
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.schema import Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Environment setup
os.environ['OPENAI_API_KEY'] = ""
VECTOR_STORAGE_DIR = "./vector_2"
DATA_STORAGE_DIR = "./data_2"
os.makedirs(DATA_STORAGE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORAGE_DIR, exist_ok=True)

def create_chat_engine():
    """Load vectors from storage or create a new index if missing.
       returns a chat engine
    """
    if os.path.exists(VECTOR_STORAGE_DIR):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORAGE_DIR)
            index = load_index_from_storage(storage_context)
            logger.info("Loaded existing vector store index.")
            return index.as_chat_engine()
        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
    
    logger.info("Vector store missing, creating a new index...")
    documents = SimpleDirectoryReader(DATA_STORAGE_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(VECTOR_STORAGE_DIR)
    return index.as_chat_engine()

from pptx import Presentation
import fitz
def load_pptx_text(file_path):
    """Given a pptx file path, extract all the text from it"""
    pres = Presentation(file_path)
    pptx_text = []
    for slide in pres.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text_frame = shape.text_frame
                for paragraph in text_frame.paragraphs:
                    for run in paragraph.runs:
                        pptx_text.append(run.text)
                        
    return pptx_text

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = []
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text.append(page.get_text())
        pdf_document.close()
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
    return "\n".join(text)

def load_any_documents(dirpath:str):
    """Given a directory path, load all the files in it: .pptx, .docx, .txt, .pdf, png"""
    documents = []
    for file_name in os.listdir(dirpath):
        file_path = os.path.join(dirpath, file_name)
        
        if file_name.endswith(".pptx"):
            # handle pptx file
            pptx_text = load_pptx_text(file_path)
            documents.append(Document(text="\n".join(pptx_text), doc_id=file_name))
        elif file_name.endswith(".pdf"):
            pdf_text = extract_text_from_pdf(file_path)
            documents.append(Document(text=pdf_text, doc_id=file_name))
        else:
            documents += (SimpleDirectoryReader(input_files=[file_path]).load_data())
            
    return documents

os.environ["PINECONE_API_KEY"] = ""
api_key = os.environ["PINECONE_API_KEY"]
index_name = "potencia"
pc = Pinecone(api_key=api_key)

index_config = pc.describe_index(index_name)
pinecone_index = pc.Index(index_name)

def upload_vectors_to_pinecone():
    """Upload vectors to a Pinecone index."""
    documents = load_any_documents(DATA_STORAGE_DIR)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    return index
        
def create_chat_engine_pinecone(index_name:str):
    """Create a chat engine using the specified pinecone index name."""
    if index_name in pc.list_indexes().names():
        # Connect to the existing index
        index = pc.Index(index_name)
        print(f"Connected to index '{index_name}'.")
    else:
        print(f"Index '{index_name}' does not exist.")
        
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return index.as_chat_engine()

