import os
import logging
import gradio as gr
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.schema import Document

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Environment setup
os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY')
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
            return index
        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
    
    logger.info("Vector store missing, creating a new index...")
    documents = SimpleDirectoryReader(DATA_STORAGE_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(VECTOR_STORAGE_DIR)
    return index.as_chat_engine()