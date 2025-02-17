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

def load_vectors():
    """Load vectors from storage or create a new index if missing."""
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
    return index

# create chat engine
chat_engine = load_vectors().as_chat_engine()

# Gradio Function
def query(selected_option, custom_input):
    """
    Handles user input, processes through ChatEngine, and returns response.
    """
    input_query = custom_input.strip() if custom_input.strip() else selected_option
    response = chat_engine.chat(f"{input_query}, format the response")
    return response.response  # Extract text response

# Define Inputs for Gradio UI
inputs = [
    gr.Dropdown(
        choices=[
            "How much do the tutors get paid?",
            "What is the recommended structure for a tutoring session?",
            "Generate a lesson plan on irregular verbs for a beginner English class, include examples",
            "What forms of support does Potencia offer to tutors during the semester?"
        ],
        label="Choose a question"
    ),
    gr.Textbox(lines=1, placeholder="Type your own question here", label="Custom Input")
]

# Gradio Interface
interface = gr.Interface(
    fn=query,
    inputs=inputs,
    outputs=gr.Textbox(label="Response"),
    title="ChatEngine Interface",
    description="Ask questions and get responses using a single ChatEngine."
)

# Launch the Gradio App
interface.launch()
