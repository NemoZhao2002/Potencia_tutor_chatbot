import os
from dotenv import load_dotenv
import logging

# Import the create_chat_engine function from your module
from rag_utils import create_chat_engine_pinecone, upload_vectors_to_pinecone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Ensure the OpenAI API key is set
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        return

    os.environ['OPENAI_API_KEY'] = openai_api_key

    # upload vectors to vector db
    # index = upload_vectors_to_pinecone()
    
    chat_engine = create_chat_engine_pinecone("potencia")
    response = chat_engine.chat("Who are the faculties at Potencia?")
    print(str(response))

if __name__ == "__main__":
    main()
