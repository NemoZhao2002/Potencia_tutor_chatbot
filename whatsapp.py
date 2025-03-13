import os
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from rag_utils import create_chat_engine

# Load environment variables from .env if present
load_dotenv()

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """
    This endpoint handles incoming WhatsApp messages sent via Twilio.
    It extracts the message, processes it through the chatbot, and returns
    a response in TwiML format.
    """
    # Retrieve incoming message and sender information
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "")
    print(f"Received message from {sender}: {incoming_msg}")

    # Prepare a Twilio MessagingResponse
    response = MessagingResponse()

    if not incoming_msg:
        response.message("Please send a valid message.")
        return str(response)

    try:
        # Process the incoming message through the chatbot logic
        chat_engine = create_chat_engine()
        rag_response = chat_engine.chat(incoming_msg)
        reply = str(rag_response)
    except Exception as e:
        print(f"Error processing message: {e}")
        reply = "Sorry, there was an error processing your message."

    response.message(reply)
    return str(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)