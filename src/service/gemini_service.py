
from src.app import app


import google.generativeai as genai

def gemini(keyword):
    # Configure the API with your key
    genai.configure(api_key='AIzaSyClbOBrF4jWqdwug_D9Xbd21R2HXNPrxNY')

    # Initialize the generative model (e.g., gemini-pro)
    model = genai.GenerativeModel('gemini-pro')

    # Start a new chat session with an empty history
    chat = model.start_chat(history=[])

    # Send the user's message to the chat model
    response = chat.send_message(keyword)

    # Print the chatbot's response
    return response.text


# If routing.py passes a keyword, call the chatbot with that keyword
# Example usage (assuming routing.py would pass the 'keyword')
if __name__ == "__main__":
    keyword_from_routing = "Hello chatbot!"  # Example keyword
    catbot(keyword=keyword_from_routing)