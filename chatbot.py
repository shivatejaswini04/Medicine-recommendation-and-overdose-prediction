import os
import re
from flask import Flask, request, render_template
import cohere

# Set your Cohere API key
COHERE_API_KEY = "8YgkYnRpWZgV4nCE9n8mjAzTTxMG4ii3rEp9wR1X"
co = cohere.Client(COHERE_API_KEY)

# Initialize the Flask app
app = Flask(__name__)

# Store chat history
chat_history = []

# Function to clean and trim incomplete sentences
def clean_response(text):
    # Trim the response at the last complete sentence
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > 1:
        return " ".join(sentences[:-1]) + sentences[-1] if text.strip()[-1] in ".!?" else " ".join(sentences[:-1])
    return text

@app.route("/", methods=["GET", "POST"])
def chat():
    global chat_history
    if request.method == "POST":
        user_input = request.form["user_input"]
        if user_input.strip():
            # Add user's message to chat history
            chat_history.append(("You", user_input))

            if user_input.lower() in ["hi", "hello", "hey", "start", "hi!", "hello!"]:
                response = "Hi! What can I help you with today?"
            else:
                # Generate response from Cohere
                result = co.generate(
                    model='command-xlarge',
                    prompt=user_input,
                    max_tokens=400  # Increased from 200 to avoid cut-offs
                )
                raw_response = result.generations[0].text.strip()
                response = clean_response(raw_response)

            # Add bot's response to chat history
            chat_history.append(("Bot", response))
        else:
            chat_history.append(("Bot", "Please ask a valid question."))

    return render_template("chatbot.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(port=5002)
