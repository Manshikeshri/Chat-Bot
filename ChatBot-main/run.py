import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download the necessary NLTK data files (only needed once)
nltk.download('punkt')
nltk.download('wordnet')

# Load intents from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('chatbot_model.h5')

# Prepare words and classes from the intents
words = []
classes = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower, and remove duplicates
ignore_words = ['?', '!']
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Function to predict response
def chatbot_response(msg):
    # Tokenize and lemmatize the input message
    p_words = nltk.word_tokenize(msg)
    p_words = [lemmatizer.lemmatize(word.lower()) for word in p_words]

    # Create the bag of words for the input message
    bag = [0] * len(words)
    for s in p_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    # Make prediction
    prediction = model.predict(np.array([bag]))[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Set a threshold for confidence
    if confidence > 0.7:
        tag = classes[predicted_class]
        responses = [intent['responses'] for intent in intents['intents'] if intent['tag'] == tag]
        return random.choice(responses[0])
    else:
        return "I'm sorry, I didn't understand that."

# Main loop to chat with the user
if __name__ == "__main__":
    print("Chatbot is ready to talk! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)
