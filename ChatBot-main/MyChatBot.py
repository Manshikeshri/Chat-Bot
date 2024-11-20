import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model  # Import Sequential and load_model
from tensorflow.keras.layers import Dense  # Import Dense layer
import tensorflow as tf

# Ensure the GPU is available and configure memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Download the necessary NLTK data files (only needed once)
nltk.download('punkt')
nltk.download('wordnet')

# Load intents from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Prepare training data
words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower, and remove duplicates
ignore_words = ['?','!']
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Initialize the bag of words
    bag = []
    pattern_words = doc[0]
    # Lemmatize each word and create the bag of words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for the current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag + output_row)

# Shuffle the training data and convert to numpy array
random.shuffle(training)
training = np.array(training)

# Split into X (features) and y (labels)
X = training[:, :-len(classes)]
y = training[:, -len(classes):]

# Create the model
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

# Load the model for later use
model = load_model('chatbot_model.h5')

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
