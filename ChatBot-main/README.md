# Chatbot Project

This is a simple chatbot built using Python, TensorFlow, and NLTK. The chatbot can understand user inputs and respond based on predefined intents. 

## Features
- Natural language processing to understand user inputs
- Machine learning model to predict responses
- Simple command-line interface for user interaction

## Technologies Used
- Python
- TensorFlow
- NLTK
- NumPy
- JSON

## Project Structure
. ├── intents.json # JSON file containing intents and responses ├── MyChatBot.py # Script to train the chatbot model ├── run.py # Script to run the trained chatbot ├── chatbot_model.h5 # Saved model file after training └── README.md # Project documentation


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Abootlha/ChatBot.git
   cd ChatBot
   
2. Install Dependencies: Make sure you have Python 3.x installed. Then, install the required libraries:
   pip install tensorflow nltk numpy
   
   ```bash
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   
3. Training the Chatbot
   Prepare your intents: Create or modify the intents.json file to include patterns and responses for your chatbot.

   Train the model: Run the training script:
   ```bash
   python MyChatBot.py
   ``` 
   This script will preprocess the intents, create a training dataset, and train the neural network model, saving it as chatbot_model.h5.
   

4. Running the Chatbot
   Once the model is trained, you can interact with your chatbot using the following command:

   ```bash
   python run.py
   ```
   Type your messages in the terminal, and the chatbot will respond based on the trained model. Type "exit" to stop the conversation.

