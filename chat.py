import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model data
FILE = "data.pth"
data = torch.load(FILE)

# Extract the model parameters and data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the neural network model and load its state
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    # Tokenize the input message
    sentence = tokenize(msg)
    # Convert the sentence to a bag of words vector
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0]) # Reshape the vector for the model
    X = torch.from_numpy(X).to(device)

    # Pass the input through the model
    output = model(X)
    # Get the predicted tag
    _, predicted = torch.max(output, dim=1)

    # Convert predicted index to tag
    tag = tags[predicted.item()]

    # Calculate the probabilities of each class
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        # If the probability is high enough, return a random response from the predicted tag
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    # Default response if the probability is too low
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # Get user input
        sentence = input("You: ")
        if sentence == "quit":
            break

        # Get the bot's response
        resp = get_response(sentence)
        print(resp)