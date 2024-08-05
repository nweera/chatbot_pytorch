import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Loop through each intent in the intents JSON
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    # Process each pattern associated with the intent
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag)) # Store the pattern and its tag as a tuple in xy

ignore_words = ['?', '.', '!']
# Stem and lower each word and remove ignored words
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # Convert each pattern sentence to a bag-of-words vector
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag) # Append the bag-of-words vector to X_train
    # Convert the tag to its corresponding index in the tags list
    label = tags.index(tag)
    y_train.append(label)# Append the label to y_train

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0]) # Number of features (length of bag-of-words vector)
hidden_size = 8 # Number of neurons in the hidden layer
output_size = len(tags) # Number of output classes (number of tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        # Initialize the dataset with training data and labels
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        # Return a sample (features and label) at a given index
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,# Number of samples per batch
                          shuffle=True,# Shuffle the data at each epoch
                          num_workers=0) # Number of worker threads to use

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words) # Get model predictions
        loss = criterion(outputs, labels) # Calculate loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print the final loss after training
print(f'final loss: {loss.item():.4f}')

# Save the model state and training parameters to a file
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
