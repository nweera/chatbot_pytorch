# AI Chatbot using PyTorch and Flask

A neural network-based conversational agent built with PyTorch and Flask, featuring a web-based chat interface. The chatbot uses natural language processing to understand user intents and provide relevant responses.

## Features

- Neural network-based intent classification
- Web-based chat interface
- Real-time response generation
- Pre-trained model for common queries
- Support for multiple conversation topics
- Easy to extend with new intents

## Technologies Used

- **Backend**:
  - Python 3.x
  - PyTorch (Neural Network)
  - Flask (Web Server)
  - NLTK (Natural Language Processing)

- **Frontend**:
  - HTML
  - CSS
  - JavaScript

## Project Structure

```
chatbot_pytorch/
├── app.py              # Flask server implementation
├── chat.py             # Chat logic and model integration
├── intents.json        # Training data and response patterns
├── model.py            # Neural network model architecture
├── nltk_utils.py       # Natural language processing utilities
├── train.py            # Model training script
├── data.pth            # Trained model data (generated after training)
└── templates/          # Frontend templates
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nweera/chatbot_pytorch.git
cd chatbot_pytorch
```

2. Install the required packages:
```bash
pip install torch flask nltk
```

3. Install NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Usage

1. Train the model (if not using pre-trained model):
```bash
python train.py
```

2. Start the Flask server:
```bash
python app.py
```

3. Open your web browser and navigate to `http://localhost:5000`

## Training Custom Intents

The chatbot can be customized by modifying the `intents.json` file. The file structure follows this format:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello!", "Hi there!", "Hey!"]
    }
  ]
}
```

After modifying the intents, retrain the model using `train.py`.

## Model Architecture

The chatbot uses a feedforward neural network with:
- Input layer: Size based on bag-of-words vocabulary
- Hidden layer: 8 neurons
- Output layer: Number of intent classes
- ReLU activation functions between layers

## API Endpoints

- `GET /` - Serves the chat interface
- `POST /predict` - Accepts messages and returns bot responses
  - Request body: `{"message": "your message here"}`
  - Response: `{"answer": "bot response here"}`
## Inspiration

This chatbot project is based on the YouTube playlist Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial, which guided the development of the project. The tutorial provides step-by-step instructions for implementing NLP techniques and deep learning models to create a chatbot.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

