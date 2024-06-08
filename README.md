# Chatbot

This project implements a chatbot using natural language processing (NLP) techniques and a trained deep learning model. The chatbot can understand and respond to user queries based on predefined intents and patterns.

## Table of Contents

- [Requirements](#requirements)
- [Files](#files)
- [Training the Model](#training-the-model)
- [Running the Chatbot](#running-the-chatbot)
- [Usage](#usage)
- [Notes](#notes)

## Requirements

Ensure you have the following libraries installed:

```bash
pip install numpy
pip install nltk
pip install tensorflow
pip install pickle-mixin
```

Additionally, download the required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Files

- **intents.json**: Contains the training data with predefined intents and their corresponding patterns and responses.
- **words.pkl**: A pickle file that stores the unique lemmatized words from the training data.
- **classes.pkl**: A pickle file that stores the unique intents from the training data.
- **chatbotmodel.h5**: The trained deep learning model for intent classification.

## Training the Model

To train the chatbot model, run the `train_chatbot.py` script:

```bash
python train_chatbot.py
```

This script will preprocess the data, train the model, and save the trained model as `chatbotmodel.h5`.

## Running the Chatbot

To start the chatbot, run the `chatbot.py` script:

```bash
python chatbot.py
```

This script will load the trained model and start a loop to interact with the user.

## Usage

1. **Training the Model**:
   - Ensure `intents.json` is in the same directory as your script.
   - Run the training script to preprocess the data and train the model.
   - The trained model will be saved as `chatbotmodel.h5`.

2. **Running the Chatbot**:
   - Ensure the trained model (`chatbotmodel.h5`), `words.pkl`, `classes.pkl`, and `intents.json` are in the same directory as your script.
   - Run the chatbot script to start interacting with the bot.
   - Type your message and press enter. To quit, type "quit".

## Notes

- Customize the `intents.json` file to add or modify intents, patterns, and responses according to your requirements.
- Adjust the model architecture and training parameters as needed for improved performance.
- The chatbot uses a simple bag-of-words model for intent classification. Consider exploring more advanced NLP techniques for better accuracy.
