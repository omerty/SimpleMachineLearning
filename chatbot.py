import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Load preprocessed data from pickle files
words = pickle.load(open('words.pkl', "rb"))
classes = pickle.load(open('classes.pkl', "rb"))
model = load_model('chatbotmodel.h5')

# Function to clean up sentences by tokenizing and lemmatizing words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert sentences into bag-of-words representation
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the intent of a sentence using the trained model
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Function to get a random response based on the predicted intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you please rephrase?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "Sorry, I didn't understand that. Can you please rephrase?"

print("Go Bot is Running!")

# Main loop to continuously get user input and provide responses
while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Goodbye!")
        break
    intents1 = predict_class(message)
    response = get_response(intents1, intents)
    print("Bot:", response)
