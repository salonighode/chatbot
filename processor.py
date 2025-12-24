import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
import os

# Optional: to disable oneDNN message effects set env var (if you want)
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)   # use punkt (not punkt_tab)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load model and data
model = tf.keras.models.load_model('ds_chatbot_model.h5')
intents = json.loads(open('job_intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model, error_threshold=0.25):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": float(r[1])})
    # If no result above threshold, return an empty list (caller should handle)
    return return_list

def getResponse(ints, intents_json):
    # ints is expected to be a list (possibly empty)
    if not ints:
        # fallback response when model is not confident
        return "Sorry, I didn't understand that. Can you rephrase?"
    tag = ints[0]['intent']
    list_of_intents = intents_json.get('intents', [])
    for i in list_of_intents:
        if i.get('tag') == tag:
            return random.choice(i.get('responses', ["I'm not sure how to respond."]))
    # fallback if tag not found in intents file
    return "I couldn't find a matching response."

def chatbot_response(msg):
    try:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res
    except Exception as e:
        # Log exception and return safe message
        print("Error in chatbot_response:", e)
        return "Something went wrong — please try again later."
