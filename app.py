
### Main application file to run the chatbot and handle interactions.
# This file loads the trained model, processes user inputs, generates predictions, 
# and returns relevant chatbot responses. It serves as the core of the chatbot, 
# enabling the application to interact with users in real-time. This file is crucial 
# for deploying the chatbot and linking the model to user interactions.


## Make the necessary import statements!
from flask import Flask, request, jsonify, render_template
import json
import pickle
import numpy as np
import random
import os
import sys
import nltk
import openai
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

# Initialize Flask app for deploying the user interface :)
app = Flask(__name__)


### IMPORTING AND SECURELY LOADING MY OPENAI API KEY FROM config.py  --> (SECURITY PURPOSES)
### This prevents hardcoding sensitive information in the code and sets the API key for making requests to the OpenAI API ... 
import openai
from config import Shahmeer_SAL_ChatBot_OpenAPI_Key
openai.api_key = Shahmeer_SAL_ChatBot_OpenAPI_Key


# Set default encoding to utf-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Constants
IGNORE_CHARS = ['?', '!', '.', '>', '<', '=']
INTENTS_FILE_PATH = './intents.json'
WORDS_PICKLE_PATH = './words_collection.pkl'
CLASSES_PICKLE_PATH = './classes.pkl'
MODEL_FILE_PATH = './chatbot_model.keras'
LOG_DIR = './logs'


# Load the intents file from the specified path.
# Additionally, load the file that stores the trained model to prepare it for making predictions.
def load_intents(file_path):
    with open(file_path) as file:
        return json.load(file)

#################################################################################################################################


# This function tokenizes and lemmatizes the user's input, then converts it into a bag-of-words 
# ... representation based on the unique words in the dataset. This creates a binary vector
# ... indicating the presence of each word from the vocabulary in the user's input, which is
# ... then returned as a NumPy array for model input.
def preprocess_user_input(user_input, unique_words):
    words = word_tokenize(user_input)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    bag = [1 if word in words else 0 for word in unique_words]
    return np.array([bag])



# This function preprocesses the user's input into a bag-of-words format and uses the trained model
# ...to predict the probability of each class (intent) based on the input. The function
# ...then filters out predictions below the error threshold 0.05 to ignore less confident
# ...predictions. The remaining predictions are sorted by probability in descending order, 
# ...with the most likely intent appearing first. The function returns a list of intents 
# ...that surpass the threshold, along with their associated probabilities.
def predict_class(model, user_input, words, classes, error_threshold=0.05):
    collection = preprocess_user_input(user_input, words)

    # Predict the probabilities for each class
    predictions = model.predict(collection)[0]

    # Filter and sort the predictions based on the error threshold
    results = [{"intent": classes[i], "probability": prob} for i, prob in enumerate(predictions) if prob > error_threshold]
    results.sort(key=lambda x: x["probability"], reverse=True)

    return results


    ### This function generates a list of potential responses based on the predicted intents from the user's input.
    # The function first checks if there are any valid predicted intents. If none are found, it returns an empty list.
    # For each predicted intent, it matches the intent's context with the corresponding context in the intents JSON.
    # Once a match is found, a random response from the available answers is selected and added to the list of 
    # ... predicted answers. The function returns a list of these potential responses for further processing which
    # ... are then fed into the OpenAI's API.
def get_response_from_chatbot(intents_list, intents_json):
    """Get the response based on the predicted class."""
    if len(intents_list) == 0:
        return []

    predicted_intents = []
    predicted_answers = []
    list_of_intents = intents_json['intents'] 
    for i in range(0, len(intents_list)):
        predicted_intents.append(intents_list[i]['intent'])
        tag = intents_list[i]['intent']
        for i in list_of_intents:
            if i['context'] == tag: 
                result = random.choice(i['answers'])
                predicted_answers.append(result)
                break
    return predicted_answers

def load_files():
    # This function loads the intents, words, classes, and model.
    try:
        with open(INTENTS_FILE_PATH) as file:
            intents = json.load(file)
        words = pickle.load(open(WORDS_PICKLE_PATH, 'rb'))
        classes = pickle.load(open(CLASSES_PICKLE_PATH, 'rb'))
        model = load_model(MODEL_FILE_PATH) # loading the pretrained model
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    return intents, words, classes, model

intents_json_file, words, classes, model = load_files()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'reply': "Please provide a valid input."})

    intents = predict_class(model, user_input, words, classes)
    response_list = get_response_from_chatbot(intents, intents_json_file)
    
    if not response_list:
        best_response = "Sorry, I have not been trained to provide an answer to that question. Please ask another question."
    else:
        try:
            # Create the OpenAI chat completion prompt
            prompt = [
                {"role": "system", "content": "You are a highly intelligent assistant for Shahmeer Airline's customer support chatbot. The user has provided a query, and the chatbot has generated multiple potential responses. Your task is to choose the best response and rephrase it in a more welcoming, friendly, and customer-oriented manner, making the user feel valued and appreciated."},
                {"role": "user", "content": f"The user asked: '{user_input}'"},
                {"role": "assistant", "content": "Here are the possible responses generated by the trained chatbot. Please choose the one that best addresses the user's query and enhance it to be more friendly, welcoming, and customer-oriented."},
            ]
            for current_response in response_list:
                prompt.append({"role": "assistant", "content": current_response})
            
            # Add a final instruction to ensure the best and friendliest response is selected
            prompt.append({"role": "system", "content": "Please select the best response from the options provided above, and rephrase it to be more friendly and welcoming to the user. Your goal is to make the user feel valued, welcomed, and appreciated by the airline. Also add a few unique emojis in between depending on what answer you generate."})
            prompt.append({"role": "system", "content": "If the user asks for contact details, the office is located at 3433 Rue Durocher in Montreal. The phone number is +1(514)-713-2441. Shahmeer Airlines is based in Montreal, Canada. Website is www.flywithshahmeerairlines.com, email address is: flywithSAL@mail.jets.ca"})
            prompt.append({"role": "system", "content": "Important: If none of the provided responses match the user's query (if the question is way too different from the available answers), kindly apologize and inform the user that the bot does not have an answer for their question. Politely suggest that they check the contact details for further assistance. Do not attempt to create or generate a new answer on your own."})
            
            # Call OpenAI API
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Using the  GPT-40-mini model (least expensive + efficient!)
                messages=prompt
            )
        
            # Next Step: Extracting  the most suitable response from OpenAI's output, which is selected based on the context
            # ... and relevance to the user's query. This response is then further refined (in wording) to ensure it is the best possible
            # ... answer to address the user's needs in a friendly and effective manner! :)
            best_response = completion.choices[0].message['content'].strip()

        except Exception as e:
            print(f"Error: {e}")
            best_response = "Sorry, I have not been trained to provide an answer to that question. Please ask another question."
    
    return jsonify({'reply': best_response})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
