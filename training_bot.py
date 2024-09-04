
### SHAHMEER AIRLINES CHATBOT! --> A chatbot of my imaginary airlines :) --> Sounds fun!

# NECESSARY IMPORT STATEMENTS
import json
import pickle
import numpy as np
import random
import os
import sys
import warnings
import seaborn
import nltk
import torch
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_tuner import RandomSearch


## A GOOD STEP --> Enhancing the model's compuitng speed!
# Set the device for training:
# - On macOS with Apple Silicon, use Metal Performance Shaders (MPS) if available for hardware acceleration.
# - On other platforms, if a CUDA-enabled GPU is available, use it for GPU acceleration.
# - If neither MPS nor CUDA is available, default to using the CPU.
# This approach ensures optimal performance by utilizing the best available hardware for model training.

device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 
                      'cpu')

print(f"Using device: {device}")


# Set default encoding to utf-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Delaration of some constants... --> file paths!
IGNORE_CHARS = ['?', '!', '.', '>', '<', '=']
INTENTS_FILE_PATH = './intents.json'
WORDS_PICKLE_PATH = './words_collection.pkl'
CLASSES_PICKLE_PATH = './classes.pkl'
MODEL_FILE_PATH = './chatbot_model.keras'
LOG_DIR = './logs'

def load_intents(file_path):
    # Load the intents file from the specified path.
    with open(file_path) as file:
        return json.load(file)


# The following function tokenizes patterns from the intents, building a collection of words, documents (word list with context),
# and a list of unique contexts for later use in training the model.
def tokenize_patterns(intents):
    
    word_collection = []
    documents = []
    contexts = []

    for intent in intents['intents']:
        for pattern in intent['questions']:
            word_list = word_tokenize(pattern)
            word_collection.extend(word_list)
            documents.append((word_list, intent['context']))
            if intent['context'] not in contexts:
                contexts.append(intent['context'])
    
    return word_collection, documents, contexts


# The following function: 1) Lemmatizes the words and filters out any characters specified in ignore_chars.
#                         2) Returns a list of lemmatized words with ignored characters removed.
def lemmatize_and_filter_words(words, ignore_chars):
    lemmatized_filtered_words = []

    for word in words:
        if word not in ignore_chars:
            lemmatized_word = lemmatizer.lemmatize(word.lower())
            lemmatized_filtered_words.append(lemmatized_word)

    return lemmatized_filtered_words


def remove_duplicates_and_sort(items):
    # Remove duplicates from the list and sort it.
    return sorted(set(items))

def save_to_pickle(data, file_path):
    # Save data to a pickle file.
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def process_words_and_contexts(word_collection, all_contexts, ignore_chars):
    """Process the words and contexts to create unique lists and save them."""
    lemmatized_words = lemmatize_and_filter_words(word_collection, ignore_chars)
    unique_words = remove_duplicates_and_sort(lemmatized_words)
    unique_contexts = remove_duplicates_and_sort(all_contexts)
    
    save_to_pickle(unique_words, WORDS_PICKLE_PATH)
    save_to_pickle(unique_contexts, CLASSES_PICKLE_PATH)
    
    return unique_words, unique_contexts

def create_training_data(documents, unique_words, unique_contexts):
    # Create the training data with a collection of words and output vectors...
    training_list = []
    output_empty = [0] * len(unique_contexts)

    for doc in documents:
        collection = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in unique_words:
            collection.append(1 if word in word_patterns else 0)
        output_row = list(output_empty)
        output_row[unique_contexts.index(doc[1])] = 1
        training_list.append([collection, output_row])
    
    # Shuffle the training list multiple times  toreduce the risk of avoid overfitting
    for i in range(0,7):
        random.shuffle(training_list)
    
    # Extracting the features and labels (storing the features + labels to train_X amd train_Y)
    train_X = [item[0] for item in training_list]
    train_Y = [item[1] for item in training_list]
    
    # Convert lists to Numpy arrays for efficient computation (input data for the machine learning model!)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    
    return train_X, train_Y

# Load intents file
intents = load_intents(INTENTS_FILE_PATH)

# Tokenize patterns and prepare data structures
word_collection, documents, all_contexts = tokenize_patterns(intents)

# Process words and contexts to create unique lists
unique_words, unique_contexts = process_words_and_contexts(word_collection, all_contexts, IGNORE_CHARS)

# Create data so that the ML Model can be trained.
trainingdata_X, trainingdata_Y = create_training_data(documents, unique_words, unique_contexts)

# Print the shape of the training data to verify
print("Training X shape:", trainingdata_X.shape)
print("Training Y shape:", trainingdata_Y.shape)


# Next Step:  (Hyperparameter tuning function) 
#  The function builds and compiles a Sequential neural network model for hyperparameter tuning.
#  The model includes multiple Dense layers with varying units, dropout for regularization,
#  and batch normalization for stability. The final layer uses softmax activation for 
#  multi-class classification. The learning rate is selected from a range of options.

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=64, max_value=256, step=64),
                    input_shape=(trainingdata_X.shape[1],), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.40))
    model.add(BatchNormalization())

    model.add(Dense(units=hp.Int('units_2', min_value=64, max_value=128, step=64),
                    activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.40))
    model.add(BatchNormalization())

    model.add(Dense(units=hp.Int('units_3', min_value=32, max_value=64, step=32),
                    activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.40))
    model.add(BatchNormalization())

    model.add(Dense(len(unique_contexts), activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Using Random Searchinorder to find the best combination of the hyper-parameters!
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='chatbot_tuning'
)

tuner.search(trainingdata_X, trainingdata_Y, epochs=50, validation_split=0.15)

# Extracting the optimal hyperparameters ...
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Building the model with the optimal hyperparameters, that we extracted  and then training the model...
model = tuner.hypermodel.build(best_hps)

# Now, we use cross validation technique, inorder to evaluate the model's performance more robustly.
# Cross Validation Technique ensure that the model perfroms well on unseen data rather than just fitting onto the data it was 
# trained on ...
# We would divide the dataset into 10 subsets/folds.
# The model is trained on the training folds and then it's evaluatd on the validation fold.
# Each of the 10 folds takes a turn as a validation (testing) dataset, while the other 9 are used for training!
kf = KFold(n_splits=10, shuffle=True, random_state=40)

for train_index, val_index in kf.split(trainingdata_X):
    X_train, X_val = trainingdata_X[train_index], trainingdata_X[val_index]
    Y_train, Y_val = trainingdata_Y[train_index], trainingdata_Y[val_index]

    history = model.fit(X_train, Y_train, epochs=500, batch_size=32, validation_data=(X_val, Y_val), callbacks=[
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
        ModelCheckpoint(MODEL_FILE_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        TensorBoard(log_dir=LOG_DIR)
    ])

# Last step: Saving the trained model in the file, This file would be loaded inorder to run the model.
model.save(MODEL_FILE_PATH)
print("Model trained and saved to", MODEL_FILE_PATH)


