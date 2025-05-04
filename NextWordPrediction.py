import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
import random
# import os

# def combine_texts_from_folder(folder_path):
#     combined_text = ""
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(folder_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 combined_text += file.read().lower().replace('\n', ' ') + " "
#     return combined_text.strip() 

# # Example usage:
# folder_path = "C:\\Users\\HP\\OneDrive\\Documents\\NLP_PROJECT\\charlotte"  # Replace with your folder path
# combined_text = combine_texts_from_folder(folder_path)

# 1. Load and preprocess text data
def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().lower().replace('\n', ' ')

corpus_text = load_corpus('C:\\Users\\HP\\OneDrive\\Documents\\NLP_PROJECT\\JamiesonSean.txt')

# # Save to a new file (optional)
# with open("combined_corpus.txt", "w", encoding='utf-8') as f:
#     f.write(combined_text)


# 2. Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([corpus_text])
total_words = len(tokenizer.word_index) + 1

# 3. Generate input sequences
input_sequences = []
tokens = tokenizer.texts_to_sequences([corpus_text])[0]

for i in range(2, len(tokens)):
    n_gram_sequence = tokens[:i+1]
    input_sequences.append(n_gram_sequence)

# Pad sequences
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

# Split into predictors and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# 4. Define the RNN model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_seq_len - 1))
model.add(SimpleRNN(256, return_sequences=True))
model.add(SimpleRNN(256))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 5. Train the model
history = model.fit(X, y, epochs=10, batch_size=128, verbose=1)

# 6. Predict next word using temperature sampling
def predict_next_word_temp(seed_text, temperature=1.0):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    preds = model.predict(token_list, verbose=0)[0]
    
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)
    predicted_index = np.argmax(probas)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# 7. Generate text
def generate_text(seed_text, next_words=1, temperature=1.0):
    for _ in range(next_words):
        next_word = predict_next_word_temp(seed_text, temperature)
        seed_text += " " + next_word
    return seed_text

# Example usage
seed_input = "What are"
print(generate_text(seed_input, next_words=1, temperature=0.8))
