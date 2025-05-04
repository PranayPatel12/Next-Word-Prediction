# 🧠 Next Word Prediction Using RNN

This project implements a next word prediction model using Recurrent Neural Networks (RNN) and word embeddings. Given a sequence of words, the model predicts the most probable next word based on a trained corpus.

## 📚 Overview

The project uses the following components:
- **Text preprocessing** with tokenization and padding
- **RNN model** with two `SimpleRNN` layers
- **Embedding layer** to convert words into dense vectors
- **Temperature sampling** for more diverse text generation

## 🛠️ Features

- Preprocess and tokenize a custom text corpus
- Build a sequential RNN model for next word prediction
- Use temperature-controlled sampling for generating next words
- Easy integration for generating custom text sequences

## 🧾 How It Works

1. Load a text file (e.g., literary works) as the training corpus.
2. Tokenize the text and create padded input sequences.
3. Train an RNN model using one-hot encoded labels.
4. Predict the next word based on a given input seed text.

## 🧪 Example

```python
seed_input = "What are"
print(generate_text(seed_input, next_words=1, temperature=0.8))
```


📂 Folder Structure
.
├── NextWordPrediction.py    # Main Python script
└── JamiesonSean.txt         # Text corpus used for training (customizable)


🧰 Requirements
  Python 3.x
  TensorFlow 2.x
  NumPy

  Install dependencies with:
  pip install tensorflow numpy


🧠 Model Architecture
  Embedding Layer (100-dimensional)
  SimpleRNN Layer (256 units, return sequences)
  SimpleRNN Layer (256 units)
  Dense Layer (Softmax activation for prediction)


🔁 Training
  Model is trained for 10 epochs with a batch size of 128. You can modify the corpus or training parameters in the script.

📈 Output
  Once trained, the model can predict the next word for a given input phrase. It supports temperature tuning to control prediction randomness.

.

📝 Customization
  To use your own text corpus:
    Replace the path in corpus_text = load_corpus('path_to_your_text.txt')
    Optionally, uncomment and use the combine_texts_from_folder function for multiple files.

📌 Notes
  The script currently uses a single .txt file as input.
  For large datasets, training time may increase. GPU acceleration is recommended.
  Model saving/loading is not yet implemented but can be added with model.save() and tf.keras.models.load_model().

