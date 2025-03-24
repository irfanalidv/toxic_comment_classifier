import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from typing import List

# === Constants ===
MAXLEN = 100
LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
VOCAB_SIZE = 20000  # Update this if your training tokenizer used a different size
EMBEDDING_DIM = 128  # Must match the training setup

# === Model Class ===
class ToxicCommentClassifier:
    def __init__(self):
        """
        Initializes the classifier by loading the tokenizer and model weights.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "model", "model.h5")
        tokenizer_path = os.path.join(base_path, "model", "tokenizer.pickle")

        # Load Tokenizer
        with open(tokenizer_path, "rb") as handle:
            self.tokenizer = pickle.load(handle)

        # Rebuild model architecture (must match training!)
        self.model = Sequential([
            Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAXLEN),
            LSTM(64, return_sequences=False),
            Dense(len(LIST_CLASSES), activation="sigmoid")
        ])

        # Load weights only
        self.model.load_weights(model_path)

    def classify(self, text: str) -> dict:
        """
        Classifies a given text and returns toxicity scores.
        """
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAXLEN)
        preds = self.model.predict(padded, verbose=0)[0]
        return {LIST_CLASSES[i]: float(preds[i]) for i in range(len(LIST_CLASSES))}

    def predict(self, text: str) -> float:
        scores = self.classify(text)
        return np.mean(list(scores.values()))

    def predict_batch(self, texts: List[str]) -> List[float]:
        scores_list = [self.classify(text) for text in texts]
        return [np.mean(list(scores.values())) for scores in scores_list]
