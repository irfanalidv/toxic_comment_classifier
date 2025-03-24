import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, Dense, Dropout
from typing import List

# === Constants ===
MAXLEN = 100
VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class ToxicCommentClassifier:
    def __init__(self):
        """
        Initializes the classifier by loading tokenizer and model weights.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "model", "model.weights.h5")
        tokenizer_path = os.path.join(base_path, "model", "tokenizer.pickle")

        # Load Tokenizer
        with open(tokenizer_path, "rb") as handle:
            self.tokenizer = pickle.load(handle)

        # Rebuild model architecture (must match training!)
        inp = Input(shape=(MAXLEN,))
        x = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, trainable=False)(inp)
        x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        output = Dense(len(LIST_CLASSES), activation="sigmoid")(x)

        self.model = Model(inputs=inp, outputs=output)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

        # Load weights (trained separately)
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
