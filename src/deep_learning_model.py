import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
df = pd.read_csv("processed_data/processed_jobs.csv")

texts = df["cleaned_text"].values
labels = df["fraudulent"].values

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)
# Optimized settings for 8 GB RAM
MAX_WORDS = 10000
MAX_LEN = 200

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)
model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=64,
        input_length=MAX_LEN
    ),
    Bidirectional(LSTM(32)),
    Dropout(0.4),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=16,
    callbacks=[early_stop]
)
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

print("BiLSTM Accuracy:", accuracy_score(y_test, y_pred))
print("BiLSTM F1 Score:", f1_score(y_test, y_pred))
model.save("models/bilstm_model_v1.h5")

import pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
