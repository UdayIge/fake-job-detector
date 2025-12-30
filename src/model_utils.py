import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200

model = tf.keras.models.load_model("models/bilstm_model_v1.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_text(text: str) -> float:
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = model.predict(padded)[0][0]
    return float(prob)
