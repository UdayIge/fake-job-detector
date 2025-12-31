from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt, JWTError

SECRET_KEY = "milestone4_secret_key"
ALGORITHM = "HS256"

security = HTTPBearer()

FAKE_USER = {
    "username": "admin",
    "password": "admin123"
}


import pickle
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_LEN = 200

# Load model and tokenizer with error handling
try:
    model_path = "models/bilstm_model_v1.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

try:
    tokenizer_path = "models/tokenizer.pkl"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    tokenizer = None
    print(f"Error loading tokenizer: {e}")

def predict_text(text: str) -> float:
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or tokenizer not loaded. Please check server logs."
        )
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text input cannot be empty"
        )
    try:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        prob = model.predict(padded, verbose=0)[0][0]
        return float(prob)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}"
        )

app = FastAPI(title="Fake Job Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_token(username: str):
    return jwt.encode({"sub": username}, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
class LoginInput(BaseModel):
    username: str
    password: str

@app.post("/")
def root():
    return {"message": "Welcome to the Fake Job Detection API"}

@app.post("/login")
def login(data: LoginInput):
    if data.username == FAKE_USER["username"] and data.password == FAKE_USER["password"]:
        token = create_token(data.username)
        return {"access_token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

class JobInput(BaseModel):
    description: str

@app.get("/health")
def health_check():
    """Health check endpoint to verify API and model status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/predict")
def predict(job: JobInput, user: str = Depends(verify_token)):
    if not job.description or not job.description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description cannot be empty"
        )
    
    probability = predict_text(job.description)
    label = "Fraudulent" if probability > 0.5 else "Legitimate"

    return {
        "prediction": label,
        "fraud_probability": round(probability, 4),
        "confidence": round(abs(probability - 0.5) * 2, 4)  # Confidence score
    }
