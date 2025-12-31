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
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

MAX_LEN = 200

# Initialize text preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Add custom stop words (matching data_preprocessing.py)
custom_stop_words = {
    'job', 'work', 'company', 'position', 'apply', 'experience',
    'candidate', 'role', 'opportunity', 'team', 'looking'
}
stop_words.update(custom_stop_words)

# Load model and tokenizer with error handling
try:
    model_path = "models/bilstm_model_v1.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path, compile=False)
    # Compile model for inference (matching training configuration)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print("MODEL LOADED SUCCESSFULLY")
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

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data (matching data_preprocessing.py).
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not text or text == '':
        return ''
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        # Fallback if tokenization fails
        tokens = text.split()
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
             if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def predict_text(text: str) -> float:
    """
    Preprocess text and make prediction using the BiLSTM model.
    
    Args:
        text: Raw input text (job description)
        
    Returns:
        Probability of being fraudulent (0-1)
    """
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
        # CRITICAL: Preprocess text to match training data format
        cleaned_text = clean_text(text)
        
        if not cleaned_text or cleaned_text.strip() == '':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text became empty after preprocessing. Please provide more meaningful text."
            )
        
        # Tokenize and pad (matching training pipeline)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        
        # Make prediction
        prob = model.predict(padded, verbose=0)[0][0]
        return float(prob)
    except HTTPException:
        raise
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
