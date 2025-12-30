# 🔍 Fake Job Detection System

A full-stack AI-powered web application that detects fraudulent job postings using deep learning and machine learning techniques. The system analyzes job descriptions and provides real-time predictions with confidence scores.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![React](https://img.shields.io/badge/React-19.2-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **AI-Powered Detection**: Uses a BiLSTM (Bidirectional LSTM) deep learning model for accurate fraud detection
- **Real-time Analysis**: Instant predictions with probability scores
- **Modern Web Interface**: Beautiful, responsive React frontend with intuitive UX
- **Secure Authentication**: JWT-based authentication system
- **RESTful API**: FastAPI backend with comprehensive error handling
- **Confidence Scoring**: Provides both fraud probability and confidence metrics
- **Health Monitoring**: Built-in health check endpoint for system status

## 🏗️ Architecture

The application follows a client-server architecture:

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   React Frontend │ ◄─────► │  FastAPI Backend │ ◄─────► │  BiLSTM Model   │
│   (Port 5173)   │  HTTP   │   (Port 8000)   │         │   (TensorFlow)  │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **TensorFlow**: Deep learning framework for the BiLSTM model
- **scikit-learn**: Machine learning utilities
- **NLTK**: Natural language processing
- **python-jose**: JWT token handling
- **Uvicorn**: ASGI server

### Frontend
- **React 19**: UI library
- **Vite**: Build tool and dev server
- **CSS3**: Modern styling with animations

### Machine Learning
- **BiLSTM Model**: Bidirectional LSTM for sequence classification
- **Tokenizer**: Custom tokenizer for text preprocessing
- **TF-IDF**: Feature extraction (for alternative models)

## 📁 Project Structure

```
proj/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── App.jsx          # Main application component
│   │   ├── App.css          # Application styles
│   │   └── main.jsx         # Entry point
│   ├── package.json         # Frontend dependencies
│   └── vite.config.js       # Vite configuration
│
├── src/                      # Backend source code
│   ├── api.py               # FastAPI application and endpoints
│   ├── data_preprocessing.py # Data preprocessing utilities
│   ├── deep_learning_model.py # BiLSTM model definition
│   ├── model_training.py    # Model training scripts
│   ├── model_evaluation.py  # Model evaluation metrics
│   └── model_utils.py       # Model utility functions
│
├── models/                   # Trained models
│   ├── bilstm_model_v1.h5   # BiLSTM model weights
│   ├── tokenizer.pkl        # Text tokenizer
│   ├── logistic_regression_v1.pkl
│   └── random_forest_v1.pkl
│
├── processed_data/           # Preprocessed datasets
│   ├── processed_jobs.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   └── y_train.npy, y_test.npy
│
├── requirements.txt         # Python dependencies
├── main.py                  # Application entry point (if applicable)
└── README.md                # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- Virtual environment (recommended)

### Backend Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd proj
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download NLTK data** (if not already downloaded):
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
   ```

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

## 💻 Usage

### Starting the Backend Server

1. **Activate your virtual environment** (if not already activated)

2. **Start the FastAPI server**:
   ```bash
   uvicorn src.api:app --reload --port 8000
   ```

   The API will be available at `http://127.0.0.1:8000`

3. **Access API documentation**:
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - ReDoc: `http://127.0.0.1:8000/redoc`

### Starting the Frontend

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

### Using the Application

1. **Login**: Use the default credentials:
   - Username: `admin`
   - Password: `admin123`

2. **Analyze a Job Posting**:
   - Paste a job description into the text area
   - Click "Analyze Job Posting"
   - View the prediction results with fraud probability and confidence score

## 📡 API Documentation

### Endpoints

#### `POST /login`
Authenticate and receive a JWT token.

**Request Body**:
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### `POST /predict`
Analyze a job posting for fraud detection.

**Headers**:
```
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "description": "Job description text here..."
}
```

**Response**:
```json
{
  "prediction": "Fraudulent",
  "fraud_probability": 0.8542,
  "confidence": 0.7084
}
```

#### `GET /health`
Check API and model status.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true
}
```

### Authentication

All prediction endpoints require JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer <your_token>
```

## 🤖 Model Details

### BiLSTM Model

The system uses a Bidirectional LSTM neural network for fraud detection:

- **Architecture**: Embedding → Bidirectional LSTM → Dense layers
- **Input**: Tokenized and padded sequences (max length: 200)
- **Output**: Probability score (0-1) indicating fraud likelihood
- **Threshold**: 0.5 (above = fraudulent, below = legitimate)

### Model Performance

The model has been trained and evaluated on a dataset of job postings. Key metrics include:
- Accuracy
- F1 Score
- ROC-AUC Score

(Note: Specific metrics should be added based on your model evaluation results)

## 🔧 Development

### Running Tests

```bash
# Backend tests (if available)
pytest

# Frontend tests (if available)
cd frontend
npm test
```

### Code Style

- Backend: Follow PEP 8 Python style guide
- Frontend: ESLint configuration included

### Building for Production

**Frontend**:
```bash
cd frontend
npm run build
```

**Backend**:
The FastAPI application can be deployed using:
- Uvicorn (production)
- Gunicorn with Uvicorn workers
- Docker (recommended for containerization)

## 🔒 Security Notes

- **Default Credentials**: Change the default admin credentials in production
- **JWT Secret**: Update `SECRET_KEY` in `src/api.py` for production use
- **CORS**: Configure CORS origins appropriately for production
- **HTTPS**: Always use HTTPS in production environments

## 📊 Data Processing

The system includes comprehensive data preprocessing:
- Text cleaning and normalization
- Stop word removal
- Tokenization
- Sequence padding
- Feature extraction (TF-IDF for alternative models)

## 🐛 Troubleshooting

### Model Not Loading
- Ensure model files exist in the `models/` directory
- Check file paths in `src/api.py`
- Verify TensorFlow installation

### CORS Errors
- Ensure backend CORS settings include your frontend URL
- Check that both servers are running

### Authentication Issues
- Verify JWT token is included in request headers
- Check token expiration
- Ensure correct credentials are used

## 📝 License

This project is provided as-is for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ using FastAPI, React, and TensorFlow**

