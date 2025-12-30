
# Local Setup & Execution Guide


This guide explains how to set up and run the **Fake Job Detection System** locally, covering all milestones from data preparation to authenticated web application.

---

## Prerequisites

Ensure the following are installed on your system:

* Python **3.10+** (tested on Python 3.12)
* Node.js **18+**
* npm (comes with Node.js)
* Git (optional)
* 8 GB RAM or higher

---

## Project Structure (Final)

```
project/
│
├── src/
│   ├── api.py
│   ├── model_utils.py
│   ├── deep_learning_model.py
│
├── models/
│   ├── bilstm_model_v1.h5
│   └── tokenizer.pkl
│
├── processed_data/
│   └── processed_jobs.csv
│
├── frontend/          # React (Vite) app
│
├── requirements.txt
├── README.md
└── guide.md
```

---

# Milestone 1 — Environment Setup & Project Initialization

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Milestone 1 is complete when:

* Project structure is created
* Virtual environment is active
* Dependencies are installed successfully

---

# Milestone 2 — Model Training & Evaluation

This milestone includes classical ML models and a deep learning BiLSTM model.

### 1. Ensure processed dataset exists

```
processed_data/processed_jobs.csv
```

Required columns:

* `cleaned_text`
* `fraudulent`

### 2. Train Deep Learning model (BiLSTM)

From project root:

```bash
python src/deep_learning_model.py
```

This will:

* Train an optimized BiLSTM (8 GB RAM safe)
* Evaluate accuracy and F1 score
* Save model to `models/`

  * `bilstm_model_v1.h5`
  * `tokenizer.pkl`

Milestone 2 is complete when:

* Training finishes without errors
* Model files are saved

---

# Milestone 3 — API & Basic Frontend

### 1. Start FastAPI backend

```bash
uvicorn src.api:app --reload
```

Backend will run at:

```
http://127.0.0.1:8000
```

### 2. Verify API

Open in browser:

```
http://127.0.0.1:8000/docs
```

You should see:

* `/login`
* `/predict` (secured endpoint)

Milestone 3 is complete when:

* API starts successfully
* Swagger UI loads

---

# Milestone 4 — Authentication & React UI (Vite)

## Backend (Authentication)

Authentication is implemented using **JWT (Bearer Token)**.

### Test login (via Swagger or frontend)

* Username: `admin`
* Password: `admin123`

---

## Frontend (React + Vite)

### 1. Navigate to frontend folder

```bash
cd frontend
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Start React app

```bash
npm run dev
```

Frontend will run at:

```
http://127.0.0.1:5173
```

### 4. Application Flow

1. Click **Login**
2. JWT token is generated
3. Enter job description
4. Click **Check Job**
5. Prediction is displayed

Milestone 4 is complete when:

* Login works
* Protected `/predict` endpoint is accessible
* Prediction result is shown in UI

---

# Running Order (Quick Reference)

```bash
# Activate environment
venv\Scripts\activate

# Backend
uvicorn src.api:app --reload

# Frontend (new terminal)
cd frontend
npm run dev
```
