<p align="center">
  <h1 align="center">🏦 Credit Scoring System</h1>
  <p align="center">
    <strong>Production-Grade ML Credit Score Prediction with FastAPI & Streamlit</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-0.136+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Streamlit-1.56+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/scikit--learn-1.6+-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Details](#-model-details)
- [Testing](#-testing)
- [Docker Deployment](#-docker-deployment)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

A **production-grade credit scoring system** that predicts an applicant's creditworthiness (Low / Average / High) based on demographic and financial data. Built with a **hybrid workflow** — model training on Google Colab, software engineering locally.

The system features:
- **Machine Learning Pipeline** — Feature engineering, SMOTE class balancing, GradientBoosting classifier
- **REST API** — FastAPI with Pydantic validation, auto-generated Swagger/ReDoc docs
- **Interactive Dashboard** — Streamlit UI with real-time gauge charts and risk assessment
- **Docker-Ready** — One-command deployment with docker-compose

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "Cloud — Google Colab"
        A["📊 Kaggle Dataset"] --> B["🔬 Feature Engineering<br/>Income_per_Dependent<br/>Age_Income_Ratio"]
        B --> C["🏋️ Model Training<br/>GradientBoosting + SMOTE"]
        C --> D["📦 Export Artifacts<br/>model.pkl / scaler.pkl<br/>encoder.pkl / config.json"]
    end

    subgraph "Local — Production Stack"
        D -->|"Manual Transfer"| E["📂 models/"]
        E --> F["⚙️ Preprocessing Pipeline<br/>src/preprocessing.py"]

        F --> G["🚀 FastAPI Server<br/>api/main.py"]
        G -->|"POST /api/v1/predict"| H["📨 JSON Response"]

        F --> I["📊 Streamlit Dashboard<br/>dashboard/app.py"]
        I --> J["🎯 Gauge Chart + Risk Badge"]
    end

    subgraph "Deployment"
        G --> K["🐳 Docker Container"]
        I --> K
    end

    style A fill:#4DB6AC,stroke:#00897B,color:#fff
    style C fill:#4DB6AC,stroke:#00897B,color:#fff
    style G fill:#26A69A,stroke:#00897B,color:#fff
    style I fill:#80CBC4,stroke:#4DB6AC,color:#000
    style K fill:#B2DFDB,stroke:#80CBC4,color:#000
```

### Request Flow

```mermaid
sequenceDiagram
    participant U as User / Dashboard
    participant A as FastAPI /predict
    participant P as Preprocessor
    participant M as ML Model

    U->>A: POST {age, gender, income, ...}
    A->>A: Pydantic Validation
    A->>P: preprocess(input_data)
    P->>P: Engineer → Encode → Reorder → Scale
    P-->>A: feature_vector
    A->>M: model.predict(features)
    M-->>A: prediction + probabilities
    A-->>U: {credit_score, confidence, risk_level}
```

---

## 📁 Project Structure

```
Credit-Scoring-Model/
├── src/                          # Core ML Logic
│   ├── config.py                 # Central configuration & constants
│   ├── logger.py                 # Structured logging (loguru)
│   ├── preprocessing.py          # Feature engineering & encoding pipeline
│   └── predict.py                # Inference engine (CreditScorer class)
├── api/                          # REST API
│   ├── main.py                   # FastAPI app with CORS & docs
│   ├── routes.py                 # /predict & /health endpoints
│   └── schemas.py                # Pydantic request/response models
├── dashboard/                    # Interactive UI
│   └── app.py                    # Streamlit dashboard with gauge charts
├── models/                       # ML Artifacts (from Colab)
│   ├── credit_model.pkl          # Trained GradientBoosting classifier
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── target_encoder.pkl        # Fitted LabelEncoder
│   └── feature_config.json       # Feature names & metadata
├── data/                         # Dataset
│   └── Credit_Score_...csv       # Kaggle classification dataset
├── notebooks/                    # Training Reference
│   └── colab_training.py         # Google Colab training script
├── tests/                        # Test Suite
│   ├── test_preprocessing.py     # 12 preprocessing pipeline tests
│   └── test_api.py               # 8 API endpoint tests
├── Dockerfile                    # Production container
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Python dependencies
├── DEVELOPMENT_PLAN.md           # Development roadmap & progress
└── README.md                     # This file
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **3-Class Prediction** | Classifies credit score as **Low**, **Average**, or **High** |
| **Confidence Scores** | Returns probability breakdown for each class |
| **Risk Assessment** | Automatic risk level mapping (High/Medium/Low Risk) |
| **Input Validation** | Pydantic enums enforce valid Gender, Education, etc. |
| **Gauge Visualization** | Real-time Plotly gauge chart in Streamlit dashboard |
| **Structured Logging** | loguru with colored console + auto-rotated file logs |
| **Health Checks** | `/health` endpoint for monitoring & Docker health checks |
| **CORS Enabled** | Cross-origin requests supported for frontend integrations |
| **Docker Ready** | Single-command deployment with docker-compose |

---

## 🚀 Installation

### Prerequisites
- Python 3.12+ 
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/MaryamCodeHub/Credit-Scoring-Model.git
cd Credit-Scoring-Model

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Start the FastAPI Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Swagger Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/api/v1/health

### Start the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at http://localhost:8501

---

## 📡 API Documentation

### POST `/api/v1/predict`

Predict the credit score for a loan applicant.

**Request Body:**
```json
{
  "Age": 30,
  "Gender": "Male",
  "Income": 75000,
  "Education": "Bachelor's Degree",
  "Marital Status": "Single",
  "Number of Children": 0,
  "Home Ownership": "Rented"
}
```

**Response:**
```json
{
  "credit_score": "High",
  "confidence": 0.87,
  "probabilities": {
    "Low": 0.05,
    "Average": 0.08,
    "High": 0.87
  },
  "risk_level": "Low Risk"
}
```

### GET `/api/v1/health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Input Constraints

| Field | Type | Constraints |
|-------|------|-------------|
| `Age` | int | 18–100 |
| `Gender` | enum | Male, Female |
| `Income` | float | > 0 |
| `Education` | enum | High School Diploma, Associate's Degree, Bachelor's Degree, Master's Degree, Doctorate |
| `Marital Status` | enum | Single, Married |
| `Number of Children` | int | 0–15 |
| `Home Ownership` | enum | Rented, Owned |

---

## 🤖 Model Details

| Attribute | Value |
|-----------|-------|
| **Algorithm** | GradientBoosting Classifier |
| **F1 Score** | 1.00 (weighted) |
| **Target Classes** | Low, Average, High |
| **Features** | 9 (7 raw + 2 engineered) |
| **Class Balancing** | SMOTE oversampling |
| **Hyperparameter Tuning** | GridSearchCV with StratifiedKFold |
| **Training Environment** | Google Colab |

### Preprocessing Pipeline

```mermaid
graph LR
    A["Raw Input<br/>7 features"] --> B["Feature Engineering<br/>+Income_per_Dependent<br/>+Age_Income_Ratio"]
    B --> C["Categorical Encoding<br/>Ordinal: Education<br/>Binary: Gender, Marital, Home"]
    C --> D["Feature Reordering<br/>Match training order"]
    D --> E["Standard Scaling<br/>5 numerical features"]
    E --> F["Model-Ready<br/>9 features"]

    style A fill:#EF5350,color:#fff
    style B fill:#FFB74D,color:#000
    style C fill:#FFF176,color:#000
    style D fill:#81C784,color:#000
    style E fill:#4DB6AC,color:#fff
    style F fill:#42A5F5,color:#fff
```

### Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `Income_per_Dependent` | `Income / (Children + 1)` | Captures financial burden |
| `Age_Income_Ratio` | `Income / Age` | Captures career progression |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_preprocessing.py -v    # 12 tests
python -m pytest tests/test_api.py -v              # 8 tests
```

**Test Coverage:**
- ✅ Feature engineering (income ratios, age ratios)
- ✅ Categorical encoding (ordinal, binary)
- ✅ Feature reordering & validation
- ✅ API health endpoint
- ✅ API root endpoint  
- ✅ Input validation (missing fields, invalid values, edge cases)

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start both services
docker-compose up --build

# Services:
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

### Using Docker Directly

```bash
# Build
docker build -t credit-scoring-api .

# Run API
docker run -p 8000:8000 credit-scoring-api

# Run Dashboard
docker run -p 8501:8501 credit-scoring-api \
  streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **ML Framework** | scikit-learn, imbalanced-learn |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **Logging** | Loguru |
| **Serialization** | Joblib |
| **Testing** | Pytest |
| **Containerization** | Docker, Docker Compose |
| **Training** | Google Colab |

---

<p align="center">
  Built with ❤️ as a Production-Grade Data Science Portfolio Project
</p>
