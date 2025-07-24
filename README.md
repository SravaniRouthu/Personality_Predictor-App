# Personality Predictor - Advanced Django ML Application

A comprehensive Django web application for personality prediction using machine learning. This application can predict whether a person is an Introvert or Extrovert based on behavioral patterns using Logistic Regression and Random Forest models.

## Features

### Core ML Capabilities
- **Logistic Regression Model**: Fast, interpretable linear classification
- **Random Forest Model**: High-accuracy ensemble method with feature importance
- **80/20 Train-Test Split**: Optimal model evaluation
- **Advanced Preprocessing**: Feature scaling, encoding, and selection
- **Model Persistence**: Save and load trained models


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone and Navigate
```bash
# Navigate to your project directory
cd "Django App-Introvert vs Extrovert"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 4: Create Superuser (Optional)
```bash
python manage.py createsuperuser
```

### Step 5: Run the Application
```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## Usage Guide

### 1. Upload Dataset
- Navigate to "Upload Data" in the navigation
- Upload the `personality_dataset.csv` file
- Choose whether to clear existing data
- Click "Upload Dataset"

### 2. Train Models
- Go to "Train Models" page
- Select model type (Logistic Regression, Random Forest, or Both)
- Enter a descriptive model name
- Set test size (recommended: 0.2 for 20%)
- Click "Start Training"

### 3. Make Predictions
- Navigate to "Predict" page
- Fill in the personality questionnaire
- Select a trained model
- Click "Predict Personality"
- View results with confidence scores

### 4. Explore Data
- Visit "Explore Data" for visualizations
- View personality distribution charts
- Analyze feature correlations
- Understand data patterns

### 5. Bulk Predictions
- Use "Predict CSV" for batch predictions
- Upload CSV file with same format as training data
- Get predictions for multiple records

## 📁 Project Structure

```
Django App-Introvert vs Extrovert/
├── personality_predictor/          # Django project settings
│   ├── settings.py                # Main settings
│   ├── urls.py                    # URL configuration
│   └── wsgi.py                    # WSGI configuration
├── ml_app/                        # Main application
│   ├── models.py                  # Database models
│   ├── views.py                   # View functions
│   ├── forms.py                   # Django forms
│   ├── urls.py                    # App URLs
│   └── ml_engine.py              # ML engine core
├── templates/                     # HTML templates
│   ├── base.html                  # Base template
│   └── ml_app/                   # App templates
├── models/                        # Saved ML models
├── media/                         # Uploaded files
├── requirements.txt               # Python dependencies
├── manage.py                      # Django management
├── personality_dataset.csv         # Dataset
└── README.md                      # This file
```

## 🔧 API Usage

### REST API Endpoint
```
POST /api/predict/
Content-Type: application/json

{
    "time_spent_alone": 5,
    "stage_fear": "Yes",
    "social_event_attendance": 3,
    "going_outside": 4,
    "drained_after_socializing": "No",
    "friends_circle_size": 8,
    "post_frequency": 2
}
```

### Response
```json
{
    "prediction": "Introvert",
    "confidence": 0.85,
    "model_used": "Personality_Model_LR",
    "model_accuracy": 0.923
}
```

## 📈 Model Performance

### Expected Results
- **Logistic Regression**: ~85-90% accuracy
- **Random Forest**: ~90-95% accuracy
- **Feature Importance**: Social events and time alone are typically most important
- **Training Time**: 30-60 seconds for both models

### Key Features
- **Feature Selection**: Automatic selection of top 5 features
- **Cross-validation**: Built-in train/test split
- **Hyperparameter Tuning**: Optimized parameters for each model type
- **Scalability**: Handles datasets up to 10,000+ records



---

**Built  using Django, scikit-learn, and Bootstrap 5** 