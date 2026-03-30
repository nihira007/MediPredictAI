# MediPredictAI 

MediPredictAI is an AI-powered disease prediction system that analyzes symptoms and provides possible conditions along with severity levels and medical suggestions.

The system uses a trained machine learning model to evaluate symptom patterns and predict possible diseases along with probability scores, severity levels, and personalized precautionary recommendations.

It also includes an AI-powered assistant that helps users understand medical conditions, offering quick explanations, guidance, and actionable suggestions in real time.

By combining machine learning with intuitive design, MediPredict AI enables users to make more informed health decisions in a simple and accessible way.

---

## Features

* Symptom-based multi-disease prediction using Machine Learning.
* Probability-based ranking of conditions.
* Severity analysis (Low → Critical).
* Precautions and disease descriptions.
* Integration of an AI assistant (LLM) for real-time medical guidance.
* Clean and interactive UI.
  
---

##  Preview

![UI](assets/main.png)

![UI](assets/symptoms.png)

![UI](assets/results.png)

![UI](assets/aichatbot.png)

---

##  How It Works

1. User selects symptoms.
2. Symptoms are converted into a feature vector.
3. ML model predicts possible diseases.
4. Returns results with probabilities, severity, and precautions.
5. Users can further interact with an AI assistant for additional guidance and explanations.

---

## Tech Stack

* Backend: Python, FastAPI
* Machine Learning: Random Forest (Tuned with GridSearchCV)
* Frontend: HTML, CSS, JavaScript
* AI Integration: Groq API
  
---

##  Model Details

* Algorithm: Random Forest
* Accuracy: ~99%
* Dataset: Kaggle Medical Dataset
* Symptoms: 132
* Diseases: 41

---

##  Setup Instructions

### Environment Variables
Create a `.env` file in the `backend/` directory (or rename `.env.example`) and add:
```bash
GROQ_API_KEY=your_api_key_here
```

### Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Frontend

Open:

```bash
frontend/index.html
```

---

## Disclaimer

This project is for educational purposes only and does not replace professional medical advice.  Always consult a healthcare provider for medical concerns.

---

##  Author

Built & Designed by **Nihira Hassan**
