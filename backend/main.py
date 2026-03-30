from fastapi import FastAPI 
from pydantic import BaseModel 
from groq import Groq 
from fastapi.middleware.cors import CORSMiddleware 
from model.predict import predict_disease 
from utils.medical_info import get_description, get_precautions 
from utils.suggestions import get_suggestions 
import os 
from dotenv import load_dotenv 

load_dotenv() 
api_key = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=api_key) 

app = FastAPI(title="DiagnosAI") 

app.add_middleware( 
                   CORSMiddleware, 
                   allow_origins=["*"], 
                   allow_credentials=True, 
                   allow_methods=["*"], 
                   allow_headers=["*"], 
                   ) 

class InputData(BaseModel): 
    symptoms: list[str] 
    
@app.get("/") 
def home(): 
    return {"status": "running"} 

# API endpoint to predict diseases from symptoms
@app.post("/predict")
def predict(data: InputData):
    preds, severity_score, severity_level, confidence_score = predict_disease(data.symptoms)
    advice = get_suggestions(severity_score)

    enriched_preds = []

    for pred in preds:
        disease = pred["disease"]

        enriched_preds.append({
            "disease": disease,
            "probability": pred["probability"],
            "description": get_description(disease),
            "precautions": get_precautions(disease)
        })


    return {
    "top_predictions": enriched_preds,
    "severity_score": severity_score,
    "severity_level": severity_level,
    "confidence": confidence_score,
    "advice": advice
} 
  
# Generate detailed disease explanation using AI (Groq)    
@app.post("/disease-info") 
def disease_info(data: dict): 
    disease = data.get("disease") 
    prompt = f""" 
    Explain {disease} in simple medical terms. 
    Include: 
    - description 
    - causes 
    - precautions 
    - when to see doctor 
    - lifestyle tips """ 
    
    res = client.chat.completions.create( 
                                         model="llama-3.1-8b-instant", 
                                         messages=[{"role": "user", "content": prompt}] 
                                         ) 
    
    return {
        "info": res.choices[0].message.content} 
    
# Return model accuracy metrics
@app.get("/metrics")
def get_metrics():
    try:
        path = os.path.join("model", "metrics.txt")
        with open(path, "r") as f:
            acc = float(f.read())
        return {"accuracy": round(acc * 100, 2)}
    except:
        return {"accuracy": 0}
        
# AI chatbot endpoint for medical queries    
@app.post("/chat") 
def chat(data: dict): 
    try: 
        print("CHAT HIT") 
        
        query = data.get("query") 
        
        res = client.chat.completions.create( 
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system", 
                            "content": """ 
                            You are a helpful AI medical assistant. 
                            
                            Rules: 
                            - Answer ONLY what the user asks 
                            - Keep responses VERY SHORT (1-3 lines max) 
                            - Provide long explanations when requested or needed, but keep initial responses concise 
                            - Use simple language
                            - Be empathetic and supportive
                            - Sound natural like a real doctor
                            - If the user asks for advice give 2-3 actionable suggestions
                            - If user says "yes", "ok", "thanks" respond naturally like no problem, happy to help, etc. 
                            Maximum 40 words per response 
                            """ 
                            }, 
                        { 
                         "role": "user", 
                         "content": query 
                         } 
                        ] 
                    ) 
        reply = res.choices[0].message.content 
        return {"reply": reply} 
    
    except Exception as e: 
        print("ERROR:", e) 
        return {"reply": f"Backend Error: {str(e)}"}
    