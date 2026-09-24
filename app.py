import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


model1 = joblib.load("email_model.pkl")   
vectorizer1 = joblib.load("email_vectorizer.pkl")
model2 = joblib.load("job_model.pkl")   
vectorizer2 = joblib.load("job_vectorizer.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/predict/spam")
def predict_spam(data: TextInput):
    text = data.text.lower()
    vec = vectorizer1.transform([text])
    pred = model1.predict(vec)[0]
    result = "spam" if pred == 1 else "ham"
    return {"result": result}

@app.post("/predict/job")
def predict_job(data: TextInput):
    text = data.text.lower()
    vec = vectorizer2.transform([text])
    pred = model2.predict(vec)[0]
    result = "fake" if pred == 1 else "real"
    return {"result": result}

