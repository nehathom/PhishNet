from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
import torch
from dotenv import load_dotenv
load_dotenv()
import traceback 


from genai import build_prompt, get_genai_reasoning

# Add this simple feature extractor inside main.py (or separate file)
from urllib.parse import urlparse

def extract_features(url: str):
    parsed = urlparse(url)
    return {
        "has_ip": any(char.isdigit() for char in parsed.netloc),
        "https_token": any(token in url.lower() for token in ["secure", "login", "verify", "update"]),
        "nb_dots": url.count('.'),
        "length_url": len(url),
    }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "distilbert-phish"  # relative to /app
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

class URLItem(BaseModel):
    url: str

@app.post("/predict")
def predict_phishing(item: URLItem):
    # Model prediction
    inputs = tokenizer(item.url, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Extract features
    features = extract_features(item.url)

    # Generate explanation safely
    try:
        prompt = build_prompt(item.url, features, predicted_class)
        explanation = get_genai_reasoning(prompt)
        print(explanation)
    except Exception as e:
        # Print full traceback to console/log
        print("Error generating explanation:", e)
        print(e)
        traceback.print_exc()

        # Return a safe default message for the frontend
        explanation = "Error generating explanation at this time. Please try again later."

    return {
        "is_phishing": bool(predicted_class),
        "prediction": predicted_class,
        "features": features,
        "explanation": explanation
    }