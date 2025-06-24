from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import os
import io
from transformers import ViTForImageClassification, ViTImageProcessor
import uvicorn
from huggingface_hub import login

app = FastAPI()

model_repo = "belpin/vitmodel_skincheck"

login(token=os.getenv("HF_TOKEN"))
processor = ViTImageProcessor.from_pretrained(model_repo)
model = ViTForImageClassification.from_pretrained(model_repo)
model.eval()
print("✅ Model loaded from Hugging Face.")

id2label = model.config.id2label  # otomatis ambil label dari config

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        label = id2label[predicted_class]  # <--- FIX DI SINI

    return JSONResponse(content={"prediction": label})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
