from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import os

# 1. INITIALIZE FASTAPI
app = FastAPI(title="Veritas AI Detector API")

# 2. CORS (CRITICAL: Allows your frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, change to your frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. GLOBAL MODEL LOADER
# We load the model once at startup so we don't reload it for every request.
MODEL_PATH = "with_flux_model.keras"
model = None

@app.on_event("startup")
async def load_ai_model():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Loading Model from {MODEL_PATH}...")
        # compile=False makes it load faster/safer for inference only
        model = load_model(MODEL_PATH, compile=False)
        print(f"✅ Model Loaded Successfully!")
    else:
        print(f"❌ CRITICAL ERROR: Model file '{MODEL_PATH}' not found.")

# 4. PREPROCESSING FUNCTION (The "Perfect" Match)
def preprocess_image(image_bytes):
    """
    Exact replica of the training preprocessing.
    1. Open Image
    2. Convert to RGB (Fixes PNG/Transparency issues)
    3. Resize to 224x224 using LANCZOS (High quality)
    4. Apply EfficientNetV2 specific preprocessing
    """
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Force RGB
        img = img.convert('RGB')
        
        # Resize (LANCZOS is what we used in training)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to Array
        img_array = np.array(img)
        
        # Expand dims (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # TF EfficientNet Preprocessing
        return preprocess_input(img_array)
        
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

# 5. THE PREDICTION ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    
    # Validation
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload JPG or PNG.")

    try:
        # Read file
        contents = await file.read()
        
        # Preprocess
        processed_image = preprocess_image(contents)
        
        # Predict
        prediction = model.predict(processed_image)
        
        # Classes (Alphabetical Order used by Keras: 0=AI, 1=Real)
        ai_score = float(prediction[0][0])
        real_score = float(prediction[0][1])
        
        # Logic
        result_class = "AI" if ai_score > real_score else "Real"
        confidence = max(ai_score, real_score) * 100
        
        return {
            "prediction": result_class,
            "confidence_percentage": round(confidence, 2),
            "probabilities": {
                "ai": round(ai_score, 4),
                "real": round(real_score, 4)
            },
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}

# Health Check
@app.get("/")
def home():
    return {"status": "online", "message": "Veritas AI Detector is running."}