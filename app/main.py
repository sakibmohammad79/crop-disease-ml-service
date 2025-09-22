from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager
import asyncio
from fastapi import UploadFile, HTTPException
# Import real model
from app.real_model import initialize_real_model, predict_with_real_model

# Global variable to track model status
model_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded
    # Startup
    print("🧠 Initializing Real CNN Model...")
    model_loaded = initialize_real_model()
    if model_loaded:
        print("✅ Real CNN model loaded successfully!")
    else:
        print("⚠️  Model loading failed, using fallback predictions")

    yield  # Application is running

    # Shutdown (if you need cleanup)
    print("👋 Shutting down service...")

app = FastAPI(
    title="Real Crop Disease Detection ML Service",
    description="CNN-based AI service for crop disease detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Real Crop Disease Detection ML Service",
        "status": "running",
        "version": "2.0.0",
        "model_type": "Convolutional Neural Network",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ml_service": "ready",
        "models_loaded": model_loaded,
        "model_type": "CNN" if model_loaded else "Fallback",
        "memory_usage": "moderate"
    }

# @app.post("/predict")
# async def predict_disease_endpoint(image: UploadFile = File(...)):
#     start_time = time.time()
    
#     try:
#         # File validation
#         if not image.content_type or not image.content_type.startswith('image/'):
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Invalid file type: {image.content_type}. Only image files allowed."
#             )
#         # Read image
#         contents = await image.read()
#         if len(contents) > 10 * 1024 * 1024:  # 10MB
#             raise HTTPException(status_code=400, detail="File too large")
        
#         print(f"🔍 Processing image: {image.filename} ({len(contents)} bytes)")
        
#         # Get prediction using real CNN model
#         result = predict_with_real_model(contents)
        
#         # Calculate processing time
#         processing_time = round(time.time() - start_time, 2)
        
#         return {
#             "success": True,
#             "message": "Image analyzed with CNN model",
#             "image_info": {
#                 "filename": image.filename,
#                 "size_bytes": len(contents)
#             },
#             "prediction": result['prediction'],
#             "treatment": result['treatment'],
#             "processing_time_seconds": processing_time,
#             "model_info": {
#                 "type": result.get('model_type', 'CNN'),
#                 "version": result['prediction']['model_version'],
#                 "details": result.get('model_info', {})
#             },
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")



@app.post("/predict")
async def predict_disease_endpoint(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    loop = asyncio.get_running_loop()
    try:
        # run predict in threadpool to avoid blocking the event loop
        result = await loop.run_in_executor(None, predict_with_real_model, contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "success": True,
        "message": "Image analyzed",
        "prediction": result['prediction'],
        "treatment": result['treatment'],
        "model_info": result.get('model_info', {})
    }


if __name__ == "__main__":
    print("🌾 Starting Real ML Service with CNN...")
    print("🔗 Visit http://localhost:8000/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

    # source ./venv/Scripts/activate
    # python -m uvicorn app.main:app --reload
