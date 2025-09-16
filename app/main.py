from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from datetime import datetime
import uvicorn

# Import our simple model
from app.simple_model import predict_with_simple_model

app = FastAPI(
    title="Crop Disease Detection ML Service",
    description="Simple AI service for crop disease detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
        "service": "Simple Crop Disease Detection ML Service",
        "status": "running",
        "version": "1.0.0",
        "model_type": "Rule-based Classifier",
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
        "models_loaded": True,
        "model_type": "Simple Rule-Based",
        "memory_usage": "low"
    }

@app.post("/predict")
async def predict_disease_endpoint(image: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        # File validation
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {image.content_type}. Only image files allowed."
            )
        
        # Read image
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File too large")
        
        print(f"Processing image: {image.filename} ({len(contents)} bytes)")
        
        # Get prediction using simple model
        result = predict_with_simple_model(contents)
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)
        
        return {
            "success": True,
            "message": "Image analyzed successfully",
            "image_info": {
                "filename": image.filename,
                "size_bytes": len(contents),
                "analysis": result.get('analysis', {})
            },
            "prediction": result['prediction'],
            "treatment": result['treatment'],
            "processing_time_seconds": processing_time,
            "model_info": {
                "type": result.get('model_type', 'Simple Classifier'),
                "version": result['prediction']['model_version']
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """
     ML model information
    """
    return {
        "models": [
            {
                "name": "Simple Crop Disease Classifier",
                "version": "1.0.0",
                "type": "Rule-Based Classification",
                "status": "loaded",
                "accuracy": "Demo Model - For Learning Purpose",
                "supported_diseases": [
                    "Healthy",
                    "Leaf Spot Disease", 
                    "Blight Disease",
                    "Rust Disease",
                    "Mosaic Virus"
                ],
                "features": [
                    "Color analysis",
                    "Basic image statistics", 
                    "Rule-based decision making"
                ]
            }
        ],
        "total_models": 1,
        "memory_usage": "Minimal"
    }

# # Server run
if __name__ == "__main__":
    print("🌾 Starting Simple ML Service for Beginners...")
    print("📚 Perfect for learning ML integration!")
    print("🔗 Visit http://localhost:8000/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
