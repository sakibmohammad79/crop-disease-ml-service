from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import time
from datetime import datetime
import uvicorn

# FastAPI application তৈরি
app = FastAPI(
    title="Crop Disease Detection ML Service",
    description="AI service for detecting diseases in crop images",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI URL
    redoc_url="/redoc"  # ReDoc URL
)

# CORS middleware - Node.js backend এর সাথে communicate করার জন্য
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:3000"],  # Node.js এবং React
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Basic routes
@app.get("/")
async def root():
    """
    Service এর basic information
    """
    return {
        "service": "Crop Disease Detection ML Service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """
    Service health check - Node.js এর ML service monitoring এর জন্য
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ml_service": "ready",
        "models_loaded": True,  # এখনো actual model নাই, পরে update করব
        "memory_usage": "normal"
    }

@app.post("/predict")
async def predict_disease(image: UploadFile = File(...)):
    """
    Main prediction endpoint
    এখানে image receive করে disease prediction করবে
    """
    start_time = time.time()
    
    try:
        # File validation
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {image.content_type}. Only image files allowed."
            )
        
        # File size check (10MB limit)
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum 10MB allowed."
            )
        
        # Image processing with PIL
        try:
            pil_image = Image.open(io.BytesIO(contents))
            
            # Image information extract করি
            image_info = {
                "format": pil_image.format,
                "mode": pil_image.mode,
                "size": pil_image.size,
                "width": pil_image.width,
                "height": pil_image.height
            }
            
            print(f"Processing image: {image.filename}")
            print(f"Image info: {image_info}")
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # এখানে actual ML model দিয়ে prediction করব
        # এখনকার জন্য dummy response
        
        # বিভিন্ন disease এর dummy data
        disease_predictions = [
            {
                "disease": "healthy",
                "disease_name": "Healthy Crop",
                "confidence": 0.85,
                "is_healthy": True,
                "severity": "NONE",
                "description": "Crop appears healthy with no signs of disease"
            },
            {
                "disease": "leaf_blight",
                "disease_name": "Leaf Blight",
                "confidence": 0.78,
                "is_healthy": False,
                "severity": "MEDIUM",
                "description": "Fungal infection affecting leaves"
            },
            {
                "disease": "brown_spot",
                "disease_name": "Brown Spot Disease",
                "confidence": 0.92,
                "is_healthy": False,
                "severity": "HIGH",
                "description": "Bacterial infection causing brown spots"
            }
        ]
        
        # Random prediction select করি (actual model এর পরিবর্তে)
        import random
        selected_prediction = random.choice(disease_predictions)
        
        # Processing time calculate
        processing_time = round(time.time() - start_time, 2)
        
        # Treatment recommendations based on disease
        treatment_recommendations = {
            "healthy": {
                "immediate_action": "No treatment required",
                "chemicals": [],
                "organic_solution": "Continue current farming practices",
                "prevention": "Regular monitoring and proper nutrition"
            },
            "leaf_blight": {
                "immediate_action": "Apply fungicide spray immediately",
                "chemicals": ["Mancozeb", "Copper Oxychloride"],
                "organic_solution": "Neem oil spray, remove affected leaves",
                "prevention": "Improve drainage, reduce humidity"
            },
            "brown_spot": {
                "immediate_action": "Remove affected parts, apply bactericide",
                "chemicals": ["Streptomycin", "Copper compounds"],
                "organic_solution": "Baking soda spray, increase air circulation",
                "prevention": "Avoid overhead watering, proper spacing"
            }
        }
        
        treatment = treatment_recommendations.get(
            selected_prediction["disease"], 
            treatment_recommendations["healthy"]
        )
        
        # Response তৈরি করি
        response = {
            "success": True,
            "message": "Image processed successfully",
            "image_info": {
                "filename": image.filename,
                "size_bytes": len(contents),
                "dimensions": f"{image_info['width']}x{image_info['height']}",
                "format": image_info["format"]
            },
            "prediction": {
                "disease": selected_prediction["disease"],
                "disease_name": selected_prediction["disease_name"],
                "confidence": selected_prediction["confidence"],
                "is_healthy": selected_prediction["is_healthy"],
                "affected_area_percentage": round(random.uniform(5, 30), 1) if not selected_prediction["is_healthy"] else 0,
                "severity": selected_prediction["severity"],
                "description": selected_prediction["description"],
                "model_version": "dummy_v1.0"
            },
            "treatment": treatment,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/models/info")
async def get_model_info():
    """
    ML model এর information
    """
    return {
        "models": [
            {
                "name": "crop_disease_classifier_v1",
                "version": "1.0.0",
                "type": "CNN Classification",
                "status": "loaded",
                "accuracy": "94.2%",
                "supported_diseases": [
                    "Healthy",
                    "Leaf Blight", 
                    "Brown Spot",
                    "Bacterial Wilt",
                    "Fungal Infection"
                ]
            }
        ],
        "total_models": 1,
        "memory_usage": "256MB"
    }

# Server run করার function
if __name__ == "__main__":
    print("Starting ML Service...")
    print("Visit http://localhost:8000/docs for API documentation")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True  # Development এর জন্য auto reload
    )