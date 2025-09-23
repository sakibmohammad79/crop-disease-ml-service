# 🌱 Crop Disease Detection - ML Service

This is the **Machine Learning (ML) service** for the **Crop Disease Detection System**.  
It uses a **TensorFlow CNN model** served through **FastAPI** to predict plant diseases from uploaded leaf images.  

---

## 🚀 Features
- FastAPI-based ML API
- TensorFlow pre-trained CNN model for plant disease classification
- Image preprocessing pipeline (resize, normalization, batching)
- CORS-enabled (works with backend/frontend)
- Docker-ready for deployment
- Easy integration with any client app

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **FastAPI** (API framework)
- **Uvicorn** (ASGI server)
- **TensorFlow / Keras** (ML model)
- **Pillow, NumPy** (image preprocessing)

---


## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sakibmohammad79/crop-disease-ml-service
   cd crop-disease-ml-service

2. Create a virtual environment
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows

3. Install dependencies
   pip install -r requirements.txt

4. (Optional) Download pre-trained model
   python download_model.py

Start the FastAPI server:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Server will run at:
👉 http://localhost:8000
👉 Docs available at: http://localhost:8000/docs


📡 API Endpoints
🔹 Upload image & get prediction

POST /predict
Request (multipart/form-data):



📌 Next Steps

Improve model accuracy with more dataset training

Add support for more crops & diseases

Deploy to cloud (AWS/GCP/Render)

🤝 Contributing

PRs are welcome! Please fork this repo and create a pull request.
