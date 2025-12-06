from fastapi import FastAPI, File, UploadFile, HTTPException
import io
from PIL import Image
from model import predict_whole
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI() 

origins = [
    "*"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/") 
async def status():
    return { "status": "ok" }

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read uploaded file bytes
    img_bytes = await image.read()

    # Convert bytes to PIL image
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")

    save_path = f"./webdata/{image.filename}"
    with open(save_path, "wb") as f:
        f.write(img_bytes)

    detections, final_image_base64 = predict_whole(save_path)
    return {
        "detections": detections,           
        "final_image_base64": final_image_base64, 
        "success": True
    }
