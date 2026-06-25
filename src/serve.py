import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "ai-lab-mlops-277707107390-20260625")
AWS_S3_MODEL_KEY = os.environ.get("AWS_S3_MODEL_KEY", "models/model.pkl")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")

model = None
model_version = "not-loaded"


def download_model():
    """Tai file model.pkl tu S3 ve may khi server khoi dong."""
    global model_version
    dir_name = os.path.dirname(MODEL_PATH)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        logger.info(f"Model file found locally at {MODEL_PATH}. Skipping download.")
        model_version = "local"
    else:
        logger.info(f"Model not found at {MODEL_PATH}. Downloading from S3 bucket {AWS_S3_BUCKET}...")
        try:
            s3 = boto3.client("s3")
            s3.download_file(AWS_S3_BUCKET, AWS_S3_MODEL_KEY, MODEL_PATH)
            logger.info("Successfully downloaded model from S3.")
            model_version = "s3"
        except ClientError as e:
            logger.error(f"Failed to download model from S3: {e}")
            model_version = "failed-s3-download"


# Tải và load model khi khởi động service
download_model()
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
else:
    logger.error("No model file available to load.")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server."""
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_version": model_version
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Endpoint suy luan."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    try:
        prediction = model.predict([req.features])
        pred_val = int(prediction[0])
        labels = {0: "thap", 1: "trung_binh", 2: "cao"}
        label = labels.get(pred_val, "unknown")
        return {
            "prediction": pred_val,
            "label": label,
            "model_version": model_version
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

