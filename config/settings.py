import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIRS = {
    os.path.join(BASE_DIR, "../datasets/mozilla/clips"): "Leitura",
    os.path.join(BASE_DIR, "../datasets/librispeech/LibriSpeech/train-clean-100"):"Leitura",
    os.path.join(BASE_DIR, "../datasets/interviews"): "Fala Espontânea"

}

UPLOAD_FOLDER = os.path.join(BASE_DIR, "../uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

LOG_MODEL_PATH = os.path.join(BASE_DIR, "../models/log_model.joblib")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "../models/xgb_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.joblib")

FEATURES_FILE = os.path.join(BASE_DIR, "../data/audio_features.csv")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
