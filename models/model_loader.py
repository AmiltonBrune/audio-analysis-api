import os
from joblib import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")

LOG_MODEL_PATH = os.path.join(MODEL_DIR, "log_model.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

def load_models():
    if not os.path.exists(LOG_MODEL_PATH):
        raise FileNotFoundError(f"❌ Modelo não encontrado: {LOG_MODEL_PATH}. Execute `python models/train_model.py` primeiro.")
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(f"❌ Modelo não encontrado: {XGB_MODEL_PATH}. Execute `python models/train_model.py` primeiro.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"❌ Scaler não encontrado: {SCALER_PATH}. Execute `python models/train_model.py` primeiro.")

    log_model = load(LOG_MODEL_PATH)
    xgb_model = load(XGB_MODEL_PATH)
    scaler = load(SCALER_PATH)

    return log_model, xgb_model, scaler
