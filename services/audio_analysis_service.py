import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from transformers import pipeline
from data.extract_features import extract_audio_features
from models.model_loader import load_models
from services.validate_with_openai import validate_with_openai

sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

log_model, xgb_model, scaler = load_models()


def analyze_audio(file_path, filename):
    try:
        logging.info(f"📂 Analisando arquivo: {file_path}")

        features = extract_audio_features(file_path)
        if features is None:
            return {"error": "Falha ao processar o áudio ou extração incompleta."}, 500

        logging.info(f"🔍 Features extraídas: {features}")

        feature_names = [
            "duration", "mean_freq", "spectral_bandwidth", "zero_crossing_rate", "energy",
            "silence_ratio", "fft_peak_freq", "pitch_mean", "spectral_entropy", "spectral_rolloff", "chroma_mean",
            "speech_rate", "avg_pause_duration", "pitch_variability"
        ]

        features_df = pd.DataFrame([features], columns=feature_names)

        transformed_features = scaler.transform(features_df)

        logging.info(f"🔍 Features após transformação: {transformed_features.shape}")

        log_prob = log_model.predict_proba(transformed_features)[0][1] * 100

        xgb_prediction = xgb_model.predict(xgb.DMatrix(transformed_features))

        if xgb_prediction is None or len(xgb_prediction) == 0:
            logging.error("❌ Erro: XGBoost retornou um array vazio ou None!")
            return {"error": "Erro interno ao processar o áudio."}, 500

        xgb_prob = xgb_prediction[0] * 100
        avg_prob = (log_prob + xgb_prob) / 2
        classification = "Leitura" if avg_prob > 50 else "Fala Espontânea"

        sentiment_result = sentiment_pipeline(f"O áudio foi classificado como {classification}.")[0]["label"]

        openai_validation = validate_with_openai(features, avg_prob, classification)

        if not openai_validation:
            logging.warning("⚠️ OpenAI retornou resposta vazia ou erro!")

        return {
            "file_name": filename,
            "probability_of_reading": round(avg_prob, 2),
            "classification": classification,
            "sentiment_analysis": sentiment_result,
            "openai_validation": openai_validation
        }, 200

    except Exception as e:
        logging.error(f"❌ Erro ao processar o áudio: {e}")
        return {"error": "Erro interno ao processar o áudio."}, 500
