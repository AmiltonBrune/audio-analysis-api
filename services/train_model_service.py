import logging
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import nltk
from transformers import pipeline
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.settings import XGB_MODEL_PATH, LOG_MODEL_PATH, SCALER_PATH, FEATURES_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

nltk.download("vader_lexicon")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"❌ Arquivo não encontrado: {FEATURES_FILE}. Execute `python extract_audio_features.py` primeiro.")

df = pd.read_csv(FEATURES_FILE)
logging.info(f"✅ Arquivo carregado com sucesso! Total de amostras: {df.shape[0]}")

X = df.drop("label", axis=1)

y = df["label"].map({"Leitura": 1, "Fala Espontânea": 0})

if y.isnull().sum() > 0:
    logging.error("❌ Algumas labels não foram corretamente mapeadas!")
    raise ValueError("Erro ao mapear labels, verifique o dataset.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logging.info(f"📊 Divisão dos dados: {X_train.shape[0]} para treino e {X_test.shape[0]} para teste.")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 4,
    "verbosity": 1
}

logging.info("🚀 Treinando modelo XGBoost...")
xgb_model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dtest, "eval")], early_stopping_rounds=50)
logging.info("✅ Treinamento concluído!")

log_pred = log_model.predict(X_test)
xgb_pred = (xgb_model.predict(dtest) > 0.5).astype(int)

log_acc = accuracy_score(y_test, log_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

logging.info(f"✅ Acurácia Logistic Regression: {log_acc * 100:.2f}%")
logging.info(f"✅ Acurácia XGBoost: {xgb_acc * 100:.2f}%")

logging.info("\n🔍 Relatório de Classificação - Logistic Regression:\n" +
             classification_report(y_test, log_pred, target_names=["Fala Espontânea", "Leitura"]))

logging.info("\n🔍 Relatório de Classificação - XGBoost:\n" +
             classification_report(y_test, xgb_pred, target_names=["Fala Espontânea", "Leitura"]))

os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
dump(log_model, LOG_MODEL_PATH)
dump(xgb_model, XGB_MODEL_PATH)
dump(scaler, SCALER_PATH)

logging.info("✅ Modelos e scaler salvos com sucesso!")
