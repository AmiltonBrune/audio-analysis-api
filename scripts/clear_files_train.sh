#!/bin/bash

echo "🧹 Limpando arquivos antigos..."
rm -rf ../models/xgb_model.joblib
rm -rf ../models/scaler.joblib
rm -rf ../models/log_model.joblib
rm -rf ../data/audio_features.csv
echo "✅ Arquivos apagados!"