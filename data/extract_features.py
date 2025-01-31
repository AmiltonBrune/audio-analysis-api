import logging
import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import joblib
from config.settings import DATASET_DIRS, SCALER_PATH, FEATURES_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

feature_names = [
    "duration", "mean_freq", "spectral_bandwidth", "zero_crossing_rate", "energy",
    "silence_ratio", "fft_peak_freq", "pitch_mean", "spectral_entropy", "spectral_rolloff", "chroma_mean",
    "speech_rate", "avg_pause_duration", "pitch_variability"
]

def convert_audio(file_path):
    try:
        logging.info(f"🔄 Convertendo {file_path} para WAV...")
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_path, format="wav")
        logging.info(f"✅ Conversão concluída: {wav_path}")
        return wav_path
    except Exception as e:
        logging.error(f"❌ Erro ao converter {file_path} para WAV: {e}")
        return None

def extract_audio_features(file_path):
    try:
        logging.info(f"📂 Processando arquivo: {file_path}")

        if not file_path.endswith(".wav"):
            file_path = convert_audio(file_path)
            if not file_path:
                return None

        y, sr = librosa.load(file_path, sr=None)
        logging.info(f"🎵 Áudio carregado com sucesso. Taxa de amostragem: {sr}")

        duration = librosa.get_duration(y=y, sr=sr)
        mean_freq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        energy = np.mean(y ** 2)
        silence_ratio = np.sum(np.abs(y) < np.percentile(np.abs(y), 10)) / len(y)

        fft_spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1 / sr)
        fft_peak_freq = freqs[np.argmax(fft_spectrum)] if len(freqs) > 0 else 0

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes > np.median(magnitudes)) else 0

        spectral_entropy = entropy(librosa.feature.spectral_flatness(y=y)[0])
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

        speech_rate = len(librosa.effects.split(y)) / duration if duration > 0 else 0  # Estimativa da velocidade da fala
        avg_pause_duration = np.mean(librosa.effects.split(y)) if len(librosa.effects.split(y)) > 1 else 0
        pitch_variability = np.std(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes > np.median(magnitudes)) else 0

        features = [
            duration, mean_freq, spectral_bandwidth, zero_crossing_rate, energy,
            silence_ratio, fft_peak_freq, pitch_mean, spectral_entropy, spectral_rolloff, chroma_mean,
            speech_rate, avg_pause_duration, pitch_variability
        ]

        logging.info(f"✅ Features extraídas com sucesso para {file_path}: {features}")
        return features
    except Exception as e:
        logging.error(f"❌ Erro ao processar {file_path}: {e}")
        return None

def process_features():
    features, labels = [], []

    for dataset_path, label in DATASET_DIRS.items():
        for file in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file)

            if not file_path.endswith((".wav", ".mp3", ".ogg", ".flac")):
                continue

            feature_data = extract_audio_features(file_path)
            if feature_data:
                features.append(feature_data)
                labels.append(label)

    if not features:
        logging.error("❌ Nenhuma feature extraída! Verifique os áudios e tente novamente.")
        return

    df = pd.DataFrame(features, columns=feature_names)
    df["label"] = labels

    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1].astype(np.float64)).astype(np.float32)

    df.to_csv(FEATURES_FILE, index=False)
    logging.info("✅ Extração concluída e normalização aplicada!")

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"✅ StandardScaler treinado e salvo em {SCALER_PATH}")

if __name__ == "__main__":
    process_features()