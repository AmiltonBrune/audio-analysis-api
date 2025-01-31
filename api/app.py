from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flasgger import Swagger
import os
import logging
from services.audio_analysis_service import analyze_audio
from config.settings import UPLOAD_FOLDER

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
swagger = Swagger(app, template_file="../docs/swagger_config.yaml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@app.route("/analyze-audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Nome do arquivo inválido."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    logging.info(f"📂 Arquivo salvo em {file_path}")

    response, status_code = analyze_audio(file_path, filename)
    return jsonify(response), status_code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
