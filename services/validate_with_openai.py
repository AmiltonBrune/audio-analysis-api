import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def validate_with_openai(features, probability, classification):
    prompt = f"""
        Um áudio foi analisado e as seguintes características foram extraídas:
        - ⏳ Duração: {features[0]:.2f}s
        - 🎼 Frequência Média: {features[1]:.2f}
        - 🎵 Espectral Banda: {features[2]:.2f}
        - 🔄 Taxa de cruzamento por zero: {features[3]:.2f}
        - ⚡ Energia Média: {features[4]:.2f}
        - 🗣 Velocidade de fala: {features[5]:.2f} palavras/minuto
        - 🔁 Padrão de repetição: {features[6]:.2f}
        - 🔇 Taxa de silêncio: {features[7]:.2f}
        - 📈 Variação de entonação: {features[8]:.2f}
        - ⏩ Variação de velocidade da fala: {features[9]:.2f}
        - 🔊 Frequência com maior amplitude (FFT Peak Freq): {features[10]:.2f} Hz
        - 🎤 Frequência Fundamental (Pitch Tracking):** {features[11]:.2f} Hz

        📖 **Regras gerais sobre Leitura e Fala Espontânea**:
        - **Leitura**: Tem pausas **mais regulares**, menor variação na **velocidade** e na **entonação**.
        - **Fala Espontânea**: Possui pausas **irregulares**, variação maior na **velocidade** e entonação mais dinâmica.

        📌 **O modelo de Machine Learning classificou esse áudio como `{classification}` com uma certeza de `{probability:.2f}%`.**

        🚀 **Tarefa para você (IA)**:
        1. **Com base nas características do áudio, a classificação `{classification}` parece correta?**  
        2. **Se não, qual seria a classificação mais apropriada?**  
        3. **Justifique a resposta de forma objetiva, usando os dados apresentados.**

        📍 **IMPORTANTE**: Se não tiver certeza, explique quais características do áudio indicam dúvida e qual seria um critério mais confiável.

        """

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Você é um especialista em análise de áudio e padrões de fala."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro ao validar com OpenAI: {str(e)}"
