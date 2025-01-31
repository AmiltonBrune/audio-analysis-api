# 📖 Análise de Áudio: Classificação entre Leitura e Fala Espontânea

Este projeto analisa arquivos de áudio para determinar se são gravações de leitura de texto ou de fala espontânea. Ele utiliza técnicas de Machine Learning, Processamento de Áudio e validação com OpenAI GPT para obter alta precisão.

---

## 🚀 Arquitetura e Design da Solução

A solução é composta pelos seguintes componentes:

1️⃣ **Coleta de Dados**  
   - Utiliza datasets como Mozilla Common Voice (leitura) e LibriSpeech (fala espontânea).

2️⃣ **Processamento e Extração de Features**  
   - Conversão de formatos de áudio para WAV com `pydub`.  
   - Extração de características acústicas com `librosa`, incluindo:  
     - Duração, energia, taxa de cruzamento por zero, taxa de silêncio, entre outros.  
     - Análise de frequência com FFT.  
     - Extração de frequência fundamental (Pitch Tracking).  

3️⃣ **Treinamento de Modelos**  
   - `XGBoost` é usado para classificação.  
   - `StandardScaler` normaliza as features.  
   - Modelos são avaliados com relatórios de classificação e métricas de acurácia.  

4️⃣ **API Flask com Swagger**  
   - Endpoint `/analyze-audio` para análise de áudio.  
   - OpenAI GPT é usado para validação refinada.  

### 📌 Diagrama da Arquitetura

[![](https://mermaid.ink/img/pako:eNp9lMty2jAUhl9Fo5nsSAokXBedIdwC4RZM0k5NFgdbgKa25Eo2ExKy6KNkushMt30Ev1hlKSRqc_Gw4Dv6fe7yHfa4T3AdrwREazRrzRlSj0wWxjDHlzJJHwTlc2yOsqfhXkYBBx_5BKU_E5_ya3NImD9nr1w0Jj3UCUB-t32culPikQVBHKUPmQu0oYCU9PpF03QdCDaAGEc-FSRO_6hEsqCJDi8tacttriEE5U0SsaHpo9aRm1hA-pj-0rQkECeC2K-1n18LVR8CLRuCt6aMoAEBwShbWfKOewUB9QF5PETjiLCGnW7X7RIBSEWIuIwB9Z3x6MPGTAT3iJQQEhbzl2babTpzm5xtiIgJikA5_9K40sEn21aysGL33HZWK30uUqsGdCG4BEvXdxtMlSABLQX5kaS_mUdNOY5HJ1tLee42IfCSAFRHWUxNjplwJoDJJRchEfLTaDA7_7DG_7tpFzdwh6brX7unXLXMij7cHw34isqYemhKVqosSTmzZCPXiYH5IHzHg4CID1MxszPr8DJAO6Gx22Ybs2KR4GEUm6YDSx9Uz4gVeKLW92nMah3-WYW3g09JzEW2yHrJ7aAXblNdDkmX1Nvv6v5OWAGnrtqWBSxoVoPKT_0CQtWo7ek6bj9Rzco8xXTzYWoHB-pOJje61hb4XBpzAx0eft5Nxs4MfQK1KdtbcghZKjt0ahSnmQI1DTQ1nBk409Az0NPQN9DXcG7gXMPTl6alYWDD0MDAZCLIhkrVlB1qG_vwHftI20dqL9WYb-HZ3tZOOwY6GsYGxhomBiavZV0DXQ0XNkxtcAxc6Pj7MWd3f4ca5mj6_pHz9hHOYXW9QqC--jbfZdI5jtckJHNcV3_VwuuP6b3SQRJzZ8s8XI9FQnJY8GS1xvUlBFJREvkQkxYFtYfhXhIB-8a5jbh-h29wvVipHZUrhWK1ki8cl_O1Yg5vcf2wWjqqFGvV4vFxuVIu5SvV-xy-1Q4KR4VS_rhWOCnlC9Vy6aRcvv8L0orulw?type=png)](https://mermaid.live/edit#pako:eNp9lMty2jAUhl9Fo5nsSAokXBedIdwC4RZM0k5NFgdbgKa25Eo2ExKy6KNkushMt30Ev1hlKSRqc_Gw4Dv6fe7yHfa4T3AdrwREazRrzRlSj0wWxjDHlzJJHwTlc2yOsqfhXkYBBx_5BKU_E5_ya3NImD9nr1w0Jj3UCUB-t32culPikQVBHKUPmQu0oYCU9PpF03QdCDaAGEc-FSRO_6hEsqCJDi8tacttriEE5U0SsaHpo9aRm1hA-pj-0rQkECeC2K-1n18LVR8CLRuCt6aMoAEBwShbWfKOewUB9QF5PETjiLCGnW7X7RIBSEWIuIwB9Z3x6MPGTAT3iJQQEhbzl2babTpzm5xtiIgJikA5_9K40sEn21aysGL33HZWK30uUqsGdCG4BEvXdxtMlSABLQX5kaS_mUdNOY5HJ1tLee42IfCSAFRHWUxNjplwJoDJJRchEfLTaDA7_7DG_7tpFzdwh6brX7unXLXMij7cHw34isqYemhKVqosSTmzZCPXiYH5IHzHg4CID1MxszPr8DJAO6Gx22Ybs2KR4GEUm6YDSx9Uz4gVeKLW92nMah3-WYW3g09JzEW2yHrJ7aAXblNdDkmX1Nvv6v5OWAGnrtqWBSxoVoPKT_0CQtWo7ek6bj9Rzco8xXTzYWoHB-pOJje61hb4XBpzAx0eft5Nxs4MfQK1KdtbcghZKjt0ahSnmQI1DTQ1nBk409Az0NPQN9DXcG7gXMPTl6alYWDD0MDAZCLIhkrVlB1qG_vwHftI20dqL9WYb-HZ3tZOOwY6GsYGxhomBiavZV0DXQ0XNkxtcAxc6Pj7MWd3f4ca5mj6_pHz9hHOYXW9QqC--jbfZdI5jtckJHNcV3_VwuuP6b3SQRJzZ8s8XI9FQnJY8GS1xvUlBFJREvkQkxYFtYfhXhIB-8a5jbh-h29wvVipHZUrhWK1ki8cl_O1Yg5vcf2wWjqqFGvV4vFxuVIu5SvV-xy-1Q4KR4VS_rhWOCnlC9Vy6aRcvv8L0orulw)

---

## 🛠 **Como Executar Localmente**

### **1️⃣ Instalar Dependências**
Certifique-se de ter o Python 3.8+ instalado e execute:

```bash
pip install -r requirements.txt
```

### **2️⃣ Configurar Variáveis de Ambiente**
Crie um arquivo `.env` na raiz do projeto com:

```bash
OPENAI_API_KEY=your_openai_api_key
```

### **3️⃣ Baixar os Datasets**

Acesse o link para baixar os datasets do Google Drive:

```bash
https://drive.google.com/drive/folders/1qnwriH2xSBYTr8id8IBuEMbJEvvEepSV
```

### **4️⃣ Processar e Treinar os Modelos**

- **Extrair Features:**
  ```bash
  python data/extract_features.py
  ```

- **Treinar Modelos:**
  ```bash
  python services/train_model_service.py
  ```

### **5️⃣ Iniciar a API**

```bash
python api/app.py
```
Acesse a API em `http://localhost:5000`

---

## 🔬 **Executando Testes**

Para rodar os testes unitários, execute:

```bash
pytest tests/
```

---

## 📊 **Processo de Análise de Áudio**

1️⃣ **Conversão do áudio para WAV** (caso necessário).

2️⃣ **Extração de Features Acústicas:**
   - Frequência média, taxa de cruzamento por zero, taxa de silêncio, pitch variability.
   - 
3️⃣ **Normalização das Features com StandardScaler.**

4️⃣ **Previsão Inicial do XGBoost.**

5️⃣ **Validação com OpenAI GPT** para refinamento do resultado.

---

## 📂 **Estrutura do Projeto**

```
audio_analysis_api/
├── api/                  # Código da API Flask
│   ├── app.py            # Arquivo principal da API
│   └── ...
├── config/               # Configurações do projeto
├── data/                 # Scripts para manipular dados e features
├── datasets/             # Pasta para armazenar datasets (não versionada)
├── docs/                 # Configurações do Swagger
├── models/               # Modelos treinados e scripts relacionados
├── scripts/              # Scripts utilitários (ex: download de datasets)
├── services/             # Serviços para análise e treinamento
└── tests/                # Testes unitários
```

---

## 🐳 **Execução com Docker**

### **1️⃣ Construir e Subir os Containers**

```bash
docker-compose up --build
```

### **2️⃣ Acessar a API**

- **URL:** `http://localhost:5000`
- **Documentação Swagger:** `http://localhost:5000/apidocs`

----
## 🚀 **Conclusão**

Este projeto combina Machine Learning, Processamento de Áudio e IA generativa para classificar gravações de leitura e fala espontânea com alta precisão. O uso de FFT e Pitch Tracking melhora a acurácia, enquanto a OpenAI GPT auxilia na validação, tornando a solução mais robusta e confiável.
