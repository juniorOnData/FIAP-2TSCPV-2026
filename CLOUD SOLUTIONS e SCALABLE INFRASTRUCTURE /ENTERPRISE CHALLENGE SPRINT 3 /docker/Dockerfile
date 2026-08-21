# Imagem base leve com Python já instalado
FROM python:3.11-slim

# Pasta onde a aplicação vai viver dentro do container
WORKDIR /app

# Instala as dependências primeiro (aproveita cache de build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API e o modelo treinado
COPY app.py .
COPY modelo_incidentes.pkl .

# Porta que o container expõe
EXPOSE 80

# Comando que sobe a API quando o container inicia
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
