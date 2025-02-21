# Usa uma imagem oficial do Python 3.10 como base
FROM python:3.12.9

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos necessários para dentro do container
COPY requirements.txt ./
COPY main.py ./
COPY .env ./

# Instala as dependências da aplicação
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta (caso precise rodar um serviço)
EXPOSE 8080

# Comando para rodar a aplicação
CMD ["python", "main.py"]
