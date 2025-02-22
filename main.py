import os
import time
import asyncio
import base64
import logging
import urllib
import schedule
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from fastapi import FastAPI

# 🔹 Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# 🔹 Carrega variáveis de ambiente
load_dotenv()

client = None
db = None
app = FastAPI()

async def connect_to_mongo():
    """Estabelece conexão com o MongoDB."""
    global client, db
    raw_uri = os.getenv('DATABASE_URL')

    if "@" in raw_uri:
        scheme, rest = raw_uri.split("://", 1)
        creds, host = rest.split("@", 1)
        user, password = creds.split(":", 1)

        user_escaped = urllib.parse.quote_plus(user)
        password_escaped = urllib.parse.quote_plus(password)

        escaped_uri = f"{scheme}://{user_escaped}:{password_escaped}@{host}"
    else:
        escaped_uri = raw_uri

    client = AsyncIOMotorClient(escaped_uri)
    db = client['hits']
    logging.info("✅ Conectado ao MongoDB")


async def close_mongo_connection():
    """Fecha a conexão com o MongoDB."""
    global client
    if client:
        client.close()
        logging.info("🔌 Desconectado do MongoDB")


# 🔹 Configuração do Elasticsearch
es = Elasticsearch(
    os.getenv('ES_URL'),
    basic_auth=(os.getenv('ES_USER'), os.getenv('ES_PASSWORD')),
    request_timeout=60,
    max_retries=10,
    retry_on_timeout=True
)

# 🔹 Configuração do OpenAI Embeddings
embedding_model = AzureOpenAIEmbeddings(
    api_version=os.getenv('OPENAI_API_VERSION_EMBEDDING'),
    api_key=os.getenv('OPENAI_API_KEY'),
    azure_endpoint=os.getenv('EMBEDDING'),
    azure_deployment="text-embedding-large",
    dimensions=3072,
    chunk_size=1
)

# 🔹 Splitter de texto
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=50, separators=["\n\n", "\n", " "])

def clean_text(text):
    """Remove caracteres indesejados e normaliza espaços e quebras de linha."""
    return ' '.join(text.replace("\n", " ").split())


async def process_document():
    """Processa um documento do MongoDB e armazena embeddings no Elasticsearch."""
    global db

    if db is None:
        logging.error("❌ Database não conectado. Tentando conectar novamente...")
        await connect_to_mongo()

    try:
        document = await db['document'].find_one_and_update({"status": 0}, {"$set": {"status": 1}})
        if not document:
            logging.info("🔍 Nenhum documento pendente para processamento.")
            return

        filename = document.get("filename")
        base64_content = document.get("file_content")
        enterprise_id = document.get("enterprise_id")

        logging.info(f"📄 Processando documento: {filename}")

        enterprise = await db['enterprise'].find_one({"_id": enterprise_id})
        if not enterprise:
            logging.warning(f"⚠️ Empresa com ID {enterprise_id} não encontrada.")
            return

        enterprise_name = enterprise.get("name")
        index_name = enterprise_name.replace(" ", "").lower()

        temp_folder = "/tmp"
        os.makedirs(temp_folder, exist_ok=True)
        file_path = os.path.join(temp_folder, filename)

        with open(file_path, "wb") as file:
            file.write(base64.b64decode(base64_content))

        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            logging.error(f"❌ Tipo de arquivo não suportado: {filename}")
            return

        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        docs_split = text_splitter.split_documents(docs)

        es_store = ElasticsearchStore.from_documents(
            docs, embedding_model, es_cloud_id=os.getenv('ES_CLOUD_ID'),
            es_user=os.getenv('ES_USER'), es_password=os.getenv('ES_PASSWORD'),
            index_name=index_name
        )
        es_store.client.indices.refresh(index=index_name)

        logging.info(f"✅ Documento '{filename}' processado e armazenado no Elasticsearch.")
        os.remove(file_path)

    except Exception as e:
        logging.error(f"❌ Erro ao processar documento: {e}")


async def process_loop():
    """Loop assíncrono para processar documentos continuamente."""
    await connect_to_mongo()
    while True:
        await process_document()
        await asyncio.sleep(60)

@app.get("/")
async def read_root():
    return {"status": "API rodando 🚀"}

@app.post("/process")
async def trigger_processing():
    await process_document()
    return {"message": "Processamento iniciado"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_loop())

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()
