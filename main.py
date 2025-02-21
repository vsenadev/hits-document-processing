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


async def connect_to_mongo():
    """Estabelece conexão com o MongoDB."""
    global client, db
    raw_uri = os.getenv('DATABASE_URL')

    if "@" in raw_uri:
        scheme, rest = raw_uri.split("://", 1)
        creds, host = rest.split("@", 1)
        user, password = creds.split(":", 1)

        # Escapando usuário e senha corretamente
        user_escaped = urllib.parse.quote_plus(user)
        password_escaped = urllib.parse.quote_plus(password)

        # Montando a URI escapada
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

# 🔹 Splitter para dividir textos corretamente
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800, chunk_overlap=50, separators=["\n\n", "\n", " "]
)


def clean_text(text):
    """Remove caracteres indesejados e normaliza espaços e quebras de linha."""
    text = text.replace("\n", " ")  # Substitui quebras de linha por espaço
    text = text.replace("\u0005", "").replace("\u0002", "").replace("\u001d", "")  # Remove caracteres invisíveis
    text = ' '.join(text.split())  # Remove múltiplos espaços
    return text


async def process_document():
    """Processa um documento do MongoDB e armazena embeddings no Elasticsearch."""
    global db

    if db is None:
        logging.error("❌ Database não conectado. Tentando conectar novamente...")
        await connect_to_mongo()

    try:
        # 🔹 Obtém um único documento com status = 0
        document = await db['document'].find_one_and_update(
            {"status": 0}, {"$set": {"status": 1}}
        )

        if not document:
            logging.info("🔍 Nenhum documento pendente para processamento.")
            return

        filename = document.get("filename")
        base64_content = document.get("file_content")
        enterprise_id = document.get("enterprise_id")

        logging.info(f"📄 Processando documento: {filename}")

        # 🔹 Obtém o nome da empresa a partir do enterprise_id
        enterprise = await db['enterprise'].find_one({"_id": enterprise_id})
        if not enterprise:
            logging.warning(f"⚠️ Empresa com ID {enterprise_id} não encontrada.")
            return

        enterprise_name = enterprise.get("name")
        index_name = enterprise_name.replace(" ", "").lower()

        logging.info(f"🏢 Empresa: {enterprise_name} | Índice no Elasticsearch: {index_name}")

        # 🔹 Decodifica o conteúdo base64 e salva o arquivo temporariamente
        temp_folder = "/tmp"
        os.makedirs(temp_folder, exist_ok=True)
        file_path = os.path.join(temp_folder, filename)

        with open(file_path, "wb") as file:
            file.write(base64.b64decode(base64_content))

        # 🔹 Verifica o tipo de arquivo e carrega corretamente
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)  # Usa PyMuPDFLoader para melhor extração
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            logging.error(f"❌ Tipo de arquivo não suportado: {filename}")
            return

        docs = loader.load()

        # 🔹 Limpa e formata os textos extraídos
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        # 🔹 Divide o texto em chunks
        docs_split = text_splitter.split_documents(docs)

        # # 🔹 Armazena os embeddings no Elasticsearch
        # if es.indices.exists(index=index_name):
        #     es.indices.delete(index=index_name)

        es_store = ElasticsearchStore.from_documents(
            docs, embedding_model, es_cloud_id=os.getenv('ES_CLOUD_ID'), es_user=os.getenv('ES_USER'),
            es_password=os.getenv('ES_PASSWORD'), index_name=index_name
        )
        es_store.client.indices.refresh(index=index_name)

        logging.info(f"✅ Documento '{filename}' processado e armazenado no Elasticsearch.")

        # 🔹 Remove o arquivo temporário
        os.remove(file_path)

    except Exception as e:
        logging.error(f"❌ Erro ao processar documento: {e}")


async def main():
    """Loop assíncrono para buscar documentos continuamente."""
    await connect_to_mongo()

    while True:
        await process_document()
        await asyncio.sleep(60)  # Espera 60 segundos antes de buscar novamente


if __name__ == "__main__":
    logging.info("🚀 Iniciando o processamento contínuo...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("🛑 Interrompido pelo usuário.")
    finally:
        asyncio.run(close_mongo_connection())
