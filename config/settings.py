import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_top_p = float(os.getenv("LLM_TOP_P", "0.9"))
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # Redis 配置
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    redis_db = int(os.getenv("REDIS_DB", "0"))
    redis_ttl = int(os.getenv("REDIS_TTL", "86400"))  # 缓存默认24小时过期

    # Neo4j 图数据库配置
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    file_storage_backend = os.getenv("FILE_STORAGE_BACKEND", "local")
    file_public_base_url = os.getenv("FILE_PUBLIC_BASE_URL", "/files")
    local_file_storage_dir = os.getenv("LOCAL_FILE_STORAGE_DIR", "rag_files")
    minio_public_base_url = os.getenv("MINIO_PUBLIC_BASE_URL", "")
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY", "")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY", "")
    minio_bucket = os.getenv("MINIO_BUCKET", "fixagent-rag")
    minio_document_bucket = os.getenv("MINIO_DOCUMENT_BUCKET", minio_bucket)
    minio_public_image_bucket = os.getenv("MINIO_PUBLIC_IMAGE_BUCKET", minio_bucket)
    minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    image_summary_llm_enabled = os.getenv("IMAGE_SUMMARY_LLM_ENABLED", "false").lower() == "true"


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
