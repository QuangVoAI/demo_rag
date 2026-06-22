"""Cấu hình chung cho Nhatrovn Room Assistant."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

# --- Đường dẫn project ---
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FEEDBACK_LOG_DIR = DATA_DIR / "state" / "retrieval_feedback"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Nạp biến môi trường ---
load_dotenv(ENV_FILE)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- HuggingFace Login ---
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    except Exception:
        pass

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", GROQ_API_KEY).split(",") if k.strip()]
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# --- Cấu hình embedding model ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024   # Chiều output của bge-m3
EMBEDDING_VERSION = int(os.getenv("EMBEDDING_VERSION", "1"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# --- Cấu hình assistant ---
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Nhatrovn Assistant")

# --- Kafka ---
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")

# --- Qdrant ---
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "listings_v1")
QDRANT_LISTINGS_COLLECTION = os.getenv("QDRANT_LISTINGS_COLLECTION", "listings_v1")
# Optional gRPC endpoint (qdrant-client + rdkafka).
# Nếu để trống, client tự suy ra từ QDRANT_URL (REST port + 1).
QDRANT_GRPC_URL = os.getenv("QDRANT_GRPC_URL", "").strip()
QDRANT_SKIP_COMPAT_CHECK = _env_bool("QDRANT_SKIP_COMPAT_CHECK", True)

# --- MongoDB (Listing Repository) ---
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "nhatrovn")
MONGODB_LISTINGS_COLLECTION = os.getenv("MONGODB_LISTINGS_COLLECTION", "listings")

# --- Retrieval (low-VRAM defaults) ---
# Bật chế độ tiết kiệm VRAM (mặc định: True cho card 4GB).
RAG_LOW_VRAM_MODE = _env_bool("RAG_LOW_VRAM_MODE", True)
# Cross-encoder reranker rất tốn VRAM — mặc định TẮT trong low-VRAM.
USE_RERANKER = _env_bool("USE_RERANKER", False)
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))
# Tổng số ký tự evidence đưa vào LLM (cắt cứng để giảm token & latency).
EVIDENCE_MAX_CHARS = int(os.getenv("EVIDENCE_MAX_CHARS", "3500"))
# Max tokens cho mỗi lượt sinh câu trả lời.
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "1024"))
# Nếu False, bỏ qua reviewer LLM (tiết kiệm token, latency).
ENABLE_REVIEWER = _env_bool("ENABLE_REVIEWER", False)
# 0 = tắt rewrite loop; legacy graph luôn dùng config này.
MAX_REWRITE_RETRIES = int(os.getenv("MAX_REWRITE_RETRIES", "0"))
# Nếu True và VRAM không đủ, fallback embedding cho router.
ROUTER_EMBEDDING_FALLBACK = _env_bool("ROUTER_EMBEDDING_FALLBACK", False)
# Embedding dtype: "fp16" nếu có GPU, "fp32" nếu CPU. None -> auto.
EMBEDDING_DTYPE = os.getenv("EMBEDDING_DTYPE", "").strip().lower() or None

# --- Hybrid Search RRF ---
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "0.6"))
RRF_SPARSE_WEIGHT = float(os.getenv("RRF_SPARSE_WEIGHT", "0.4"))
RRF_K = int(os.getenv("RRF_K", "60"))

# --- Metadata Search (boost theo tín hiệu rõ từ query) ---
METADATA_BOOST = float(os.getenv("METADATA_BOOST", "0.5"))
METADATA_FIELDS = tuple(
    f.strip() for f in os.getenv("METADATA_FIELDS", "listing_id,district,title,amenities,address").split(",") if f.strip()
)

# --- Feedback retry loop (bounded, có log JSONL) ---
ENABLE_FEEDBACK_RETRY = _env_bool("ENABLE_FEEDBACK_RETRY", True)
FEEDBACK_MAX_RETRIEVAL_RETRIES = int(os.getenv("FEEDBACK_MAX_RETRIEVAL_RETRIES", "1"))
LOW_CONFIDENCE_MIN_DOCS = int(os.getenv("LOW_CONFIDENCE_MIN_DOCS", "1"))
LOW_CONFIDENCE_MIN_SCORE = float(os.getenv("LOW_CONFIDENCE_MIN_SCORE", "0.015"))
FEEDBACK_LOG_PATH = Path(
    os.getenv("FEEDBACK_LOG_PATH", str(FEEDBACK_LOG_DIR / "retrieval_feedback.jsonl"))
)

# --- Upstash Redis Cache ---
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", str(7 * 24 * 3600)))
REDIS_SESSION_TTL_SECONDS = int(os.getenv("REDIS_SESSION_TTL_SECONDS", str(24 * 3600)))
ENABLE_DYNAMIC_LISTING_ANSWER_CACHE = os.getenv("ENABLE_DYNAMIC_LISTING_ANSWER_CACHE", "false").lower() == "true"

# --- Langfuse Observability ---
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Tham số tìm kiếm và rewrite ---
# Số listing tối thiểu để coi là kết quả "đủ tốt" (giữ alias, dùng cho grader/feedback).
MIN_GOOD_LISTINGS = 1
# Ngưỡng điểm để coi một listing là phù hợp (ánh xạ sang LOW_CONFIDENCE_MIN_SCORE).
GRADE_SCORE_THRESHOLD = LOW_CONFIDENCE_MIN_SCORE

# --- Listing Indexing Events ---
LISTING_CHANGED_TOPIC = os.getenv("LISTING_CHANGED_TOPIC", "listing.changed")
LISTING_INDEX_DLQ_TOPIC = os.getenv("LISTING_INDEX_DLQ_TOPIC", "listing.index.dlq")
LISTING_INDEX_MAX_ATTEMPTS = int(os.getenv("LISTING_INDEX_MAX_ATTEMPTS", "3"))
