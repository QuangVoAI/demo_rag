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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
_groq_keys_str = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in _groq_keys_str.split(",") if k.strip()]

_all_keys = []
if GROQ_API_KEY and GROQ_API_KEY not in _all_keys:
    _all_keys.append(GROQ_API_KEY)
for k in GROQ_API_KEYS:
    if k not in _all_keys:
        _all_keys.append(k)

# Phân chia Role-based keys (Key 1 cho tác vụ nhanh, Key 2 cho sinh text)
GROQ_KEY_FAST = _all_keys[0] if _all_keys else ""
GROQ_KEY_SMART = _all_keys[1] if len(_all_keys) > 1 else GROQ_KEY_FAST

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
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rooms_v1")
QDRANT_ROOMS_COLLECTION = os.getenv("QDRANT_ROOMS_COLLECTION", "rooms_v1")
# Optional gRPC endpoint (qdrant-client + rdkafka).
# Nếu để trống, client tự suy ra từ QDRANT_URL (REST port + 1).
QDRANT_GRPC_URL = os.getenv("QDRANT_GRPC_URL", "").strip()
QDRANT_SKIP_COMPAT_CHECK = _env_bool("QDRANT_SKIP_COMPAT_CHECK", True)

# --- MongoDB (Room Repository) ---
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", os.getenv("MONGODB_DB_NAME", "demo_rag"))
MONGODB_DB_NAME = MONGODB_DATABASE
MONGODB_ROOMS_COLLECTION = os.getenv("MONGODB_ROOMS_COLLECTION", "rooms")

# --- Retrieval (low-VRAM defaults) ---
# Bật chế độ tiết kiệm VRAM (mặc định: True cho card 4GB).
RAG_LOW_VRAM_MODE = _env_bool("RAG_LOW_VRAM_MODE", True)
# Cross-encoder reranker rất tốn VRAM — mặc định TẮT trong low-VRAM.
USE_RERANKER = _env_bool("USE_RERANKER", False)
RERANK_CANDIDATE_POOL = int(os.getenv("RERANK_CANDIDATE_POOL", "20"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
# Số phòng ứng viên sau lọc cứng MongoDB trước khi semantic rank (~9k rows).
RETRIEVAL_CANDIDATE_LIMIT = int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "100"))
# Tổng số ký tự evidence đưa vào LLM (cắt cứng để giảm token & latency).
EVIDENCE_MAX_CHARS = int(os.getenv("EVIDENCE_MAX_CHARS", "3500"))
# Max tokens cho mỗi lượt sinh câu trả lời.
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "1024"))
# Cắt sớm input quá dài để tránh prompt injection/log bloat và giữ latency ổn định.
MAX_USER_QUESTION_CHARS = int(os.getenv("MAX_USER_QUESTION_CHARS", "1200"))
# Nếu False, bỏ qua reviewer LLM (tiết kiệm token, latency).
ENABLE_REVIEWER = _env_bool("ENABLE_REVIEWER", False)
# Khi bật, bot từ chối trả lời nếu thiếu bằng chứng hoặc verifier phát hiện claim không an toàn.
ENABLE_ABSTAIN = _env_bool("ENABLE_ABSTAIN", True)
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
    f.strip() for f in os.getenv("METADATA_FIELDS", "room_id,room_code,district,title,amenities,address").split(",") if f.strip()
)

# --- Feedback retry loop (bounded, có log JSONL) ---
ENABLE_FEEDBACK_RETRY = _env_bool("ENABLE_FEEDBACK_RETRY", True)
FEEDBACK_MAX_RETRIEVAL_RETRIES = int(os.getenv("FEEDBACK_MAX_RETRIEVAL_RETRIES", "1"))
LOW_CONFIDENCE_MIN_DOCS = int(os.getenv("LOW_CONFIDENCE_MIN_DOCS", "1"))
LOW_CONFIDENCE_MIN_SCORE = float(os.getenv("LOW_CONFIDENCE_MIN_SCORE", "0.015"))
FEEDBACK_LOG_PATH = Path(
    os.getenv("FEEDBACK_LOG_PATH", str(FEEDBACK_LOG_DIR / "retrieval_feedback.jsonl"))
)

# --- LLM network safety ---
GROQ_REQUEST_TIMEOUT_SECONDS = float(os.getenv("GROQ_REQUEST_TIMEOUT_SECONDS", "60"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_RETRY_BASE_DELAY_SECONDS = float(os.getenv("GROQ_RETRY_BASE_DELAY_SECONDS", "2"))
STRICT_CONFIG_VALIDATION = _env_bool("STRICT_CONFIG_VALIDATION", False)

# --- Upstash Redis Cache ---
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", str(7 * 24 * 3600)))
REDIS_SESSION_TTL_SECONDS = int(os.getenv("REDIS_SESSION_TTL_SECONDS", str(24 * 3600)))

# --- Langfuse Observability ---
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "").strip()
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "").strip() or LANGFUSE_BASE_URL or "https://cloud.langfuse.com"
if LANGFUSE_HOST and not os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

# --- Tham số tìm kiếm và rewrite ---
# Số phòng tối thiểu để coi là kết quả "đủ tốt".
MIN_GOOD_ROOMS = 1
# Ngưỡng điểm để coi một phòng là phù hợp (ánh xạ sang LOW_CONFIDENCE_MIN_SCORE).
GRADE_SCORE_THRESHOLD = LOW_CONFIDENCE_MIN_SCORE

# --- Room Indexing Events ---
ROOM_CHANGED_TOPIC = os.getenv("ROOM_CHANGED_TOPIC", "room.changed")
ROOM_INDEX_DLQ_TOPIC = os.getenv("ROOM_INDEX_DLQ_TOPIC", "room.index.dlq")
ROOM_INDEX_MAX_ATTEMPTS = int(os.getenv("ROOM_INDEX_MAX_ATTEMPTS", "3"))


def validate_runtime_config(strict: bool | None = None) -> dict[str, list[str]]:
    """Validate runtime knobs without exposing configured secrets."""
    errors: list[str] = []
    warnings: list[str] = []

    if TOP_K_RETRIEVAL < 1:
        errors.append("TOP_K_RETRIEVAL must be >= 1")
    if RERANK_CANDIDATE_POOL < 1:
        errors.append("RERANK_CANDIDATE_POOL must be >= 1")
    if TOP_K_RERANK < 1:
        errors.append("TOP_K_RERANK must be >= 1")
    if EVIDENCE_MAX_CHARS < 500:
        errors.append("EVIDENCE_MAX_CHARS must be >= 500")
    if ANSWER_MAX_TOKENS < 64:
        errors.append("ANSWER_MAX_TOKENS must be >= 64")
    if MAX_USER_QUESTION_CHARS < 100:
        errors.append("MAX_USER_QUESTION_CHARS must be >= 100")
    if FEEDBACK_MAX_RETRIEVAL_RETRIES < 0:
        errors.append("FEEDBACK_MAX_RETRIEVAL_RETRIES must be >= 0")
    if GROQ_REQUEST_TIMEOUT_SECONDS < 5:
        errors.append("GROQ_REQUEST_TIMEOUT_SECONDS must be >= 5")
    if GROQ_MAX_RETRIES < 1:
        errors.append("GROQ_MAX_RETRIES must be >= 1")
    if GROQ_RETRY_BASE_DELAY_SECONDS < 0:
        errors.append("GROQ_RETRY_BASE_DELAY_SECONDS must be >= 0")
    if not GROQ_API_KEYS:
        warnings.append("GROQ_API_KEYS is empty; generation will fall back to deterministic templates")
    if not MONGODB_URI:
        warnings.append("MONGODB_URI is empty; production repository will return no rooms")

    should_raise = STRICT_CONFIG_VALIDATION if strict is None else strict
    if should_raise and errors:
        raise RuntimeError("Invalid runtime config: " + "; ".join(errors))
    return {"errors": errors, "warnings": warnings}
