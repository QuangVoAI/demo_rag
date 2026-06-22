"""
Metadata-first retrieval: trích tín hiệu rõ từ câu hỏi trước khi vào hybrid search.

Mục tiêu:
  - Nếu người dùng nhắc tới mã phòng (#A101), arxiv-style id (2604.08423v1) hoặc
    quận/huyện/thành phố rõ ràng, kéo document đó lên đầu với điểm boost.
  - Không thay thế semantic search; chỉ thêm 1 lớp "biết đường" trước khi gọi dense+sparse.

Public API:
  - extract_metadata_signals(query) -> dict[str, list[str]]
  - metadata_lookup(signals, repository) -> dict[str, dict]  (id -> payload)
  - score_metadata_hit(payload, signals, fields) -> float (0..1)
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# --- Patterns ---------------------------------------------------------------
# Listing id dạng #A101, #B_202, #listing-12 (bắt cả tiếng Việt có dấu) — case-insensitive.
_LISTING_ID_RE = re.compile(r"#\s*([A-Za-z0-9_\-]{1,32})")
# arXiv-style: YYMM.NNNNN(vN). Cho phép cả "2604.08423" và "2604.08423v1".
_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
# Mã phòng không có dấu "#": A101, B202 ở đầu câu hoặc sau khoảng trắng.
_BARE_ID_RE = re.compile(r"\b([A-Z]{1,3}\d{2,5})\b")

# Quận/huyện/thành phố phổ biến (lowercase, accent-insensitive).
_LOCATION_KEYWORDS: tuple[str, ...] = (
    "quận", "huyện", "thị xã", "thành phố", "tp.", "tp ",
    "district", "ward", "province", "city",
)

_SOFT_LOCATIONS: tuple[str, ...] = (
    "bình thạnh", "bình thạn", "binh thanh",
    "quận 1", "quận 7", "quận 10", "quận 12",
    "gò vấp", "go vap", "tân bình", "tan binh",
    "phú nhuận", "phu nhuan", "thủ đức", "thu duc",
    "bình tân", "bình chánh", "củ chi", "hóc môn",
    "nhà bè", "cần giờ", "tân phú",
)

_TERM_STOPWORDS = {
    "can", "co", "cho", "gan", "gia", "hoi", "la", "minh", "mot", "nha",
    "o", "phong", "quan", "the", "tim", "toi", "tro", "va", "voi",
}


def _strip_accents(text: str) -> str:
    import unicodedata
    norm = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def _norm(text: str) -> str:
    return _strip_accents((text or "").lower())


def extract_metadata_signals(query: str) -> dict[str, list[str]]:
    """
    Trích tín hiệu rõ ràng từ câu hỏi.

    Returns:
        {
          "listing_id": [...],   # mã phòng (#A101 hoặc bare A101)
          "arxiv_like": [...],   # chuỗi YYMM.NNNNN hoặc YYMM.NNNNNvN
          "district":   [...],   # snippet có từ khoá khu vực
        }
    """
    q = (query or "").strip()
    q_lower = q.lower()
    q_norm = _norm(q)

    listing_ids: list[str] = []
    for m in _LISTING_ID_RE.findall(q):
        listing_ids.append(m.upper())
    for m in _BARE_ID_RE.findall(q):
        upper = m.upper()
        # Bỏ qua nếu trùng listing_id đã bắt được.
        if upper not in listing_ids:
            listing_ids.append(upper)

    arxiv_ids = [(a + b) for a, b in _ARXIV_RE.findall(q)]

    districts: list[str] = []
    for needle in _SOFT_LOCATIONS:
        if needle in q_norm and needle not in districts:
            districts.append(needle)
    if any(kw in q_lower for kw in _LOCATION_KEYWORDS):
        # Nếu câu chứa từ khoá khu vực nhưng chưa match snippet nào,
        # giữ lại cả câu để bước sau fuzzy-match trên district/ward/province.
        districts.append(q_lower.strip())

    return {
        "listing_id": listing_ids,
        "arxiv_like": arxiv_ids,
        "district": districts,
        "query": [q] if q else [],
    }


def metadata_signal_present(signals: dict[str, list[str]]) -> bool:
    """Return true when the query has explicit metadata-like hints."""
    return bool(
        signals.get("listing_id")
        or signals.get("arxiv_like")
        or signals.get("district")
    )


def _terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9]{2,}", _norm(text)))
    return {term for term in terms if term not in _TERM_STOPWORDS}


def _get_field(payload: dict[str, Any], field: str) -> Any:
    """Lấy field từ payload, hỗ trợ dotted path (vd: location.district)."""
    if payload is None:
        return None
    cur: Any = payload
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def score_metadata_hit(
    payload: dict[str, Any] | None,
    signals: dict[str, list[str]],
    fields: Iterable[str] = ("listing_id", "district", "title", "amenities"),
) -> float:
    """
    Tính điểm boost metadata cho 1 payload. Trả về giá trị 0..1.

    Trọng số mặc định:
      listing_id  = 1.0  (match chính xác)
      arxiv_like  = 1.0  (match chính xác)
      district    = 0.5  (substring match, accent-insensitive)
      title       = 0.3
      amenities   = 0.2
    """
    if not payload or not signals:
        return 0.0

    score = 0.0
    payload_listing_id = str(payload.get("listing_id") or payload.get("id") or "").upper()
    searchable_values = []
    for field in fields:
        value = _get_field(payload, field)
        if isinstance(value, list):
            searchable_values.extend(str(item) for item in value)
        elif value is not None:
            searchable_values.append(str(value))
    payload_norm = _norm(" ".join(searchable_values))
    payload_amenities = " ".join(payload.get("amenities") or [])
    payload_amenities_norm = _norm(payload_amenities)

    for lid in signals.get("listing_id", []) or []:
        if lid and lid.upper() == payload_listing_id:
            score = max(score, 1.0)
    for aid in signals.get("arxiv_like", []) or []:
        if aid and aid == str(payload.get("arxiv_id") or ""):
            score = max(score, 1.0)
        elif aid and aid in _norm(str(payload.get("title") or "")):
            score = max(score, 0.6)
    for snippet in signals.get("district", []) or []:
        if not snippet:
            continue
        snippet_norm = _norm(snippet)
        if snippet_norm in payload_norm or snippet_norm in payload_amenities_norm:
            score = max(score, 0.5)

    if "title" in fields:
        title = payload.get("title")
        if title:
            title_norm = _norm(str(title))
            for snippet in signals.get("district", []) or []:
                snippet_norm = _norm(snippet)
                if snippet_norm and snippet_norm in title_norm:
                    score = max(score, 0.3)
    if "amenities" in fields:
        for snippet in signals.get("district", []) or []:
            if _norm(snippet) in payload_amenities_norm:
                score = max(score, 0.2)

    for query in signals.get("query", []) or []:
        query_terms = _terms(query)
        if len(query_terms) < 2:
            continue
        payload_terms = _terms(payload_norm)
        if not payload_terms:
            continue
        overlap = len(query_terms & payload_terms) / max(len(query_terms), 1)
        if overlap >= 0.45:
            score = max(score, min(0.4, 0.15 + overlap * 0.25))

    return min(score, 1.0)


def metadata_lookup(
    signals: dict[str, list[str]],
    repository: Any,
) -> dict[str, dict[str, Any]]:
    """
    Tra cứu payload theo tín hiệu rõ (listing_id chính là khóa chính của repository).

    Args:
        signals: Output của extract_metadata_signals.
        repository: Bất kỳ object nào có get_by_id(listing_id) -> dict | None.

    Returns:
        Mapping listing_id (upper) -> payload (dict). Bỏ qua những id không tìm thấy.
    """
    found: dict[str, dict[str, Any]] = {}
    for lid in signals.get("listing_id", []) or []:
        if not lid:
            continue
        if not hasattr(repository, "get_by_id"):
            continue
        try:
            doc = repository.get_by_id(lid)
        except Exception:
            doc = None
        if doc:
            key = str(doc.get("listing_id") or lid).upper()
            found[key] = doc
    return found


__all__ = [
    "extract_metadata_signals",
    "metadata_lookup",
    "metadata_signal_present",
    "score_metadata_hit",
]
