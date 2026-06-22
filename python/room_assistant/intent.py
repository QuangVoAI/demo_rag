"""Phân tích intent và constraint patch cho Nhatrovn Assistant.

Chiến lược routing hai tầng:
  1. Regex nhanh — xử lý ~85% trường hợp rõ ràng mà không cần gọi LLM.
  2. LLM fallback — chỉ gọi khi regex không tự tin, trả về JSON nhỏ gọn.

Quy tắc: Module này chỉ đọc dữ liệu (read-only), không ghi side effect nào
vào session state — việc đó thuộc về session_store.py.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .schemas import ALLOWED_OPERATION_PATHS, INTENTS, OP_TYPES, ParsedRequest


# ---------------------------------------------------------------------------
# Từ khóa hành động nghiệp vụ mà assistant KHÔNG được tự thực hiện
# ---------------------------------------------------------------------------
ACTION_KEYWORDS = {
    "dat_lich": (
        "đặt lịch", "dat lich", "hẹn xem", "hen xem",
        "xem phòng giúp", "lịch hẹn", "lich hen",
    ),
    "message_owner": (
        "nhắn chủ", "nhan chu", "gửi tin", "gui tin",
        "liên hệ chủ", "lien he chu", "nhắn tin cho chủ",
    ),
    "save_favorite": (
        "lưu phòng", "luu phong", "yêu thích", "yeu thich",
        "đánh dấu phòng", "danh dau phong",
    ),
    "hold_room": (
        "giữ chỗ", "giu cho", "giữ phòng", "giu phong",
        "đặt cọc giữ phòng", "dat coc giu phong",
    ),
    "payment": (
        "thanh toán", "thanh toan", "đặt cọc", "dat coc",
        "chuyển khoản", "chuyen khoan", "trả tiền thuê",
    ),
    "edit_listing": (
        "sửa thông tin phòng", "sua thong tin phong",
        "đổi giá phòng", "doi gia phong", "cập nhật tin đăng",
    ),
    "negotiate": (
        "thương lượng", "thuong luong", "trả giá", "tra gia",
        "ép giá", "ep gia", "bớt giá", "bot gia",
    ),
}

# ---------------------------------------------------------------------------
# Ánh xạ tên tiện ích người dùng → tên chuẩn trong hệ thống
# ---------------------------------------------------------------------------
AMENITY_ALIASES: dict[str, str] = {
    # Làm mát
    "máy lạnh": "air_conditioner",
    "may lanh": "air_conditioner",
    "điều hòa": "air_conditioner",
    "dieu hoa": "air_conditioner",
    # Ban công / sân thượng
    "ban công": "balcony",
    "ban cong": "balcony",
    "sân thượng": "rooftop",
    "san thuong": "rooftop",
    # Giặt giũ
    "máy giặt": "washing_machine",
    "may giat": "washing_machine",
    # Vệ sinh
    "wc riêng": "private_bathroom",
    "toilet riêng": "private_bathroom",
    "nhà vệ sinh riêng": "private_bathroom",
    "nha ve sinh rieng": "private_bathroom",
    # Không gian
    "gác": "mezzanine",
    "gac": "mezzanine",
    "gác lửng": "mezzanine",
    # Bếp
    "bếp": "kitchen",
    "bep": "kitchen",
    "bếp riêng": "kitchen",
    # Cửa sổ / ánh sáng
    "cửa sổ": "window",
    "cua so": "window",
    # An ninh
    "camera": "camera",
    "thang máy": "elevator",
    "thang may": "elevator",
    "bảo vệ": "security",
    "bao ve": "security",
    # Internet
    "wifi": "wifi",
    "internet": "wifi",
    # Chỗ đậu xe
    "chỗ để xe": "parking",
    "cho de xe": "parking",
    "hầm xe": "parking",
    "ham xe": "parking",
    # Thú cưng
    "nuôi thú cưng": "pets_allowed",
    "nuoi thu cung": "pets_allowed",
}

# Tùy chọn mềm — không loại phòng nhưng dùng để ranking
SOFT_PREFERENCE_ALIASES: dict[str, str] = {
    "yên tĩnh": "quiet",
    "yen tinh": "quiet",
    "thoáng": "airy",
    "thoang": "airy",
    "sáng": "bright",
    "sang": "bright",
    "gần tiện ích": "near_amenities",
    "gan tien ich": "near_amenities",
    "học tập": "study_friendly",
    "hoc tap": "study_friendly",
    "an ninh": "safe_neighborhood",
    "an toàn": "safe_neighborhood",
    "an toan": "safe_neighborhood",
    "mới": "newly_renovated",
    "moi": "newly_renovated",
}

# ---------------------------------------------------------------------------
# Từ khóa phân loại intent theo regex (không dùng LLM)
# ---------------------------------------------------------------------------

# Mapping: intent_key → tuple các cụm từ cần khớp
_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "COMPARE_ROOMS": (
        "so sanh", "khac nhau", "nen chon phong nao",
        "hon phong", "vs phong", "so sanh phong",
    ),
    "CALCULATE_COST": (
        "tong chi phi", "chi phi", "tien coc", "phi hang thang",
        "uoc tinh", "bao nhieu tien", "tinh tien", "phi phat sinh",
        "mat bao nhieu", "mat bao nhieu tien",
    ),
    "FIND_SIMILAR": (
        "tuong tu", "giong phong", "phong giong", "phong khac tuong tu",
        "co phong nao", "tim phong khac",
    ),
    "SUMMARIZE_ROOM": (
        "tom tat", "uu diem", "han che", "diem manh", "diem yeu",
        "nhan xet", "danh gia phong", "phu hop khong",
    ),
    "ASK_ABOUT_ROOM": (
        "phong nay", "dang xem", "phong do", "phong tren",
        "co cho nuoi", "cho nuoi thu", "can ho nay", "phong nay co",
    ),
    "REFINE_SEARCH": (
        "them ", "bo ", "khong can", "tang ngan sach", "giam ngan sach",
        "doi sang", "thay doi", "chinh sua dieu kien", "bo tieu chi",
    ),
    "SEARCH_ROOM": (
        "tim phong", "phong tro", "nha tro", "can ho", "studio",
        "thue phong", "muon thue", "can thue", "tim nha",
        "phong cho thue", "nha cho thue",
    ),
    "REQUEST_FAQ": (
        "quy dinh", "dieu khoan", "chinh sach", "huong dan",
        "lam the nao", "can lam gi", "thu tuc", "ho so can gi",
        "ky hop dong", "hop dong thue", "bao nhieu coc",
    ),
}


def _strip_accents(value: str) -> str:
    """Loại bỏ dấu thanh tiếng Việt để so sánh không phân biệt."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value)
        if unicodedata.category(ch) != "Mn"
    )


def _norm(text: str) -> str:
    """Chuẩn hoá text: bỏ dấu, viết thường, xóa khoảng trắng thừa."""
    return re.sub(r"\s+", " ", _strip_accents(text).lower()).strip()


def _money_to_vnd(raw: str, unit: str | None) -> int:
    """Chuyển chuỗi số tiền sang VND nguyên."""
    raw = raw.strip()
    unit_norm = _norm(unit or "")
    separators = raw.count(".") + raw.count(",")
    if separators > 1:
        return int(re.sub(r"\D", "", raw))
    normalized_raw = raw.replace(",", ".")
    if separators == 1 and not unit_norm:
        whole, frac = re.split(r"[\.,]", raw, maxsplit=1)
        if len(frac) == 3 and len(whole) <= 3:
            return int(whole + frac)
    value = float(normalized_raw)
    if unit_norm in {"tr", "trieu", "million", "m"}:
        return int(value * 1_000_000)
    if unit_norm in {"k", "nghin"}:
        return int(value * 1_000)
    if value < 1000:
        return int(value * 1_000_000)
    return int(value)


def _append_unique(ops: list[dict[str, Any]], op: str, path: str, value: Any = None) -> None:
    """Thêm operation vào danh sách nếu hợp lệ và chưa tồn tại."""
    if op not in OP_TYPES or path not in ALLOWED_OPERATION_PATHS:
        return
    item: dict[str, Any] = {"op": op, "path": path}
    if op != "clear":
        item["value"] = value
    if item not in ops:
        ops.append(item)


def _extract_listing_ids(text: str) -> list[str]:
    """Trích xuất mã phòng từ nội dung câu hỏi (VD: #A101, phòng B202)."""
    ids: list[str] = []
    patterns = [
        r"#([A-Za-z0-9][A-Za-z0-9_-]{1,40})",
        r"\b(?:listing|phòng|phong|mã|ma)\s+([A-Za-z0-9][A-Za-z0-9_-]{1,40})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            listing_id = match.group(1).strip()
            if listing_id not in ids:
                ids.append(listing_id)
    return ids


def _extract_budget(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất ngân sách tối đa / tối thiểu từ câu hỏi."""
    money = r"(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?"

    # Xử lý dạng rút gọn "3tr5" → 3.5 triệu
    compact_million = re.search(r"\b(\d+)\s*(?:tr|trieu)\s*(\d+)\b", normalized)
    if compact_million:
        value = f"{compact_million.group(1)}.{compact_million.group(2)}"
        _append_unique(ops, "set", "budget.max", _money_to_vnd(value, "trieu"))
        return

    max_patterns = (
        rf"(?:tối đa|toi da|duoi|dưới|không quá|khong qua|ngân sách|ngan sach|budget).*?{money}",
        rf"{money}\s*(?:đổ lại|do lai|tro xuong|trở xuống)",
    )
    min_patterns = (
        rf"(?:tối thiểu|toi thieu|trên|tren|hơn|hon|từ|tu)\s*{money}",
    )

    for pattern in max_patterns:
        match = re.search(pattern, normalized)
        if match:
            _append_unique(ops, "set", "budget.max", _money_to_vnd(match.group(1), match.group(2)))
            break
    for pattern in min_patterns:
        match = re.search(pattern, normalized)
        if match:
            _append_unique(ops, "set", "budget.min", _money_to_vnd(match.group(1), match.group(2)))
            break

    # Xử lý "tăng ngân sách lên X triệu"
    if "tang ngan sach" in normalized or "tăng ngân sách" in text.lower():
        match = re.search(rf"(?:lên|len)\s*{money}", normalized)
        if match:
            _append_unique(ops, "set", "budget.max", _money_to_vnd(match.group(1), match.group(2)))


def _extract_location(normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất quận/huyện và mốc địa lý (gần trường, gần chợ...) từ câu hỏi."""
    # Nhận diện quận/huyện — hỗ trợ cả "quận 3", "Q3", "huyện Bình Chánh"
    district_pattern = re.compile(
        r"\b(?:quan|q\.?|huyen|huyện)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]{1,30})",
        re.IGNORECASE,
    )
    districts = []
    for match in district_pattern.finditer(normalized):
        value = match.group(1).strip()
        # Cắt tại từ ngăn cách để tránh lấy thừa
        value = re.split(r"\b(?:gan|duoi|tren|co|va|,|\.)\b", value)[0].strip()
        if value and value not in districts:
            districts.append(f"quan {value}")
    for district in districts:
        _append_unique(ops, "append", "location.districts", district)

    # Nhận diện mốc địa lý gần (gần ĐHQG, gần Vincom...)
    landmarks = []
    for match in re.finditer(r"\b(?:gan|gần)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]{2,40})", normalized):
        value = match.group(1).strip()
        value = re.split(r"\b(?:duoi|tren|co|va|,|\.)\b", value)[0].strip()
        if value and value not in landmarks:
            landmarks.append(value)
    for landmark in landmarks:
        _append_unique(ops, "append", "location.near_landmarks", landmark)


def _extract_people_and_pets(normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất số người ở, thú cưng và phương tiện từ câu hỏi."""
    # Số người ở
    match = re.search(r"(\d+)\s*(?:nguoi|người|ban|bạn)\b", normalized)
    if match:
        _append_unique(ops, "set", "occupants", int(match.group(1)))

    # Thú cưng
    if re.search(r"nuoi\s*(meo|mèo|cat)", normalized):
        _append_unique(ops, "append", "pets_required", "cat")
    if re.search(r"nuoi\s*(cho|chó|dog)", normalized):
        _append_unique(ops, "append", "pets_required", "dog")

    # Phương tiện đặc biệt
    if re.search(r"xe\s*(dien|điện|electric)", normalized):
        _append_unique(ops, "append", "vehicles", "electric_bike")
    if re.search(r"xe\s*(may|máy|motor)", normalized):
        _append_unique(ops, "append", "vehicles", "motorbike")
    if re.search(r"\b(o\s*to|oto|car|xe\s*hoi)\b", normalized):
        _append_unique(ops, "append", "vehicles", "car")


def _extract_amenities(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất tiện ích bắt buộc và tùy chọn từ câu hỏi."""
    # Xác định người dùng đang bỏ / xóa tiện ích hay thêm vào
    remove_mode = bool(re.search(r"\b(bo|bỏ|khong can|không cần|loai|loại|xoa|xóa)\b", normalized))

    for alias, canonical in AMENITY_ALIASES.items():
        if _norm(alias) in normalized:
            _append_unique(
                ops,
                "remove" if remove_mode else "append",
                "amenities_required",
                canonical,
            )

    for alias, canonical in SOFT_PREFERENCE_ALIASES.items():
        if _norm(alias) in normalized:
            _append_unique(ops, "append", "amenities_preferred", canonical)

    # Lệnh xóa toàn bộ điều kiện tìm kiếm
    if re.search(r"(xoa het|xóa hết|clear|bo het|bỏ hết)\s*(dieu kien|điều kiện)", normalized):
        for path in (
            "location.districts",
            "location.wards",
            "location.near_landmarks",
            "vehicles",
            "pets_required",
            "amenities_required",
            "amenities_preferred",
            "excluded_features",
        ):
            _append_unique(ops, "clear", path)


def _requested_action(normalized: str) -> str | None:
    """Kiểm tra xem người dùng có yêu cầu thao tác nghiệp vụ không."""
    for action, keywords in ACTION_KEYWORDS.items():
        if any(_norm(keyword) in normalized for keyword in keywords):
            return action
    return None


# ---------------------------------------------------------------------------
# Tầng 1: Phân loại intent bằng regex
# Trả về (intent, confidence) — confidence < 1.0 thì cần LLM verify
# ---------------------------------------------------------------------------

def _regex_classify(
    normalized: str,
    action: str | None,
    ids: list[str],
    current_state: dict[str, Any] | None,
) -> tuple[str, float]:
    """
    Phân loại intent nhanh bằng regex.

    Returns:
        (intent, confidence) — confidence = 1.0 nếu chắc chắn,
        nhỏ hơn nếu cần LLM kiểm tra lại.
    """
    if action:
        return "REQUEST_ACTION", 1.0

    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            # ASK_ABOUT_ROOM cũng cần có context phòng
            if intent == "ASK_ABOUT_ROOM" and not (ids or current_state):
                continue
            return intent, 1.0

    # Có mã phòng nhưng không khớp intent nào rõ ràng → hỏi về phòng
    if ids:
        return "ASK_ABOUT_ROOM", 0.85

    # Kế thừa intent từ lượt trước nếu đang trong luồng tìm kiếm
    if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"}:
        return "REFINE_SEARCH", 0.75

    return "GENERAL_HELP", 0.6


# ---------------------------------------------------------------------------
# Tầng 2: LLM fallback khi regex không chắc chắn
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """Bạn là bộ phân loại intent cho chatbot tìm phòng trọ tại Việt Nam (nhatro.vn).
Nhiệm vụ: Phân loại đúng intent từ câu hỏi tiếng Việt của người dùng.

Danh sách intent hợp lệ:
- SEARCH_ROOM: Tìm / lọc phòng theo tiêu chí
- REFINE_SEARCH: Điều chỉnh tiêu chí tìm kiếm đang có
- ASK_ABOUT_ROOM: Hỏi chi tiết về một phòng cụ thể
- CALCULATE_COST: Tính chi phí thuê (tiền cọc, phí phát sinh)
- COMPARE_ROOMS: So sánh nhiều phòng với nhau
- FIND_SIMILAR: Tìm phòng tương tự phòng đang xem
- SUMMARIZE_ROOM: Tóm tắt ưu / nhược điểm phòng
- REQUEST_FAQ: Hỏi về quy trình thuê, hợp đồng, thủ tục
- REQUEST_ACTION: Yêu cầu hành động nghiệp vụ (đặt lịch, nhắn chủ, thanh toán...)
- GENERAL_HELP: Câu hỏi chung hoặc không xác định được

Trả về JSON duy nhất, không giải thích thêm:
{"intent": "<INTENT>", "confidence": <0.0-1.0>}"""


async def _llm_classify_intent(question: str, current_state: dict[str, Any] | None) -> tuple[str, float]:
    """Gọi LLM để phân loại intent khi regex không chắc chắn."""
    try:
        from agents.llm_client import groq_complete, GROQ_MODEL_FAST

        context_hint = ""
        if current_state and current_state.get("last_intent"):
            context_hint = f"\nIntent lượt trước: {current_state['last_intent']}"

        prompt = f"Câu hỏi: {question}{context_hint}"
        raw = await groq_complete(
            prompt=prompt,
            system_prompt=_LLM_SYSTEM_PROMPT,
            model=GROQ_MODEL_FAST,
            max_tokens=60,
            temperature=0.0,
        )
        # Trích xuất JSON từ phản hồi
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            data = json.loads(match.group(0))
            intent = data.get("intent", "GENERAL_HELP")
            confidence = float(data.get("confidence", 0.7))
            if intent in INTENTS:
                return intent, confidence
    except Exception:
        pass
    return "GENERAL_HELP", 0.5


# ---------------------------------------------------------------------------
# API chính: phân tích một lượt người dùng
# ---------------------------------------------------------------------------

def parse_intent_and_constraint_patch(
    question: str,
    current_state: dict[str, Any] | None = None,
) -> ParsedRequest:
    """
    Phân tích đồng bộ (regex-only) một lượt người dùng.

    Dùng phiên bản đồng bộ này trong các code path không async.
    Để dùng LLM fallback, gọi parse_intent_async().
    """
    text = question or ""
    normalized = _norm(text)
    operations: list[dict[str, Any]] = []

    _extract_budget(text, normalized, operations)
    _extract_location(normalized, operations)
    _extract_people_and_pets(normalized, operations)
    _extract_amenities(text, normalized, operations)

    referenced_listing_ids = _extract_listing_ids(text)
    action = _requested_action(normalized)
    intent, _ = _regex_classify(normalized, action, referenced_listing_ids, current_state)

    if intent not in INTENTS:
        intent = "GENERAL_HELP"

    current_listing_id = referenced_listing_ids[0] if referenced_listing_ids else None

    return {
        "intent": intent,
        "operations": operations,
        "current_listing_id": current_listing_id,
        "referenced_listing_ids": referenced_listing_ids,
        "requested_action": action,
    }


async def parse_intent_async(
    question: str,
    current_state: dict[str, Any] | None = None,
) -> ParsedRequest:
    """
    Phân tích bất đồng bộ với LLM fallback.

    Quy trình:
      1. Regex phân loại nhanh → trả về ngay nếu confidence >= 0.9.
      2. Nếu confidence < 0.9 → gọi LLM để xác nhận / đính chính.
      3. Kết hợp kết quả: ưu tiên LLM nếu confidence LLM > regex.
    """
    text = question or ""
    normalized = _norm(text)
    operations: list[dict[str, Any]] = []

    _extract_budget(text, normalized, operations)
    _extract_location(normalized, operations)
    _extract_people_and_pets(normalized, operations)
    _extract_amenities(text, normalized, operations)

    referenced_listing_ids = _extract_listing_ids(text)
    action = _requested_action(normalized)

    regex_intent, regex_conf = _regex_classify(normalized, action, referenced_listing_ids, current_state)

    # Nếu regex đã chắc → không cần LLM
    if regex_conf >= 0.9 or action:
        final_intent = regex_intent
    else:
        # Gọi LLM để phân loại chính xác hơn
        llm_intent, llm_conf = await _llm_classify_intent(question, current_state)
        # Chọn kết quả có độ tin cậy cao hơn
        final_intent = llm_intent if llm_conf >= regex_conf else regex_intent

    if final_intent not in INTENTS:
        final_intent = "GENERAL_HELP"

    current_listing_id = referenced_listing_ids[0] if referenced_listing_ids else None

    return {
        "intent": final_intent,
        "operations": operations,
        "current_listing_id": current_listing_id,
        "referenced_listing_ids": referenced_listing_ids,
        "requested_action": action,
    }
