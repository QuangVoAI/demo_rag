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
        "đặt phòng", "dat phong", "book phòng", "book phong",
        "booking phòng", "booking phong",
        "ghé xem", "ghe xem", "qua xem", "xem truc tiep", "xem trực tiếp",
    ),
    "message_owner": (
        "nhắn chủ", "nhan chu", "gửi tin", "gui tin",
        "liên hệ chủ", "lien he chu", "nhắn tin cho chủ",
        "chat ngay", "gọi hỗ trợ", "goi ho tro", "gọi tư vấn",
        "goi tu van", "gọi tổng đài", "goi tong dai",
        "số điện thoại", "so dien thoai", "xin số", "xin so",
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
    "edit_room": (
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
    "giặt đồ": "washing_machine",
    "giat do": "washing_machine",
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
    "kệ bếp": "kitchen",
    "ke bep": "kitchen",
    "tủ lạnh": "refrigerator",
    "tu lanh": "refrigerator",
    "nước nóng": "hot_water",
    "nuoc nong": "hot_water",
    "giường": "bed",
    "giuong": "bed",
    "nệm": "mattress",
    "nem": "mattress",
    "tủ quần áo": "wardrobe",
    "tu quan ao": "wardrobe",
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
    "ho boi": "pool",
    "hồ bơi": "pool",
    "be boi": "pool",
    "bể bơi": "pool",
    # Thú cưng
    "nuôi thú cưng": "pets_allowed",
    "nuoi thu cung": "pets_allowed",
    # Sạc xe điện
    "sạc xe điện": "ev_charging",
    "sac xe dien": "ev_charging",
    "nhận xe điện": "ev_charging",
    "nhan xe dien": "ev_charging",
    # Giờ giấc
    "giờ tự do": "free_hours",
    "gio tu do": "free_hours",
    "tự do giờ giấc": "free_hours",
    "tu do gio giac": "free_hours",
    "giờ giấc tự do": "free_hours",
    "gio giac tu do": "free_hours",
    "chìa khóa trao tay": "free_hours",
    "chia khoa trao tay": "free_hours",
    # Mới bổ sung
    "khóa vân tay": "fingerprint_lock",
    "khoa van tay": "fingerprint_lock",
    "máy sấy": "dryer",
    "may say": "dryer",
    "chỗ phơi đồ": "laundry_area",
    "cho phoi do": "laundry_area",
    "sân phơi": "laundry_area",
    "san phoi": "laundry_area",
    "bếp điện": "electric_stove",
    "bep dien": "electric_stove",
}

ROOM_TYPE_ALIASES: dict[str, str] = {
    "phòng trọ": "phong_tro",
    "phong tro": "phong_tro",
    "nhà trọ": "phong_tro",
    "nha tro": "phong_tro",
    "căn hộ": "can_ho",
    "can ho": "can_ho",
    "studio": "studio",
    "chdv": "chdv",
    "căn hộ dịch vụ": "chdv",
    "can ho dich vu": "chdv",
    "1pn": "1pn",
    "1 phòng ngủ": "1pn",
    "2pn": "2pn",
    "2 phòng ngủ": "2pn",
    "3pn": "3pn",
    "3 phòng ngủ": "3pn",
    "duplex": "duplex",
    "giường nam": "giuong_nam",
    "giuong nam": "giuong_nam",
    "giường nữ": "giuong_nu",
    "giuong nu": "giuong_nu",
    "giường": "giuong_tang",
    "giuong": "giuong_tang",
    "ktx": "giuong_tang",
    "ký túc xá": "giuong_tang",
    "ky tuc xa": "giuong_tang",
    "sleepbox nữ": "sleepbox_nu",
    "sleepbox nu": "sleepbox_nu",
    "sleepbox nam": "sleepbox_nam",
    "sleepbox": "sleepbox",
    "nhà phố": "nha_pho",
    "nha pho": "nha_pho",
    "nhà nguyên căn": "nha_pho",
    "mặt bằng": "mat_bang",
    "mat bang": "mat_bang",
    "penthouse": "penthouse",
}

KNOWN_DISTRICT_ALIASES: tuple[str, ...] = (
    "binh thanh",
    "go vap",
    "tan binh",
    "tan phu",
    "phu nhuan",
    "binh tan",
    "thu duc",
    "nha be",
    "hoc mon",
    "binh chanh",
    "can gio",
    "cu chi",
)

KNOWN_WARD_ALIASES: tuple[str, ...] = (
    "thao dien",
)

DISTRICT_PREFIXES: tuple[str, ...] = (
    "thanh pho",
    "tp",
    "thi xa",
    "tx",
    "quan",
    "huyen",
)

WARD_PREFIXES: tuple[str, ...] = (
    "thi tran",
    "tt",
    "phuong",
    "xa",
)

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
    "không chung chủ": "no_owner_shared",
    "khong chung chu": "no_owner_shared",
    "không chủ chung": "no_owner_shared",
    "khong chu chung": "no_owner_shared",
}

# ---------------------------------------------------------------------------
# Từ khóa phân loại intent theo regex (không dùng LLM)
# ---------------------------------------------------------------------------

# Mapping: intent_key → tuple các cụm từ cần khớp
_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "COMPARE_ROOMS": (
        "so sanh", "khac nhau", "nen chon phong nao",
        "phong nao phu hop nhat", "phong nao hop nhat",
        "hon phong", "vs phong", "so sanh phong",
    ),
    "CALCULATE_COST": (
        "tong chi phi", "chi phi", "tien coc", "phi hang thang",
        "uoc tinh", "bao nhieu tien", "tinh tien", "phi phat sinh",
        "mat bao nhieu", "mat bao nhieu tien", "gia dien", "gia nuoc",
        "phi dien", "phi nuoc", "phi quan ly", "phi gui xe",
        "tien dien", "tien nuoc", "wifi bao nhieu", "gui xe bao nhieu",
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
        "phong cho thue", "nha cho thue", "chdv", "duplex",
        "sleepbox", "giuong", "mat bang", "nha pho",
        "co phong", "con phong", "phong trong", "sap het", "hot",
    ),
    "REQUEST_FAQ": (
        "quy dinh", "dieu khoan", "chinh sach", "huong dan",
        "lam the nao", "can lam gi", "thu tuc", "ho so can gi",
        "ky hop dong", "hop dong thue", "bao nhieu coc",
        "phong that", "thong tin that", "da xac thuc", "xac thuc",
        "dan xem", "xem tan noi", "mien phi", "mat phi",
        "anhome", "tai app", "ky gui", "lap mang", "chu nha",
        "dang tin", "hoa hong",
    ),
}

DETAIL_FIELD_KEYWORDS: tuple[str, ...] = (
    "dien tich", "gia phong", "gia bao nhieu", "con phong", "phong trong",
    "ma phong", "vi tri", "dia chi", "gan dau", "gan truong", "gan cho",
    "gan sieu thi", "toilet", "wc", "nha ve sinh", "gio giac", "cua so",
    "ban cong", "tien ich", "co gi", "thu cung", "nuoi meo", "nuoi cho", "de xe", "cho de xe",
    "gui xe", "xe dien", "may giat", "wifi", "tu lanh", "nuoc nong",
    "gac", "noi that", "full noi that", "nem", "giuong", "tu quan ao", "ke bep", "thang may",
    "may lanh", "dien", "nuoc", "quan ly", "xac thuc", "anhome ho tro",
)

ROOM_REFERENCE_KEYWORDS: tuple[str, ...] = (
    "phong nay", "phong do", "phong tren", "dang xem", "can ho nay",
    "cho nay", "nha nay", "muc nay", "tin nay",
    "can vua nay", "căn vừa nãy", "can nay", "can do", "phong vua roi", "phong nãy", "phong nay con", "can nay con"
)

COST_FIELD_KEYWORDS: tuple[str, ...] = (
    "chi phi", "gia dien", "gia nuoc", "phi dien", "phi nuoc",
    "phi quan ly", "phi gui xe", "tien dien", "tien nuoc",
    "tien coc", "bao nhieu tien",
)

CHITCHAT_KEYWORDS: tuple[str, ...] = (
    "xin chao", "chao", "hello", "hi",
    "cam on", "thank", "thanks", "ok", "oke", "vâng", "vang",
    "duoc roi", "được rồi", "tam biet", "tạm biệt", "bye",
)

FAQ_KEYWORDS: tuple[str, ...] = (
    "thu tuc",
    "quy trinh",
    "quy dinh",
    "hop dong",
    "ky hop dong",
    "xac thuc",
    "xem phong truoc",
    "xem tan noi",
    "thu phi",
    "mat phi",
    "phi nguoi thue",
    "mien phi",
    "lien he chu nha",
    "chu nha",
    "dat lich xem phong",
    "xem phong",
)

SIMILAR_KEYWORDS: tuple[str, ...] = (
    "tuong tu",
    "giong vay",
    "giong nhu vay",
    "giong phong nay",
    "giong phong do",
    "giong phong tren",
    "phong tuong tu",
)


def _strip_accents(value: str) -> str:
    """Loại bỏ dấu thanh tiếng Việt để so sánh không phân biệt."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value)
        if unicodedata.category(ch) != "Mn"
    )


def _norm(text: str) -> str:
    """Chuẩn hoá text: bỏ dấu, viết thường, xóa khoảng trắng thừa."""
    return re.sub(r"\s+", " ", _strip_accents(text).lower().replace("đ", "d")).strip()


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


def _extract_room_ids(text: str) -> list[str]:
    """Trích xuất mã phòng từ nội dung câu hỏi (VD: #A101, phòng B202)."""
    ids: list[str] = []
    patterns = [
        r"#([A-Za-z0-9][A-Za-z0-9_-]{1,40})",
        r"\b(?:phòng|phong|mã|ma)\s+([A-Za-z0-9][A-Za-z0-9_-]{1,40})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            room_id = match.group(1).strip()
            if not _looks_like_room_id(room_id):
                continue
            if room_id not in ids:
                ids.append(room_id)
    return ids


def _strip_followup_room_detail_clause(text: str) -> str:
    raw = str(text or "")
    patterns = (
        r"[,.;]\s*ph[oòóỏõọôồốổỗộơờớởỡợ]ng\s+(?:[^\n]{0,80}?)\b(?:co|có|khong|không|bao nhieu|bao nhiêu|the nao|thế nào)\b.*$",
    )
    trimmed = raw
    for pattern in patterns:
        trimmed = re.sub(pattern, "", trimmed, flags=re.IGNORECASE)
    return trimmed.strip() or raw


def _normalize_location_shorthand(text: str) -> str:
    normalized = str(text or "")
    normalized = re.sub(r"\bp\.\s*([A-Za-zÀ-ỹà-ỹ][A-Za-zÀ-ỹà-ỹ\s]{1,30})", r"phường \1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bq\.\s*(\d{1,2})\b", r"quận \1", normalized, flags=re.IGNORECASE)
    return normalized


def _looks_like_room_id(value: str) -> bool:
    return any(ch.isdigit() for ch in value)


def _extract_budget(text: str, normalized: str, ops: list[dict[str, Any]], current_state: dict[str, Any] | None = None) -> None:
    """Trích xuất ngân sách tối đa / tối thiểu từ câu hỏi."""
    money = r"(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?"

    # Xử lý "tăng ngân sách lên X triệu"
    up_match = re.search(rf"(?:tăng|tang|nới|noi)\s*(?:ngân sách|ngan sach)?\s*(?:lên|len)\s*{money}", normalized)
    if up_match:
        _append_unique(ops, "set", "budget.max", _money_to_vnd(up_match.group(1), up_match.group(2)))
        _append_unique(ops, "set", "budget.max_operator", "lte")
        return

    # Xử lý "thêm X triệu" (thêm vào ngân sách hiện tại)
    add_match = re.search(rf"(?:thêm|them|nới|noi|tăng|tang)\s*(?:thêm|them|ngân sách|ngan sach)?\s*{money}", normalized)
    if add_match and "len" not in add_match.group(0) and "lên" not in add_match.group(0) and add_match.group(1):
        val = _money_to_vnd(add_match.group(1), add_match.group(2))
        current_budget = current_state.get("constraints", {}).get("budget", {}).get("max") if current_state else None
        if current_budget:
            _append_unique(ops, "set", "budget.max", current_budget + val)
        else:
            _append_unique(ops, "set", "budget.max", val)
        _append_unique(ops, "set", "budget.max_operator", "lte")
        return

    # Xử lý "giảm/bớt X triệu" (trừ vào ngân sách hiện tại)
    sub_match = re.search(rf"(?:giảm|giam|bớt|bot)\s*(?:ngân sách|ngan sach)?\s*(?:đi|di)?\s*{money}", normalized)
    if sub_match and sub_match.group(1):
        val = _money_to_vnd(sub_match.group(1), sub_match.group(2))
        current_budget = current_state.get("constraints", {}).get("budget", {}).get("max") if current_state else None
        if current_budget:
            new_budget = max(0, current_budget - val)
            _append_unique(ops, "set", "budget.max", new_budget)
            _append_unique(ops, "set", "budget.max_operator", "lte")
            return

    range_match = re.search(
        rf"(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?\s*(?:-|–|đến|den|tới|toi|~)\s*(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?",
        normalized,
    )
    if not range_match:
        # Xử lý khoảng giá ngầm định (VD: tầm 2 3 triệu)
        range_match = re.search(
            rf"(?:tầm|tam|từ|tu|khoảng|khoang)\s+(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?\s+(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?",
            normalized,
        )
    if range_match:
        left_unit = range_match.group(2)
        right_unit = range_match.group(4)
        shared_unit = right_unit or left_unit or "trieu"
        min_val = _money_to_vnd(range_match.group(1), left_unit or shared_unit)
        max_val = _money_to_vnd(range_match.group(3), right_unit or shared_unit)
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        _append_unique(ops, "set", "budget.min", min_val)
        _append_unique(ops, "set", "budget.min_operator", "gte")
        _append_unique(ops, "set", "budget.max", max_val)
        _append_unique(ops, "set", "budget.max_operator", "lte")
        return

    # Xử lý dạng rút gọn "3tr5" → 3.5 triệu
    compact_million = re.search(r"\b(\d+)\s*(?:tr|trieu)\s*(\d+)\b(?!\s*thang)", normalized)
    if compact_million:
        value = f"{compact_million.group(1)}.{compact_million.group(2)}"
        _append_unique(ops, "set", "budget.max", _money_to_vnd(value, "trieu"))
        _append_unique(ops, "set", "budget.max_operator", "lte")
        return

    max_patterns = (
        rf"(?:tối đa|toi da|duoi|dưới|không quá|khong qua|ngân sách|ngan sach|budget).*?{money}",
        rf"{money}\s*(?:đổ lại|do lai|tro xuong|trở xuống)",
    )
    min_patterns = (
        rf"(?:tối thiểu|toi thieu|trên|tren|hơn|hon|từ|tu)\s*{money}",
    )

    budget_set = False
    for pattern in max_patterns:
        match = re.search(pattern, normalized)
        if match:
            max_val = _money_to_vnd(match.group(1), match.group(2))
            _append_unique(ops, "set", "budget.max", max_val)
            matched_segment = match.group(0)
            if "duoi" in matched_segment or "dưới" in matched_segment:
                _append_unique(ops, "set", "budget.max_operator", "lt")
            else:
                _append_unique(ops, "set", "budget.max_operator", "lte")
            budget_set = True
            break
    for pattern in min_patterns:
        match = re.search(pattern, normalized)
        if match:
            min_val = _money_to_vnd(match.group(1), match.group(2))
            _append_unique(ops, "set", "budget.min", min_val)
            matched_segment = match.group(0)
            if "tren" in matched_segment or "trên" in matched_segment or "hon" in matched_segment or "hơn" in matched_segment:
                _append_unique(ops, "set", "budget.min_operator", "gt")
            else:
                _append_unique(ops, "set", "budget.min_operator", "gte")
            budget_set = True
            break

    if not budget_set:
        money_with_unit = r"(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)"
        exact_patterns = (
            rf"(?:giá|gia|khoảng|khoang|tầm|tam|quanh|mức|muc)\s*{money}",
            rf"\b{money_with_unit}\b",
        )
        for pattern in exact_patterns:
            match = re.search(pattern, normalized)
            if match and match.group(1):
                val = _money_to_vnd(match.group(1), match.group(2))
                _append_unique(ops, "set", "budget.max", val)
                _append_unique(ops, "set", "budget.max_operator", "lte")
                break


def _extract_location(normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất đơn vị hành chính cấp quận/huyện và cấp phường/xã/thị trấn."""
    districts: list[str] = []
    explicit_district_found = False
    wards = []

    ward_matches = _iter_prefixed_location_units(normalized, WARD_PREFIXES)
    masked_chars = list(normalized)
    for match in ward_matches:
        if re.search(r"(?:khong|không|ko|trừ|tru)\s*(?:o|ở|tại|tai|tim|tìm|lay|lấy)?\s*$", normalized[:match["start"]]):
            continue
        ward, district = _split_ward_and_trailing_district(match["value"])
        ward = _normalize_ward_candidate(ward)
        if ward and ward not in wards:
            wards.append(ward)
        if district and district not in districts:
            districts.append(district)
            _append_unique(ops, "append", "location.districts", district)
            explicit_district_found = True
        for idx in range(match["start"], match["end"]):
            masked_chars[idx] = " "
    normalized_for_district = "".join(masked_chars)

    compact_districts = []
    for match in re.finditer(r"\bq\.?\s*(\d{1,2})\b", normalized_for_district, flags=re.IGNORECASE):
        if re.search(r"(?:khong|không|ko|trừ|tru)\s*(?:o|ở|tại|tai|tim|tìm|lay|lấy)?\s*$", normalized_for_district[:match.start()]):
            continue
        value = f"quan {match.group(1)}"
        if value not in compact_districts:
            compact_districts.append(value)

    for item in compact_districts:
        if item not in districts:
            districts.append(item)
            explicit_district_found = True

    for match in _iter_prefixed_location_units(normalized_for_district, DISTRICT_PREFIXES):
        if re.search(r"(?:khong|không|ko|trừ|tru)\s*(?:o|ở|tại|tai|tim|tìm|lay|lấy)?\s*$", normalized_for_district[:match["start"]]):
            continue
        district, trailing = _normalize_district_candidate(match["prefix"], match["value"])
        if district and district not in districts:
            districts.append(district)
            explicit_district_found = True
        if trailing:
            _append_unique(ops, "append", "location.near_landmarks", trailing)
    for district in districts:
        _append_unique(ops, "append", "location.districts", district)

    if not explicit_district_found:
        for alias in KNOWN_DISTRICT_ALIASES:
            if _contains_phrase(normalized, alias):
                _append_unique(ops, "append", "location.districts", alias)

    for match in re.finditer(r"\bp\.?\s*(\d{1,2})\b", normalized, flags=re.IGNORECASE):
        value = match.group(1).strip()
        if value and value not in wards:
            wards.append(value)
    for ward in wards:
        _append_unique(ops, "append", "location.wards", ward)

    if not districts and not wards:
        for alias in KNOWN_WARD_ALIASES:
            if _contains_phrase(normalized, alias):
                _append_unique(ops, "append", "location.wards", alias)
                wards.append(alias)
                break

    if not districts and not wards:
        bare_location_match = re.search(
            r"\b(?:o|ở|tai|tại)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]{3,35})",
            normalized,
            flags=re.IGNORECASE,
        )
        if bare_location_match and re.search(r"(?:khong|không|ko|trừ|tru)\s*$", normalized[:bare_location_match.start()]):
            bare_location_match = None
        if bare_location_match:
            candidate = _trim_location_segment(bare_location_match.group(1))
            candidate = re.split(
                r"\b(?:duoi|dưới|tren|trên|toi da|tối đa|tu|từ|gia|ngan sach|ngân sách)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            candidate_norm = _normalize_admin_value(candidate)
            if candidate_norm and _looks_like_location_hint(candidate_norm):
                if re.match(r"^(?:le van|nguyen|tran|pham|vo van|hai ba|hoang|huynh|phan)\b", _norm(candidate_norm)):
                    _append_unique(ops, "append", "location.near_landmarks", candidate_norm)
                else:
                    _append_unique(ops, "append", "location.wards", candidate_norm)

    # Nhận diện mốc địa lý gần (gần ĐHQG, gần Vincom...)
    landmarks = []
    for match in re.finditer(r"\b(?:gan|gần)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]{2,40})", normalized):
        value = match.group(1).strip()
        value = re.split(r"\b(?:duoi|tren|co|va|,|\.)\b", value)[0].strip()
        if _norm(value).startswith("giong"):
            continue
        if value and value not in landmarks:
            landmarks.append(value)
    for landmark in landmarks:
        _append_unique(ops, "append", "location.near_landmarks", landmark)

    street_patterns = (
        r"\b(?:duong|đường|mat tien|mặt tiền|hem|hẻm)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ/-]{2,45})",
    )
    for pattern in street_patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            prefix_window = normalized[max(0, match.start() - 12):match.start()]
            suffix_window = normalized[match.end():match.end() + 30]
            if re.search(r"\bphong\s*$", prefix_window, flags=re.IGNORECASE) and re.search(
                r"\b(co|có|khong|không|bao nhieu|bao nhiêu|the nao|thế nào)\b",
                suffix_window,
                flags=re.IGNORECASE,
            ):
                continue
            value = match.group(0).strip()
            value = re.sub(r"\bphuong\s+[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+\b.*$", "", value, flags=re.IGNORECASE).strip()
            value = re.sub(r"\bp\.?\s+[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+\b.*$", "", value, flags=re.IGNORECASE).strip()
            value = re.split(r"\b(?:duoi|tren|co|va|gia|la|là|dc|đc|duoc|được|,|\.)\b", value)[0].strip()
            if value:
                _append_unique(ops, "append", "location.near_landmarks", value)


def _normalize_ward_candidate(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""

    value = re.split(
        r"\b(?:gan|gần|duoi|dưới|tren|trên|co|có|va|và|gia|giá|la|là|dc|đc|duoc|được|khong can|không cần|ko can|hok can|khoang|khoảng|tam|tầm|ngan sach|ngân sách|muc|mức)\b|,|\.",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    value = re.sub(r"\s+(?:khong|không)\??$", "", value, flags=re.IGNORECASE).strip()
    if not value:
        return ""

    numeric_ward = re.match(r"(\d{1,2})\b", value)
    if numeric_ward:
        return numeric_ward.group(1)
    return value


def _normalize_district_candidate(prefix: str, value: str) -> tuple[str, str]:
    value = _trim_location_segment(value)
    if not value:
        return "", ""

    numeric_district = re.match(r"(\d{1,2})\b", value)
    if numeric_district:
        trailing = _normalize_ward_candidate(value[numeric_district.end():].strip())
        if trailing and not _looks_like_location_hint(trailing):
            trailing = ""
        return f"quan {numeric_district.group(1)}", trailing

    normalized = _normalize_admin_value(value)
    if normalized.split()[0] in {"khac", "do", "nay"}:
        return "", ""
    return normalized, ""


def _iter_prefixed_location_units(normalized: str, prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    if prefixes == WARD_PREFIXES:
        pattern = r"thi tran|tt\.?|phuong|(?<!thi\s)xa"
    else:
        pattern = r"thanh pho|tp\.?|thi xa|tx\.?|quan|huyen"
    regex = re.compile(
        rf"\b(?P<prefix>{pattern})\.?\s+(?P<value>[a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+)",
        re.IGNORECASE,
    )
    return [
        {
            "prefix": match.group("prefix"),
            "value": match.group("value"),
            "start": match.start(),
            "end": match.end(),
        }
        for match in regex.finditer(normalized)
    ]


def _split_ward_and_trailing_district(value: str) -> tuple[str, str]:
    candidate = _trim_location_segment(value)
    if not candidate:
        return "", ""

    prefix_pattern = re.compile(r"\b(thanh pho|tp\.?|thi xa|tx\.?|quan|q\.?|huyen)\b", re.IGNORECASE)
    prefix_matches = [item for item in prefix_pattern.finditer(candidate) if item.start() > 0]
    if not prefix_matches:
        return candidate, ""

    match = prefix_matches[-1]
    ward = candidate[:match.start()].strip()
    district = _normalize_district_candidate(match.group(1), candidate[match.end():].strip())[0]
    return ward, district


def _trim_location_segment(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    value = re.split(
        r"\b(?:gan|gần|duoi|dưới|tren|trên|co|có|va|và|gia|giá|khong can|không cần|ko can|hok can|khoang|khoảng|tam|tầm|ngan sach|ngân sách|muc|mức)\b|,|\.",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return re.sub(r"\s+(?:khong|không)\??$", "", value, flags=re.IGNORECASE).strip()


def _normalize_admin_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).strip()


def _looks_like_location_hint(value: str) -> bool:
    normalized = _norm(value)
    if not normalized:
        return False
    if normalized in AMENITY_ALIASES or normalized in SOFT_PREFERENCE_ALIASES or normalized in ROOM_TYPE_ALIASES:
        return False
    if any(token in normalized for token in ("phong", "phong tro", "can ho", "studio")):
        return False
    return len(normalized.split()) >= 2


def _soft_preference_present(normalized: str, alias: str) -> bool:
    alias_norm = _norm(alias)
    if not _contains_phrase(normalized, alias_norm):
        return False
    if alias_norm == "sang" and re.search(r"\b(?:doi|đổi|chuyen|chuyển)\s+sang\b", normalized):
        return False
    return True


def _has_explicit_location_ops(operations: list[dict[str, Any]]) -> bool:
    return any(
        str(item.get("path", "")).startswith("location.")
        and item.get("op") in {"append", "set", "replace"}
        for item in operations
    )


def _reset_location_for_fresh_search(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _has_explicit_location_ops(operations):
        return operations

    reset_ops: list[dict[str, Any]] = []
    for path in (
        "location.province",
        "location.districts",
        "location.wards",
        "location.near_landmarks",
        "location.max_distance_km",
    ):
        _append_unique(reset_ops, "clear", path)
    return reset_ops + operations


def _extract_move_in_date(normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất thời gian dự kiến vào ở theo các lựa chọn trên form đặt lịch."""
    if re.search(r"(o ngay|ở ngay|can phong o ngay|cần phòng ở ngay)", normalized):
        _append_unique(ops, "set", "move_in_date", "immediate")
        return
    match = re.search(r"\b(\d{1,2})\s*(?:ngay|ngày)\b", normalized)
    if match:
        _append_unique(ops, "set", "move_in_date", f"in_{int(match.group(1))}_days")
        return
    for phrase, value in (
        ("trong thang nay", "this_month"),
        ("trong tháng này", "this_month"),
        ("thang nay", "this_month"),
        ("tháng này", "this_month"),
        ("dau thang sau", "early_next_month"),
        ("đầu tháng sau", "early_next_month"),
        ("giua thang sau", "mid_next_month"),
        ("giữa tháng sau", "mid_next_month"),
        ("cuoi thang sau", "late_next_month"),
        ("cuối tháng sau", "late_next_month"),
        ("giua thang nay", "mid_this_month"),
        ("giữa tháng này", "mid_this_month"),
        ("cuoi thang nay", "late_this_month"),
        ("cuối tháng này", "late_this_month"),
    ):
        if phrase in normalized:
            _append_unique(ops, "set", "move_in_date", value)
            return


def _extract_people_and_pets(normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất số người ở, thú cưng và phương tiện từ câu hỏi."""
    # Số người ở
    match = re.search(r"(\d+)\s*(?:nguoi|người|bạn)\b", normalized)
    if match:
        _append_unique(ops, "set", "occupants", int(match.group(1)))
    elif re.search(r"\b(1\s*minh|mot\s*minh|doc\s*than)\b", normalized):
        _append_unique(ops, "set", "occupants", 1)
    elif re.search(r"\b(vo\s*chong|cap\s*doi|hai\s*nguoi)\b", normalized):
        _append_unique(ops, "set", "occupants", 2)

    # Thú cưng
    if re.search(r"nuoi\s*(meo|mèo|cat)", normalized):
        _append_unique(ops, "append", "pets_required", "cat")
    if re.search(r"nuoi\s*(cho|chó|dog)", normalized):
        _append_unique(ops, "append", "pets_required", "dog")
    if re.search(r"(thu\s*cung|pet|nuoi\s*thu\s*cung)", normalized):
        _append_unique(ops, "append", "pets_required", "pet")

    # Phương tiện đặc biệt
    if re.search(r"(gui\s*xe|cho\s*de\s*xe)\s*(dien|electric)", normalized):
        _append_unique(ops, "append", "vehicles", "electric_bike")
    if re.search(r"(xe\s*(may|máy|motor)|\b\d+\s*xe\b|cho\s*de\s*xe|gui\s*xe|ham\s*xe)", normalized):
        _append_unique(ops, "append", "vehicles", "motorbike")
    if re.search(r"\b(o\s*to|oto|car|xe\s*hoi)\b", normalized):
        _append_unique(ops, "append", "vehicles", "car")


def _extract_area_preferences(normalized: str, ops: list[dict[str, Any]]) -> None:
    if re.search(r"\b(rong hon|rộng hơn|lon hon|lớn hơn)\b", normalized) or "dien tich lon hon" in normalized or "diện tích lớn hơn" in normalized:
        _append_unique(ops, "set", "area.preference", "larger")


def _contains_phrase(normalized: str, phrase: str) -> bool:
    escaped = re.escape(_norm(phrase))
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", normalized))


def _extract_categories(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất loại phòng từ câu hỏi."""
    for alias, canonical in ROOM_TYPE_ALIASES.items():
        if _contains_phrase(normalized, alias):
            _append_unique(ops, "append", "categories", canonical)


def _extract_amenities(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất tiện ích bắt buộc và tùy chọn từ câu hỏi."""
    negated_features: set[str] = set()
    if re.search(r"khong\s*may\s*lanh", normalized) or "không máy lạnh" in text.lower():
        negated_features.add("air_conditioner")
    if re.search(r"khong\s*gac", normalized) or "không gác" in text.lower():
        negated_features.add("mezzanine")

    for alias, canonical in AMENITY_ALIASES.items():
        if _contains_phrase(normalized, alias):
            if canonical in negated_features:
                continue
            _append_unique(
                ops,
                "remove" if _is_amenity_remove_request(normalized, alias) else "append",
                "amenities_required",
                canonical,
            )

    for feature in sorted(negated_features):
        _append_unique(ops, "append", "excluded_features", feature)

    for alias, canonical in SOFT_PREFERENCE_ALIASES.items():
        if _soft_preference_present(normalized, alias):
            _append_unique(ops, "append", "amenities_preferred", canonical)

    # Lệnh xóa toàn bộ điều kiện tìm kiếm
    if re.search(r"(xoa het|xóa hết|clear|bo het|bỏ hết)\s*(dieu kien|điều kiện)", normalized):
        for path in (
            "categories",
            "location.districts",
            "location.near_landmarks",
            "vehicles",
            "pets_required",
            "amenities_required",
            "amenities_preferred",
            "excluded_features",
        ):
            _append_unique(ops, "clear", path)


def _is_amenity_remove_request(normalized: str, alias: str) -> bool:
    alias_norm = _norm(alias)
    if not alias_norm:
        return False
    escaped = re.escape(alias_norm)
    remove_prefix = r"(?:bo|khong can|ko can|loai|xoa|khong lay|ko lay|khong co|ko co)"
    filler = r"(?:\s+\w+){0,4}\s+"
    return bool(re.search(rf"\b{remove_prefix}\b{filler}{escaped}(?!\w)", normalized))


def _requested_action(normalized: str) -> str | None:
    """Kiểm tra xem người dùng có yêu cầu thao tác nghiệp vụ không."""
    for action, keywords in ACTION_KEYWORDS.items():
        if any(_norm(keyword) in normalized for keyword in keywords):
            return action
    return None


def _has_current_room(current_state: dict[str, Any] | None) -> bool:
    if not current_state:
        return False
    return bool(current_state.get("current_room_id") or current_state.get("selected_room_ids") or current_state.get("last_result_ids"))


def _has_search_filter_operations(operations: list[dict[str, Any]]) -> bool:
    search_paths = {
        "budget.min",
        "budget.max",
        "location.province",
        "location.districts",
        "location.wards",
        "location.near_landmarks",
        "move_in_date",
        "occupants",
        "amenities_required",
        "categories",
    }
    return any(str(item.get("path")) in search_paths for item in operations)


def _soften_commute_landmark_constraints(normalized: str, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if "tien di chuyen" not in normalized and "tiện di chuyển" not in normalized:
        return operations
    has_admin_location = any(
        item.get("path") in {"location.districts", "location.wards"} and item.get("op") == "append"
        for item in operations
    )
    if not has_admin_location:
        return operations
    return [
        item for item in operations
        if not (item.get("path") == "location.near_landmarks" and item.get("op") == "append")
    ]


def _has_keyword(normalized: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in normalized for keyword in keywords)


def _is_chitchat_or_closing(normalized: str) -> bool:
    compact = normalized.strip(" .,!?\t\r\n")
    if not compact:
        return True
    return any(_contains_phrase(compact, keyword) for keyword in CHITCHAT_KEYWORDS)


def _is_room_detail_question(normalized: str, ids: list[str], current_state: dict[str, Any] | None) -> bool:
    if _has_keyword(normalized, COST_FIELD_KEYWORDS):
        return False
    if not _has_keyword(normalized, DETAIL_FIELD_KEYWORDS):
        return False
    if ids or _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
        return True
    return _has_current_room(current_state) and bool(re.search(r"\b(co|có|khong|không|bao nhieu|bao nhiêu|the nao|thế nào|la gi|là gì|cho toi biet|cho minh biet|noi ro|nói rõ)\b", normalized))


def _is_result_set_compare_request(normalized: str) -> bool:
    if not re.search(r"\bso\s*sanh\b", normalized):
        return False
    return bool(
        re.search(r"\b(?:3|ba)\s*phong\b", normalized)
        or re.search(r"\b(?:cac|nhung|may)\s+phong\b", normalized)
        or re.search(r"\bphong\s+(?:nay|tren|vua|dau tien)\b", normalized)
    )


def _is_compare_preference_request(normalized: str, current_state: dict[str, Any] | None) -> bool:
    if not current_state or len(current_state.get("last_result_ids") or []) < 2:
        return False
    if "phong nao" not in normalized and "so sanh" not in normalized:
        return False
    return bool(
        re.search(r"\bphong nao\b.*\b(?:hon|hơn)\b", normalized)
        or "phu hop hon" in normalized
        or "tot hon" in normalized
        or "re hon" in normalized
        or "rẻ hơn" in normalized
        or "nhieu tien ich hon" in normalized
        or "nhiều tiện ích hơn" in normalized
        or "khu vuc tot hon" in normalized
        or "khu vực tốt hơn" in normalized
    )


def _is_faq_question(normalized: str) -> bool:
    return any(_contains_phrase(normalized, keyword) for keyword in FAQ_KEYWORDS)


def _is_similar_request(normalized: str, current_state: dict[str, Any] | None) -> bool:
    if not _has_current_room(current_state):
        return False
    if any(_contains_phrase(normalized, keyword) for keyword in SIMILAR_KEYWORDS):
        return True
    if "gan giong" in normalized or "gần giống" in normalized:
        return True
    if "giong vay" in normalized or "giống vậy" in normalized:
        return True
    if "giong" in normalized and _contains_phrase(normalized, "tim phong"):
        return True
    if "phong nay" in normalized and ("re hon" in normalized or "rong hon" in normalized or "may lanh" in normalized):
        return True
    if "giong" in normalized and any(token in normalized for token in ("re hon", "rong hon", "may lanh", "ban cong", "quan khac", "quận khác")):
        return True
    return False


def _selected_room_id_from_ordinal(normalized: str, current_state: dict[str, Any] | None) -> str | None:
    if not current_state:
        return None
    match = re.search(r"\b(?:chon|chọn|lay|lấy)\s+(?:phong|phòng)?\s*(?:so|số|#)?\s*(\d{1,2})\b", normalized)
    if not match:
        match = re.search(r"\b(?:phong|phòng)\s*(?:so|số|thu|thứ|#)\s*(\d{1,2})\b", normalized)
    if not match:
        match = re.search(r"\b(?:phong|phòng)\s+(?:dau tien|đầu tiên|thu nhat|thứ nhất|so mot|số một|1)\b", normalized)
    word_index = None
    if not match:
        word_match = re.search(r"\b(?:phong|phòng)\s+(?:thu|thứ|so|số)?\s*(hai|ba|bon|bốn|tu|tư|nam|năm)\b", normalized)
        if word_match:
            word_index = {
                "hai": 1,
                "ba": 2,
                "bon": 3,
                "bốn": 3,
                "tu": 3,
                "tư": 3,
                "nam": 4,
                "năm": 4,
            }.get(word_match.group(1))
    if not match:
        if word_index is None:
            return None
        index = word_index
    else:
        index = int(match.group(1)) - 1 if match.groups() and match.group(1) else 0
    ids = current_state.get("last_result_ids") or []
    if 0 <= index < len(ids):
        return str(ids[index])
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
    if _is_result_set_compare_request(normalized):
        return "COMPARE_ROOMS", 1.0
    if _is_compare_preference_request(normalized, current_state):
        return "COMPARE_ROOMS", 0.95
    if _is_similar_request(normalized, current_state):
        return "FIND_SIMILAR", 0.95
    if _selected_room_id_from_ordinal(normalized, current_state):
        return "ASK_ABOUT_ROOM", 1.0
    if _is_room_detail_question(normalized, ids, current_state):
        return "ASK_ABOUT_ROOM", 1.0
    if _is_faq_question(normalized):
        return "REQUEST_FAQ", 0.95

    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            # ASK_ABOUT_ROOM cũng cần có context phòng
            if intent == "ASK_ABOUT_ROOM" and not (ids or current_state):
                continue
            if intent == "SEARCH_ROOM" and _is_faq_question(normalized):
                continue
            return intent, 1.0

    # Có mã phòng nhưng không khớp intent nào rõ ràng → hỏi về phòng
    if ids:
        return "ASK_ABOUT_ROOM", 0.85

    if _is_chitchat_or_closing(normalized):
        return "GENERAL_HELP", 0.9

    # Kế thừa intent từ lượt trước nếu đang trong luồng tìm kiếm
    if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"}:
        return "REFINE_SEARCH", 0.75

    return "GENERAL_HELP", 0.6


# ---------------------------------------------------------------------------
# Tầng 2: LLM fallback khi regex không chắc chắn
# ---------------------------------------------------------------------------

from .prompts import INTENT_CLASSIFIER_PROMPT

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
            system_prompt=INTENT_CLASSIFIER_PROMPT,
            model=GROQ_MODEL_FAST,
            max_tokens=60,
            temperature=0.0,
        )
        cleaned = raw.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        data = json.loads(cleaned)
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
    location_text = _normalize_location_shorthand(_strip_followup_room_detail_clause(text))
    normalized_for_location = _norm(location_text)
    operations: list[dict[str, Any]] = []

    _extract_budget(text, normalized, operations, current_state)
    _extract_location(normalized_for_location, operations)
    _extract_move_in_date(normalized, operations)
    _extract_people_and_pets(normalized, operations)
    _extract_area_preferences(normalized, operations)
    _extract_amenities(text, normalized, operations)
    _extract_categories(text, normalized, operations)
    operations = _soften_commute_landmark_constraints(normalized, operations)

    referenced_room_ids = _extract_room_ids(text)
    selected_room_id = _selected_room_id_from_ordinal(normalized, current_state)
    if selected_room_id and selected_room_id not in referenced_room_ids:
        referenced_room_ids = [selected_room_id]
    action = _requested_action(normalized)
    intent, _ = _regex_classify(normalized, action, referenced_room_ids, current_state)
    if operations and intent == "GENERAL_HELP":
        intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"
    if (
        operations and intent == "ASK_ABOUT_ROOM"
        and not referenced_room_ids
        and not _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS)
        and (
            _has_search_filter_operations(operations)
            or re.search(r"\b(?:tim|can|muon|thue|o)\b", normalized)
            or _has_keyword(normalized, _INTENT_KEYWORDS["SEARCH_ROOM"])
            or not _is_room_detail_question(normalized, referenced_room_ids, current_state)
        )
    ):
        intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"

    if intent not in INTENTS:
        intent = "GENERAL_HELP"

    if intent == "SEARCH_ROOM":
        operations = _reset_location_for_fresh_search(operations)

    current_room_id = referenced_room_ids[0] if referenced_room_ids else None

    return {
        "intent": intent,
        "operations": operations,
        "current_room_id": current_room_id,
        "referenced_room_ids": referenced_room_ids,
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
    location_text = _normalize_location_shorthand(_strip_followup_room_detail_clause(text))
    normalized_for_location = _norm(location_text)
    operations: list[dict[str, Any]] = []

    _extract_budget(text, normalized, operations, current_state)
    _extract_location(normalized_for_location, operations)
    _extract_move_in_date(normalized, operations)
    _extract_people_and_pets(normalized, operations)
    _extract_area_preferences(normalized, operations)
    _extract_amenities(text, normalized, operations)
    _extract_categories(text, normalized, operations)
    operations = _soften_commute_landmark_constraints(normalized, operations)

    referenced_room_ids = _extract_room_ids(text)
    selected_room_id = _selected_room_id_from_ordinal(normalized, current_state)
    if selected_room_id and selected_room_id not in referenced_room_ids:
        referenced_room_ids = [selected_room_id]
    action = _requested_action(normalized)

    regex_intent, regex_conf = _regex_classify(normalized, action, referenced_room_ids, current_state)
    if operations and regex_intent == "GENERAL_HELP":
        regex_intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"
        regex_conf = max(regex_conf, 0.85)
    if (
        operations and regex_intent == "ASK_ABOUT_ROOM"
        and not referenced_room_ids
        and not _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS)
        and (
            _has_search_filter_operations(operations)
            or re.search(r"\b(?:tim|can|muon|thue|o)\b", normalized)
            or _has_keyword(normalized, _INTENT_KEYWORDS["SEARCH_ROOM"])
            or not _is_room_detail_question(normalized, referenced_room_ids, current_state)
        )
    ):
        regex_intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"
        regex_conf = max(regex_conf, 0.85)

    # Nếu câu dài hơn 5 từ, ép giảm confidence của regex để bắt LLM phải check lại
    # Tránh trường hợp "Hôm trước mình thuê phòng, giờ muốn lấy cọc" bị regex "thuê phòng" bắt nhầm
    word_count = len(question.strip().split())
    if word_count > 5 and not action and regex_intent not in {"ASK_ABOUT_ROOM", "COMPARE_ROOMS"}:
        regex_conf = min(regex_conf, 0.8)

    # Nếu regex đã chắc → không cần LLM
    if regex_conf >= 0.9 or action:
        final_intent = regex_intent
    else:
        # Gọi LLM để phân loại chính xác hơn
        llm_intent, llm_conf = await _llm_classify_intent(question, current_state)
        # Chọn kết quả có độ tin cậy cao hơn
        final_intent = llm_intent if llm_conf >= regex_conf else regex_intent

    if final_intent == "SUMMARIZE_ROOM" and _is_room_detail_question(normalized, referenced_room_ids, current_state):
        final_intent = "ASK_ABOUT_ROOM"

    if final_intent not in INTENTS:
        final_intent = "GENERAL_HELP"

    if final_intent == "SEARCH_ROOM":
        operations = _reset_location_for_fresh_search(operations)

    current_room_id = referenced_room_ids[0] if referenced_room_ids else None

    return {
        "intent": final_intent,
        "operations": operations,
        "current_room_id": current_room_id,
        "referenced_room_ids": referenced_room_ids,
        "requested_action": action,
    }
