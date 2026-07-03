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
from .landmark_aliases import normalize_landmark, _norm_landmark, _should_drop_unresolved_landmark


# ---------------------------------------------------------------------------
# Từ khóa hành động nghiệp vụ mà assistant KHÔNG được tự thực hiện
# ---------------------------------------------------------------------------
ACTION_KEYWORDS = {
    # Kiểm tra hủy/đổi trước dat_lich vì các cụm này có thể chứa "lịch hẹn".
    "huy_lich": (
        "hủy lịch", "huy lich", "huỷ lịch", "hủy hẹn", "huy hen",
        "hủy lịch hẹn", "huy lich hen", "hủy hẹn xem", "huy hen xem",
        "bỏ lịch", "bo lich", "bỏ hẹn", "bo hen",
    ),
    "doi_lich": (
        "đổi lịch", "doi lich", "đổi hẹn", "doi hen",
        "đổi lịch hẹn", "doi lich hen", "dời lịch", "doi lich xem",
        "đổi giờ xem", "doi gio xem", "đổi ngày xem", "doi ngay xem",
    ),
    "dat_lich": (
        "đặt lịch", "dat lich", "hẹn xem", "hen xem",
        "xem phòng giúp", "lịch hẹn", "lich hen",
        "đặt phòng", "dat phong", "book phòng", "book phong",
        "booking phòng", "booking phong",
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
        "giảm giá", "giam gia",
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
    # Không gian / nội thất
    "gác": "mezzanine",
    "gac": "mezzanine",
    "gác lửng": "mezzanine",
    "nội thất": "furnished",
    "noi that": "furnished",
    "full nội thất": "furnished",
    "full noi that": "furnished",
    "có nội thất": "furnished",
    "co noi that": "furnished",
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
}

DISTRICT_ALIASES: dict[str, str] = {
    "binh thanh": "binh thanh", "b.thanh": "binh thanh", "bình thạnh": "binh thanh",
    "tan binh": "tan binh", "tân bình": "tan binh",
    "binh tan": "binh tan", "bình tân": "binh tan",
    "q1": "quan 1", "q.1": "quan 1", "quận 1": "quan 1", "quan 1": "quan 1",
    "q2": "quan 2", "q.2": "quan 2", "quận 2": "quan 2", "quan 2": "quan 2",
    "q3": "quan 3", "q.3": "quan 3", "quận 3": "quan 3", "quan 3": "quan 3",
    "q4": "quan 4", "q.4": "quan 4", "quận 4": "quan 4", "quan 4": "quan 4",
    "q5": "quan 5", "q.5": "quan 5", "quận 5": "quan 5", "quan 5": "quan 5",
    "q6": "quan 6", "q.6": "quan 6", "quận 6": "quan 6", "quan 6": "quan 6",
    "q7": "quan 7", "q.7": "quan 7", "quận 7": "quan 7", "quan 7": "quan 7",
    "q8": "quan 8", "q.8": "quan 8", "quận 8": "quan 8", "quan 8": "quan 8",
    "q9": "quan 9", "q.9": "quan 9", "quận 9": "quan 9", "quan 9": "quan 9",
    "q10": "quan 10", "q.10": "quan 10", "quận 10": "quan 10", "quan 10": "quan 10",
    "q11": "quan 11", "q.11": "quan 11", "quận 11": "quan 11", "quan 11": "quan 11",
    "q12": "quan 12", "q.12": "quan 12", "quận 12": "quan 12", "quan 12": "quan 12",
    "go vap": "go vap", "gò vấp": "go vap", "gv": "go vap",
    "phu nhuan": "phu nhuan", "phú nhuận": "phu nhuan", "pn": "phu nhuan",
    "tan phu": "tan phu", "tân phú": "tan phu",
    "thu duc": "thu duc", "thủ đức": "thu duc",
    "nha be": "nha be", "nhà bè": "nha be",
    "hoc mon": "hoc mon", "hóc môn": "hoc mon",
    "binh chanh": "binh chanh", "bình chánh": "binh chanh",
    "can gio": "can gio", "cần giờ": "can gio",
    "cu chi": "cu chi", "củ chi": "cu chi"
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
        "tim phong khac",
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
        "them dieu kien", "them tieu chi", "them loc", "them yeu cau",
        "them ngan sach", "bo loc", "bo ", "khong can", "tang ngan sach",
        "giam ngan sach", "doi sang", "thay doi", "chinh sua dieu kien", "bo tieu chi",
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
        # Tư vấn thuê trọ chung (không gắn phòng cụ thể)
        "luu y", "nen chu y", "meo thue", "kinh nghiem thue",
        "lua dao", "luừa đảo", "nhan biet tin", "tin gia",
        "hop ly", "bao nhieu thang", "sinh vien",
        "gia han hop dong", "cham dut hop dong", "hoan coc", "hoàn cọc",
        "khieu nai", "khiếu nại", "bi lua", "bị lừa",
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
    "hoi them", "cho biet them", "cho minh biet", "chi tiet", "them chi tiet",
    "thong tin them", "noi ro them", "noi ro hon",
)

ROOM_REFERENCE_KEYWORDS: tuple[str, ...] = (
    "phong nay", "phong do", "phong tren", "phong kia", "can tren", "can kia",
    "cai tren", "cai duoi", "cai kia", "cai vua roi", "vua roi", "vua noi",
    "dang xem", "can ho nay", "can nay", "cho nay", "nha nay", "muc nay", "tin nay",
    "khu nay",
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

OFF_TOPIC_KEYWORDS: tuple[str, ...] = (
    "giai bai",
    "giai bai tap",
    "lam bai tap",
    "lam ho bai",
    "viet code",
    "code giup",
    "lap trinh giup",
    "debug code",
    "fix bug",
    "viet bai van",
    "viet essay",
    "dich doan van",
    "dich bai",
    "lam slide",
    "lam powerpoint",
    "lam cv",
    "viet cv",
    "viet email",
    "tom tat tai lieu",
    "giai toan",
    "giai ly",
    "giai hoa",
    "lam de thi",
    "xem boi",
    "coi tarot",
    "tu van tinh cam",
)

DOMAIN_KEYWORDS: tuple[str, ...] = (
    "phong",
    "nha tro",
    "phong tro",
    "can ho",
    "chdv",
    "thue",
    "gia phong",
    "tien coc",
    "tien ich",
    "quan",
    "phuong",
    "dia chi",
    "xem phong",
    "dat lich",
    "chu nha",
    "hop dong",
    "wifi",
    "may lanh",
    "ban cong",
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


_SOFT_PREFERENCE_DETAIL_KEYWORDS: tuple[str, ...] = tuple(
    sorted({_norm(alias) for alias in SOFT_PREFERENCE_ALIASES})
)


def _is_off_topic_request(normalized: str) -> bool:
    if not normalized:
        return False
    has_off_topic_signal = any(phrase in normalized for phrase in OFF_TOPIC_KEYWORDS)
    if not has_off_topic_signal:
        return False
    has_domain_signal = any(phrase in normalized for phrase in DOMAIN_KEYWORDS)
    return not has_domain_signal


def _money_to_vnd(raw: str, unit: str | None) -> int:
    from .money import money_to_vnd

    return money_to_vnd(raw, unit)


def _has_strict_upper_budget_phrase(normalized: str) -> bool:
    return bool(re.search(
        r"\b(?:duoi|re hon|thap hon|nho hon|it hon|mem hon|khong den|khong toi|chua toi)\b",
        normalized,
    ))


def _has_inclusive_upper_budget_phrase(normalized: str) -> bool:
    return bool(re.search(r"\b(?:toi da|khong qua|ngan sach|budget|tro xuong|do lai)\b", normalized))


def _has_strict_lower_budget_phrase(normalized: str) -> bool:
    if _has_strict_upper_budget_phrase(normalized):
        return False
    return bool(re.search(r"\b(?:tren|cao hon|dat hon|lon hon|hon)\b", normalized))


def _has_inclusive_lower_budget_phrase(normalized: str) -> bool:
    return bool(re.search(r"\b(?:toi thieu|tu)\b", normalized))


def _is_relative_budget_increase(normalized: str) -> bool:
    return bool(
        re.search(r"\b(?:noi|tang|them)\b.*\bngan sach\b.*\bthem\b", normalized)
        or re.search(r"\b(?:noi|tang|them)\s+them\b", normalized)
    )


def _is_price_complaint(normalized: str) -> bool:
    """Than phiền giá ('giá cao thế', 'đắt quá') — không phải hỏi chính sách/FAQ."""
    if re.search(r"\b(?:giam gia|bot gia|thuong luong)\b", normalized):
        return False
    return bool(re.search(
        r"\b(?:gia cao|dat qua|mac qua|qua tam tien|sao (?:ma )?thue noi|thue khong noi)\b",
        normalized,
    ))


def _has_search_constraint_operations(ops: list[dict[str, Any]]) -> bool:
    """Câu có ràng buộc tìm kiếm cụ thể → là search/refine, không phải FAQ hay hỏi chi tiết phòng."""
    _SEARCH_OP_PATHS = {
        "location.districts",
        "location.near_landmarks",
        "location.wards",
        "location.province",
        "budget.max",
        "budget.min",
        "categories",
        "amenities_required",
        "vehicles",
        "occupants",
        "pets_required",
    }
    return any(
        op.get("path") in _SEARCH_OP_PATHS
        and op.get("op") in {"set", "append", "replace", "remove"}
        for op in ops
    )


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
    """Trích xuất mã phòng từ nội dung câu hỏi (VD: #A101, phòng B202, P.305)."""
    ids: list[str] = []
    patterns = [
        r"#([A-Za-z0-9][A-Za-z0-9_.-]{1,40})",
        r"\b(?:phòng|phong|mã|ma)\s+([A-Za-z0-9][A-Za-z0-9_.-]{1,40})\b",
        r"\b([a-f0-9]{24})\b",
        r"\b([A-Za-z]\.\d{2,5})\b",
        r"\b([A-Za-z]{1,3}\d{2,5})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            room_id = match.group(1).strip()
            if not _looks_like_room_id(room_id):
                continue
            if room_id not in ids:
                ids.append(room_id)
    return ids


def _looks_like_area_measurement(value: str) -> bool:
    compact = str(value or "").strip().lower().replace(" ", "")
    return bool(re.fullmatch(r"\d{1,3}m2", compact)) or bool(re.fullmatch(r"\d{1,3}m²", compact))


def _looks_like_room_id(value: str) -> bool:
    if _looks_like_area_measurement(value):
        return False
    return any(ch.isdigit() for ch in value)


def _looks_like_mongo_object_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{24}", str(value or "").lower()))


def _looks_like_room_code(value: str) -> bool:
    compact = str(value or "").strip().upper().replace(".", "")
    if not compact or _looks_like_mongo_object_id(compact):
        return False
    return bool(re.fullmatch(r"[A-Z]{1,3}\d{2,5}", compact))


def _build_exact_reference(referenced_ids: list[str]) -> dict[str, Any] | None:
    if not referenced_ids:
        return None
    ref = referenced_ids[0]
    if _looks_like_mongo_object_id(ref):
        return {"room_id": ref, "confidence": 1.0}
    if _looks_like_room_code(ref):
        return {"room_code": ref, "confidence": 1.0}
    return {"room_id": ref, "confidence": 0.85}


def _normalize_room_reference(value: Any) -> str:
    return str(value or "").strip().lstrip("#").strip()


def _extract_budget(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất ngân sách tối đa / tối thiểu từ câu hỏi."""
    from .money import extract_colloquial_budget_vnd

    if re.search(r"\b(?:lon hon|lớn hơn|rong hon|rộng hơn|nho hon|nhỏ hơn|it hon|ít hơn)\s*\d+\s*m2\b", normalized):
        return

    if _is_relative_budget_increase(normalized):
        return

    colloquial = extract_colloquial_budget_vnd(normalized)
    if colloquial is not None:
        if _has_strict_lower_budget_phrase(normalized) or _has_inclusive_lower_budget_phrase(normalized):
            _append_unique(ops, "set", "budget.min", colloquial)
            operator = "gte" if _has_inclusive_lower_budget_phrase(normalized) else "gt"
            _append_unique(ops, "set", "budget.min_operator", operator)
            return
        _append_unique(ops, "set", "budget.max", colloquial)
        if _has_strict_upper_budget_phrase(normalized):
            _append_unique(ops, "set", "budget.max_operator", "lt")
        else:
            _append_unique(ops, "set", "budget.max_operator", "lte")
        return

    money = r"(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?"

    range_match = re.search(
        rf"(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?\s*(?:-|–|đến|den|tới|toi|~)\s*(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?",
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
        (
            rf"(?:duoi|re hon|thap hon|nho hon|it hon|mem hon|khong den|khong toi|chua toi)\s*{money}",
            "lt",
        ),
        (
            rf"(?:toi da|khong qua|ngan sach|budget).*?{money}",
            "lte",
        ),
        (
            rf"{money}\s*(?:do lai|tro xuong)",
            "lte",
        ),
    )
    min_patterns = (
        (
            rf"(?:tren|cao hon|dat hon|lon hon|hon)\s*{money}",
            "gt",
        ),
        (
            rf"(?:toi thieu|tu)\s*{money}",
            "gte",
        ),
    )

    matched_single_bound = False
    for pattern, operator in max_patterns:
        match = re.search(pattern, normalized)
        if match:
            max_val = _money_to_vnd(match.group(1), match.group(2))
            _append_unique(ops, "set", "budget.max", max_val)
            _append_unique(ops, "set", "budget.max_operator", operator)
            matched_single_bound = True
            break
    if not matched_single_bound:
        for pattern, operator in min_patterns:
            match = re.search(pattern, normalized)
            if match:
                min_val = _money_to_vnd(match.group(1), match.group(2))
                _append_unique(ops, "set", "budget.min", min_val)
                _append_unique(ops, "set", "budget.min_operator", operator)
                break

    # Xử lý "tăng ngân sách lên X triệu"
    if "tang ngan sach" in normalized or "tăng ngân sách" in text.lower():
        match = re.search(rf"(?:lên|len)\s*{money}", normalized)
        if match:
            _append_unique(ops, "set", "budget.max", _money_to_vnd(match.group(1), match.group(2)))


def _apply_relative_budget_refinement(
    text: str,
    normalized: str,
    current_state: dict[str, Any] | None,
    ops: list[dict[str, Any]],
) -> None:
    if not current_state:
        return
    budget = (current_state.get("constraints") or {}).get("budget") or {}
    current_max = budget.get("max")
    if not isinstance(current_max, (int, float)) or current_max <= 0:
        return

    match = re.search(
        r"(?:noi|nới|tang|tăng|them|thêm)\s+(?:ngan sach|ngân sách)?\s*(?:them|thêm)?\s*(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?",
        normalized,
    )
    if not match:
        return
    if "ngan sach" not in normalized and "ngân sách" not in text.lower():
        return

    increased_max = int(current_max) + _money_to_vnd(match.group(1), match.group(2))
    ops[:] = [op for op in ops if op.get("path") != "budget.max"]
    _append_unique(ops, "set", "budget.max", increased_max)
    if budget.get("max_operator"):
        ops[:] = [op for op in ops if op.get("path") != "budget.max_operator"]
        _append_unique(ops, "set", "budget.max_operator", budget.get("max_operator"))


def _maybe_replace_budget_filters(
    normalized: str,
    current_state: dict[str, Any] | None,
    ops: list[dict[str, Any]],
) -> None:
    if not current_state:
        return
    budget = (current_state.get("constraints") or {}).get("budget") or {}
    set_paths = {
        op.get("path")
        for op in ops
        if op.get("op") in {"set", "replace"} and str(op.get("path") or "").startswith("budget.")
    }
    sets_min = "budget.min" in set_paths
    sets_max = "budget.max" in set_paths
    if sets_min == sets_max:
        return

    def _prepend_clear(path: str) -> None:
        clear_op = {"op": "clear", "path": path}
        if clear_op not in ops:
            ops.insert(0, clear_op)

    if sets_max and budget.get("min") is not None and not _is_relative_budget_increase(normalized):
        _prepend_clear("budget.min_operator")
        _prepend_clear("budget.min")
        return

    if sets_min and budget.get("max") is not None:
        new_min = next((op.get("value") for op in ops if op.get("path") == "budget.min"), None)
        current_max = budget.get("max")
        if (
            isinstance(new_min, (int, float))
            and isinstance(current_max, (int, float))
            and new_min >= current_max
        ):
            _prepend_clear("budget.max_operator")
            _prepend_clear("budget.max")


def _extract_location(normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất quận/huyện và mốc địa lý (gần trường, gần chợ...) từ câu hỏi."""
    # Nhận diện quận/huyện — hỗ trợ cả "quận 3", "Q3", "huyện Bình Chánh"
    compact_districts = []
    for match in re.finditer(r"\bq\.?\s*(\d{1,2})\b", normalized, flags=re.IGNORECASE):
        value = f"quan {match.group(1)}"
        if value not in compact_districts:
            compact_districts.append(value)

    district_pattern = re.compile(
        r"\b(?:quan|q\.?|huyen|huyện)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]{1,30})",
        re.IGNORECASE,
    )
    districts = compact_districts
    for match in district_pattern.finditer(normalized):
        value = match.group(1).strip()
        # Cắt tại từ ngăn cách để tránh lấy thừa
        value = re.split(r"\b(?:gan|duoi|tren|co|va|gia|,|\.)\b", value)[0].strip()
        numeric_district = re.match(r"(\d{1,2})\b", value)
        if numeric_district:
            value = numeric_district.group(1)
        if value and value not in districts:
            districts.append(f"quan {value}")
    for district in districts:
        _append_unique(ops, "append", "location.districts", district)

    for alias, standard_name in DISTRICT_ALIASES.items():
        if _contains_phrase(normalized, alias):
            _append_unique(ops, "append", "location.districts", standard_name)

    wards = []
    for match in re.finditer(
        r"\b(?:phuong|phường|xa|xã)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]{1,35})",
        normalized,
        re.IGNORECASE,
    ):
        value = match.group(1).strip()
        value = re.split(r"\b(?:gan|duoi|tren|co|va|gia|,|\.)\b", value)[0].strip()
        if value and value not in wards:
            wards.append(value)
    for ward in wards:
        _append_unique(ops, "append", "location.wards", ward)

    # Nhận diện mốc địa lý gần (gần ĐHQG, gần Vincom...)
    landmarks = []
    for match in re.finditer(r"\b(?:gan|gần)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]{2,40})", normalized):
        value = match.group(1).strip()
        value = re.split(r"\b(?:duoi|tren|co|va|khong|ko|k|,|\.)\b", value)[0].strip()
        value = _clean_landmark(value)
        if value and value not in landmarks:
            landmarks.append(value)
    for landmark in landmarks:
        _append_unique(ops, "append", "location.near_landmarks", landmark)

    street_patterns = (
        r"\b(?:duong|đường|mat tien|mặt tiền|hem|hẻm)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ/-]{2,45})",
    )
    for pattern in street_patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            value = match.group(1).strip()
            value = re.split(r"\b(?:duoi|tren|co|va|gia|,|\.)\b", value)[0].strip()
            value = _clean_landmark(value)
            if value:
                _append_unique(ops, "append", "location.near_landmarks", value)


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
    match = re.search(r"(\d+)\s*(?:nguoi|người|ban|bạn)\b", normalized)
    if match:
        _append_unique(ops, "set", "occupants", int(match.group(1)))

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


def _contains_phrase(normalized: str, phrase: str) -> bool:
    escaped = re.escape(_norm(phrase))
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", normalized))


# Từ đệm / đại từ chỉ định bị regex "gần …", "đường …" vô tình gom vào địa danh.
# Loại chúng để tránh tạo near_landmark rác như "do" (từ "gần đó"), "tdtu a" (từ "gần TDTU á").
_LANDMARK_TRAILING_FILLER: frozenset[str] = frozenset({
    "a", "ah", "vay", "v", "z", "nha", "nhe", "nhi", "ne",
    "ko", "k", "kg", "khong", "oi", "luon", "do", "day", "kia",
})
# Đại từ chỉ định thuần — không phải địa danh thật ("gần đó", "gần đây", "gần kia").
_LANDMARK_REFERENCE_ONLY: frozenset[str] = frozenset({
    "do", "day", "kia", "nay",
})


def _clean_landmark(value: str) -> str | None:
    """Làm sạch địa danh: bỏ từ đệm cuối câu và loại đại từ chỉ định.

    Trả về None nếu chuỗi không còn là địa danh thật (vd: "đó", "đây").
    """
    cleaned = re.sub(r"\s+", " ", str(value or "").strip())
    if not cleaned:
        return None
    tokens = cleaned.split(" ")
    while tokens and tokens[-1] in _LANDMARK_TRAILING_FILLER:
        tokens.pop()
    result = " ".join(tokens).strip()
    if len(result) < 2:
        return None
    if result in _LANDMARK_REFERENCE_ONLY:
        return None
    normalized = normalize_landmark(result)
    if normalized:
        return normalized
    if _should_drop_unresolved_landmark(_norm_landmark(result)):
        return None
    return result


def _extract_categories(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất loại phòng từ câu hỏi."""
    for alias, canonical in ROOM_TYPE_ALIASES.items():
        if _contains_phrase(normalized, alias):
            _append_unique(ops, "append", "categories", canonical)


def _extract_amenities(
    text: str,
    normalized: str,
    ops: list[dict[str, Any]],
    current_state: dict[str, Any] | None = None,
) -> None:
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
        if _is_location_pivot_phrase(normalized) and canonical == "bright":
            continue
        if canonical == "newly_renovated" and re.search(r"\bmoi\s+dung\b", normalized):
            continue
        if _contains_phrase(normalized, alias):
            if _looks_like_room_attribute_question(normalized, current_state):
                continue
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
    remove_prefix = r"(?:bo|khong can|loai|xoa)"
    filler = r"(?:\s+\w+){0,4}\s+"
    return bool(re.search(rf"\b{remove_prefix}\b{filler}{escaped}(?!\w)", normalized))


def _first_or_none(values: list[Any]) -> Any | None:
    return values[0] if values else None


def _current_context_room_id(current_state: dict[str, Any] | None) -> str | None:
    if not current_state:
        return None
    current = str(current_state.get("current_room_id") or "").strip()
    if current:
        return current
    return _first_or_none(current_state.get("last_result_ids") or [])


def _resolve_parsed_current_room_id(
    *,
    intent: str,
    referenced_room_ids: list[str],
    current_state: dict[str, Any] | None,
    ordinal_out_of_range: bool,
) -> str | None:
    if ordinal_out_of_range:
        return None
    if referenced_room_ids:
        return referenced_room_ids[0]
    if current_state and intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST", "FIND_SIMILAR"}:
        return _current_context_room_id(current_state)
    return None


def _has_soft_preference_keyword(normalized: str) -> bool:
    return _has_keyword(normalized, _SOFT_PREFERENCE_DETAIL_KEYWORDS)


def _is_search_availability_question(normalized: str) -> bool:
    """'Có phòng ... không?' — tìm kiếm, không phải hỏi thuộc tính phòng đang xem."""
    if _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
        return False
    if re.search(r"\b(?:can|phong|nha)\s+(?:nay|do|kia|tren|duoi(?!\s*(?:\d|trieu|tr\b|t\b|m\b|cu\b|chuc\b|tram\b))|dang xem)\b", normalized):
        return False
    return bool(re.search(r"\b(?:co|con)\s+(?:can|phong|nha)\b", normalized))


def _looks_like_room_attribute_question(
    normalized: str,
    current_state: dict[str, Any] | None,
) -> bool:
    if not _has_current_room(current_state):
        return False
    if _looks_like_session_constraint_refinement(normalized, current_state):
        return False
    if _is_location_pivot_phrase(normalized):
        return False
    if _is_search_availability_question(normalized):
        return False
    if _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
        return True
    if not _has_soft_preference_keyword(normalized):
        return False
    if re.search(r"\b(?:can|phong|nha)\s+nao\b", normalized):
        return False
    return bool(
        re.search(
            r"\b(co|có|khong|không|the nao|thế nào|bao nhieu|bao nhiêu|la gi|là gì|"
            r"duoc khong|được không|co khong|có không|an toan|an ninh|yen tinh)\b",
            normalized,
        )
    )


def _is_negotiate_capability_question(normalized: str) -> bool:
    """Hỏi khả năng mặc cả — tư vấn, không phải yêu cầu thực hiện."""
    if not re.search(r"\b(giam gia|giảm giá|thuong luong|thương lượng|tra gia|trả giá|bot gia|bớt giá)\b", normalized):
        return False
    return bool(
        re.search(
            r"\b(co the|liệu có|duoc khong|được không|co khong|có không|co duoc|có được|"
            r"bao nhieu|bao nhiêu|the nao|thế nào|ra sao)\b",
            normalized,
        )
        or re.search(r"\b(co|con)\s+(?:giam gia|thuong luong|tra gia)\b", normalized)
    )


def _requested_action(normalized: str) -> str | None:
    """Kiểm tra xem người dùng có yêu cầu thao tác nghiệp vụ không."""
    if _is_action_capability_question(normalized):
        return None
    if _is_negotiate_capability_question(normalized):
        return None
    for action, keywords in ACTION_KEYWORDS.items():
        if any(_norm(keyword) in normalized for keyword in keywords):
            return action
    return None


def _is_action_capability_question(normalized: str) -> bool:
    return bool(
        re.search(r"\b(co the|liệu có|qua web|tren web|trên web|quy trinh|quy định|thu tuc|thủ tục)\b", normalized)
        or re.search(r"\b(bao nhieu|bao nhiêu|the nao|thế nào|ra sao)\b", normalized)
    )


def _has_current_room(current_state: dict[str, Any] | None) -> bool:
    if not current_state:
        return False
    return bool(current_state.get("current_room_id") or current_state.get("selected_room_ids") or current_state.get("last_result_ids"))


def _has_keyword(normalized: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in normalized for keyword in keywords)


def _is_chitchat_or_closing(normalized: str) -> bool:
    compact = normalized.strip(" .,!?\t\r\n")
    if not compact:
        return True
    return any(_contains_phrase(compact, keyword) for keyword in CHITCHAT_KEYWORDS)


def _is_location_pivot_phrase(normalized: str) -> bool:
    if re.search(
        r"\b(?:doi|đổi|chuyen|chuyển|sang|ve|về)\b.*\b(?:quan|q\.?|phuong|phường|khu)\b",
        normalized,
    ):
        return True
    if re.match(r"^(?:khong|không)\s*,", normalized) and re.search(
        r"\b(?:quan|q\.?|phuong|phường)\b",
        normalized,
    ):
        return True
    if re.search(r"\bmoi\s+dung\b", normalized) and re.search(r"\b(?:quan|q\.?)\b", normalized):
        return True
    return False


def _is_attribute_confirmation_question(normalized: str) -> bool:
    """'Có máy lạnh không?' — hỏi thuộc tính phòng đang xem, không phải refine tìm kiếm."""
    if re.match(r"^(?:khong|không)\s*,", normalized):
        return False
    if re.search(r"\b(?:duoc khong|được không|co khong|có không)\s*\??\s*$", normalized):
        return True
    if re.search(r"\b(?:khong|không)\s*\??\s*$", normalized):
        return True
    return False


def _looks_like_session_constraint_refinement(
    normalized: str,
    current_state: dict[str, Any] | None,
) -> bool:
    """Refine điều kiện tìm kiếm giữa phiên — không nhầm với hỏi chi tiết một phòng."""
    if not current_state:
        return False
    if current_state.get("last_intent") not in {"SEARCH_ROOM", "REFINE_SEARCH"}:
        return False
    if _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
        return False
    if _selected_room_id_from_ordinal(normalized, current_state):
        return False
    if _selected_room_id_from_deictic(normalized, current_state):
        return False
    if _is_attribute_confirmation_question(normalized):
        return False
    if _is_location_pivot_phrase(normalized):
        return True
    if re.search(
        r"\b(?:them|thêm|bo|bỏ|uu tien|ưu tiên|loai|loại|xoa|xóa|tang|tăng|giam|giảm|"
        r"nang|nâng|chi|chỉ|muon|muốn|can them|cần thêm)\b",
        normalized,
    ):
        return True
    if re.search(
        r"\b(?:co|con)\s+(?:may lanh|máy lạnh|noi that|nội thất|cho de xe|chỗ để xe|"
        r"ban cong|ban công|may giat|máy giặt|tu lanh|tủ lạnh|wifi)\b",
        normalized,
    ):
        return True
    if re.search(r"\b(?:muon|muốn|can|need)\s+.*\b(?:co|con)\b", normalized):
        return True
    if re.search(
        r"\b(?:lay|lấy)\s+\d{1,2}\s+(?:phong|phòng|can|căn)\s+(?:tot|tốt|phu hop|phù hợp)",
        normalized,
    ):
        return True
    return False


def _is_room_detail_question(normalized: str, ids: list[str], current_state: dict[str, Any] | None) -> bool:
    if _looks_like_session_constraint_refinement(normalized, current_state):
        return False
    if _is_location_pivot_phrase(normalized):
        return False
    if re.search(r"\b(?:co|con)\s+(?:can|phong|nha)\s+nao\b", normalized):
        return False
    if re.search(r"\b(?:can|phong|nha)\s+nao\s+(?:co|con)\b", normalized):
        return False
    if re.search(r"\b(?:them|thêm|bo|bỏ|loai|loại|xoa|xóa)\s+(?:dieu kien|điều kiện)\b", normalized):
        return False
    if re.search(r"\b(?:tim|tìm|loc|lọc)\s+phong\b", normalized):
        return False
    if _has_keyword(normalized, COST_FIELD_KEYWORDS):
        return False
    if _has_soft_preference_keyword(normalized) and (
        ids
        or _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS)
        or _has_current_room(current_state)
    ):
        if re.search(r"\b(?:can|phong|nha)\s+nao\b", normalized):
            return False
        if _is_search_availability_question(normalized):
            return False
        return True
    if not _has_keyword(normalized, DETAIL_FIELD_KEYWORDS):
        return False
    if ids or _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
        return True
    return _has_current_room(current_state) and bool(re.search(r"\b(co|có|khong|không|bao nhieu|bao nhiêu|the nao|thế nào|la gi|là gì|cho toi biet|cho minh biet|noi ro|nói rõ)\b", normalized))


def _is_result_set_compare_request(normalized: str) -> bool:
    if not re.search(r"\bso\s*sanh\b", normalized):
        return False
    if len(_ordinal_indices_from_text(normalized)) >= 2:
        return False
    if re.search(r"\bphong\s+(?:so|số|thu|thứ)\s*\d", normalized):
        return False
    return bool(
        re.search(r"\b(?:3|ba)\s*phong\b", normalized)
        or re.search(r"\b(?:2|hai)\s*phong\b", normalized)
        or re.search(r"\b(?:cac|nhung|may)\s+phong\b", normalized)
        or re.search(r"\bphong\s+(?:nay|tren|vua|dau tien)\b", normalized)
    )


def _requested_compare_count(normalized: str, available_count: int) -> int:
    if match := re.search(r"\b(\d{1,2})\s*phong\b", normalized):
        return min(int(match.group(1)), available_count, 3)
    if re.search(r"\b(?:3|ba)\s*phong\b", normalized):
        return min(3, available_count)
    if re.search(r"\b(?:2|hai)\s*phong\b", normalized):
        return min(2, available_count)
    if re.search(r"\bphong\s+(?:dau tien|đầu tiên)\b", normalized):
        return min(1, available_count)
    return min(3, available_count)


_ORDINAL_WORD_INDEX: dict[str, int] = {
    "mot": 0,
    "một": 0,
    "nhat": 0,
    "nhất": 0,
    "hai": 1,
    "ba": 2,
    "bon": 3,
    "bốn": 3,
    "tu": 3,
    "tư": 3,
    "nam": 4,
    "năm": 4,
}


def _ordinal_indices_from_text(normalized: str) -> list[int]:
    hits: list[tuple[int, int]] = []
    for match in re.finditer(r"\b(?:phong|phòng)\s*(?:so|số|thu|thứ|#)?\s*(\d{1,2})\b", normalized):
        hits.append((match.start(), int(match.group(1)) - 1))
    for match in re.finditer(
        r"\b(?:phong|phòng)\s+(?:thu|thứ|so|số)?\s*(mot|một|nhat|nhất|hai|ba|bon|bốn|tu|tư|nam|năm)\b",
        normalized,
    ):
        index = _ORDINAL_WORD_INDEX.get(match.group(1))
        if index is not None:
            hits.append((match.start(), index))
    first_match = re.search(r"\b(?:phong|phòng)\s+(?:dau tien|đầu tiên)\b", normalized)
    if first_match:
        hits.append((first_match.start(), 0))
    for match in re.finditer(
        r"\b(?:can|căn|cai|cái|muc|mục)\s*(?:so|số|thu|thứ|#)?\s*(\d{1,2})\b",
        normalized,
    ):
        hits.append((match.start(), int(match.group(1)) - 1))
    for match in re.finditer(
        r"\b(?:can|căn|cai|cái|muc|mục)\s+(?:thu|thứ|so|số)?\s*(mot|một|nhat|nhất|hai|ba|bon|bốn|tu|tư|nam|năm)\b",
        normalized,
    ):
        index = _ORDINAL_WORD_INDEX.get(match.group(1))
        if index is not None:
            hits.append((match.start(), index))
    first_item = re.search(r"\b(?:can|căn|cai|cái|muc|mục)\s+(?:dau tien|đầu tiên)\b", normalized)
    if first_item:
        hits.append((first_item.start(), 0))
    if re.search(r"\bso\s*sanh\b", normalized):
        for match in re.finditer(r"\b(?:so|số)\s*(\d{1,2})\b", normalized):
            index = int(match.group(1)) - 1
            hits.append((match.start(), index))
        for match in re.finditer(
            r"\b(?:thu|thứ)\s*(mot|một|nhat|nhất|hai|ba|bon|bốn|tu|tư|nam|năm)\b",
            normalized,
        ):
            index = _ORDINAL_WORD_INDEX.get(match.group(1))
            if index is not None:
                hits.append((match.start(), index))
        for match in re.finditer(r"\b(?:va|và|and)\s*(\d{1,2})\b", normalized):
            hits.append((match.start(), int(match.group(1)) - 1))
    hits.sort(key=lambda item: item[0])
    indices: list[int] = []
    for _, index in hits:
        if index not in indices:
            indices.append(index)
    return indices


def _ordinal_request_index(normalized: str, current_state: dict[str, Any] | None) -> int | None:
    if not current_state:
        return None
    if re.search(r"\b(?:lay|lấy|cho|đưa|dua)\s+\d{1,2}\s+(?:phong|phòng|can|căn)\b", normalized):
        return None
    match = re.search(r"\b(?:chon|chọn|lay|lấy)\s+(?:phong|phòng)?\s*(?:so|số|#)?\s*(\d{1,2})\b", normalized)
    if not match:
        match = re.search(r"\b(?:phong|phòng)\s*(?:so|số|thu|thứ|#)\s*(\d{1,2})\b", normalized)
    if not match:
        match = re.search(r"\b(?:phong|phòng)\s*(\d{1,2})\b", normalized)
    if match:
        return int(match.group(1)) - 1
    if re.search(r"\b(?:phong|phòng)\s+(?:dau tien|đầu tiên|thu nhat|thứ nhất|so mot|số một|1)\b", normalized):
        return 0
    word_match = re.search(
        r"\b(?:phong|phòng)\s+(?:thu|thứ|so|số)?\s*(hai|ba|bon|bốn|tu|tư|nam|năm)\b",
        normalized,
    )
    if word_match:
        return _ORDINAL_WORD_INDEX.get(word_match.group(1))
    return None


_DETAIL_SEARCH_OP_PATHS = frozenset({
    "amenities_required",
    "amenities_preferred",
    "categories",
    "excluded_features",
    "budget.min",
    "budget.max",
    "budget.min_operator",
    "budget.max_operator",
    "location.districts",
    "location.wards",
    "location.province",
    "location.near_landmarks",
    "vehicles",
    "occupants",
    "pets_required",
})


def _strip_detail_question_search_ops(
    operations: list[dict[str, Any]],
    *,
    intent: str,
    normalized: str,
    referenced_room_ids: list[str],
    current_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    about_current_room = (
        intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}
        or _looks_like_room_attribute_question(normalized, current_state)
        or (
            _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS)
            and _has_current_room(current_state)
            and not any(kw in normalized for kw in _INTENT_KEYWORDS["SEARCH_ROOM"])
            and not re.search(r"\b(?:co|con)\s+(?:can|phong|nha)\s+nao\b", normalized)
        )
    )
    if not about_current_room:
        return operations
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"} and not _is_room_detail_question(
        normalized, referenced_room_ids, current_state
    ):
        if not _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
            return operations
    return [op for op in operations if op.get("path") not in _DETAIL_SEARCH_OP_PATHS]


def _selected_room_id_from_ordinal(normalized: str, current_state: dict[str, Any] | None) -> str | None:
    if not current_state:
        return None
    index = _ordinal_request_index(normalized, current_state)
    if index is None:
        return None
    ids = current_state.get("last_result_ids") or []
    if 0 <= index < len(ids):
        return str(ids[index])
    return None


def _selected_room_ids_from_ordinals(
    normalized: str,
    current_state: dict[str, Any] | None,
) -> tuple[list[str], bool]:
    if not current_state:
        return [], False
    ids = current_state.get("last_result_ids") or []
    if not ids:
        return [], False

    indices = _ordinal_indices_from_text(normalized)
    if not indices:
        return [], False

    resolved = []
    for index in indices:
        if 0 <= index < len(ids):
            resolved_id = str(ids[index])
            if resolved_id not in resolved:
                resolved.append(resolved_id)
    unresolved = len(resolved) < len(indices)
    return resolved, unresolved


def _selected_room_id_from_deictic(normalized: str, current_state: dict[str, Any] | None) -> str | None:
    """Resolve chỉ định ngữ cảnh: phòng này/đó, cái trên, phòng kia, căn vừa rồi…"""
    if not current_state:
        return None

    context_room = _current_context_room_id(current_state)
    if context_room and (
        re.search(r"\b(?:can|phong|cai|tin|muc|nha|cho|khu)\s+nay\b", normalized)
        or re.search(r"\bdang\s+xem\b", normalized)
        or re.search(r"\b(?:can|phong|cai)\s+do\b", normalized)
    ):
        return str(context_room)

    ids = [str(item) for item in (current_state.get("last_result_ids") or []) if item]
    if not ids:
        return None

    def _pick(index: int) -> str | None:
        if index < 0:
            index = len(ids) + index
        if 0 <= index < len(ids):
            return ids[index]
        return None

    if re.search(r"\b(?:can|phong|cai)\s+(?:o\s+)?tren\b", normalized):
        return _pick(0)

    if re.search(r"\b(?:can|phong|cai)\s+(?:o\s+)?duoi(?!\s*(?:\d|trieu|tr\b|t\b|m\b|cu\b|chuc\b|tram\b))\b", normalized):
        return _pick(1) if len(ids) > 1 else _pick(-1)

    if re.search(r"\b(?:can|phong)\s+cuoi\b", normalized):
        return _pick(-1)

    if re.search(r"\b(?:can|phong)\s+kia\b", normalized):
        current = str(current_state.get("current_room_id") or "").strip()
        if current and current in ids and len(ids) >= 2:
            current_index = ids.index(current)
            other_index = 1 - current_index if len(ids) == 2 else (current_index + 1) % len(ids)
            return ids[other_index]
        return _pick(1) if len(ids) > 1 else _pick(0)

    if re.search(r"\b(?:cai|can)\s+vua\s+(?:roi|noi|goi)\b", normalized) or re.search(
        r"\bvua\s+(?:roi|noi)\b",
        normalized,
    ):
        current = str(current_state.get("current_room_id") or "").strip()
        if current and current in ids:
            return current
        return _pick(0)

    return None


def _maybe_replace_location_filters(
    normalized: str,
    current_state: dict[str, Any] | None,
    ops: list[dict[str, Any]],
) -> None:
    if not current_state:
        return
    current_location = (current_state.get("constraints") or {}).get("location") or {}
    current_districts = [str(item).strip() for item in (current_location.get("districts") or []) if item]
    current_wards = [str(item).strip() for item in (current_location.get("wards") or []) if item]
    current_landmarks = [str(item).strip() for item in (current_location.get("near_landmarks") or []) if item]
    new_districts = [op["value"] for op in ops if op.get("op") == "append" and op.get("path") == "location.districts"]
    new_landmarks = [op["value"] for op in ops if op.get("op") == "append" and op.get("path") == "location.near_landmarks"]
    if not new_districts and not new_landmarks:
        return

    additive = bool(re.search(r"\b(them|thêm|hoac|hoặc|ca|cả)\b", normalized))
    if additive:
        return

    def _prepend_clear(path: str) -> None:
        clear_op = {"op": "clear", "path": path}
        if clear_op not in ops:
            ops.insert(0, clear_op)

    def _district_sets_differ(current: list[str], new: list[str]) -> bool:
        if not current or not new:
            return bool(current) != bool(new)
        return {_norm(item) for item in current} != {_norm(item) for item in new}

    # Lượt tìm kiếm neo theo quận: reset phường/mốc cũ; khi đổi quận thì reset thêm tiện ích ưu tiên.
    if new_districts:
        if current_districts and _district_sets_differ(current_districts, new_districts):
            _prepend_clear("location.districts")
            if (current_state.get("constraints") or {}).get("amenities_preferred"):
                _prepend_clear("amenities_preferred")
            _maybe_clear_stale_scope_on_pivot(ops)
        if current_wards:
            _prepend_clear("location.wards")
        if current_landmarks:
            _prepend_clear("location.near_landmarks")
        return

    # Mốc mới (không kèm quận): thay mốc cũ thay vì cộng dồn — trừ khi chỉ lặp lại đúng mốc hiện tại.
    if new_landmarks and current_landmarks:
        normalized_new = {_norm(item) for item in new_landmarks}
        normalized_current = {_norm(item) for item in current_landmarks}
        if not normalized_new.issubset(normalized_current):
            _prepend_clear("location.near_landmarks")


def _maybe_clear_stale_scope_on_pivot(ops: list[dict[str, Any]]) -> None:
    """Khi đổi quận/khu vực, bỏ loại phòng từ lượt trước nếu lượt này không nhắc lại."""
    if any(op.get("path") == "categories" for op in ops):
        return
    clear_op = {"op": "clear", "path": "categories"}
    if clear_op not in ops:
        ops.insert(0, clear_op)


def _maybe_clear_stale_search_scope(normalized: str, ops: list[dict[str, Any]]) -> None:
    """Tìm kiếm mới (có 'tìm phòng' + điều kiện) nhưng không nhắc loại phòng → bỏ category cũ trong session."""
    if _is_relative_budget_increase(normalized):
        return
    if any(op.get("path") == "categories" for op in ops):
        return
    has_search_pivot = any(
        op.get("path") in {"location.districts", "budget.max", "location.near_landmarks"}
        and op.get("op") in {"append", "set", "clear", "replace"}
        for op in ops
    )
    if not has_search_pivot:
        return
    is_full_search = bool(
        re.search(r"\b(?:tim|muon thue|can thue|thue phong|phong tro|nha tro|can ho)\b", normalized)
    )
    if not is_full_search:
        return
    clear_op = {"op": "clear", "path": "categories"}
    if clear_op not in ops:
        ops.insert(0, clear_op)


def _coerce_llm_payload(raw: Any) -> tuple[dict[str, Any], float]:
    if isinstance(raw, dict):
        confidence = raw.get("confidence")
        try:
            parsed_confidence = float(confidence)
        except (TypeError, ValueError):
            parsed_confidence = 0.0
        return raw, parsed_confidence
    if isinstance(raw, tuple) and raw:
        intent = raw[0] if len(raw) >= 1 else "GENERAL_HELP"
        confidence = raw[1] if len(raw) >= 2 else 0.0
        try:
            parsed_confidence = float(confidence)
        except (TypeError, ValueError):
            parsed_confidence = 0.0
        return {"intent": intent, "operations": []}, parsed_confidence
    return {}, 0.0


def _sanitize_llm_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in operations or []:
        op = item.get("op")
        path = item.get("path")
        value = item.get("value")
        if path == "location.university":
            path = "location.near_landmarks"
            op = "append"
        if path == "location.districts" and isinstance(value, str):
            value = DISTRICT_ALIASES.get(_norm(value), value.strip())
        if op not in OP_TYPES or path not in ALLOWED_OPERATION_PATHS:
            continue
        clean_item = {"op": op, "path": path}
        if op != "clear":
            clean_item["value"] = _clean_value_for_operation(value)
        if clean_item not in sanitized:
            sanitized.append(clean_item)
    return sanitized


def _clean_value_for_operation(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _partition_operations(operations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard_ops: list[dict[str, Any]] = []
    soft_ops: list[dict[str, Any]] = []
    for item in operations or []:
        path = item.get("path")
        if path in HARD_OPERATION_PATHS:
            if item not in hard_ops:
                hard_ops.append(item)
        elif path in SOFT_OPERATION_PATHS:
            if item not in soft_ops:
                soft_ops.append(item)
    return hard_ops, soft_ops


def _operation_signature(item: dict[str, Any]) -> tuple[Any, Any, Any]:
    return item.get("op"), item.get("path"), _clean_value_for_operation(item.get("value"))


def _same_operation_set(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return {_operation_signature(item) for item in left} == {_operation_signature(item) for item in right}


def _router_candidate_summary(parsed: dict[str, Any], confidence: float) -> dict[str, Any]:
    operations = list(parsed.get("operations", []))
    hard_ops, soft_ops = _partition_operations(operations)
    return {
        "intent": parsed.get("intent", "GENERAL_HELP"),
        "confidence": confidence,
        "hard_operations": hard_ops,
        "soft_operations": soft_ops,
        "referenced_room_ids": list(parsed.get("referenced_room_ids", []) or []),
        "requested_action": parsed.get("requested_action"),
        "current_room_id": parsed.get("current_room_id"),
    }


def _has_router_conflict(regex_candidate: dict[str, Any], llm_candidate: dict[str, Any]) -> bool:
    if regex_candidate.get("intent") != llm_candidate.get("intent"):
        return True
    if list(regex_candidate.get("referenced_room_ids") or []) != list(llm_candidate.get("referenced_room_ids") or []):
        return True
    if regex_candidate.get("requested_action") != llm_candidate.get("requested_action"):
        return True
    return not _same_operation_set(
        list(regex_candidate.get("hard_operations") or []),
        list(llm_candidate.get("hard_operations") or []),
    )


def _build_llm_candidate(
    llm_intent: str,
    llm_confidence: float,
    llm_operations: list[dict[str, Any]],
    referenced_room_ids: list[str],
    requested_action: str | None,
) -> dict[str, Any]:
    current_room_id = referenced_room_ids[0] if referenced_room_ids else None
    return _router_candidate_summary(
        {
            "intent": llm_intent,
            "operations": llm_operations,
            "referenced_room_ids": referenced_room_ids,
            "requested_action": requested_action,
            "current_room_id": current_room_id,
        },
        llm_confidence,
    )


def _merge_router_decision(
    regex_parsed: dict[str, Any],
    regex_candidate: dict[str, Any],
    llm_candidate: dict[str, Any],
    verifier: dict[str, Any],
    current_state: dict[str, Any] | None,
) -> dict[str, Any]:
    approved_intent = regex_parsed.get("intent", "GENERAL_HELP")
    if verifier.get("approved_intent") in INTENTS:
        approved_intent = verifier["approved_intent"]
    elif verifier.get("use_llm_intent") and llm_candidate.get("intent") in INTENTS:
        approved_intent = llm_candidate["intent"]
    elif regex_candidate.get("intent") in INTENTS:
        approved_intent = regex_candidate["intent"]

    operations = list(regex_parsed.get("operations", []))
    regex_hard_ops = list(regex_candidate.get("hard_operations") or [])
    if regex_hard_ops:
        _, regex_soft_ops = _partition_operations(operations)
        operations = regex_hard_ops + [
            op for op in regex_soft_ops
            if op.get("path") not in {item.get("path") for item in regex_hard_ops}
        ]
    elif verifier.get("use_llm_hard_slots"):
        _, regex_soft_ops = _partition_operations(operations)
        operations = list(llm_candidate.get("hard_operations", [])) + regex_soft_ops
    if verifier.get("allow_llm_soft_slots", True):
        for item in llm_candidate.get("soft_operations", []):
            if item not in operations:
                operations.append(item)
    regex_hard_paths = {item.get("path") for item in regex_hard_ops} if regex_hard_ops else set()
    for item in verifier.get("approved_operations", []) or []:
        path = item.get("path")
        if path in HARD_OPERATION_PATHS and regex_hard_paths and path in regex_hard_paths:
            continue
        if item not in operations and item.get("path") in ALLOWED_OPERATION_PATHS and item.get("op") in OP_TYPES:
            operations.append(item)

    referenced_room_ids = list(regex_parsed.get("referenced_room_ids", []) or [])
    regex_room_ids = [
        _normalize_room_reference(item)
        for item in referenced_room_ids
        if _normalize_room_reference(item)
    ]
    approved_ids = [
        _normalize_room_reference(item)
        for item in (verifier.get("approved_room_ids") or [])
        if _normalize_room_reference(item)
    ]
    if regex_room_ids and float(regex_candidate.get("confidence") or 0) >= 0.85:
        referenced_room_ids = regex_room_ids
    elif approved_ids:
        referenced_room_ids = approved_ids
    elif verifier.get("use_llm_hard_slots"):
        referenced_room_ids = list(llm_candidate.get("referenced_room_ids", []) or referenced_room_ids)

    requested_action = regex_parsed.get("requested_action")
    approved_action = verifier.get("approved_requested_action")
    if regex_parsed.get("requested_action") and float(regex_candidate.get("confidence") or 0) >= 0.85:
        requested_action = regex_parsed.get("requested_action")
    elif approved_action:
        requested_action = approved_action
    elif verifier.get("use_llm_hard_slots") and llm_candidate.get("requested_action"):
        requested_action = llm_candidate.get("requested_action")

    current_room_id = regex_parsed.get("current_room_id")
    if regex_parsed.get("ordinal_out_of_range") or regex_parsed.get("compare_unresolved"):
        current_room_id = None
        if regex_parsed.get("ordinal_out_of_range"):
            referenced_room_ids = []
    elif referenced_room_ids:
        current_room_id = referenced_room_ids[0]
    elif not current_room_id and current_state and approved_intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST", "FIND_SIMILAR"}:
        current_room_id = _current_context_room_id(current_state)

    merged = {
        "intent": approved_intent if approved_intent in INTENTS else "GENERAL_HELP",
        "operations": operations,
        "current_room_id": current_room_id,
        "referenced_room_ids": referenced_room_ids,
        "requested_action": requested_action,
    }
    for key in ("exact_reference", "ordinal_out_of_range", "ordinal_requested", "ordinal_available_count", "compare_unresolved"):
        if key in regex_parsed:
            merged[key] = regex_parsed[key]
    return merged


def _default_verifier_decision(regex_candidate: dict[str, Any], llm_candidate: dict[str, Any]) -> dict[str, Any]:
    regex_intent = regex_candidate.get("intent", "GENERAL_HELP")
    llm_intent = llm_candidate.get("intent", "GENERAL_HELP")
    regex_conf = float(regex_candidate.get("confidence") or 0.0)
    llm_conf = float(llm_candidate.get("confidence") or 0.0)
    same_hard = _same_operation_set(
        list(regex_candidate.get("hard_operations") or []),
        list(llm_candidate.get("hard_operations") or []),
    ) and list(regex_candidate.get("referenced_room_ids") or []) == list(llm_candidate.get("referenced_room_ids") or [])

    if regex_intent == llm_intent and same_hard:
        return {
            "approved_intent": regex_intent,
            "use_llm_intent": False,
            "use_llm_hard_slots": False,
            "allow_llm_soft_slots": True,
            "approved_room_ids": regex_candidate.get("referenced_room_ids", []),
            "approved_requested_action": regex_candidate.get("requested_action"),
            "hard_conflict": False,
            "reason": "regex_and_llm_agree",
        }

    if regex_intent == "GENERAL_HELP" and llm_intent != "GENERAL_HELP" and llm_conf >= 0.9:
        return {
            "approved_intent": llm_intent,
            "use_llm_intent": True,
            "use_llm_hard_slots": same_hard,
            "allow_llm_soft_slots": True,
            "approved_room_ids": llm_candidate.get("referenced_room_ids", []),
            "approved_requested_action": llm_candidate.get("requested_action"),
            "hard_conflict": not same_hard,
            "reason": "llm_rescue_for_unclear_regex",
        }

    llm_room_ids = list(llm_candidate.get("referenced_room_ids") or [])
    if (
        llm_intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"}
        and regex_intent in {"REFINE_SEARCH", "REQUEST_FAQ", "GENERAL_HELP"}
        and llm_room_ids
        and llm_conf >= 0.9
        and same_hard
    ):
        return {
            "approved_intent": llm_intent,
            "use_llm_intent": True,
            "use_llm_hard_slots": False,
            "allow_llm_soft_slots": True,
            "approved_room_ids": llm_room_ids,
            "approved_requested_action": llm_candidate.get("requested_action"),
            "hard_conflict": False,
            "reason": "llm_room_detail_over_regex_mismatch",
        }

    return {
        "approved_intent": regex_intent,
        "use_llm_intent": False,
        "use_llm_hard_slots": False,
        "allow_llm_soft_slots": not _has_router_conflict(regex_candidate, llm_candidate) or same_hard,
        "approved_room_ids": regex_candidate.get("referenced_room_ids", []),
        "approved_requested_action": regex_candidate.get("requested_action"),
        "hard_conflict": not same_hard,
        "reason": "prefer_regex_hard_slots",
    }


def _should_call_router_verifier(regex_candidate: dict[str, Any], llm_candidate: dict[str, Any]) -> bool:
    if not _has_router_conflict(regex_candidate, llm_candidate):
        return False
    regex_intent = regex_candidate.get("intent", "GENERAL_HELP")
    llm_intent = llm_candidate.get("intent", "GENERAL_HELP")
    regex_conf = float(regex_candidate.get("confidence") or 0.0)
    llm_conf = float(llm_candidate.get("confidence") or 0.0)

    if llm_conf < max(0.85, regex_conf):
        return False
    if regex_intent == "GENERAL_HELP" and llm_intent != "GENERAL_HELP":
        return False
    if regex_candidate.get("requested_action") != llm_candidate.get("requested_action"):
        return True
    if not _same_operation_set(
        list(regex_candidate.get("hard_operations") or []),
        list(llm_candidate.get("hard_operations") or []),
    ):
        return True
    if list(regex_candidate.get("referenced_room_ids") or []) != list(llm_candidate.get("referenced_room_ids") or []):
        return True
    return regex_intent != llm_intent and llm_conf >= 0.95


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
    if _is_off_topic_request(normalized):
        return "GENERAL_HELP", 1.0
    if action:
        return "REQUEST_ACTION", 1.0
    if _is_result_set_compare_request(normalized):
        return "COMPARE_ROOMS", 1.0
    for priority_intent in ("FIND_SIMILAR", "SUMMARIZE_ROOM", "CALCULATE_COST"):
        if any(kw in normalized for kw in _INTENT_KEYWORDS[priority_intent]):
            return priority_intent, 1.0
    if _selected_room_id_from_ordinal(normalized, current_state):
        return "ASK_ABOUT_ROOM", 1.0
    if _selected_room_id_from_deictic(normalized, current_state):
        return "ASK_ABOUT_ROOM", 1.0
    if _is_room_detail_question(normalized, ids, current_state):
        return "ASK_ABOUT_ROOM", 1.0
    if ids and _has_keyword(normalized, DETAIL_FIELD_KEYWORDS):
        return "ASK_ABOUT_ROOM", 1.0

    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            # ASK_ABOUT_ROOM cũng cần có context phòng
            if intent == "ASK_ABOUT_ROOM" and not (ids or current_state):
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

from agents.prompts import (
    INTENT_CLASSIFIER_PROMPT as _LLM_SYSTEM_PROMPT,
    INTENT_ROUTER_VERIFIER_PROMPT as _LLM_ROUTER_VERIFIER_PROMPT,
)


HARD_OPERATION_PATHS: set[str] = {
    "budget.min",
    "budget.min_operator",
    "budget.max",
    "budget.max_operator",
    "location.districts",
    "location.wards",
}

SOFT_OPERATION_PATHS: set[str] = {
    "location.near_landmarks",
    "location.province",
    "location.max_distance_km",
    "categories",
    "occupants",
    "vehicles",
    "pets_required",
    "amenities_required",
    "amenities_preferred",
    "excluded_features",
    "move_in_date",
}


async def _llm_classify_intent(question: str, current_state: dict[str, Any] | None) -> dict[str, Any]:
    """Gọi LLM để phân tích intent và extract constraints."""
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
            max_tokens=300,
            temperature=0.0,
        )
        
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = __import__('json').loads(raw[start:end])
            return data
    except Exception:
        pass
    return {}


def _is_valid_verifier_payload(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("approved_intent") not in INTENTS:
        return False
    required = {"approved_intent", "use_llm_intent", "hard_conflict", "reason"}
    return all(key in data for key in required)


async def _llm_verify_routing_decision(
    question: str,
    current_state: dict[str, Any] | None,
    regex_candidate: dict[str, Any],
    llm_candidate: dict[str, Any],
) -> dict[str, Any]:
    try:
        from agents.llm_client import groq_complete, GROQ_MODEL_FAST

        payload = {
            "question": question,
            "current_state": {
                "last_intent": (current_state or {}).get("last_intent"),
                "current_room_id": (current_state or {}).get("current_room_id"),
                "constraints": (current_state or {}).get("constraints", {}),
            },
            "regex_candidate": regex_candidate,
            "llm_candidate": llm_candidate,
        }
        raw = await groq_complete(
            prompt=json.dumps(payload, ensure_ascii=False),
            system_prompt=_LLM_ROUTER_VERIFIER_PROMPT,
            model=GROQ_MODEL_FAST,
            max_tokens=350,
            temperature=0.0,
        )
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            if _is_valid_verifier_payload(data):
                return data
    except Exception:
        pass
    return {}


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

    if _is_off_topic_request(normalized):
        return {
            "intent": "GENERAL_HELP",
            "operations": [],
            "current_room_id": None,
            "referenced_room_ids": [],
            "requested_action": None,
        }

    _extract_budget(text, normalized, operations)
    _extract_location(normalized, operations)
    _extract_move_in_date(normalized, operations)
    _extract_people_and_pets(normalized, operations)
    _extract_amenities(text, normalized, operations, current_state)
    _extract_categories(text, normalized, operations)
    _apply_relative_budget_refinement(text, normalized, current_state, operations)
    _maybe_replace_budget_filters(normalized, current_state, operations)
    _maybe_replace_location_filters(normalized, current_state, operations)
    _maybe_clear_stale_search_scope(normalized, operations)

    referenced_room_ids = [_normalize_room_reference(item) for item in _extract_room_ids(text)]
    ordinal_out_of_range = False
    compare_unresolved = False
    ordinal_requested: int | None = None
    ordinal_available_count = len((current_state or {}).get("last_result_ids") or [])

    ordinal_index = _ordinal_request_index(normalized, current_state)
    is_result_set_compare = _is_result_set_compare_request(normalized)
    if ordinal_index is not None and not re.search(r"\bso\s*sanh\b", normalized):
        ids = (current_state or {}).get("last_result_ids") or []
        if ordinal_index >= len(ids):
            ordinal_out_of_range = True
            ordinal_requested = ordinal_index + 1
            referenced_room_ids = []

    selected_room_id = None
    compare_room_ids: list[str] = []
    if not is_result_set_compare:
        selected_room_id = None if ordinal_out_of_range else _selected_room_id_from_ordinal(normalized, current_state)
        if not selected_room_id:
            selected_room_id = _selected_room_id_from_deictic(normalized, current_state)
        compare_room_ids, compare_unresolved = _selected_room_ids_from_ordinals(normalized, current_state)
    elif current_state:
        pool = current_state.get("last_result_ids") or current_state.get("selected_room_ids") or []
        if pool:
            referenced_room_ids = pool[:_requested_compare_count(normalized, len(pool))]
    if len(compare_room_ids) >= 2:
        referenced_room_ids = compare_room_ids
    elif selected_room_id and selected_room_id not in referenced_room_ids:
        referenced_room_ids = [selected_room_id]
    action = _requested_action(normalized)
    intent, _ = _regex_classify(normalized, action, referenced_room_ids, current_state)
    if len(compare_room_ids) >= 2:
        intent = "COMPARE_ROOMS"
    elif compare_unresolved and re.search(r"\bso\s*sanh\b", normalized):
        intent = "COMPARE_ROOMS"
    try:
        from room_assistant.staff_knowledge import is_policy_question
        if (
            is_policy_question(text, current_state)
            and not action
            and not referenced_room_ids
            and not _has_search_constraint_operations(operations)
            and not _is_price_complaint(normalized)
            and not _is_room_detail_question(normalized, referenced_room_ids, current_state)
        ):
            intent = "REQUEST_FAQ"
    except Exception:
        pass
    if _is_action_capability_question(normalized) and any(_norm(keyword) in normalized for keywords in ACTION_KEYWORDS.values() for keyword in keywords):
        intent = "REQUEST_FAQ"
    if not action and (_has_keyword(normalized, COST_FIELD_KEYWORDS) or _is_action_capability_question(normalized)):
        if "dat coc" in normalized or "đặt cọc" in text.lower():
            intent = "REQUEST_FAQ"
    # "sinh viên"/"hợp lý"… là keyword FAQ, nhưng câu có ràng buộc tìm kiếm cụ thể
    # (quận + ngân sách) là một search; than phiền giá là GENERAL_HELP (script đồng cảm).
    if intent == "REQUEST_FAQ" and not action:
        if _has_search_constraint_operations(operations):
            intent = (
                "REFINE_SEARCH"
                if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"}
                else "SEARCH_ROOM"
            )
        elif _is_price_complaint(normalized):
            intent = "GENERAL_HELP"
    if operations and intent == "GENERAL_HELP":
        intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"
    if (
        operations and intent == "ASK_ABOUT_ROOM"
        and not referenced_room_ids
        and not _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS)
        and (
            _has_keyword(normalized, _INTENT_KEYWORDS["SEARCH_ROOM"])
            or not _is_room_detail_question(normalized, referenced_room_ids, current_state)
        )
    ):
        intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"

    if intent not in INTENTS:
        intent = "GENERAL_HELP"

    operations = _strip_detail_question_search_ops(
        operations,
        intent=intent,
        normalized=normalized,
        referenced_room_ids=referenced_room_ids,
        current_state=current_state,
    )

    current_room_id = _resolve_parsed_current_room_id(
        intent=intent,
        referenced_room_ids=referenced_room_ids,
        current_state=current_state,
        ordinal_out_of_range=ordinal_out_of_range,
    )
    if (
        current_room_id
        and not referenced_room_ids
        and intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST", "FIND_SIMILAR"}
    ):
        referenced_room_ids = [current_room_id]

    return {
        "intent": intent,
        "operations": operations,
        "current_room_id": current_room_id,
        "referenced_room_ids": referenced_room_ids,
        "requested_action": action,
        "exact_reference": _build_exact_reference(referenced_room_ids),
        "ordinal_out_of_range": ordinal_out_of_range,
        "ordinal_requested": ordinal_requested,
        "ordinal_available_count": ordinal_available_count,
        "compare_unresolved": compare_unresolved,
    }


async def parse_intent_async(
    question: str,
    current_state: dict[str, Any] | None = None,
    status_callback: Any = None,
) -> ParsedRequest:
    """
    Phân tích một lượt người dùng bằng regex trước, chỉ dùng LLM để cứu
    các trường hợp regex không đủ chắc hoặc thiếu tín hiệu.
    """
    text = question or ""
    normalized = _norm(text)
    if _is_off_topic_request(normalized):
        return {
            "intent": "GENERAL_HELP",
            "operations": [],
            "current_room_id": None,
            "referenced_room_ids": [],
            "requested_action": None,
        }
    regex_parsed = parse_intent_and_constraint_patch(text, current_state)
    regex_intent, regex_confidence = _regex_classify(
        normalized,
        regex_parsed.get("requested_action"),
        regex_parsed.get("referenced_room_ids", []),
        current_state,
    )

    if status_callback:
        await status_callback("[status:ĐỊNH HƯỚNG|intent_router] Đang nhận diện nhu cầu và điều kiện chính...\n")
        await status_callback("[status:LÀM RÕ|intent_router] Đang phân tích ngữ cảnh hội thoại...\n")

    raw_llm = await _llm_classify_intent(text, current_state)
    data, llm_confidence = _coerce_llm_payload(raw_llm)
    llm_intent = data.get("intent", "GENERAL_HELP")
    if llm_intent not in INTENTS:
        llm_intent = "GENERAL_HELP"
    llm_operations = _sanitize_llm_operations(data.get("operations", []))
    llm_room_ids = [
        _normalize_room_reference(item)
        for item in (data.get("referenced_room_ids") or [])
        if _normalize_room_reference(item)
    ]
    if not llm_room_ids:
        llm_room_ids = list(regex_parsed.get("referenced_room_ids", []))
    llm_requested_action = data.get("requested_action") or regex_parsed.get("requested_action")

    regex_candidate = _router_candidate_summary(regex_parsed, regex_confidence)
    llm_candidate = _build_llm_candidate(
        llm_intent=llm_intent,
        llm_confidence=llm_confidence,
        llm_operations=llm_operations,
        referenced_room_ids=llm_room_ids,
        requested_action=llm_requested_action,
    )

    verifier: dict[str, Any] = {}
    if _should_call_router_verifier(regex_candidate, llm_candidate):
        if status_callback:
            await status_callback("[status:ĐỐI CHIẾU|intent_router] Đang kiểm tra logic điều hướng...\n")
        verifier = await _llm_verify_routing_decision(
            text,
            current_state,
            regex_candidate,
            llm_candidate,
        )
    if not verifier:
        verifier = _default_verifier_decision(regex_candidate, llm_candidate)

    merged = _merge_router_decision(
        regex_parsed=regex_parsed,
        regex_candidate=regex_candidate,
        llm_candidate=llm_candidate,
        verifier=verifier,
        current_state=current_state,
    )
    merged["operations"] = _strip_detail_question_search_ops(
        merged.get("operations", []),
        intent=merged.get("intent", "GENERAL_HELP"),
        normalized=normalized,
        referenced_room_ids=merged.get("referenced_room_ids", []),
        current_state=current_state,
    )
    return merged

