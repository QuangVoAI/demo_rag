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
        "con phong", "phong trong", "sap het", "hot",
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
    "ban cong", "thu cung", "nuoi meo", "nuoi cho", "de xe", "cho de xe",
    "gui xe", "xe dien", "may giat", "wifi", "tu lanh", "nuoc nong",
    "gac", "nem", "giuong", "tu quan ao", "ke bep", "thang may",
    "may lanh", "dien", "nuoc", "quan ly", "xac thuc", "anhome ho tro",
)

ROOM_REFERENCE_KEYWORDS: tuple[str, ...] = (
    "phong nay", "phong do", "phong tren", "dang xem", "can ho nay",
    "cho nay", "nha nay", "muc nay", "tin nay",
)

COST_FIELD_KEYWORDS: tuple[str, ...] = (
    "chi phi", "gia dien", "gia nuoc", "phi dien", "phi nuoc",
    "phi quan ly", "phi gui xe", "tien dien", "tien nuoc",
    "tien coc", "bao nhieu tien",
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
    compact_million = re.search(r"\b(\d+)\s*(?:tr|trieu)\s*(\d+)\b(?!\s*thang)", normalized)
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
        value = re.split(r"\b(?:gan|duoi|tren|co|va|gia|,|\.)\b", value)[0].strip()
        if value and value not in districts:
            districts.append(f"quan {value}")
    for district in districts:
        _append_unique(ops, "append", "location.districts", district)

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
        value = re.split(r"\b(?:duoi|tren|co|va|,|\.)\b", value)[0].strip()
        if value and value not in landmarks:
            landmarks.append(value)
    for landmark in landmarks:
        _append_unique(ops, "append", "location.near_landmarks", landmark)

    street_patterns = (
        r"\b(?:duong|đường|mat tien|mặt tiền|hem|hẻm)\s+([a-z0-9 àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ/-]{2,45})",
    )
    for pattern in street_patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            value = match.group(0).strip()
            value = re.split(r"\b(?:duoi|tren|co|va|gia|,|\.)\b", value)[0].strip()
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


def _extract_categories(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất loại phòng từ câu hỏi."""
    for alias, canonical in ROOM_TYPE_ALIASES.items():
        if _contains_phrase(normalized, alias):
            _append_unique(ops, "append", "categories", canonical)


def _extract_amenities(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    """Trích xuất tiện ích bắt buộc và tùy chọn từ câu hỏi."""
    # Xác định người dùng đang bỏ / xóa tiện ích hay thêm vào
    remove_mode = bool(re.search(r"\b(bo|bỏ|khong can|không cần|loai|loại|xoa|xóa)\b", normalized))

    for alias, canonical in AMENITY_ALIASES.items():
        if _contains_phrase(normalized, alias):
            _append_unique(
                ops,
                "remove" if remove_mode else "append",
                "amenities_required",
                canonical,
            )

    if re.search(r"khong\s*may\s*lanh", normalized) or "không máy lạnh" in text.lower():
        _append_unique(ops, "append", "excluded_features", "air_conditioner")
    if re.search(r"khong\s*gac", normalized) or "không gác" in text.lower():
        _append_unique(ops, "append", "excluded_features", "mezzanine")

    for alias, canonical in SOFT_PREFERENCE_ALIASES.items():
        if _contains_phrase(normalized, alias):
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


def _requested_action(normalized: str) -> str | None:
    """Kiểm tra xem người dùng có yêu cầu thao tác nghiệp vụ không."""
    for action, keywords in ACTION_KEYWORDS.items():
        if any(_norm(keyword) in normalized for keyword in keywords):
            return action
    return None


def _has_current_listing(current_state: dict[str, Any] | None) -> bool:
    if not current_state:
        return False
    return bool(current_state.get("current_listing_id") or current_state.get("selected_listing_ids") or current_state.get("last_result_ids"))


def _has_keyword(normalized: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in normalized for keyword in keywords)


def _is_room_detail_question(normalized: str, ids: list[str], current_state: dict[str, Any] | None) -> bool:
    if _has_keyword(normalized, COST_FIELD_KEYWORDS):
        return False
    if not _has_keyword(normalized, DETAIL_FIELD_KEYWORDS):
        return False
    if ids or _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
        return True
    return _has_current_listing(current_state) and bool(re.search(r"\b(co|có|khong|không|bao nhieu|bao nhiêu|the nao|thế nào|la gi|là gì)\b", normalized))


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
    if _is_room_detail_question(normalized, ids, current_state):
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

    # Kế thừa intent từ lượt trước nếu đang trong luồng tìm kiếm
    if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"}:
        return "REFINE_SEARCH", 0.75

    return "GENERAL_HELP", 0.6


# ---------------------------------------------------------------------------
# Tầng 2: LLM fallback khi regex không chắc chắn
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """Bạn là bộ phân loại intent cho chatbot tìm phòng trọ tại Việt Nam (nhatrovn.vn).
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
    _extract_move_in_date(normalized, operations)
    _extract_people_and_pets(normalized, operations)
    _extract_amenities(text, normalized, operations)
    _extract_categories(text, normalized, operations)

    referenced_listing_ids = _extract_listing_ids(text)
    action = _requested_action(normalized)
    intent, _ = _regex_classify(normalized, action, referenced_listing_ids, current_state)
    if operations and intent == "GENERAL_HELP":
        intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"
    if operations and intent == "ASK_ABOUT_ROOM" and not referenced_listing_ids and not _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
        intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"

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
    _extract_move_in_date(normalized, operations)
    _extract_people_and_pets(normalized, operations)
    _extract_amenities(text, normalized, operations)
    _extract_categories(text, normalized, operations)

    referenced_listing_ids = _extract_listing_ids(text)
    action = _requested_action(normalized)

    regex_intent, regex_conf = _regex_classify(normalized, action, referenced_listing_ids, current_state)
    if operations and regex_intent == "GENERAL_HELP":
        regex_intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"
        regex_conf = max(regex_conf, 0.85)
    if operations and regex_intent == "ASK_ABOUT_ROOM" and not referenced_listing_ids and not _has_keyword(normalized, ROOM_REFERENCE_KEYWORDS):
        regex_intent = "REFINE_SEARCH" if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} else "SEARCH_ROOM"
        regex_conf = max(regex_conf, 0.85)

    # Nếu câu dài hơn 5 từ, ép giảm confidence của regex để bắt LLM phải check lại
    # Tránh trường hợp "Hôm trước mình thuê phòng, giờ muốn lấy cọc" bị regex "thuê phòng" bắt nhầm
    word_count = len(question.strip().split())
    if word_count > 5 and not action:
        regex_conf = min(regex_conf, 0.8)

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
