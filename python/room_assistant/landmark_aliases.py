"""Chuẩn hóa địa danh / mốc tìm kiếm theo cách ghi trong MongoDB.

Dữ liệu phòng thường ghi dạng viết tắt (TDTU, HUIT, XVNT…) trong ``embedding_text``
và ``tien_ich_xq``, trong khi người dùng hay hỏi tên đầy đủ ("trường đại học TDT",
"đại học Công Thương", "TTTM Aeon Tân Phú"…). Module này map câu hỏi → token
tìm kiếm thực tế có trong DB.
"""

from __future__ import annotations

import re
import unicodedata

# canonical → các chuỗi con xuất hiện trong embedding_text / tien_ich_xq (đã mine MongoDB)
LANDMARK_SEARCH_TOKENS: dict[str, tuple[str, ...]] = {
    # --- Trường đại học / cao đẳng ---
    "tdtu": (
        "tdtu", "tdt", "ton duc thang", "dai hoc ton duc thang", "dh ton duc thang",
        "truong dai hoc ton duc thang", "dhtd",
    ),
    "hutech": ("hutech", "dh cong nghe", "dai hoc cong nghe"),
    "ueh": ("ueh", "kinh te tp", "dai hoc kinh te", "dh kinh te"),
    "uel": ("uel", "dai hoc luat", "dh luat"),
    "huit": (
        "huit", "cong thuong", "dai hoc cong thuong", "dh cong thuong", "dhct",
    ),
    "fpt": ("fpt", "dai hoc fpt", "dh fpt"),
    "van_hien": ("van hien", "dai hoc van hien", "dh van hien", "dvh"),
    "rmit": ("rmit",),
    "nttu": ("nttu", "nguyen tat thanh", "dai hoc nguyen tat thanh", "dh ntt"),
    "ufm": ("ufm", "tai chinh marketing", "dai hoc tai chinh marketing"),
    "huflit": ("huflit", "dai hoc ngoai ngu", "dh ngoai ngu"),
    "hoa_sen": ("hoa sen", "dai hoc hoa sen", "dh hoa sen"),
    "hong_bang": ("hong bang", "dai hoc hong bang", "dh hong bang"),
    "gtvt": ("gtvt", "giao thong van tai", "dai hoc giao thong"),
    "hcmus": ("hcmus", "khoa hoc tu nhien", "dh khoa hoc tu nhien"),
    "hcmut": ("hcmut", "bach khoa", "dai hoc bach khoa", "dh bach khoa"),
    "dhqg": ("dhqg", "dai hoc quoc gia", "vnu", "dh quoc gia"),
    "pasteur": ("pasteur", "y duoc pasteur", "cao dang y duoc pasteur"),
    # --- TTTM / siêu thị ---
    "aeon_tan_phu": (
        "aeon tan phu", "aeon mall tan phu", "aeon mall tp", "aeon tp",
    ),
    "aeon_binh_tan": ("aeon binh tan", "aeon mall binh tan"),
    "lotte": ("lotte", "lotte mart", "lottemart"),
    "vincom": ("vincom",),
    "coopmart": ("coopmart", "co op mart", "co-op mart"),
    "landmark_81": ("landmark 81", "landmark81", "landmark"),
    "van_hanh_mall": ("van hanh mall", "van hanh", "van hanh plaza"),
    "gigamall": ("gigamall", "giga mall"),
    "sc_vivo": ("sc vivo", "vivo city", "vivocity"),
    "estella": ("estella", "estella place"),
    "thiso": ("thiso", "thiso mall"),
    "mega_market": ("mega market", "go! di an", "sieu thi go"),
    "big_c": ("big c", "bigc", "go!"),
    # --- Đường lớn (tien_ich_xq / embedding_text) ---
    "nguyen_huu_tho": ("nguyen huu tho",),
    "le_van_luong": ("le van luong",),
    "phan_van_hon": ("phan van hon",),
    "tan_ky_tan_quy": ("tan ky tan quy", "tk tq"),
    "truong_chinh": ("truong chinh",),
    "huynh_tan_phap": ("huynh tan phat",),
    "nguyen_thi_thap": ("nguyen thi thap",),
    "cach_mang_thang_8": ("cach mang thang 8", "cach mang 8", "cmtt 8"),
    "xvnt": ("xvnt", "xo viet nghe tinh", "xo viet"),
    "duong_3_2": ("3/2", "duong 3/2", "d3/2"),
    "nguyen_van_cu": ("nguyen van cu",),
    "le_van_viet": ("le van viet",),
    "nguyen_anh_thu": ("nguyen anh thu", "nguyen ảnh thủ"),
    "pham_van_dong": ("pham van dong",),
    "au_co": ("au co",),
    "song_hanh": ("song hanh",),
    "vo_van_kiet": ("vo van kiet",),
    "dinh_bo_linh": ("dinh bo linh",),
    "nguyen_xi": ("nguyen xi",),
    "hoang_quoc_viet": ("hoang quoc viet",),
    "nguyen_huu_can": ("nguyen huu can",),
    # --- Bệnh viện / chợ (mốc hay hỏi) ---
    "cho_ray": ("cho ray", "benh vien cho ray", "bv cho ray"),
    "benh_vien_175": ("benh vien 175", "bv 175", "quan y 175"),
    "cho_an_duong": ("cho an duong", "an duong"),
    "cho_binh_tay": ("cho binh tay", "binh tay"),
    "cho_hoa_cau": ("cho hoa cau", "hoa cau"),
    "san_bay": ("san bay", "tan son nhat", "tsn"),
}

# Alias người dùng / regex parser → canonical (bổ sung ngoài search tokens)
EXTRA_USER_ALIASES: dict[str, str] = {
  # TDTU
    "truong dai hoc tdt": "tdtu",
    "truong dai hoc tdtu": "tdtu",
    "dai hoc tdt": "tdtu",
    "dai hoc tdtu": "tdtu",
    "truong tdt": "tdtu",
    "truong tdtu": "tdtu",
    "tdt": "tdtu",
    # HUIT
    "dai hoc cong thuong": "huit",
    "truong dai hoc cong thuong": "huit",
    "truong cong thuong": "huit",
    "dh cong thuong": "huit",
    # Hutech
    "truong hutech": "hutech",
    "dai hoc hutech": "hutech",
    "truong dai hoc hutech": "hutech",
    # FPT
    "dai hoc fpt": "fpt",
    "truong fpt": "fpt",
    # Văn Hiến
    "dai hoc van hien": "van_hien",
    "truong van hien": "van_hien",
    # UEH / UEL
    "dai hoc kinh te": "ueh",
    "truong kinh te": "ueh",
    "dai hoc luat": "uel",
    # TTTM
    "tttm aeon tan phu": "aeon_tan_phu",
    "tttm aeon": "aeon_tan_phu",
    "aeon mall": "aeon_tan_phu",
    "trung tam thuong mai aeon": "aeon_tan_phu",
    "tttm vincom": "vincom",
    "trung tam thuong mai vincom": "vincom",
    "tttm landmark": "landmark_81",
    "tttm van hanh": "van_hanh_mall",
    # Đường — parser hay thêm prefix "duong"
    "duong nguyen huu tho": "nguyen_huu_tho",
    "duong le van luong": "le_van_luong",
    "duong phan van hon": "phan_van_hon",
    "duong tan ky tan quy": "tan_ky_tan_quy",
    "duong truong chinh": "truong_chinh",
    "duong huynh tan phat": "huynh_tan_phap",
    "duong nguyen thi thap": "nguyen_thi_thap",
    "duong cach mang thang 8": "cach_mang_thang_8",
    "duong xvnt": "xvnt",
    "duong xo viet nghe tinh": "xvnt",
    "duong 3/2": "duong_3_2",
    "duong hoang quoc viet": "hoang_quoc_viet",
    # Chợ / BV
    "cho ray": "cho_ray",
    "benh vien cho ray": "cho_ray",
    "bv cho ray": "cho_ray",
    "benh vien 175": "benh_vien_175",
}

_STRIP_PREFIXES: tuple[str, ...] = (
    "truong dai hoc ",
    "dai hoc ",
    "cao dang ",
    "cd ",
    "dh ",
    "dhcn ",
    "tttm ",
    "trung tam thuong mai ",
    "duong ",
    "d ",
    "mat tien ",
    "hem ",
    "gan ",
    "gần ",
    "cach ",
    "cách ",
    "doi dien ",
    "đối diện ",
    "benh vien ",
    "bệnh viện ",
    "bv ",
    "cho ",
    "chợ ",
    "truong ",
    "trường ",
)


def _norm_landmark(value: str) -> str:
    text = str(value or "").lower().strip()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    text = text.replace("đ", "d").replace("Đ", "d")
    return re.sub(r"\s+", " ", text).strip()


def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, tokens in LANDMARK_SEARCH_TOKENS.items():
        index[_norm_landmark(canonical)] = canonical
        for token in tokens:
            index[_norm_landmark(token)] = canonical
    for alias, canonical in EXTRA_USER_ALIASES.items():
        index[_norm_landmark(alias)] = canonical
    return index


_ALIAS_TO_CANONICAL = _build_alias_index()

# Sắp alias dài trước để khớp cụm đầy đủ trước token ngắn.
_SORTED_ALIASES: tuple[tuple[str, str], ...] = tuple(
    sorted(_ALIAS_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True)
)


def _strip_known_prefixes(text: str) -> str:
    current = text
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if current.startswith(prefix):
                current = current[len(prefix):].strip()
                changed = True
    return current


def _resolve_canonical(norm: str) -> str | None:
    if not norm:
        return None
    if norm in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[norm]
    for alias, canonical in _SORTED_ALIASES:
        if alias in norm or norm in alias:
            return canonical
    stripped = _strip_known_prefixes(norm)
    if stripped != norm:
        if stripped in _ALIAS_TO_CANONICAL:
            return _ALIAS_TO_CANONICAL[stripped]
        for alias, canonical in _SORTED_ALIASES:
            if alias in stripped or stripped in alias:
                return canonical
    return None


def normalize_landmark(value: str) -> str | None:
    """Chuẩn hóa địa danh người dùng → token tìm kiếm chính trong DB."""
    cleaned = re.sub(r"\s+", " ", str(value or "").strip())
    if len(cleaned) < 2:
        return None
    norm = _norm_landmark(cleaned)
    canonical = _resolve_canonical(norm)
    if canonical:
        tokens = LANDMARK_SEARCH_TOKENS.get(canonical)
        if tokens:
            return tokens[0]
        return canonical.replace("_", " ")
    return cleaned


def expand_landmark_tokens(value: str) -> list[str]:
    """Trả về danh sách token để regex-search trong DB (OR)."""
    raw = str(value or "").strip()
    if not raw:
        return []
    canonical = _resolve_canonical(_norm_landmark(raw))
    if not canonical:
        canonical = _resolve_canonical(_norm_landmark(_strip_known_prefixes(_norm_landmark(raw))))
    if canonical and canonical in LANDMARK_SEARCH_TOKENS:
        tokens = list(LANDMARK_SEARCH_TOKENS[canonical])
        # canonical slug cũng là token fallback
        if canonical not in tokens:
            tokens.append(canonical.replace("_", " "))
        return list(dict.fromkeys(tokens))
    return [raw]


def landmark_matches_text(landmark: str, text: str) -> bool:
    """Kiểm tra landmark (đã hoặc chưa chuẩn hóa) có trong đoạn text không."""
    haystack = _norm_landmark(text)
    if not haystack:
        return False
    for token in expand_landmark_tokens(landmark):
        needle = _norm_landmark(token)
        if needle and needle in haystack:
            return True
    return False


def primary_display_token(canonical: str) -> str:
    """Token hiển thị ngắn gọn (vd: ``TDTU``)."""
    tokens = LANDMARK_SEARCH_TOKENS.get(canonical)
    if tokens:
        return tokens[0].upper()
    return canonical.replace("_", " ").upper()
