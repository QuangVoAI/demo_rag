"""Chuẩn hóa địa danh / mốc tìm kiếm theo cách ghi trong MongoDB.

Dữ liệu phòng thường ghi dạng viết tắt (TDTU, CTU, XVNT…) trong ``tien_ich_xq``
và đôi khi ``embedding_text``, trong khi người dùng hay hỏi tên đầy đủ.
Module map câu hỏi → token tìm kiếm an toàn (word-boundary, có scope tỉnh khi cần).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# canonical → token xuất hiện trong tien_ich_xq / embedding_text (đã mine MongoDB)
LANDMARK_SEARCH_TOKENS: dict[str, tuple[str, ...]] = {
    # --- Trường đại học / cao đẳng (TP.HCM) ---
    "tdtu": (
        "tdtu", "tdt", "ton duc thang", "dai hoc ton duc thang", "dh ton duc thang",
        "truong dai hoc ton duc thang", "dhtd",
    ),
    "hutech": ("hutech", "dh cong nghe tp", "dai hoc cong nghe tp"),
    "ueh": ("ueh", "kinh te tp", "dai hoc kinh te tp", "dh kinh te tp"),
    "uel": ("uel", "kinh te luat", "dai hoc kinh te luat"),
    "huit": (
        "huit", "dai hoc cong thuong tp", "dh cong thuong tp",
        "truong dai hoc cong thuong tp",
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
    "hcmut": (
        "hcmut", "bach khoa tp", "dai hoc bach khoa tp", "dh bach khoa tp",
        "truong dai hoc bach khoa tp",
    ),
    "dhqg": ("dhqg", "dai hoc quoc gia tp", "vnu hcm", "dh quoc gia tp"),
    "pasteur": ("pasteur", "y duoc pasteur", "cao dang y duoc pasteur"),
    # --- Trường ngoài TP.HCM ---
    "ctu": (
        "ctu", "dhct", "dai hoc can tho", "dh can tho", "truong dai hoc can tho",
    ),
    "hust": (
        "hust", "bach khoa ha noi", "dai hoc bach khoa ha noi", "dhbk ha noi",
        "dh bach khoa ha noi",
    ),
    "dut": (
        "dut", "bach khoa da nang", "dai hoc bach khoa da nang", "dhbk da nang",
        "dh bach khoa da nang",
    ),
    "due": ("due", "dai hoc kinh te da nang", "dh kinh te da nang"),
    "hlu_hanoi": ("dai hoc luat ha noi", "dh luat ha noi", "hoc vien tu phap"),
    # --- TTTM / siêu thị ---
    "aeon_tan_phu": (
        "aeon tan phu", "aeon mall tan phu", "aeon mall tp", "aeon tp",
    ),
    "aeon_binh_tan": ("aeon binh tan", "aeon mall binh tan"),
    "lotte": ("lotte", "lotte mart", "lottemart"),
    "vincom": ("vincom",),
    "coopmart": ("coopmart", "co op mart", "co-op mart"),
    "landmark_81": ("landmark 81", "landmark81"),
    "van_hanh_mall": ("van hanh mall", "van hanh plaza"),
    "gigamall": ("gigamall", "giga mall"),
    "sc_vivo": ("sc vivo", "vivo city", "vivocity"),
    "estella": ("estella", "estella place"),
    "thiso": ("thiso", "thiso mall"),
    "mega_market": ("mega market", "sieu thi mega"),
    "go_di_an": ("go! di an", "go di an", "sieu thi go di an"),
    "big_c": ("big c", "bigc"),
    # --- Sân bay ---
    "tan_son_nhat": ("tan son nhat", "tsn", "san bay tan son nhat"),
    "noi_bai": ("noi bai", "san bay noi bai"),
    "can_tho_airport": ("san bay can tho",),
    "phu_quoc_airport": ("san bay phu quoc",),
    # --- Đường lớn ---
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
    "nguyen_anh_thu": ("nguyen anh thu",),
    "pham_van_dong": ("pham van dong",),
    "au_co": ("au co",),
    "song_hanh": ("song hanh",),
    "vo_van_kiet": ("vo van kiet",),
    "dinh_bo_linh": ("dinh bo linh",),
    "nguyen_xi": ("nguyen xi",),
    "hoang_quoc_viet": ("hoang quoc viet",),
    "nguyen_huu_canh": ("nguyen huu canh", "nguyen huu can"),
    # --- Bệnh viện / chợ ---
    "cho_ray": ("cho ray", "benh vien cho ray", "bv cho ray"),
    "benh_vien_175": ("benh vien 175", "bv 175", "quan y 175"),
    "cho_an_duong": ("cho an duong", "an duong"),
    "cho_binh_tay": ("cho binh tay",),
    "cho_hoa_cau": ("cho hoa cau",),
}

# province slug → canonical chỉ hợp lệ trong tỉnh đó (rỗng = không ràng buộc)
LANDMARK_PROVINCES: dict[str, tuple[str, ...]] = {
    "tdtu": ("ho chi minh",),
    "hutech": ("ho chi minh",),
    "ueh": ("ho chi minh",),
    "uel": ("ho chi minh",),
    "huit": ("ho chi minh",),
    "hcmut": ("ho chi minh",),
    "hcmus": ("ho chi minh",),
    "dhqg": ("ho chi minh",),
    "aeon_tan_phu": ("ho chi minh",),
    "aeon_binh_tan": ("ho chi minh",),
    "tan_son_nhat": ("ho chi minh",),
    "landmark_81": ("ho chi minh",),
    "van_hanh_mall": ("ho chi minh",),
    "cho_ray": ("ho chi minh",),
    "ctu": ("can tho",),
    "can_tho_airport": ("can tho",),
    "hust": ("ha noi",),
    "noi_bai": ("ha noi",),
    "hlu_hanoi": ("ha noi",),
    "dut": ("da nang",),
    "due": ("da nang",),
    "go_di_an": ("binh duong",),
    "phu_quoc_airport": ("kien giang",),
}

EXTRA_USER_ALIASES: dict[str, str] = {
    "truong dai hoc tdt": "tdtu",
    "truong dai hoc tdtu": "tdtu",
    "dai hoc tdt": "tdtu",
    "dai hoc tdtu": "tdtu",
    "truong tdt": "tdtu",
    "truong tdtu": "tdtu",
    "truong hutech": "hutech",
    "dai hoc hutech": "hutech",
    "dai hoc fpt": "fpt",
    "truong fpt": "fpt",
    "dai hoc van hien": "van_hien",
    "truong van hien": "van_hien",
    "dai hoc cong thuong tp": "huit",
    "truong dai hoc cong thuong tp": "huit",
    "dai hoc cong thuong": "huit",
    "truong dai hoc cong thuong": "huit",
    "dh cong thuong": "huit",
    "dai hoc kinh te tp": "ueh",
    "dai hoc kinh te luat": "uel",
    "dai hoc bach khoa tp": "hcmut",
    "dai hoc bach khoa ha noi": "hust",
    "dai hoc bach khoa da nang": "dut",
    "dai hoc kinh te da nang": "due",
    "dai hoc can tho": "ctu",
    "dai hoc luat ha noi": "hlu_hanoi",
    "tttm aeon tan phu": "aeon_tan_phu",
    "aeon mall tan phu": "aeon_tan_phu",
    "aeon mall binh tan": "aeon_binh_tan",
    "tttm vincom": "vincom",
    "tttm landmark 81": "landmark_81",
    "tttm van hanh": "van_hanh_mall",
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
    "cho ray": "cho_ray",
    "benh vien cho ray": "cho_ray",
    "san bay tan son nhat": "tan_son_nhat",
    "san bay noi bai": "noi_bai",
    "san bay can tho": "can_tho_airport",
    "san bay phu quoc": "phu_quoc_airport",
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
    "san bay ",
    "ga ",
)

# Alias mơ hồ — chỉ resolve khi có qualifier tỉnh trong câu hoặc province_slug
_AMBIGUOUS_ALIASES: frozenset[str] = frozenset({
    "bach khoa", "dai hoc bach khoa", "dh bach khoa",
    "dai hoc kinh te", "dh kinh te",
    "dai hoc luat", "dh luat",
    "aeon mall", "aeon", "landmark", "san bay", "fpt",
})

_KNOWN_SHORT_ABBREVS: frozenset[str] = frozenset({
    "tdt", "ctu", "dut", "ueh", "uel", "fpt", "ufm", "rmit", "nttu", "dvh",
    "hcm", "tsn", "xvnt", "kcn", "vsip", "kcx",
})

_LANDMARK_QUALIFIERS: tuple[str, ...] = (
    "aeon", "vincom", "lotte", "go! di", "go di an", "mega market", "big c", "coopmart",
    "dai hoc", "truong dai hoc", "dh ", "truong ", "benh vien", "bv ", "san bay",
    "ga ", "kcn", "vsip", "tttm", "landmark 81", "landmark81", "cho ",
)


def _norm_landmark(value: str) -> str:
    text = str(value or "").lower().strip()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    text = text.replace("đ", "d").replace("Đ", "d")
    return re.sub(r"\s+", " ", text).strip()


def _norm_province(value: str | None) -> str | None:
    if not value:
        return None
    norm = _norm_landmark(value)
    replacements = {
        "thanh pho ho chi minh": "ho chi minh",
        "tp ho chi minh": "ho chi minh",
        "tp hcm": "ho chi minh",
        "tp.hcm": "ho chi minh",
        "tphcm": "ho chi minh",
        "tinh binh duong": "binh duong",
        "thanh pho can tho": "can tho",
        "thanh pho ha noi": "ha noi",
        "thanh pho da nang": "da nang",
        "tinh kien giang": "kien giang",
        "ba ria vung tau": "ba ria vung tau",
        "tinh ba ria vung tau": "ba ria vung tau",
    }
    return replacements.get(norm, norm)


def _load_district_blocklist() -> frozenset[str]:
    """Tên quận/huyện trần từ city.txt — không được map thành landmark."""
    names: set[str] = set()
    city_file = Path(__file__).resolve().parents[2] / "city.txt"
    if not city_file.exists():
        return frozenset()
    prefix_re = re.compile(
        r"^(?:quan|huyen|thi xa|thanh pho|thi tran)\s+",
        re.IGNORECASE,
    )
    for line in city_file.read_text(encoding="utf-8").splitlines()[1:]:
        if "\t" not in line:
            continue
        districts_part = line.split("\t", 2)[-1]
        for chunk in districts_part.split(";"):
            raw = chunk.strip().rstrip(".")
            if not raw:
                continue
            norm_raw = _norm_landmark(raw)
            bare = prefix_re.sub("", norm_raw).strip()
            if bare:
                names.add(bare)
            names.add(norm_raw)
    return frozenset(names)


_DISTRICT_BLOCKLIST = _load_district_blocklist()


def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, tokens in LANDMARK_SEARCH_TOKENS.items():
        index[_norm_landmark(canonical)] = canonical
        for token in tokens:
            norm = _norm_landmark(token)
            if len(norm) >= 2:
                index[norm] = canonical
    for alias, canonical in EXTRA_USER_ALIASES.items():
        index[_norm_landmark(alias)] = canonical
    return index


_ALIAS_TO_CANONICAL = _build_alias_index()

_SORTED_ALIASES: tuple[tuple[str, str], ...] = tuple(
    sorted(_ALIAS_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True)
)


def _phrase_in_text(phrase: str, text: str) -> bool:
    if len(phrase) < 2:
        return False
    escaped = re.escape(phrase)
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", text))


def _should_drop_unresolved_landmark(norm: str) -> bool:
    if _is_bare_district(norm) and not _has_landmark_qualifier(norm):
        return True
    if norm in _AMBIGUOUS_ALIASES:
        return True
    if len(norm) <= 2 and norm not in _KNOWN_SHORT_ABBREVS:
        return True
    return False


def _has_landmark_qualifier(norm: str) -> bool:
    for qualifier in _LANDMARK_QUALIFIERS:
        q = qualifier.strip()
        if not q:
            continue
        if _phrase_in_text(q, norm):
            return True
    return False


def _is_bare_district(norm: str) -> bool:
    return norm in _DISTRICT_BLOCKLIST


def _province_hint_in_text(norm: str) -> str | None:
    hints = (
        ("ha noi", ("ha noi", "hanoi", "hn")),
        ("ho chi minh", ("ho chi minh", "hcm", "tp hcm", "sai gon", "tp hcm")),
        ("can tho", ("can tho", "ct")),
        ("da nang", ("da nang", "dn")),
        ("binh duong", ("binh duong", "bd")),
        ("kien giang", ("kien giang", "phu quoc", "rach gia")),
    )
    for province, tokens in hints:
        for token in tokens:
            if _phrase_in_text(token, norm):
                return province
    return None


def _province_matches(canonical: str, province: str | None) -> bool:
    allowed = LANDMARK_PROVINCES.get(canonical)
    if not allowed:
        return True
    if not province:
        return len(allowed) == 1
    return province in allowed


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


def _collect_candidates(norm: str) -> list[str]:
    candidates: list[str] = []
    if norm in _ALIAS_TO_CANONICAL:
        candidates.append(_ALIAS_TO_CANONICAL[norm])
    for alias, canonical in _SORTED_ALIASES:
        if alias == norm:
            continue
        if len(alias) < 2:
            continue
        if _phrase_in_text(alias, norm) and canonical not in candidates:
            candidates.append(canonical)
    stripped = _strip_known_prefixes(norm)
    if stripped != norm:
        if stripped in _ALIAS_TO_CANONICAL and _ALIAS_TO_CANONICAL[stripped] not in candidates:
            candidates.append(_ALIAS_TO_CANONICAL[stripped])
        for alias, canonical in _SORTED_ALIASES:
            if len(alias) < 2:
                continue
            if _phrase_in_text(alias, stripped) and canonical not in candidates:
                candidates.append(canonical)
    return candidates


def _pick_canonical(
    candidates: list[str],
    norm: str,
    province_slug: str | None,
) -> str | None:
    if not candidates:
        return None
    province = _norm_province(province_slug) or _province_hint_in_text(norm)

    if norm in _AMBIGUOUS_ALIASES and not province:
        return None

    scoped = [c for c in candidates if _province_matches(c, province)]
    if not scoped:
        return None
    unique = list(dict.fromkeys(scoped))
    if len(unique) == 1:
        return unique[0]
    if province:
        filtered = [c for c in unique if c in LANDMARK_PROVINCES and province in LANDMARK_PROVINCES[c]]
        if len(filtered) == 1:
            return filtered[0]
    return None


def _resolve_canonical(
    norm: str,
    province_slug: str | None = None,
    district_slug: str | None = None,
) -> str | None:
    if not norm or len(norm) < 2:
        return None
    if _is_bare_district(norm) and not _has_landmark_qualifier(norm):
        return None
    if district_slug and _norm_landmark(district_slug) == norm and not _has_landmark_qualifier(norm):
        return None
    return _pick_canonical(_collect_candidates(norm), norm, province_slug)


def normalize_landmark(
    value: str,
    province_slug: str | None = None,
    district_slug: str | None = None,
) -> str | None:
    """Chuẩn hóa địa danh người dùng → token tìm kiếm chính trong DB."""
    cleaned = re.sub(r"\s+", " ", str(value or "").strip())
    if len(cleaned) < 2:
        return None
    norm = _norm_landmark(cleaned)
    canonical = _resolve_canonical(norm, province_slug, district_slug)
    if canonical:
        tokens = LANDMARK_SEARCH_TOKENS.get(canonical)
        if tokens:
            return tokens[0]
        return canonical.replace("_", " ")
    if _should_drop_unresolved_landmark(norm):
        return None
    return cleaned


def expand_landmark_tokens(
    value: str,
    province_slug: str | None = None,
    district_slug: str | None = None,
) -> list[str]:
    """Trả về danh sách token để regex-search trong DB (OR)."""
    raw = str(value or "").strip()
    if not raw:
        return []
    norm = _norm_landmark(raw)
    if len(norm) < 2:
        return []
    canonical = _resolve_canonical(norm, province_slug, district_slug)
    if not canonical:
        stripped = _strip_known_prefixes(norm)
        if stripped != norm:
            canonical = _resolve_canonical(stripped, province_slug, district_slug)
    if canonical and canonical in LANDMARK_SEARCH_TOKENS:
        tokens = list(LANDMARK_SEARCH_TOKENS[canonical])
        if canonical.replace("_", " ") not in tokens:
            tokens.append(canonical.replace("_", " "))
        # Giữ qualifier chi nhánh nếu user gõ đầy đủ (vd Vincom Bà Rịa)
        if norm and len(norm.split()) > 1:
            norm_tokens = {_norm_landmark(item) for item in tokens}
            if norm not in norm_tokens and _has_landmark_qualifier(norm):
                tokens.insert(0, norm)
        return list(dict.fromkeys(tokens))
    if _should_drop_unresolved_landmark(norm):
        return []
    return [norm]


def landmark_matches_text(landmark: str, text: str) -> bool:
    """Kiểm tra landmark (đã hoặc chưa chuẩn hóa) có trong đoạn text không."""
    haystack = _norm_landmark(text)
    if not haystack:
        return False
    for token in expand_landmark_tokens(landmark):
        needle = _norm_landmark(token)
        if len(needle) < 2:
            continue
        if _phrase_in_text(needle, haystack) or needle in haystack:
            return True
    return False


def primary_display_token(canonical: str) -> str:
    """Token hiển thị ngắn gọn (vd: ``TDTU``)."""
    tokens = LANDMARK_SEARCH_TOKENS.get(canonical)
    if tokens:
        return tokens[0].upper()
    return canonical.replace("_", " ").upper()


def merge_nearby_into_embedding_text(room: dict) -> str:
    """Ghép tien_ich_xq vào embedding text để index vector / hiển thị đầy đủ."""
    base = str(room.get("embedding_text") or "").strip()
    nearby = str(room.get("tien_ich_xq") or "").strip()
    if not nearby:
        return base
    marker = "## Tiện ích xung quanh"
    if marker in base or nearby in base:
        return base or f"{marker}\n- {nearby}"
    if base:
        return f"{base}\n\n{marker}\n- {nearby}"
    return f"{marker}\n- {nearby}"
