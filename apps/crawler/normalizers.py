import re
import unicodedata

CATEGORY_MAP = {
    "phong tro": "phong_tro",
    "cho thue phong tro": "phong_tro",
    "can ho": "can_ho",
    "nha pho": "nha_pho",
    "mat bang": "mat_bang",
    "giuong nam": "giuong_nam",
    "giuong nu": "giuong_nu",
    "sleepbox nam": "sleepbox_nam",
    "sleepbox nu": "sleepbox_nu",
    "studio": "studio",
    "chdv 1 pn": "chdv_1pn",
    "chdv 1 phòng ngủ": "chdv_1pn",
    "chdv 2 pn": "chdv_2pn",
    "chdv 2 phòng ngủ": "chdv_2pn",
    "chdv 3 pn": "chdv_3pn",
    "chdv 3 phòng ngủ": "chdv_3pn",
    "duplex": "duplex",
}

AMENITY_MAP = {
    "wifi": "wifi",
    "mạng": "wifi",
    "internet": "wifi",
    "máy lạnh": "may_lanh",
    "điều hoà": "may_lanh",
    "điều hòa": "may_lanh",
    "giường": "giuong",
    "nệm": "nem",
    "đệm": "nem",
    "tủ quần áo": "tu_quan_ao",
    "tủ áo": "tu_quan_ao",
    "thang máy": "thang_may",
    "kệ bếp": "ke_bep",
    "bếp": "ke_bep",
    "nước nóng": "nuoc_nong",
    "nóng lạnh": "nuoc_nong",
    "tủ lạnh": "tu_lanh",
    "gác": "gac",
    "máy giặt": "may_giat",
    "khóa vân tay": "khoa_van_tay",
    "an ninh": "an_ninh",
    "chỗ để xe": "cho_de_xe",
}

def remove_vietnamese_accents(txt):
    if not txt:
        return ""
    normalized = unicodedata.normalize('NFKD', txt)
    cleaned = "".join([c for c in normalized if not unicodedata.combining(c)])
    cleaned = cleaned.replace('đ', 'd').replace('Đ', 'D')
    return cleaned

def slugify(txt):
    cleaned = remove_vietnamese_accents(txt).lower()
    cleaned = re.sub(r'[^a-z0-9\s-]', '', cleaned)
    cleaned = re.sub(r'[\s-]+', '-', cleaned).strip('-')
    return cleaned

def normalize_category(category_name):
    if not category_name:
        return "phong_tro"
    normalized = remove_vietnamese_accents(category_name).lower().strip()
    for key, val in CATEGORY_MAP.items():
        if key in normalized:
            return val
    return "phong_tro"

def parse_price(price_str):
    if not price_str:
        return 0, 0, "Thỏa thuận"
    
    price_str = price_str.lower().strip()
    
    if "thỏa thuận" in price_str or "thoa thuan" in price_str or "liên hệ" in price_str:
        return 0, 0, "Thỏa thuận"
        
    # Check if range
    range_match = re.split(r'đến|-|tới', price_str)
    
    # Check unit context for the whole string
    global_has_trieu = any(x in price_str for x in ["triệu", "trieu", "tr"])
    global_has_k = any(x in price_str for x in ["nghìn", "ngan", "k"])
    
    def parse_single_price(s):
        s = s.strip()
        cleaned = re.sub(r'[^0-9\.,]', '', s)
        if not cleaned:
            return None
            
        has_trieu = global_has_trieu or any(x in s for x in ["triệu", "trieu", "tr"])
        has_k = global_has_k or any(x in s for x in ["nghìn", "ngan", "k"])
        
        if has_trieu:
            cleaned = cleaned.replace(',', '.')
            try:
                return int(float(cleaned) * 1000000)
            except ValueError:
                pass
        elif has_k:
            cleaned = cleaned.replace(',', '.')
            try:
                return int(float(cleaned) * 1000)
            except ValueError:
                pass
        
        # No explicit or inherited unit. Strip separators if it's format like 4,500,000
        cleaned_no_sep = cleaned.replace('.', '').replace(',', '')
        try:
            val = int(cleaned_no_sep)
            if val < 100:
                return val * 1000000
            elif val < 10000:
                return val * 1000
            return val
        except ValueError:
            return None

    if len(range_match) >= 2:
        val1 = parse_single_price(range_match[0])
        val2 = parse_single_price(range_match[1])
        if val1 is not None and val2 is not None:
            min_v = min(val1, val2)
            max_v = max(val1, val2)
            return min_v, max_v, f"{min_v:,} - {max_v:,}"
        elif val1 is not None:
            return val1, val1, f"{val1:,}"
        elif val2 is not None:
            return val2, val2, f"{val2:,}"
            
    val = parse_single_price(price_str)
    if val is not None:
        return val, val, f"{val:,}"
        
    return 0, 0, price_str

def parse_area(area_str):
    if not area_str:
        return None
    nums = re.findall(r'(\d+(?:\.\d+)?)', area_str)
    if nums:
        return float(nums[0])
    return None

def normalize_address(address_str):
    if not address_str:
        return {
            "full": "",
            "street": "",
            "ward": "",
            "district": "",
            "city": "",
            "region": "Miền Nam",
            "country": "Việt Nam",
            "slug_city": "",
            "slug_district": ""
        }
    
    parts = [p.strip() for p in address_str.split(',')]
    
    city = parts[-1] if len(parts) >= 1 else ""
    district = parts[-2] if len(parts) >= 2 else ""
    ward = parts[-3] if len(parts) >= 3 else ""
    street = ", ".join(parts[:-3]) if len(parts) >= 4 else (parts[0] if len(parts) == 3 else "")
    
    city_normalized = remove_vietnamese_accents(city).lower()
    region = "Miền Nam"
    if "ha noi" in city_normalized:
        region = "Miền Bắc"
    elif "da nang" in city_normalized or "binh dinh" in city_normalized or "hue" in city_normalized:
        region = "Miền Trung"
    
    return {
        "full": address_str,
        "street": street,
        "ward": ward,
        "district": district,
        "city": city,
        "region": region,
        "country": "Việt Nam",
        "slug_city": slugify(city),
        "slug_district": slugify(district)
    }

def normalize_amenities(amenities_list):
    if not amenities_list:
        return []
    result = set()
    for item in amenities_list:
        item_lower = item.lower()
        for raw, std in AMENITY_MAP.items():
            if raw in item_lower:
                result.add(std)
    return list(result)

def build_embedding_text(listing):
    title = listing.get('title', '')
    category = listing.get('category', '')
    addr = listing.get('address', {})
    district = addr.get('district', '')
    city = addr.get('city', '')
    price = listing.get('price', {})
    price_min = price.get('min', 0)
    price_max = price.get('max', 0)
    price_text = f"từ {price_min:,}đ" if price_min == price_max else f"từ {price_min:,}đ đến {price_max:,}đ"
    
    prop_info = listing.get('property_info', {})
    area = prop_info.get('area_m2')
    area_text = f"{area}m2" if area else "chưa xác định"
    
    amenities = ", ".join(listing.get('amenities', []))
    
    rules = listing.get('rules', {})
    rules_text = f"Toilet: {rules.get('toilet', 'chưa xác định')}. Giờ giấc: {rules.get('curfew', 'chưa xác định')}."
    if rules.get('pet_allowed'):
        rules_text += " Cho phép nuôi thú cưng."
    
    desc = listing.get('description', '')
    nearby = ", ".join(listing.get('nearby_places', []))
    
    embedding_text = f"{title}. Loại: {category}. Địa chỉ: {district}, {city}. Giá: {price_text}. Diện tích: {area_text}. Tiện ích: {amenities}. Điều kiện: {rules_text}. Mô tả: {desc}."
    if nearby:
        embedding_text += f" Địa điểm gần đó: {nearby}."
        
    return embedding_text

def build_search_text(listing):
    title = listing.get('title', '')
    addr = listing.get('address', {})
    district = addr.get('district', '')
    city = addr.get('city', '')
    amenities = " ".join(listing.get('amenities', []))
    desc = listing.get('description', '')
    
    combined = f"{title} {district} {city} {amenities} {desc}"
    cleaned = re.sub(r'[^a-zA-Z0-9\sđĐàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]', ' ', combined)
    words = [w.lower() for w in cleaned.split() if w]
    return " ".join(words[:50])
