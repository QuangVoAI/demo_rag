import re
from datetime import datetime
from bs4 import BeautifulSoup
from apps.crawler.normalizers import (
    normalize_category, parse_price, parse_area, normalize_address,
    normalize_amenities, build_embedding_text, build_search_text
)

def clean_nearby_places(desc_text):
    """
    Extract nearby places from description text.
    Look for lines mentioning locations near by.
    """
    places = []
    if not desc_text:
        return places
        
    lines = [line.strip() for line in desc_text.split('\n') if line.strip()]
    for line in lines:
        line_lower = line.lower()
        # Look for typical point of interest indicators
        if any(indicator in line_lower for indicator in ["gan", "gần", "doi dien", "đối diện", "giao", "cach", "cách"]):
            # Clean indicators from the line
            cleaned = line
            for indicator in ["giao", "đối diện", "doi dien", "gần", "gan", "thuận tiện đi", "thuan tien di"]:
                cleaned = re.sub(rf'(?i)\b{indicator}\b', '', cleaned)
            # Remove leading/trailing punctuations
            cleaned = re.sub(r'^[\s,;:\.-]+|[\s,;:\.-]+$', '', cleaned).strip()
            if cleaned and len(cleaned) < 50:
                places.append(cleaned)
    return places

def parse_detail_page_content(html, url, parent_category=None, parent_city=None):
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. Source details
    # Listing code from h1.room-code or URL
    code_el = soup.find('h1', class_='room-code')
    listing_code = code_el.text.strip() if code_el else ""
    if not listing_code:
        # Extract from URL, e.g. /chi-tiet/6a37b10088ac4122462ef477/
        match = re.search(r'/chi-tiet/([^/]+)/', url)
        if match:
            listing_code = match.group(1)
            
    # 2. Title
    # Extract from active breadcrumb or document title
    title = ""
    breadcrumb_active = soup.find('li', class_='breadcrumb-item active')
    if breadcrumb_active:
        title = breadcrumb_active.text.strip()
    else:
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.text.replace(' - Nhatrovn', '').strip()
            
    # 3. Category
    # Extract from title first, fallback to parent if generic
    category = normalize_category(title)
    if category == "phong_tro" and parent_category:
        category = parent_category
        
    # 4. Description
    desc_el = soup.find(class_='rs-summary')
    desc_text = desc_el.text.strip() if desc_el else ""
    if not desc_text:
        # Fallback to rs-summary__lead
        lead_el = soup.find(class_='rs-summary__lead')
        desc_text = lead_el.text.strip() if lead_el else ""
        
    # 5. Address
    address_full = ""
    addr_el = soup.find(class_='rs-card-address')
    if addr_el:
        spans = addr_el.find_all('span')
        if spans:
            address_full = spans[-1].text.strip()
            
    address_normalized = normalize_address(address_full)
    # Overwrite city if parent specifies it
    if parent_city and not address_normalized.get('city'):
        address_normalized['city'] = parent_city.replace('-', ' ').title()
        address_normalized['slug_city'] = parent_city
        
    # 6. Price
    price_val_el = soup.find('span', class_='rs-card-price__value')
    price_text = price_val_el.text.strip() if price_val_el else ""
    # If not found, try card-price element text
    if not price_text:
        price_card_el = soup.find(class_='rs-card-price')
        if price_card_el:
            price_text = price_card_el.text.replace('Giá', '').replace(':', '').strip()
            
    p_min, p_max, display_txt = parse_price(price_text)
    price_data = {
        "min": p_min,
        "max": p_max,
        "currency": "VND",
        "period": "month",
        "display_text": display_txt
    }
    
    # 7. Property Info (Area, Floor, Rooms counts)
    area_m2 = None
    floor_position = None
    for badge in soup.find_all(class_='rs-info-badge'):
        lbl_el = badge.find(class_='rs-info-badge__label')
        val_el = badge.find(class_='rs-info-badge__val')
        if lbl_el and val_el:
            lbl = lbl_el.text.strip().lower()
            val = val_el.text.strip()
            if "diện tích" in lbl or "dien tich" in lbl:
                area_m2 = parse_area(val)
            elif "vị trí" in lbl or "vị tri" in lbl or "lầu" in lbl or "floor" in lbl:
                floor_position = val
                
    available_room_count = None
    vacant_el = soup.find(class_='rn-vacant-badge')
    if vacant_el:
        match = re.search(r'(\d+)', vacant_el.text)
        if match:
            available_room_count = int(match.group(1))
            
    total_room_count = None
    total_el = soup.find(class_='rn-property-total')
    if total_el:
        match = re.search(r'(\d+)', total_el.text)
        if match:
            total_room_count = int(match.group(1))
            
    property_info = {
        "area_m2": area_m2,
        "floor_position": floor_position,
        "num_rooms": None,
        "num_bedrooms": None,
        "num_bathrooms": None,
        "max_people": None,
        "max_vehicles": None,
        "available_room_count": available_room_count,
        "total_room_count": total_room_count
    }
    
    # 8. Amenities
    raw_amenities = []
    for chip in soup.find_all(class_='rs-amenity-chip'):
        if 'rs-amenity-chip--inactive' not in chip.get('class', []):
            lbl_el = chip.find(class_='rs-amenity-chip__label')
            if lbl_el:
                raw_amenities.append(lbl_el.text.strip())
            elif chip.get('data-tooltip'):
                raw_amenities.append(chip.get('data-tooltip'))
    amenities = normalize_amenities(raw_amenities)
    
    # 9. Fees
    fees = {
        "electricity": "Chưa xác định",
        "water": "Chưa xác định",
        "management": "Chưa xác định",
        "parking": "Chưa xác định",
        "wifi": "Chưa xác định",
        "washing_machine": "Chưa xác định"
    }
    for item in soup.find_all(class_='rs-fee-item'):
        key = item.get('data-cost-key')
        strong = item.find('strong')
        if key and strong:
            val = strong.text.strip()
            if key == "electricity":
                fees["electricity"] = val
            elif key == "water":
                fees["water"] = val
            elif key == "management":
                fees["management"] = val
            elif key == "parking":
                fees["parking"] = val
            elif key == "wifi":
                fees["wifi"] = val
            elif key == "cleaning":
                fees["washing_machine"] = val
                
    # 10. Rules / details
    rules_raw = {}
    for item in soup.find_all(class_='rs-detail-item'):
        text = item.text.strip()
        if ':' in text:
            k, v = text.split(':', 1)
            rules_raw[k.strip().lower()] = v.strip()
            
    def is_yes(val_str):
        if not val_str:
            return False
        val_lower = val_str.lower()
        return "có" in val_lower or "cho phép" in val_lower or "riêng" in val_lower or "tự do" in val_lower
        
    rules = {
        "toilet": rules_raw.get("toilet", "Chung" if is_yes(rules_raw.get("toilet")) else "Riêng"),
        "curfew": rules_raw.get("giờ giấc", "Tự do"),
        "window": is_yes(rules_raw.get("cửa sổ")),
        "balcony": is_yes(rules_raw.get("ban công")),
        "pet_allowed": is_yes(rules_raw.get("thú cưng")),
        "shared_parking": "chung" in rules_raw.get("để xe", "").lower() or is_yes(rules_raw.get("để xe")),
        "electric_vehicle_allowed": is_yes(rules_raw.get("xe điện"))
    }
    
    # Check toilet rule exactly
    if "toilet" in rules_raw:
        rules["toilet"] = rules_raw["toilet"]
        
    # 11. Audience info
    desc_lower = desc_text.lower()
    student_friendly = "sinh viên" in desc_lower or "sv" in desc_lower or "spkt" in desc_lower or "ueh" in desc_lower or "bach khoa" in desc_lower
    family_friendly = "gia đình" in desc_lower or "vợ chồng" in desc_lower or "vo chong" in desc_lower
    audience = {
        "gender_restriction": None,
        "student_friendly": student_friendly or None,
        "family_friendly": family_friendly or None
    }
    
    # 12. Nearby places
    nearby_places = clean_nearby_places(desc_text)
    
    # 13. Media images
    images = []
    # Find in hero carousel or property-hero
    hero = soup.find(class_='property-hero') or soup.find(class_='property-carousel')
    if hero:
        for img in hero.find_all('img'):
            src = img.get('src')
            if src and src.startswith('http') and src not in images:
                images.append(src)
                
    # Also fallback to rs-card-thumb-wrap img
    thumb_wrap = soup.find(class_='rs-card-thumb-wrap')
    if thumb_wrap:
        img = thumb_wrap.find('img')
        if img:
            src = img.get('src')
            if src and src.startswith('http') and src not in images:
                images.append(src)
                
    cover_image = images[0] if images else None
    media = {
        "cover_image": cover_image,
        "images": images,
        "image_count": len(images)
    }
    
    # 14. Available rooms list
    available_rooms = []
    seen_room_codes = set()
    for r_item in soup.find_all(class_='room-item'):
        r_code_el = r_item.find(class_='room-code')
        r_price_el = r_item.find(class_='room-price')
        if r_code_el and r_price_el:
            code = r_code_el.text.strip()
            if code not in seen_room_codes:
                seen_room_codes.add(code)
                r_p_min, _, _ = parse_price(r_price_el.text)
                available_rooms.append({
                    "room_code": code,
                    "price": r_p_min
                })
            
    # 15. Tags
    tags = []
    # Read tags from class presence
    if soup.find(class_='status-hot'):
        tags.append("hot")
    if soup.find(class_='status-new'):
        tags.append("new")
    if soup.find(class_='is-auth') or soup.find(class_='support-indicator'):
        tags.append("da_xac_thuc")
    if student_friendly:
        tags.append("gan_truong_dai_hoc")
        
    # Build listing document
    listing = {
        "source": {
            "site": "nhatrovn",
            "url": url,
            "listing_code": listing_code,
            "crawl_time": datetime.utcnow().isoformat() + "Z",
            "status": "active"
        },
        "category": category,
        "subtype": None,
        "title": title,
        "property_name": None,
        "description": desc_text,
        "summary": f"{title} tại {address_normalized.get('district', '')}, giá {display_txt} VND.",
        "address": address_normalized,
        "price": price_data,
        "property_info": property_info,
        "amenities": amenities,
        "fees": fees,
        "rules": rules,
        "audience": audience,
        "nearby_places": nearby_places,
        "media": media,
        "available_rooms": available_rooms,
        "tags": tags
    }
    
    # Generate embedding_text & search_text
    listing["embedding_text"] = build_embedding_text(listing)
    listing["search_text"] = build_search_text(listing)
    
    return listing
