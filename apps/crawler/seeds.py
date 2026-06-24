from apps.rooms.services import get_collection

CITIES = ["ho-chi-minh", "ha-noi", "binh-duong", "da-nang", "can-tho"]

def seed_list_urls():
    collection = get_collection("crawl_urls")
    collection.delete_many({"page_type": "list"})
    
    seeded_count = 0
    # Seed pages 1 to 10 for each city
    for city in CITIES:
        for page in range(1, 11):
            url = f"https://nhatrovn.vn/cho-thue-phong-tro/{city}/?page={page}"
            existing = collection.find_one({"url": url})
            if not existing:
                doc = {
                    "url": url,
                    "page_type": "list",
                    "category": "phong_tro",
                    "city": city,
                    "status": "pending",
                    "retry_count": 0
                }
                collection.insert_one(doc)
                seeded_count += 1
                
    return seeded_count
