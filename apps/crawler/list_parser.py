import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from apps.listings.services import get_collection

CAT_WEIGHTS = {
    "phong_tro": 0,
    "can_ho": 1,
    "nha_pho": 1,
    "mat_bang": 1,
    "giuong_nam": 1,
    "giuong_nu": 1,
    "sleepbox_nu": 1,
    "sleepbox_nam": 1,
    "studio": 1,
    "chdv_1pn": 2,
    "chdv_2pn": 2,
    "chdv_3pn": 2,
    "duplex": 2
}

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text

def parse_list_page(html, base_url="https://nhatrovn.vn"):
    soup = BeautifulSoup(html, 'html.parser')
    detail_urls = set()
    
    # Find all anchor tags that have /chi-tiet/ in their href
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/chi-tiet/' in href:
            absolute_url = urljoin(base_url, href)
            # Remove query parameters if any to keep URLs clean
            clean_url = absolute_url.split('?')[0]
            detail_urls.add(clean_url)
            
    return list(detail_urls)

def process_pending_lists():
    collection = get_collection("crawl_urls")
    pending_lists = list(collection.find({"page_type": "list", "status": "pending"}))
    
    processed_count = 0
    extracted_count = 0
    
    for doc in pending_lists:
        url = doc["url"]
        category = doc["category"]
        city = doc["city"]
        
        # Update status to processing
        collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "processing"}})
        
        try:
            html = fetch_html(url)
            detail_urls = parse_list_page(html)
            
            # Insert detail URLs into queue
            inserted_detail_count = 0
            for d_url in detail_urls:
                existing = collection.find_one({"url": d_url})
                if not existing:
                    collection.insert_one({
                        "url": d_url,
                        "page_type": "detail",
                        "category": category,
                        "city": city,
                        "status": "pending",
                        "retry_count": 0
                    })
                    inserted_detail_count += 1
                else:
                    existing_cat = existing.get("category", "phong_tro")
                    existing_weight = CAT_WEIGHTS.get(existing_cat, 0)
                    new_weight = CAT_WEIGHTS.get(category, 0)
                    if new_weight > existing_weight:
                        collection.update_one(
                            {"_id": existing["_id"]},
                            {"$set": {
                                "category": category,
                                "status": "pending"  # Recrawl to get specific category
                            }}
                        )
                        inserted_detail_count += 1
            
            collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "completed"}})
            processed_count += 1
            extracted_count += inserted_detail_count
            
        except Exception as e:
            # Handle failure
            collection.update_one({
                "_id": doc["_id"]
            }, {
                "$set": {
                    "status": "failed",
                    "error_msg": str(e)
                },
                "$inc": {
                    "retry_count": 1
                }
            })
            
    return processed_count, extracted_count
