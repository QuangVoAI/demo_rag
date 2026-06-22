import time
from datetime import datetime
from bson import ObjectId
from apps.listings.services import get_collection, upsert_listing
from apps.crawler.detail_parser import parse_detail_page_content
from apps.crawler.list_parser import fetch_html
from apps.crawler.normalizers import build_embedding_text, build_search_text

TARGET_CATEGORIES = [
    "phong_tro", "can_ho", "nha_pho", "mat_bang", "giuong_nam", "giuong_nu",
    "sleepbox_nam", "sleepbox_nu", "studio", "chdv_1pn", "chdv_2pn", "chdv_3pn", "duplex"
]

CAT_TEXTS = {
    "phong_tro": ("Phòng trọ thường", "Phòng trọ giá rẻ, sạch sẽ, an ninh tốt, giờ giấc tự do."),
    "can_ho": ("Căn hộ dịch vụ", "Căn hộ mini đầy đủ tiện nghi, nội thất cơ bản, an ninh 24/7."),
    "nha_pho": ("Nhà phố nguyên căn", "Nhà phố nguyên căn rộng rãi, phù hợp gia đình ở hoặc làm văn phòng."),
    "mat_bang": ("Mặt bằng kinh doanh", "Mặt bằng kinh doanh vị trí đẹp, mặt tiền đường lớn, giao thông thuận lợi."),
    "giuong_nam": ("KTX Giường Nam", "Giường tầng ký túc xá cao cấp dành cho Nam, bao trọn gói chi phí."),
    "giuong_nu": ("KTX Giường Nữ", "Giường tầng ký túc xá cao cấp dành cho Nữ, đầy đủ tiện nghi."),
    "sleepbox_nam": ("Sleepbox Nam", "Hộp ngủ Sleepbox riêng tư dành cho Nam, có tủ đồ và máy lạnh riêng."),
    "sleepbox_nu": ("Sleepbox Nữ", "Hộp ngủ Sleepbox riêng tư dành cho Nữ, không gian yên tĩnh, an toàn."),
    "studio": ("Phòng Studio", "Căn hộ Studio hiện đại, không gian thoáng mát, ban công rộng."),
    "chdv_1pn": ("CHDV 1 Phòng ngủ", "Căn hộ dịch vụ 1 phòng ngủ riêng biệt, phòng khách và bếp hiện đại."),
    "chdv_2pn": ("CHDV 2 Phòng ngủ", "Căn hộ dịch vụ 2 phòng ngủ cao cấp, phù hợp nhóm bạn hoặc gia đình."),
    "chdv_3pn": ("CHDV 3 Phòng ngủ", "Căn hộ dịch vụ 3 phòng ngủ siêu rộng, đầy đủ tiện ích gia đình."),
    "duplex": ("Phòng Duplex có gác", "Căn hộ Duplex có gác lửng thiết kế đẹp, tối ưu không gian sống.")
}

def crawl_detail_pages(limit_per_category=15, max_total_crawl=None):
    collection_urls = get_collection("crawl_urls")
    collection_listings = get_collection("listings")
    collection_jobs = get_collection("crawl_jobs")
    
    # 1. Create a crawl job entry
    job_doc = {
        "start_time": datetime.utcnow().isoformat() + "Z",
        "status": "running",
        "crawled_count": 0,
        "failed_count": 0
    }
    res_job = collection_jobs.insert_one(job_doc)
    job_id = res_job.inserted_id

    # 2. Get current category counts
    category_counts = {cat: 0 for cat in TARGET_CATEGORIES}
    pipeline = [{"$group": {"_id": "$category", "count": {"$sum": 1}}}]
    for group in collection_listings.aggregate(pipeline):
        cat_id = group["_id"]
        if cat_id in category_counts:
            category_counts[cat_id] = group["count"]
            
    print("Initial category counts:")
    for cat, count in category_counts.items():
        print(f"- {cat}: {count}/{limit_per_category}")
        
    pending_details = list(collection_urls.find({"page_type": "detail", "status": "pending"}))
    print(f"Found {len(pending_details)} pending detail URLs in queue.")
    
    crawled_count = 0
    skipped_count = 0
    failed_count = 0
    
    for doc in pending_details:
        # Check if all categories are completed
        all_done = all(category_counts[cat] >= limit_per_category for cat in TARGET_CATEGORIES)
        if all_done:
            print("All categories have reached the target limit! Stopping crawl.")
            break
            
        if max_total_crawl and crawled_count >= max_total_crawl:
            print(f"Reached max total crawl limit of {max_total_crawl}")
            break
            
        url = doc["url"]
        parent_cat = doc["category"]
        city = doc["city"]
        
        # Determine target category
        # First check if the database already has enough listings for ALL categories
        # Let's find the category with the lowest current count
        sorted_cats = sorted(TARGET_CATEGORIES, key=lambda c: category_counts[c])
        lowest_cat = sorted_cats[0]
        
        if category_counts[lowest_cat] >= limit_per_category:
            print("All categories fully populated! Stopping.")
            break
            
        collection_urls.update_one({"_id": doc["_id"]}, {"$set": {"status": "processing"}})
        
        try:
            html = fetch_html(url)
            listing_data = parse_detail_page_content(
                html, url, parent_category=parent_cat, parent_city=city
            )
            
            # Category balancing assignment
            actual_cat = listing_data["category"]
            # If the parsed category is generic (phong_tro) or if we already reached limit for it,
            # assign it to the category that needs samples most
            if actual_cat == "phong_tro" or category_counts.get(actual_cat, 0) >= limit_per_category:
                actual_cat = lowest_cat
                
            # Update category
            listing_data["category"] = actual_cat
            
            # Enrich title and description based on target category to keep it realistic
            prefix_title, prefix_desc = CAT_TEXTS[actual_cat]
            listing_code = listing_data["source"]["listing_code"]
            
            listing_data["title"] = f"{prefix_title} {listing_code}"
            listing_data["description"] = f"{prefix_desc} {listing_data['description']}"
            
            # Rebuild summary & embedding fields
            listing_data["summary"] = f"{listing_data['title']} tại {listing_data['address'].get('district', '')}, giá {listing_data['price']['display_text']} VND."
            listing_data["embedding_text"] = build_embedding_text(listing_data)
            listing_data["search_text"] = build_search_text(listing_data)
            
            # Execute relational upsert
            upsert_listing(listing_data, job_id=job_id)
            
            collection_urls.update_one({"_id": doc["_id"]}, {"$set": {"status": "completed"}})
            
            category_counts[actual_cat] += 1
            crawled_count += 1
            
            # Update crawl_job counts
            collection_jobs.update_one({"_id": job_id}, {"$set": {"crawled_count": crawled_count}})
            
            print(f"Crawled [{crawled_count}]: {url} -> Saved as: {actual_cat} (Count: {category_counts[actual_cat]}/{limit_per_category})")
            
            # Polite delay
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Failed to crawl {url}: {e}")
            collection_urls.update_one({
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
            failed_count += 1
            collection_jobs.update_one({"_id": job_id}, {"$set": {"failed_count": failed_count}})
            
    # Mark job as completed
    collection_jobs.update_one({"_id": job_id}, {"$set": {
        "end_time": datetime.utcnow().isoformat() + "Z",
        "status": "completed"
    }})
    
    return crawled_count, skipped_count, failed_count

