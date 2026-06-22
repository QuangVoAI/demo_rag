import random
from datetime import datetime, timedelta
from bson import ObjectId
from apps.listings.services import (
    get_users_collection, get_sessions_collection, get_properties_collection,
    get_listings_collection, get_bookings_collection, get_payments_collection,
    get_vouchers_collection, get_user_voucher_logs_collection, get_reviews_collection,
    get_comments_collection, get_support_tickets_collection, get_consignments_collection,
    get_jobs_collection, get_job_applications_collection, get_media_assets_collection
)

MOCK_TENANTS = [
    {"name": "Nguyễn Văn A", "username": "tenant_a", "email": "tenant_a@example.com", "phone": "0912345678"},
    {"name": "Lê Thị B", "username": "tenant_b", "email": "tenant_b@example.com", "phone": "0987654321"},
    {"name": "Trần Văn C", "username": "tenant_c", "email": "tenant_c@example.com", "phone": "0901234567"},
    {"name": "Phạm Minh D", "username": "tenant_d", "email": "tenant_d@example.com", "phone": "0934567890"},
    {"name": "Đỗ Hoàng E", "username": "tenant_e", "email": "tenant_e@example.com", "phone": "0976543210"}
]

MOCK_LANDLORDS = [
    {"name": "Nguyễn Văn Hùng", "username": "landlord_hung", "email": "hung@example.com", "phone": "0987654321"},
    {"name": "Trần Thị Phương", "username": "landlord_phuong", "email": "phuong@example.com", "phone": "0965432109"},
    {"name": "Lê Hoàng Minh", "username": "landlord_minh", "email": "minh@example.com", "phone": "0943210987"}
]

MOCK_VIETNAMESE_REVIEWS = [
    "Phòng đẹp, sạch sẽ, thoáng mát, chủ nhà nhiệt tình hỗ trợ dắt xe.",
    "Khu vực an ninh tốt, gần trường đại học tiện đi lại, phòng có ban công rộng rãi.",
    "Giờ giấc tự do thoải mái, giá cả phù hợp với sinh viên học tập.",
    "Máy lạnh chạy êm, tủ lạnh sạch sẽ, nội thất đầy đủ y hình.",
    "Vị trí giao thông thuận tiện, nước nôi điện đài ổn định, sẽ tiếp tục gia hạn."
]

MOCK_CANDIDATES = [
    {"name": "Nguyễn Quốc Anh", "phone": "0988123456", "birthday": "2002-04-12", "education": "Đại học", "gender": "Nam", "note": "Mong muốn ứng tuyển vị trí Sales Rep để học hỏi."},
    {"name": "Lê Thị Hồng", "phone": "0977234567", "birthday": "2001-08-20", "education": "Cao đẳng", "gender": "Nữ", "note": "Tôi có kinh nghiệm bán hàng tại cửa hàng tiện lợi."},
    {"name": "Trần Minh Quang", "phone": "0966345678", "birthday": "2003-01-15", "education": "Trung cấp", "gender": "Nam", "note": "Ứng tuyển KTX để đi làm thêm gần quận 7."},
    {"name": "Phạm Phương Thảo", "phone": "0955456789", "birthday": "2000-11-30", "education": "Đại học", "gender": "Nữ", "note": "Mong được hỗ trợ ứng tuyển và sắp xếp lịch phỏng vấn."}
]

def seed_platform_mocks():
    users_col = get_users_collection()
    sessions_col = get_sessions_collection()
    properties_col = get_properties_collection()
    listings_col = get_listings_collection()
    bookings_col = get_bookings_collection()
    payments_col = get_payments_collection()
    vouchers_col = get_vouchers_collection()
    logs_col = get_user_voucher_logs_collection()
    reviews_col = get_reviews_collection()
    comments_col = get_comments_collection()
    tickets_col = get_support_tickets_collection()
    consignments_col = get_consignments_collection()
    jobs_col = get_jobs_collection()
    apps_col = get_job_applications_collection()
    media_col = get_media_assets_collection()

    # Clear target collections
    for col in [users_col, sessions_col, bookings_col, payments_col, vouchers_col, logs_col, reviews_col, comments_col, tickets_col, consignments_col, apps_col, media_col]:
        col.delete_many({})

    print("Cleared existing mocks database collections.")

    # 1. Seed Users & Sessions
    landlord_ids = []
    tenant_ids = []
    
    # Landlords
    for item in MOCK_LANDLORDS:
        doc = {**item, "role": "landlord", "status": "active", "created_at": datetime.utcnow().isoformat() + "Z"}
        res = users_col.insert_one(doc)
        landlord_ids.append(res.inserted_id)
        # Session
        sessions_col.insert_one({
            "user_id": res.inserted_id,
            "token": f"sess_landlord_{item['username']}_{ObjectId()}",
            "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z",
            "ip_address": "192.168.1.50",
            "user_agent": "Mozilla/5.0 Chrome/120"
        })
        
    # Tenants
    for item in MOCK_TENANTS:
        doc = {**item, "role": "tenant", "status": "active", "created_at": datetime.utcnow().isoformat() + "Z"}
        res = users_col.insert_one(doc)
        tenant_ids.append(res.inserted_id)
        # Session
        sessions_col.insert_one({
            "user_id": res.inserted_id,
            "token": f"sess_tenant_{item['username']}_{ObjectId()}",
            "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z",
            "ip_address": "192.168.1.10",
            "user_agent": "Mozilla/5.0 Safari/605"
        })
        
    # Admin
    admin_doc = {"name": "Admin Nhatrovn", "username": "admin_nhatrovn", "email": "admin@nhatrovn.vn", "phone": "19005303", "role": "admin", "status": "active", "created_at": datetime.utcnow().isoformat() + "Z"}
    res_admin = users_col.insert_one(admin_doc)
    sessions_col.insert_one({
        "user_id": res_admin.inserted_id,
        "token": f"sess_admin_{ObjectId()}",
        "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z",
        "ip_address": "127.0.0.1",
        "user_agent": "Mozilla/5.0 Firefox/115"
    })
    
    print(f"Seeded {len(landlord_ids)} landlords, {len(tenant_ids)} tenants, and 1 admin.")

    # 2. Relink Properties to Landlords
    properties = list(properties_col.find({}))
    if properties:
        for prop in properties:
            landlord_id = random.choice(landlord_ids)
            properties_col.update_one({"_id": prop["_id"]}, {"$set": {"landlord_id": landlord_id}})
            
            # Seed media assets metadata for cover image
            cover_url = prop.get("media", {}).get("cover_image")
            if cover_url:
                media_col.insert_one({
                    "filename": cover_url.split('/')[-1],
                    "url": cover_url,
                    "content_type": "image/jpeg",
                    "size_bytes": random.randint(80000, 250000),
                    "uploaded_at": datetime.utcnow().isoformat() + "Z"
                })
        print(f"Linked {len(properties)} properties to landlords and seeded media_assets.")

    # 3. Seed Vouchers
    vouchers_col.insert_one({
        "code": "NHATRO500",
        "discount_amount": 500000,
        "discount_type": "flat",
        "min_rent_price": 3000000,
        "start_date": (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z",
        "end_date": (datetime.utcnow() + timedelta(days=90)).isoformat() + "Z",
        "status": "active"
    })
    vouchers_col.insert_one({
        "code": "STUDENT200",
        "discount_amount": 200000,
        "discount_type": "flat",
        "min_rent_price": 1500000,
        "start_date": (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z",
        "end_date": (datetime.utcnow() + timedelta(days=90)).isoformat() + "Z",
        "status": "active"
    })
    v_post = vouchers_col.insert_one({
        "code": "LANDLORD_POST",
        "discount_amount": 100000,
        "discount_type": "flat",
        "min_rent_price": 100000,
        "start_date": (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z",
        "end_date": (datetime.utcnow() + timedelta(days=90)).isoformat() + "Z",
        "status": "active"
    })
    
    print("Seeded 3 vouchers.")

    # 4. Seed Bookings & Payments & Vouchers Log
    listings = list(listings_col.find({}))
    bookings_seeded = 0
    payments_seeded = 0
    
    if listings:
        # Create bookings for first 20 listings
        for idx, room in enumerate(listings[:20]):
            tenant_id = random.choice(tenant_ids)
            prop_id = room["property_id"]
            price_val = room.get("price", {}).get("min") or 3000000
            
            # Booking Form inputs match
            status = "confirmed" if idx % 3 != 0 else ("pending" if idx % 3 == 1 else "cancelled")
            
            b_doc = {
                "tenant_id": tenant_id,
                "property_id": prop_id,
                "listing_id": room["_id"],
                "status": status,
                "number_people": random.randint(1, 4),
                "number_vehicles": random.randint(0, 2),
                "have_pet": random.choice([True, False]),
                "viewing_time": (datetime.utcnow() + timedelta(days=random.randint(1, 5), hours=random.randint(9, 17))).isoformat() + "Z",
                "estimated_move_in": random.choice(["ONgay", "3D", "StartNextMonth"]),
                "deposit_amount": price_val,
                "remark": "Tôi cần xem phòng vào buổi chiều, phòng tầng trung hoặc cao.",
                "created_at": (datetime.utcnow() - timedelta(days=random.randint(1, 3))).isoformat() + "Z"
            }
            res_b = bookings_col.insert_one(b_doc)
            bookings_seeded += 1
            
            if status == "confirmed":
                # Create corresponding payment
                payments_col.insert_one({
                    "booking_id": res_b.inserted_id,
                    "user_id": tenant_id,
                    "amount": price_val,
                    "payment_method": "bank_transfer",
                    "status": "success",
                    "transaction_id": f"TXN_{random.randint(10000000, 99999999)}",
                    "created_at": datetime.utcnow().isoformat() + "Z"
                })
                payments_seeded += 1
                
                # Apply voucher log sometimes
                if idx % 2 == 0:
                    v_doc = vouchers_col.find_one({"code": "NHATRO500"})
                    if v_doc:
                        logs_col.insert_one({
                            "user_id": tenant_id,
                            "voucher_id": v_doc["_id"],
                            "booking_id": res_b.inserted_id,
                            "used_at": datetime.utcnow().isoformat() + "Z"
                        })
                        
        print(f"Seeded {bookings_seeded} bookings, {payments_seeded} payments, and user voucher logs.")

    # 5. Seed Reviews & Comments
    reviews_seeded = 0
    comments_seeded = 0
    if properties:
        for prop in properties[:15]:
            # Seed 1-2 reviews per property
            for _ in range(random.randint(1, 2)):
                tenant_id = random.choice(tenant_ids)
                reviews_col.insert_one({
                    "user_id": tenant_id,
                    "property_id": prop["_id"],
                    "rating": random.choice([4, 5]),
                    "comment": random.choice(MOCK_VIETNAMESE_REVIEWS),
                    "created_at": (datetime.utcnow() - timedelta(days=random.randint(10, 30))).isoformat() + "Z"
                })
                reviews_seeded += 1
                
        print(f"Seeded {reviews_seeded} reviews.")

    if listings:
        for room in listings[:15]:
            tenant_id = random.choice(tenant_ids)
            # Create a question comment
            q_res = comments_col.insert_one({
                "user_id": tenant_id,
                "listing_id": room["_id"],
                "content": "Phòng này còn trống từ ngày 1 tới không ạ? Tôi muốn thuê lâu dài.",
                "parent_id": None,
                "created_at": (datetime.utcnow() - timedelta(days=4)).isoformat() + "Z"
            })
            comments_seeded += 1
            
            # Create landlord reply comment
            prop_doc = properties_col.find_one({"_id": room["property_id"]})
            landlord_id = prop_doc.get("landlord_id") if prop_doc else random.choice(landlord_ids)
            comments_col.insert_one({
                "user_id": landlord_id,
                "listing_id": room["_id"],
                "content": "Chào bạn, phòng này vẫn còn trống nhé bạn. Bạn có thể đặt hẹn đi xem phòng thực tế.",
                "parent_id": q_res.inserted_id,
                "created_at": (datetime.utcnow() - timedelta(days=3)).isoformat() + "Z"
            })
            comments_seeded += 1
            
        print(f"Seeded {comments_seeded} comments.")

    # 6. Seed Support Tickets
    for i in range(4):
        tenant_id = random.choice(tenant_ids)
        tickets_col.insert_one({
            "user_id": tenant_id,
            "subject": random.choice(["Lỗi tải ảnh hợp đồng", "Không áp dụng được mã NHATRO500", "Báo cáo thông tin phòng ảo"]),
            "message": "Tôi gặp sự cố khi thanh toán/sử dụng chức năng trên ứng dụng, vui lòng kiểm tra giúp tôi.",
            "status": random.choice(["open", "resolved"]),
            "created_at": (datetime.utcnow() - timedelta(days=i)).isoformat() + "Z"
        })
    print("Seeded 4 support tickets.")

    # 7. Seed Consignments
    province_options = [
        {"code": "79", "name": "Thành phố Hồ Chí Minh"},
        {"code": "74", "name": "Tỉnh Bình Dương"},
        {"code": "92", "name": "Thành phố Cần Thơ"},
        {"code": "48", "name": "Thành phố Đà Nẵng"}
    ]
    consignments_col.insert_one({
        "organization_name": "Nhà trọ Hùng Vương Q10",
        "phone": "0987654321",
        "province": "79",
        "province_name": "Thành phố Hồ Chí Minh",
        "room_types": ["phong_tro", "studio"],
        "total_rooms": 30,
        "vacant_rooms": 5,
        "need_management_service": True,
        "message": "Tôi cần ký gửi hỗ trợ lấp đầy 5 phòng trống và tư vấn dịch vụ vận hành.",
        "submitted_time": (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z",
        "status": "pending"
    })
    consignments_col.insert_one({
        "organization_name": "Căn hộ dịch vụ An Bình",
        "phone": "0965432109",
        "province": "74",
        "province_name": "Tỉnh Bình Dương",
        "room_types": ["can_ho", "chdv_1pn"],
        "total_rooms": 20,
        "vacant_rooms": 2,
        "need_management_service": False,
        "message": "Cần tìm khách thuê cho 2 căn 1PN trống.",
        "submitted_time": (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z",
        "status": "contacted"
    })
    print("Seeded 2 consignments.")

    # 8. Seed Job Applications
    jobs = list(jobs_col.find({}))
    apps_seeded = 0
    if jobs:
        for job in jobs[:10]:
            for candidate in MOCK_CANDIDATES[:2]:
                apps_col.insert_one({
                    "job_id": job["_id"],
                    "candidate": candidate,
                    "applied_time": (datetime.utcnow() - timedelta(days=random.randint(1, 5))).isoformat() + "Z"
                })
                apps_seeded += 1
        print(f"Seeded {apps_seeded} job applications.")

    return True
