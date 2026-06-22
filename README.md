# Demo RAG Django + MongoDB Cho Website Cho Thuê Nhà Trọ

## 1. Mục tiêu demo

Xây dựng một chatbot tìm nhà trọ cho website thương mại. Khi khách hàng chat như:

- "Tìm phòng trọ 2 phòng ngủ gần Quận 7"
- "Cần phòng giá dưới 5 triệu, có wifi và máy lạnh"
- "Tìm nhà trọ gần Thủ Đức, phù hợp cho sinh viên"

hệ thống sẽ:

1. Hiểu yêu cầu của người dùng
2. Truy xuất dữ liệu nhà trọ trong MongoDB
3. Kết hợp filter có cấu trúc và RAG semantic search
4. Trả về danh sách nhà trọ phù hợp nhất
5. Sinh câu trả lời tự nhiên qua chatbot

## 2. Kiến trúc tổng thể

```text
Frontend Chat UI
    ->
Django API
    ->
Intent + Filter Extractor
    ->
Hybrid Retrieval
    |- MongoDB structured filter
    |- Vector DB semantic search
    ->
Rerank / merge results
    ->
LLM response generation
    ->
Frontend trả câu trả lời cho người dùng
```

## 3. Thành phần chính

### A. Frontend chatbot

- Ô chat cho người dùng nhập câu hỏi
- Gọi API Django, ví dụ `POST /api/chat/ask`
- Hiển thị danh sách nhà trọ gợi ý

### B. Django backend

Đóng vai trò trung tâm điều phối:

- Nhận tin nhắn từ frontend
- Tách ý định tìm kiếm
- Query MongoDB
- Gọi embedding model
- Search vector DB
- Tạo prompt cho LLM
- Trả response về frontend

### C. MongoDB

Lưu dữ liệu gốc của các bài đăng nhà trọ:

- tiêu đề
- mô tả
- địa chỉ
- quận/huyện
- thành phố
- giá
- số phòng
- diện tích
- tiện ích
- liên hệ

### D. Vector DB

Lưu vector embedding của mỗi bài đăng nhà trọ để semantic search.

Trong phạm vi demo này, lựa chọn phù hợp nhất là `Chroma` vì:

- miễn phí
- chạy local nhanh
- dễ tích hợp với Python/Django
- đủ tốt cho bài toán demo chatbot tìm nhà trọ

Nếu sau này muốn mở rộng gần hơn với production, có thể chuyển sang Qdrant hoặc MongoDB Atlas Vector Search.

### E. Embedding model

Dùng để biến:

- mô tả nhà trọ
- yêu cầu chat của người dùng

thành vector để so sánh độ tương đồng ngữ nghĩa.

### F. LLM

Dùng để:

- tạo câu trả lời tự nhiên
- tóm tắt kết quả truy xuất
- hỏi lại người dùng nếu thiếu điều kiện

## 4. Luồng dữ liệu indexing

Luồng này chạy khi thêm mới hoặc cập nhật bài đăng nhà trọ.

1. Admin thêm/sửa nhà trọ trong hệ thống
2. Django nhận document nhà trọ
3. Tạo một trường text tổng hợp, ví dụ `embedding_text`
4. Gọi embedding model để tạo vector
5. Lưu vector vào vector DB kèm `listing_id`
6. Khi update document thì cập nhật lại vector
7. Khi xóa document thì xóa vector tương ứng

### Ví dụ `embedding_text`

```text
Phòng trọ 2 phòng ngủ tại Quận 7, gần Lotte Mart, giá 4.8 triệu,
diện tích 28m2, có wifi, máy lạnh, chỗ để xe, phù hợp cho sinh viên.
```

## 5. Luồng chat RAG khi người dùng tìm nhà trọ

### Bước 1. Người dùng gửi câu hỏi

Ví dụ:

```text
Tìm phòng trọ 2 phòng ngủ gần Thủ Đức, giá dưới 6 triệu, có wifi.
```

### Bước 2. Django nhận request

API ví dụ:

```http
POST /api/chat/ask
```

Body:

```json
{
  "session_id": "abc123",
  "message": "Tìm phòng trọ 2 phòng ngủ gần Thủ Đức, giá dưới 6 triệu, có wifi"
}
```

### Bước 3. Phân tích ý định và tách filter

Cần tách ra 2 loại thông tin:

#### a. Filter có cấu trúc

- `location = Thu Duc`
- `num_rooms = 2`
- `price <= 6000000`
- `amenities contains wifi`

#### b. Nhu cầu semantic

Nếu người dùng nói:

- "gần trường đại học"
- "khu an ninh"
- "phù hợp cho sinh viên"
- "không gian yên tĩnh"

thì phần này khó query bằng filter cứng, nên dùng vector search.

### Bước 4. Structured retrieval từ MongoDB

Dùng các field rõ ràng để lọc trước:

- địa điểm
- giá
- số phòng
- diện tích
- tiện ích

Mục tiêu là loại bỏ các kết quả chắc chắn sai.

### Bước 5. Semantic retrieval từ vector DB

1. Embedding câu hỏi của người dùng
2. Search vector DB
3. Lấy top K document có nghĩa gần nhất

Mục tiêu là tìm các bài đăng có mô tả phù hợp, dù người dùng không dùng đúng keyword.

### Bước 6. Hybrid merge

Gộp 2 nguồn kết quả:

- kết quả filter MongoDB
- kết quả vector DB

Sau đó chấm điểm lại:

- đúng địa điểm
- đúng số phòng
- đúng mức giá
- phù hợp semantic

Có thể ưu tiên:

- document thỏa filter có cấu trúc
- document có similarity score cao

### Bước 7. Đưa context vào LLM

Prompt gửi cho LLM gồm:

- câu hỏi của người dùng
- danh sách 3-5 nhà trọ phù hợp nhất
- instruction: chỉ được trả lời dựa trên dữ liệu retrieve

### Bước 8. Trả lời chatbot

Chatbot trả lời ví dụ:

```text
Mình tìm được 3 phòng trọ phù hợp gần Thủ Đức, giá dưới 6 triệu.

1. Phòng A - 5.5 triệu - 2 phòng ngủ - có wifi - cách Đại học SPKT 1.2 km
2. Phòng B - 5.8 triệu - 2 phòng ngủ - có wifi, máy lạnh
3. Phòng C - 4.9 triệu - 1 phòng ngủ lớn, phù hợp 2 người ở

Bạn muốn ưu tiên phòng gần trường hay phòng rẻ hơn?
```

## 6. Nguyên tắc retrieval nên áp dụng

Không nên chỉ dùng vector search.

Nên dùng `hybrid retrieval`:

1. Filter cứng bằng MongoDB cho các field rõ ràng
2. Vector search cho nhu cầu mô tả tự nhiên
3. Rerank để chọn top kết quả cuối

Lý do:

- `price`, `num_rooms`, `district` là dữ liệu có cấu trúc
- `phù hợp cho sinh viên`, `gần trung tâm`, `yên tĩnh` là dữ liệu nên tìm theo ngữ nghĩa

## 7. Thiết kế database MongoDB cho toàn bộ website

Vì `nhatrovn.vn` có nhiều danh mục như phòng trọ, căn hộ, nhà phố, mặt bằng, giường nam, giường nữ, sleepbox, studio, CHDV và duplex, nên không nên thiết kế database chỉ cho một loại phòng.

Nên dùng một collection chung tên là `listings` với schema tổng quát, sau đó phân loại bằng `category` và `subtype`.

### 7.1. Collection chính: `listings`

```json
{
  "_id": "listing_6a37b10088ac4122462ef477",
  "source": {
    "site": "nhatrovn.vn",
    "url": "https://nhatrovn.vn/cho-thue-phong-tro/ho-chi-minh/quan-10/chi-tiet/6a37b10088ac4122462ef477/",
    "listing_code": "102",
    "crawl_time": "2026-06-22T10:00:00Z",
    "status": "active"
  },
  "category": "phong_tro",
  "subtype": "phong_thuong",
  "title": "Phòng thường 102",
  "property_name": null,
  "description": "Giao Nguyễn Tri Phương, Nguyễn Chí Thanh, Nguyễn Duy Dương; đối diện ĐH UEH; thuận tiện đi các quận trung tâm.",
  "summary": "Phòng thường tại Quận 10, giá 4.5 triệu, có wifi, máy lạnh, thang máy.",
  "address": {
    "full": "400/xx Ngô Gia Tự, Phường 04, Quận 10, Thành phố Hồ Chí Minh",
    "street": "400/xx Ngô Gia Tự",
    "ward": "Phường 04",
    "district": "Quận 10",
    "city": "Thành phố Hồ Chí Minh",
    "region": "Miền Nam",
    "country": "Việt Nam",
    "slug_city": "ho-chi-minh",
    "slug_district": "quan-10"
  },
  "price": {
    "min": 4500000,
    "max": 4500000,
    "currency": "VND",
    "period": "month",
    "display_text": "4,500,000"
  },
  "property_info": {
    "area_m2": 20,
    "floor_position": "Lầu 1",
    "num_rooms": null,
    "num_bedrooms": null,
    "num_bathrooms": null,
    "max_people": null,
    "max_vehicles": null,
    "available_room_count": 3,
    "total_room_count": 10
  },
  "amenities": [
    "giuong",
    "nem",
    "tu_quan_ao",
    "thang_may",
    "wifi",
    "may_lanh",
    "ke_bep",
    "nuoc_nong",
    "tu_lanh",
    "gac"
  ],
  "fees": {
    "electricity": "4k/kWh",
    "water": "100k/ng",
    "management": "150k/ph",
    "parking": "Free",
    "wifi": "Free",
    "washing_machine": "Không có"
  },
  "rules": {
    "toilet": "Riêng",
    "curfew": "Tự do",
    "window": false,
    "balcony": false,
    "pet_allowed": false,
    "shared_parking": true,
    "electric_vehicle_allowed": false
  },
  "audience": {
    "gender_restriction": null,
    "student_friendly": true,
    "family_friendly": null
  },
  "nearby_places": [
    "Nguyễn Tri Phương",
    "Nguyễn Chí Thanh",
    "Nguyễn Duy Dương",
    "ĐH UEH"
  ],
  "media": {
    "cover_image": null,
    "images": [],
    "image_count": 0
  },
  "available_rooms": [
    {
      "room_code": "102",
      "price": 4500000
    },
    {
      "room_code": "103",
      "price": 4500000
    },
    {
      "room_code": "P.001",
      "price": 4500000
    }
  ],
  "tags": [
    "da_xac_thuc",
    "hot",
    "gan_truong_dai_hoc"
  ],
  "embedding_text": "Phòng thường 102 tại Quận 10, Thành phố Hồ Chí Minh, giá 4.5 triệu mỗi tháng, diện tích 20m2, lầu 1, có wifi, máy lạnh, thang máy, gần ĐH UEH, thuận tiện đi các quận trung tâm.",
  "search_text": "Phòng thường 102 Quận 10 Hồ Chí Minh 4.5 triệu wifi máy lạnh thang máy UEH"
}
```

### 7.2. Các giá trị `category` nên chuẩn hóa

- `phong_tro`
- `can_ho`
- `nha_pho`
- `mat_bang`
- `giuong_nam`
- `giuong_nu`
- `sleepbox_nam`
- `sleepbox_nu`
- `studio`
- `chdv_1pn`
- `chdv_2pn`
- `chdv_3pn`
- `duplex`

### 7.3. Collection phụ nên có

#### `crawl_jobs`

Lưu mỗi lần crawl:

- thời gian bắt đầu
- thời gian kết thúc
- số URL list đã crawl
- số URL detail đã crawl
- số bản ghi insert mới
- số bản ghi update
- số bản ghi lỗi

#### `crawl_urls`

Lưu hàng đợi URL:

- `url`
- `page_type`: `list` hoặc `detail`
- `category`
- `city`
- `district`
- `status`
- `retry_count`

#### `chat_logs`

Lưu lịch sử chat để đánh giá RAG:

- `session_id`
- `user_message`
- `filters_extracted`
- `retrieved_listing_ids`
- `final_answer`

## 8. Thiết kế crawl toàn site

Mục tiêu là crawl đủ dữ liệu cho demo, nhưng vẫn có cấu trúc rõ ràng để sau này mở rộng.

### 8.1. Crawl 2 tầng

#### Tầng 1: trang danh sách theo thành phố/quận/danh mục

Ví dụ:

- danh sách theo thành phố
- danh sách theo quận/huyện
- danh sách theo danh mục

Tầng này dùng để lấy:

- URL detail
- category
- city
- district
- title ngắn
- giá ngắn
- số phòng trống
- số phòng tổng
- tag như `Hot`, `Mới`, `Đã xác thực`

#### Tầng 2: trang chi tiết từng listing/phòng

Tầng này dùng để lấy:

- địa chỉ đầy đủ
- diện tích
- giá chi tiết
- vị trí lầu
- tiện ích
- chi phí điện nước
- điều kiện thuê
- mô tả tóm tắt
- danh sách phòng trống
- ảnh

### 8.2. Thứ tự crawl nên áp dụng

1. Crawl trang chủ để lấy danh sách thành phố
2. Với mỗi thành phố, crawl danh sách quận/huyện
3. Với mỗi quận/huyện, crawl theo từng danh mục
4. Từ từng trang danh sách, lấy URL detail
5. Crawl từng trang detail để tạo document đầy đủ
6. Chuẩn hóa dữ liệu trước khi lưu MongoDB
7. Tạo `embedding_text`
8. Lưu vào Chroma

### 8.3. Dữ liệu nào lấy từ list page, dữ liệu nào lấy từ detail page

#### Lấy từ list page

- `title`
- `address.full`
- `price.min`, `price.max`
- `property_info.available_room_count`
- `property_info.total_room_count`
- `tags`
- `source.url`

#### Lấy từ detail page

- `description`
- `summary`
- `amenities`
- `fees`
- `rules`
- `nearby_places`
- `media.images`
- `available_rooms`
- `property_info.area_m2`
- `property_info.floor_position`

## 9. Kế hoạch lấy mẫu dữ liệu đầy đủ cho demo

Bạn yêu cầu mỗi danh mục khoảng 15 mẫu data và đủ nhiều địa điểm. Với demo đầu tiên, nên đặt chỉ tiêu như sau:

### 9.1. Chỉ tiêu theo danh mục

- `phong_tro`: 15 mẫu
- `can_ho`: 15 mẫu
- `nha_pho`: 15 mẫu
- `mat_bang`: 15 mẫu
- `giuong_nam`: 15 mẫu
- `giuong_nu`: 15 mẫu
- `sleepbox_nam`: 15 mẫu
- `sleepbox_nu`: 15 mẫu
- `studio`: 15 mẫu
- `chdv_1pn`: 15 mẫu
- `chdv_2pn`: 15 mẫu
- `chdv_3pn`: 15 mẫu
- `duplex`: 15 mẫu

Tổng mục tiêu ban đầu: khoảng `195 listing`.

### 9.2. Chỉ tiêu phủ địa điểm

Ít nhất nên có dữ liệu từ:

- `Thành phố Hồ Chí Minh`
- `Hà Nội`
- `Đà Nẵng`
- `Bình Dương`
- `Cần Thơ`

Nếu không đủ dữ liệu cho tất cả danh mục ở mọi địa phương, ưu tiên:

1. Hồ Chí Minh
2. Hà Nội
3. Bình Dương
4. Đà Nẵng
5. Cần Thơ

### 9.3. Quy tắc phân bổ mẫu

Với mỗi `category`, cố gắng chia:

- 6 mẫu ở Hồ Chí Minh
- 3 mẫu ở Hà Nội
- 2 mẫu ở Bình Dương
- 2 mẫu ở Đà Nẵng
- 2 mẫu ở Cần Thơ

Nếu site không đủ dữ liệu ở một nơi, cho phép bù sang địa điểm khác nhưng vẫn phải có tối thiểu 3 địa phương khác nhau cho mỗi danh mục.

### 9.4. Quy tắc dừng crawl

Dừng khi đạt đủ một trong hai điều kiện:

1. đủ `15 mẫu hợp lệ` cho một danh mục
2. đã duyệt hết các quận/huyện của 5 địa phương ưu tiên

## 10. Quy tắc làm sạch và chuẩn hóa dữ liệu

### 10.1. Chuẩn hóa category

Tên hiển thị trên web cần map về slug cố định:

- `Phòng trọ` -> `phong_tro`
- `Căn hộ` -> `can_ho`
- `Nhà phố` -> `nha_pho`
- `Mặt bằng` -> `mat_bang`
- `Giường Nam` -> `giuong_nam`
- `Giường Nữ` -> `giuong_nu`
- `Sleepbox Nam` -> `sleepbox_nam`
- `Sleepbox Nữ` -> `sleepbox_nu`
- `Studio (Phòng có nội thất)` -> `studio`
- `CHDV 1 Phòng ngủ` -> `chdv_1pn`
- `CHDV 2 Phòng ngủ` -> `chdv_2pn`
- `CHDV 3 Phòng ngủ` -> `chdv_3pn`
- `Duplex (Phòng có gác và nội thất)` -> `duplex`

### 10.2. Chuẩn hóa giá

- bỏ dấu phẩy
- đổi về `int`
- tách `min` và `max`
- nếu có dạng `Giá từ 3.3 đến 3.5 triệu` thì lưu:
  - `min = 3300000`
  - `max = 3500000`

### 10.3. Chuẩn hóa địa chỉ

Tách:

- `street`
- `ward`
- `district`
- `city`

Ngoài ra giữ lại `full` để trả lời chatbot giống nguyên bản.

### 10.4. Chuẩn hóa boolean

Các field như:

- `Có ban công`
- `Có cửa sổ`
- `Nuôi thú cưng`
- `Nhận xe điện`
- `Giờ tự do`

nên lưu dạng boolean hoặc enum để filter dễ.

### 10.5. Chuẩn hóa text cho RAG

Tạo `embedding_text` theo format:

```text
{title}. Loại: {category}. Địa chỉ: {district}, {city}. Giá từ {price_min} đến {price_max}.
Diện tích: {area_m2}m2. Tiện ích: {amenities}. Điều kiện: {rules}. Mô tả: {description}.
Địa điểm gần đó: {nearby_places}.
```

## 11. Cấu trúc module Django gợi ý

```text
demo_rag/
├─ apps/
│  ├─ listings/
│  │  ├─ models.py
│  │  ├─ serializers.py
│  │  ├─ views.py
│  │  └─ services.py
│  ├─ crawler/
│  │  ├─ seeds.py
│  │  ├─ list_parser.py
│  │  ├─ detail_parser.py
│  │  ├─ normalizers.py
│  │  ├─ jobs.py
│  │  └─ management/commands/
│  ├─ chat/
│  │  ├─ views.py
│  │  ├─ services.py
│  │  └─ prompts.py
│  ├─ search/
│  │  ├─ filters.py
│  │  ├─ retriever.py
│  │  └─ reranker.py
│  └─ embeddings/
│     ├─ services.py
│     └─ sync.py
├─ config/
└─ requirements.txt
```

## 12. API và command nên có

### Command 1. Seed URL crawl

```bash
python manage.py seed_crawl_urls
```

Chức năng:

- sinh URL list page theo city, district, category
- đưa vào collection `crawl_urls`

### Command 2. Crawl list page

```bash
python manage.py crawl_list_pages
```

Chức năng:

- đọc `crawl_urls` loại `list`
- lấy URL detail
- upsert queue detail

### Command 3. Crawl detail page

```bash
python manage.py crawl_detail_pages
```

Chức năng:

- crawl trang chi tiết
- parse dữ liệu đầy đủ
- lưu `listings`
- tạo `embedding_text`
- sync Chroma

### API 1. Chat tìm nhà

```http
POST /api/chat/ask
```

Chức năng:

1. nhận message
2. extract filters
3. structured query
4. vector query
5. merge + rerank
6. gọi LLM
7. trả response

### API 2. Reindex vector

```http
POST /api/embeddings/reindex
```

Chức năng:

- đọc tất cả listing
- tạo lại vector cho demo khi cần

## 13. Pseudocode cho crawl và chat

### 13.1. Pseudocode crawl

```python
def crawl_one_detail(url: str):
    html = fetch_html(url)
    listing = parse_detail_page(html, url=url)
    listing = normalize_listing(listing)
    listing["embedding_text"] = build_embedding_text(listing)
    upsert_listing(listing)
    sync_listing_to_chroma(listing)
```

### 13.2. Pseudocode chat

```python
def chat_search(message: str):
    filters = extract_filters(message)

    mongo_results = query_mongodb(filters)

    query_vector = embed_text(message)
    vector_results = search_vector_db(query_vector, top_k=10)

    final_results = merge_and_rerank(
        mongo_results=mongo_results,
        vector_results=vector_results,
        filters=filters
    )

    context = build_context(final_results[:5])
    answer = generate_llm_answer(user_message=message, context=context)

    return {
        "answer": answer,
        "results": final_results[:5]
    }
```

## 14. Thứ tự làm demo để dễ thành công

Nên làm theo 4 phase:

### Phase 1. Thiết kế dữ liệu và crawler

- tạo schema `listings`
- tạo queue crawl
- crawl 2 tầng list/detail
- thu đủ khoảng 15 mẫu cho mỗi danh mục

Mục tiêu: có bộ dữ liệu thật, đủ rộng và đủ sạch.

### Phase 2. Retrieval có cấu trúc

- query theo `category`, `city`, `district`, `price`, `amenities`
- tạo API lọc cơ bản

Mục tiêu: người dùng có thể tìm đúng dữ liệu bằng filter cứng.

### Phase 3. Thêm RAG

- tạo `embedding_text`
- tạo vector cho listing
- search Chroma
- hybrid search với MongoDB

Mục tiêu: chatbot hiểu được các query tự nhiên hơn.

### Phase 4. Chatbot hoàn chỉnh

- thêm LLM sinh câu trả lời
- thêm memory session nếu cần
- thêm câu hỏi gợi ý tiếp theo

Mục tiêu: trả lời tự nhiên như trợ lý tìm nhà.

## 15. Gợi ý kỹ thuật cho bạn

Nếu bạn muốn demo nhanh và dễ:

- Backend: Django REST Framework
- Database: MongoDB Atlas
- Vector DB: Chroma
- Embedding: model embedding qua API
- LLM: model chat để tổng hợp câu trả lời

Stack đề xuất cho bản demo hiện tại:

- `Django + MongoDB Atlas + Chroma + Railway`

## 16. Kiến trúc triển khai đề xuất

Để demo nhanh, dễ quản lý và dễ trình bày, nên triển khai theo kiến trúc sau:

- `Django`: xử lý API, crawler, logic tìm kiếm và chatbot
- `MongoDB Atlas`: lưu dữ liệu gốc của listing, queue crawl, log chat
- `Chroma`: lưu vector embedding để semantic search
- `Railway`: deploy Django app và chạy cron crawl định kỳ

### 16.1. Luồng triển khai

```text
Frontend / Chat UI
    ->
Railway (Django API)
    |- MongoDB Atlas
    |- Chroma
    ->
LLM / Embedding API
```

### 16.2. Vai trò từng thành phần

#### Django

- cung cấp API chat
- crawl dữ liệu từ website nguồn
- chuẩn hóa dữ liệu trước khi lưu
- tạo `embedding_text`
- gọi embedding model và LLM
- truy vấn MongoDB Atlas và Chroma

#### MongoDB Atlas

- lưu `listings`
- lưu `crawl_urls`
- lưu `crawl_jobs`
- lưu `chat_logs`

#### Chroma

- lưu vector embedding của mỗi listing
- tìm top K listing gần nghĩa nhất

#### Railway

- deploy backend Django
- lưu biến môi trường
- chạy web service
- có thể cấu hình cron để crawl theo lịch

### 16.3. Biến môi trường gợi ý

```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/demo_rag_nhatro?retryWrites=true&w=majority
MONGODB_DB=demo_rag_nhatro
CHROMA_DIR=/app/data/chroma
EMBEDDING_API_KEY=your_embedding_key
LLM_API_KEY=your_llm_key
DJANGO_SECRET_KEY=your_secret_key
DEBUG=False
ALLOWED_HOSTS=your-railway-domain.up.railway.app
```

## 17. Kết luận

Đối với bài toán này, phần quan trọng nhất không chỉ là chatbot mà là:

1. thiết kế schema đủ rộng cho nhiều danh mục
2. crawl đủ dữ liệu thật từ nhiều địa điểm
3. chuẩn hóa dữ liệu để query tốt
4. tạo `embedding_text` để semantic search hoạt động đúng
5. kết hợp MongoDB Atlas filter + Chroma retrieval + LLM answer

Đây là một bài toán `hybrid RAG` dựa trên dữ liệu crawl thực tế, không phải chỉ vector search đơn thuần.
