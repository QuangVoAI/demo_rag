# Nhatrovn Assistant — Phân tích Domain Fit (Cho thuê phòng)

> Knowledge graph: .understand-anything/knowledge-graph.json
> Commit: 10fb9a6f98b9b666f750cc3876f00762ea6e3513
> Thời điểm phân tích: 2026-06-22T06:51:41Z

## 1. Kết luận tổng quan

**Hệ thống đã phù hợp 100% với lĩnh vực cho thuê phòng (nhatro.vn).** Toàn bộ production path từ frontend đến AI core đều xoay quanh domain phòng trọ Việt Nam — entity Listing, tiếng Việt, tiền VND, đơn vị hành chính VN (quận/huyện/phường/xã), semantic search với canonical text tiếng Việt, intent parser dựa trên keyword tiếng Việt.

Tuy nhiên có **một số điểm cần để ý** ở phần "legacy_data_processing/" và một số file "orphans" trong knowledge graph — phần lớn là legacy/hoặc chưa kết nối, cần được xử lý khi cập nhật dữ liệu thật.

## 2. Bằng chứng domain fit (production path)

### 2.1. Domain entity — Listing (phòng trọ)
python/room_assistant/schemas.py định nghĩa read model rất sát thực tế phòng trọ VN:
- ent_price, deposit (tiền cọc),  rea_m2 (diện tích)
- province, district, ward (đơn vị hành chính VN, không phải zipcode/state)
- amenities: đầy đủ tiện ích phòng trọ VN — air_conditioner, washing_machine, private_bathroom, mezzanine (gác lửng), balcony, parking, elevator, wifi, security
- pets_allowed, electric_bike_allowed, max_occupants, available_from

### 2.2. Vietnamese-aware throughout
- python/room_assistant/intent.py: regex parser dùng từ khóa tiếng Việt — "tìm phòng", "so sánh", "tính tiền", "tương tự", "đặt lịch", "nhắn chủ", "giữ chỗ", "thanh toán", "thương lượng", "trả giá"…
- Tiền tệ: parser xử lý "1.500.000", "1tr5", "3 triệu" → VND integer (_money_to_vnd)
- Vietnamese district normalization: bỏ dấu, xử lý "Quan X", "Q. X", "quận X" (_normalize_location_value)
- Currency chính: VND (format 4,5 triệu/tháng cho dễ đọc)
- python/retrieval/qdrant_client.py: stop-word list bao gồm cả tiếng Việt ("của", "và", "là", "được", "có", "không", "cho", "với", "tôi", "bạn"…)

### 2.3. Đúng 10 Intent cho use-case phòng trọ
SEARCH_ROOM, REFINE_SEARCH, ASK_ABOUT_ROOM, CALCULATE_COST, COMPARE_ROOMS, FIND_SIMILAR, SUMMARIZE_ROOM, REQUEST_FAQ, GENERAL_HELP, REQUEST_ACTION

Đặc biệt: CALCULATE_COST (tính cọc + phí phát sinh), FIND_SIMILAR (tìm phòng tương tự quanh khu vực + ngân sách ±15%) là các use-case rất cụ thể của phòng trọ.

### 2.4. Hard NO actions (bảo vệ user)
python/room_assistant/intent.py định nghĩa ACTION_KEYWORDS cho dat_lich, message_owner, save_favorite, hold_room, payment, edit_listing, negotiate — tất cả đều được phân loại thành REQUEST_ACTION và hệ thống chỉ trả lời hướng dẫn UI, KHÔNG tự thực hiện.

Invariant cứng (đã enforce trong tools.py):
- write_tool_calls_per_turn = 0
- max_read_tool_calls_per_turn <= 3

### 2.5. Production flow khớp với domain
- Frontend (Tailwind + WebSocket) → Rust Actix gateway → Kafka → Python query_worker → room_assistant.workflow → trả về grounded answer tiếng Việt.
- Continuous indexing: listing.changed Kafka event → index worker → embed bằng BAAI/bge-m3 → upsert/delete Qdrant → DLQ nếu fail.
- Session state lưu location, budget, pets_required, move_in_date… (rất phù hợp hành vi tìm phòng qua nhiều turn).

## 3. Bằng chứng code-level khác

| File | Dấu hiệu domain fit |
|---|---|
| python/room_assistant/schemas.py | LIST_OPERATION_PATHS = districts, wards, vehicles, pets, amenities — chính xác use-case phòng trọ |
| python/room_assistant/tools.py | 7 tool: search_listings, get_listing_detail, retrieve_listing_context, retrieve_faq, calculate_cost_estimate, compare_listings, find_similar_listings |
| python/room_assistant/repository.py | Mongo query với $or nhiều field cho "cùng 1 listing có thể lưu ở schema khác nhau" |
| python/room_assistant/indexing.py | build_canonical_embedding_text dùng label tiếng Việt "Tiêu đề", "Mô tả", "Đặc điểm không gian", "Tiện ích", "Khu vực xung quanh", "Quy định dạng văn bản" |
| python/agents/router.py | Seed embedding đề cập "nhatro.vn", "phòng trọ", "thuê nhà", "căn hộ quận 7" |
| python/agents/response_writer.py | Tone theo mood (frustrated/urgent/normal); "Dùng 'mình/bạn', không dùng 'chúng tôi/quý khách'" — bám sát giọng chat tiếng Việt |
| python/room_assistant/intent.py | AMENITY_ALIASES map tiếng Việt → canonical: "máy lạnh"/"điều hòa" → air_conditioner, "nhà vệ sinh riêng" → private_bathroom, "hầm xe" → parking |
| frontend/index.html (meta) | "Trợ lý AI read-only giúp khách hàng tìm và so sánh phòng trọ" |
| README.md | Mở đầu: "Read-only AI assistant for customers looking for rooms on Nhatrovn." |

## 4. Điểm cần lưu ý (legacy / off-path / chưa kết nối)

Khi chạy knowledge graph, phát hiện một số điểm "lệch domain" hoặc cần làm rõ:

### 4.1. python/legacy_data_processing/ — Legacy MyKingdom (Đã dọn dẹp)
- Đã đổi tên thư mục từ `python/data_processing` thành `python/legacy_data_processing` để làm rõ phạm vi: đây là phần pipeline training cũ dùng cho chatbot MyKingdom, hoàn toàn độc lập và **không ảnh hưởng/không import** trong production của `nhatro.vn`.

### 4.2. Các file agents quan trọng (Đã được cập nhật cho Nhatrovn)
- `python/agents/state.py` và `python/agents/grader.py` **không còn là legacy/dead code**. Chúng đã được refactor và viết lại hoàn toàn để phục vụ nghiệp vụ nhatro.vn (quản lý trạng thái tìm phòng trọ và đánh giá chất lượng kết quả tìm kiếm phòng trọ).

### 4.3. Frontend self-contained
- frontend/app.js không import gì (toàn inline) — OK cho static serving, nhưng cần kiểm tra: có dùng top_k mà backend WsQueryMessage đã khai báo default_top_k = 5? Frontend có truyền top_k không? (Không — frontend hard-code top-k thông qua server-side retrieval. Đây là design choice, không phải bug.)

### 4.4. Cơ chế đồng bộ dữ liệu MongoDB -> Qdrant mới (Đã cập nhật)
- Thay vì `scripts/backfill_listings.py` cũ, hệ thống hiện tại sử dụng:
  - `python/scripts/index_mongo_to_qdrant.py` để đồng bộ toàn phần dữ liệu từ MongoDB Atlas (`demo_rag.listings`) vào Qdrant.
  - `python/scripts/kafka_indexer.py` làm consumer thời gian thực, tự động cập nhật Qdrant mỗi khi nhận tin nhắn thay đổi từ Kafka/Redpanda (tránh re-index từ đầu).

## 5. Knowledge graph summary

| Thống kê | Giá trị |
|---|---|
| Tổng nodes | 59 |
| Tổng edges | 84 |
| Tổng layers | 9 |
| Tour steps | 5 |
| Issues | 0 |
| Orphans | 0 |
| File nodes | 46 |
| Concept nodes | 4 |
| Domain tag domain:nhatro-vn | 31 nodes (53%) |
| Edge types | contains, calls, serves, imports, triggers, defines_schema, validates, implements, tested_by, configures, documents |

**Layer breakdown:**
1. layer:frontend — 3 nodes
2. layer:rust-gateway — 10 nodes
3. layer:kafka-workers — 3 nodes
4. layer:room-assistant (production domain core) — 9 nodes
5. layer:shared-agents — 10 nodes
6. layer:retrieval-infra — 4 nodes
7. layer:config-and-utils — 4 nodes
8. layer:training-data-pipeline (legacy / off-path) — 6 nodes
9. layer:scripts-and-infra — 6 nodes

## 6. Câu trả lời trực tiếp cho câu hỏi "đã phù hợp với lĩnh vực cho thuê phòng chưa?"

**Đã phù hợp hoàn toàn về code/kiến trúc**, nhưng cần:

### 6.1. Không cần sửa về domain fit
- Toàn bộ production code (room_assistant/, agents/ production paths, Rust gateway, frontend) đều viết đúng cho nhatro.vn.
- Mọi schema, intent, tool, query đều chuẩn phòng trọ Việt Nam.
- Câu trả lời của LLM được enforce chỉ dùng data đã verify (grounded) và không bịa.

### 6.2. Cần làm khi cập nhật dữ liệu (theo dữ liệu MongoDB và Qdrant mới)
1. Đảm bảo cấu hình đúng MongoDB Atlas (`demo_rag.listings`) trong file `.env`.
2. Khởi động Docker (Qdrant, Redpanda, MongoDB) bằng lệnh `docker-compose up -d`.
3. Chạy `python python/scripts/index_mongo_to_qdrant.py` để đồng bộ toàn bộ dữ liệu hiện tại lên Qdrant.
4. Chạy `python python/scripts/kafka_indexer.py` ngầm để đồng bộ thời gian thực mỗi khi crawler có bài viết mới.
5. Kiểm tra Redis (Upstash) cho session state và Langfuse cho observability.

### 6.3. Khuyến nghị dọn dẹp và kiểm thử bổ sung
1. Thư mục `python/data_processing/` đã được di chuyển thành `python/legacy_data_processing/` để tránh nhầm lẫn.
2. Cân nhắc viết thêm các ca kiểm thử (tests) kiểm tra luồng phản hồi theo tâm trạng (mood-aware response) của Assistant.

### 6.4. Ghi chú thêm
- Các file log phát sinh cục bộ, dữ liệu local của Qdrant (`qdrant_storage/`), dữ liệu local của DB Rust (`rust_backend/chat_history.db*`) không nên được commit lên Git.

## 7. Cách dùng knowledge graph

Mở dashboard tương tác:

```bash
# Sau khi cài pnpm (cần cho build lần đầu)
cd C:\Users\Admin\.understand-anything
# Cài dependencies
pnpm install --frozen-lockfile
# Build core
pnpm --filter @understand-anything/core build
# Sau đó có thể dùng các lệnh của plugin understand-anything
```

Trong khi chờ môi trường đầy đủ, file knowledge-graph.json đã có thể dùng để:
- Search nhanh các node theo tag (vd: "domain:nhatro-vn").
- Xem toàn bộ quan hệ imports/calls giữa các module.
- Đối chiếu với code thật khi refactor.
