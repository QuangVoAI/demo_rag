# File chứa template trả lời theo kịch bản và nhãn dùng trong room_assistant workflow.

# ==========================================
# 1. INPUT / TOOL LIMITS
# ==========================================
INPUT_TOO_LONG_ANSWER = (
    "Câu hỏi hơi dài nên mình chưa xử lý để tránh sai lệch dữ liệu. "
    "Bạn rút gọn dưới {max_chars} ký tự và gửi lại giúp mình nhé."
)

TOOL_BUDGET_EXCEEDED_ANSWER = (
    "Mình cần giới hạn số lần đọc dữ liệu trong một lượt. "
    "Bạn thử hỏi lại hẹp hơn với tối đa 3 phòng hoặc một nhu cầu cụ thể nhé."
)

# ==========================================
# 2. SEARCH — NO RESULT
# ==========================================
SEARCH_NO_RESULT_BUDGET_LINE = (
    "Dạ em tìm trong **{area}** với ngân sách **{budget}/tháng** "
    "mà chưa thấy căn trống khớp ạ. "
    "Anh/chị thử nới thêm khoảng 500k–1 triệu hoặc xem khu lân cận, em lọc lại ngay nha."
)
SEARCH_NO_RESULT_BUDGET_URGENT_PREFIX = "Dạ em hiểu mình cần gấp ạ. {body}"
SEARCH_NO_RESULT_BUDGET_FRUSTRATED_PREFIX = "Dạ em hiểu mình tìm mãi cũng mệt ạ. {body}"

SEARCH_NO_RESULT_BUDGET_ONLY = (
    "Dạ em chưa thấy căn nào trong tầm **{budget}/tháng** ạ. "
    "Anh/chị cho em biết khu vực ưu tiên hoặc nới ngân sách thêm chút, em lọc lại liền nha."
)
SEARCH_NO_RESULT_BUDGET_ONLY_EMPATHY_PREFIX = "Dạ em hiểu mà ạ. {body}"

SEARCH_NO_RESULT_URGENT = (
    "Dạ em hiểu mình đang cần gấp ạ. Em chưa thấy căn khớp 100% ngay, "
    "nhưng nếu mình nới ngân sách một chút hoặc mở rộng khu vực, em lọc lại liền "
    "để tìm phòng còn trống sớm nhất cho mình nha."
)
SEARCH_NO_RESULT_FRUSTRATED = (
    "Dạ em hiểu mình tìm mãi cũng hơi mệt ạ. Em chưa thấy căn khớp trọn điều kiện, "
    "nhưng mình thử nới ngân sách hoặc bỏ bớt 1–2 tiêu chí, em lọc lại ngay — "
    "chắc chắn sẽ có thêm lựa chọn phù hợp hơn ạ."
)
SEARCH_NO_RESULT_DEFAULT = (
    "Dạ em tìm mỏi mắt mà chưa thấy phòng nào khớp 100% điều kiện của mình ạ. "
    "Anh/chị thử nới ngân sách hoặc mở rộng khu vực giúp em nhé, đảm bảo sẽ có nhiều căn đẹp lắm ạ!"
)
SEARCH_NO_RESULT_SALES_HANDOFF_SUFFIX = (
    "Bên em có đội ngũ sales kiểm tra thêm nguồn phòng nội bộ — đôi khi họ tìm được căn "
    "phù hợp hơn ạ. Anh/chị chờ em chút để được tư vấn thêm nha."
)
SEARCH_NO_RESULT_SALES_HANDOFF: dict[str, str] = {
    "normal": (
        "Dạ em tìm mỏi mắt mà chưa thấy phòng nào phù hợp đang hiển thị trên hệ thống ạ. "
        + SEARCH_NO_RESULT_SALES_HANDOFF_SUFFIX
    ),
    "urgent": (
        "Dạ em hiểu mình cần gấp ạ. Em đã lọc kỹ inventory đang public nhưng chưa thấy căn khớp ngay. "
        + SEARCH_NO_RESULT_SALES_HANDOFF_SUFFIX
    ),
    "frustrated": (
        "Dạ em hiểu mình tìm mãi cũng mệt ạ. Em rà soát hết phòng đang hiển thị mà vẫn chưa có căn khớp trọn điều kiện. "
        + SEARCH_NO_RESULT_SALES_HANDOFF_SUFFIX
    ),
}

ORDINAL_OUT_OF_RANGE = (
    "Dạ trong danh sách vừa rồi em chỉ có **{available}** phòng, "
    "chưa có phòng số **{requested}** ạ. Anh/chị chọn lại giúp em (ví dụ phòng số 1–{available}) nha."
)
COMPARE_ORDINAL_UNRESOLVED = (
    "Dạ em chưa xác định đủ phòng để so sánh ạ. "
    "Anh/chị nói rõ giúp em (ví dụ: so sánh phòng số 1 và phòng số 3 trong danh sách vừa rồi) nha."
)

# ==========================================
# 3. SEARCH — ALTERNATIVE / SUCCESS
# ==========================================
SEARCH_ALTERNATIVE_OPENING: dict[str, str] = {
    "frustrated": (
        "Dạ em hiểu điều kiện hơi khó nên mình hơi mệt khi chưa thấy căn ưng ý ạ. "
        "Em gợi ý mấy căn gần đúng nhất để mình tham khảo nha:"
    ),
    "urgent": (
        "Dạ em hiểu mình cần gấp ạ. Chưa có căn khớp 100% nhưng em tìm được vài căn "
        "gần đúng nhất để mình xem trước nha:"
    ),
    "normal": (
        "Dạ điều kiện hiện tại hơi khó nên em chưa thấy căn khớp 100% ạ. "
        "Em gợi ý mấy căn gần đúng nhất để mình tham khảo nha:"
    ),
}

SEARCH_ALTERNATIVE_ROOM_LINE = (
    "{index}. **{title}** (#{room_id}) — chỉ {rent}/tháng, {district}."
)
SEARCH_ALTERNATIVE_CTA = (
    "\nNếu mình nới ngân sách hoặc bỏ bớt 1–2 tiêu chí, em sẽ tìm được nhiều căn ưng hơn ạ 😊"
)

SEARCH_RELAXED_OPENING = "Dạ {relaxed_note} Mấy căn cùng khu vực vẫn ngon mà hợp lý nè:"
SEARCH_SUCCESS_OPENING = "Dạ còn phòng ạ! Em vừa lọc ra mấy căn sạch đẹp, giá cực tốt cho mình đây:"
SEARCH_ROOM_LINE = (
    "{index}. **{title}** (#{room_id}) — chỉ {rent}/tháng, {district}{landmark_suffix}."
)
SEARCH_SUCCESS_CTA = (
    "\nAnh/chị ưng căn nào chưa ạ? Nếu rảnh thì sắp xếp ghé qua xem thực tế nha, "
    "phòng bên ngoài đẹp hơn ảnh nhiều ạ 😊"
)

UNKNOWN_DISTRICT = "chưa rõ khu vực"
LANDMARK_HINT_SUFFIX = ", {hint}"

# ==========================================
# 4. RELAXED SEARCH NOTES
# ==========================================
RELAX_FIELD_LABELS: dict[str, str] = {
    "near_landmarks": "vị trí gần mốc bạn nói",
    "wards": "phường bạn chọn",
    "amenities_preferred": "vài tiện nghi ưu tiên",
    "amenities_required": "đủ tiện nghi yêu cầu",
    "excluded_features": "điều kiện loại trừ",
    "vehicles": "chỗ để xe",
    "budget": "mức ngân sách",
}

RELAXED_NOTE_DEFAULT = (
    "em chưa thấy căn khớp đúng 100% nên xin phép nới nhẹ tiêu chí cho mình ạ."
)
RELAXED_NOTE_WITH_FIELDS = (
    "em chưa thấy căn khớp đúng {fields} nên em xin phép gợi ý mấy căn gần đúng nhất nha."
)

# ==========================================
# 5. INTENT-SPECIFIC ANSWER TEMPLATES
# ==========================================
REQUEST_ACTION_ANSWER = (
    "Dạ tính năng thao tác tự động em chưa được học ạ. "
    "Với yêu cầu '{action}', anh/chị thao tác trực tiếp trên giao diện giúp em nha! "
    "Nhưng nếu ưng phòng rồi, chiều nay ghé xem thực tế luôn cho tiện anh/chị nhỉ?"
)
REQUEST_ACTION_DEFAULT = "thao tác nghiệp vụ"

REQUEST_ACTION_LABELS: dict[str, str] = {
    "dat_lich": "đặt lịch xem phòng",
    "huy_lich": "hủy lịch hẹn",
    "doi_lich": "đổi lịch hẹn",
    "message_owner": "nhắn tin cho chủ nhà",
    "save_favorite": "lưu phòng yêu thích",
    "hold_room": "giữ chỗ",
    "payment": "thanh toán / đặt cọc",
    "edit_room": "sửa thông tin tin đăng",
    "negotiate": "thương lượng / giảm giá",
}

REQUEST_ACTION_ANSWERS: dict[str, str] = {
    "dat_lich": (
        "Dạ em **chưa được phép tự đặt lịch hộ** anh/chị trên hệ thống ạ. "
        "Anh/chị chọn căn ưng ý rồi bấm **Đặt lịch xem phòng** trên tin đăng, "
        "hoặc nhắn em khu vực + giờ rảnh để em gợi ý mấy căn phù hợp trước nha 😊"
    ),
    "huy_lich": (
        "Dạ em **chưa hủy lịch hộ** được ạ. Anh/chị vào mục **Lịch hẹn** trên tài khoản nhatrovn "
        "để hủy, hoặc liên hệ hotline nếu cần hỗ trợ gấp nha."
    ),
    "doi_lich": (
        "Dạ em **chưa đổi lịch hộ** được ạ. Anh/chị mở **Lịch hẹn** trên app/website để đổi giờ xem, "
        "hoặc hủy lịch cũ rồi đặt lại giúp em nha."
    ),
    "negotiate": (
        "Dạ em **không có quyền thương lượng hay giảm giá** thay chủ nhà, "
        "nên em chưa hỗ trợ bớt giá được ạ. "
        "Mình xem phòng ưng ý trước, rồi trao đổi trực tiếp với chủ nhà khi ký HĐ nha. "
        "Nếu cần tầm giá dễ thở hơn, em lọc thêm khu lân cận giúp anh/chị được ạ."
    ),
    "message_owner": (
        "Dạ em **chưa nhắn tin hộ** chủ nhà được ạ. Anh/chị bấm **Chat/Liên hệ** trên tin đăng "
        "để trao đổi trực tiếp với chủ nhà nha."
    ),
    "payment": (
        "Dạ em **không thu tiền hay giữ cọc hộ**, nên em chưa hỗ trợ thanh toán được ạ. "
        "Anh/chị chỉ thanh toán sau khi xác minh trực tiếp "
        "với bên cho thuê hoặc qua kênh chính thức trên nhatrovn nha."
    ),
    "hold_room": (
        "Dạ em **chưa giữ chỗ hộ** được ạ. Anh/chị đặt lịch xem phòng trước, ưng ý rồi trao đổi "
        "cọc/giữ phòng trực tiếp với chủ nhà nha."
    ),
    "save_favorite": (
        "Dạ em **chưa lưu yêu thích hộ** được ạ. Anh/chị bấm biểu tượng **Yêu thích** trên tin đăng "
        "để lưu lại căn mình quan tâm nha."
    ),
    "edit_room": (
        "Dạ em **chỉ tư vấn tin đăng**, không sửa thông tin hộ chủ nhà được ạ. "
        "Chủ nhà cần đăng nhập tài khoản để cập nhật tin nha."
    ),
}


def format_request_action_answer(action: str | None) -> str:
    key = str(action or "").strip()
    if key in REQUEST_ACTION_ANSWERS:
        return REQUEST_ACTION_ANSWERS[key]
    label = REQUEST_ACTION_LABELS.get(key, REQUEST_ACTION_DEFAULT)
    return REQUEST_ACTION_ANSWER.format(action=label)

FIND_SIMILAR_MISSING_SOURCE = (
    "Dạ em chưa rõ anh/chị muốn tìm phòng tương tự căn nào. "
    "Anh/chị gửi mã phòng hoặc chọn một phòng trong danh sách giúp em nha!"
)
ASK_ROOM_MISSING_ID = (
    "Dạ em chưa rõ anh/chị đang quan tâm căn nào. Anh/chị gửi mã phòng cho em nha!"
)
ASK_ROOM_HEADER = "**{title}** (#{room_id}) — giá {rent}/tháng."
ASK_ROOM_LOCATION = "Khu vực: {location}."
ASK_ROOM_AMENITIES = (
    "Dạ tiện ích có đủ: {features}. Mình dọn vào là ở thoải mái luôn ạ."
)
ASK_ROOM_UNKNOWN_FIELDS = "_Dữ liệu chưa xác nhận: {fields}._"
UNKNOWN_LOCATION = "chưa rõ"

CALCULATE_COST_MISSING_INFO = (
    "Dạ em chưa đủ thông tin tính chi phí căn này. Anh/chị cho em xin mã phòng nhé!"
)
CALCULATE_COST_OPENING = "Dạ em tính sương sương chi phí dự kiến cho anh/chị nhé:"
CALCULATE_COST_LINE = "- {label}: {amount}"
CALCULATE_COST_DEPOSIT = "- Tiền cọc: {amount}"
CALCULATE_COST_RENTAL_PERIOD = "- Thời gian thuê: {months} tháng"
CALCULATE_COST_RECURRING_FEES = "- Phí cố định {months} tháng: {amount}"
CALCULATE_COST_TOTAL_PERIOD = "\n**Tổng tạm tính {months} tháng:** {amount}"
CALCULATE_COST_TOTAL_INITIAL = "\n**Tổng tạm tính ban đầu:** {amount}"
CALCULATE_COST_UNKNOWN = "_Chưa có dữ liệu: {fields}._"
CALCULATE_COST_NOT_CALCULATED = "_Có dữ liệu nhưng chưa tính vào tổng: {details}._"

COST_FIXED_FIELD_LABELS: dict[str, str] = {
    "monthly_rent": "Tiền thuê mỗi tháng",
    "parking": "Phí gửi xe",
    "management": "Phí quản lý",
    "water": "Tiền nước",
    "wifi": "Wifi",
    "washing_machine": "Máy giặt",
}

COST_ITEM_LABELS: dict[str, str] = {
    "rent_first_month": "Tiền thuê tháng đầu",
    "deposit": "Tiền cọc",
    "fee_electricity": "Tiền điện",
    "fee_water": "Tiền nước",
    "fee_management": "Phí quản lý",
    "fee_parking": "Phí gửi xe",
    "fee_wifi": "Wifi",
    "fee_washing_machine": "Máy giặt",
}

COMPARE_MISSING_ROOMS = (
    "Dạ em chưa tìm thấy dữ liệu phòng: {room_ids} ạ. Anh/chị kiểm tra lại mã giúp em nha."
)
COMPARE_NEED_ROOM_IDS = (
    "Dạ để em so sánh chuẩn xác, anh/chị gửi giúp em tối đa 3 mã phòng nha "
    "(ví dụ: `so sánh #A #B`)."
)
COMPARE_OPENING = "Dạ em gửi anh/chị bảng so sánh chi tiết:"
COMPARE_ROW = (
    "- **{title}** (#{room_id}): {rent}/tháng, {area} m², {district}, {status}."
)
COMPARE_BEST_PICK = (
    "\n✨ **Gợi ý cực hợp lý:** Căn **{title}** (#{room_id}) với giá {rent}/tháng{area_suffix}."
)
COMPARE_BEST_PICK_AREA_SUFFIX = ", diện tích {area} m²"
COMPARE_MISSING_DATA = "_Chưa có dữ liệu cho: {room_ids}._"
COMPARE_NOT_COMPARED = (
    "_Mình chỉ so sánh tối đa 3 phòng/lượt nên chưa so sánh: {room_ids}._"
)
COMPARE_STATUS_AVAILABLE = "Còn phòng"
COMPARE_STATUS_UNKNOWN = "chưa rõ trạng thái"
COMPARE_UNKNOWN_AREA = "chưa rõ"

COMPARE_INSIGHT_CHEAPER = (
    "- **Giá:** {cheaper_title} rẻ hơn {pricier_title} khoảng {diff}/tháng."
)
COMPARE_INSIGHT_LARGER = (
    "- **Diện tích:** {larger_title} rộng hơn {smaller_title} khoảng {diff} m²."
)
COMPARE_INSIGHT_FIRST_ADVANTAGE = "- **Ưu điểm {title}:** có thêm {features}."
COMPARE_INSIGHT_SECOND_ADVANTAGE = "- **Ưu điểm {title}:** có thêm {features}."
COMPARE_INSIGHT_TIE = (
    "- Hai căn này khá ngang nhau về dữ liệu hiện có; nếu cần mình có thể đào sâu thêm "
    "vào phí, nội thất và tiện ích chi tiết."
)

REQUEST_FAQ_FALLBACK = (
    "Dạ anh/chị cần hỏi thêm về quy trình thuê, hợp đồng hay tiền cọc không ạ? "
    "Anh/chị cứ nhắn, em tư vấn kỹ cho nha."
)

# ==========================================
# 6. GENERAL HELP SCENARIOS
# ==========================================
GENERAL_HELP_PRICE_OBJECTION = (
    "Dạ em hiểu mà ạ, tầm giá này với sinh viên thì mình phải cân lên đặt xuống dữ lắm. "
    "Nếu mình ưu tiên tiết kiệm, em có thể lọc giúp các căn mềm hơn một chút hoặc tìm khu vực "
    "lân cận để giá dễ chịu hơn.\n"
    "Mình nói em mức ngân sách dễ thở nhất với khu anh/chị muốn ở, em lọc lại ngay mấy căn "
    "hợp túi tiền cho mình nha 😊"
)

GENERAL_HELP_OFF_TOPIC = (
    "Dạ em chỉ hỗ trợ tư vấn phòng trọ, giá thuê, chi phí, tiện ích và khu vực phù hợp thôi ạ. "
    "Mấy việc như giải bài, viết code hay xử lý nội dung ngoài thuê phòng thì em chưa hỗ trợ được.\n"
    "Nếu anh/chị đang cần tìm phòng, cứ nhắn khu vực, ngân sách hoặc tiện ích mong muốn, "
    "em lọc ngay cho mình nha 😊"
)

GENERAL_HELP_DEFAULT = (
    "Dạ em có thể tìm phòng, so sánh giá, tư vấn chi phí và tiện ích chi tiết ạ. "
    "Anh/chị đang cần tìm phòng quanh khu vực nào để em hỗ trợ gửi phòng đẹp ngay nhé 😊"
)

# ==========================================
# 7. LLM CONTEXT BUILDING ([DỮ LIỆU ĐÃ XÁC MINH])
# ==========================================
LLM_CONTEXT_ROOM_LIST_HEADER = "Danh sách phòng phù hợp:"
LLM_CONTEXT_ROOM_STATUS_AVAILABLE = "Trạng thái: còn phòng"
LLM_CONTEXT_ROOM_STATUS_UNAVAILABLE = "Trạng thái: hết phòng"
LLM_CONTEXT_VERIFIED_AMENITIES = "Tiện ích xác minh: {amenities}"
LLM_CONTEXT_ROOM_FEATURES = "Thông tin phòng: {features}"
LLM_CONTEXT_COST_HEADER = "\nƯớc tính chi phí:"
LLM_CONTEXT_COST_LINE = "  - {name}: {amount}"
LLM_CONTEXT_COST_TOTAL_INITIAL = "  Tổng: {amount}"
LLM_CONTEXT_COST_TOTAL_PERIOD = "  Tổng {months} tháng: {amount}"
LLM_CONTEXT_COST_UNKNOWN = "  Chưa có dữ liệu: {fields}"
LLM_CONTEXT_COST_NOT_CALCULATED = "  Có dữ liệu nhưng chưa tính vào tổng: {details}"
LLM_CONTEXT_COMPARE_HEADER = "\nBảng so sánh:"
LLM_CONTEXT_COMPARE_ROW = (
    "  - #{room_id}: {rent}/tháng, {area} m², {district}"
)
LLM_CONTEXT_COMPARE_MISSING = "  Chưa có dữ liệu: {room_ids}"
LLM_CONTEXT_COMPARE_NOT_COMPARED = (
    "  Chưa so sánh do giới hạn tối đa 3 phòng: {room_ids}"
)
LLM_CONTEXT_FAQ_HEADER = "\nThông tin FAQ:"
LLM_CONTEXT_FAQ_LINE = "  [{topic}] {answer}"
LLM_CONTEXT_UNKNOWN_FIELDS = "\nCác trường chưa có dữ liệu: {fields}"
LLM_CONTEXT_EMPTY = "Không có dữ liệu phù hợp."

# ==========================================
# 8. SUGGESTED QUESTIONS
# ==========================================
SUGGESTED_QUESTIONS_WITH_ROOM = [
    "Tính tổng chi phí cho #{room_id}",
    "Tóm tắt ưu điểm và hạn chế của #{room_id}",
    "Tìm phòng tương tự nhưng rẻ hơn",
]
SUGGESTED_QUESTIONS_GENERAL_HELP = [
    "Tìm phòng dưới 5 triệu ở quận Bình Thạnh",
    "So sánh #A #B #C",
    "Phòng này có cho nuôi mèo không?",
]
SUGGESTED_QUESTIONS_NO_RESULT = [
    "Nới ngân sách thêm 1 triệu",
    "Bỏ yêu cầu máy lạnh",
    "Đổi sang khu vực gần trường hơn",
]
SUGGESTED_QUESTIONS_INPUT_TOO_LONG = [
    "Tìm phòng dưới 5 triệu ở Bình Thạnh",
    "So sánh #A101 #B202",
    "Phòng này có máy lạnh không?",
]

# ==========================================
# 9. AMENITY / FEATURE LABELS
# ==========================================
AMENITY_LABELS: dict[str, str] = {
    "air_conditioner": "Máy lạnh",
    "balcony": "Ban công",
    "window": "Cửa sổ",
    "washing_machine": "Máy giặt",
    "private_bathroom": "WC riêng",
    "mezzanine": "Gác",
    "kitchen": "Bếp",
    "refrigerator": "Tủ lạnh",
    "hot_water": "Nước nóng",
    "bed": "Giường",
    "mattress": "Nệm",
    "wardrobe": "Tủ quần áo",
    "elevator": "Thang máy",
    "wifi": "Wifi",
    "ev_charging": "Sạc xe điện",
    "free_hours": "Giờ tự do",
    "pets_allowed": "Cho nuôi thú cưng",
}

FEATURE_FACT_LABELS: tuple[str, ...] = (
    "Máy lạnh",
    "Ban công",
    "Cửa sổ",
    "Wifi",
    "Gác",
    "Toilet",
    "Giờ giấc",
    "Máy giặt",
    "Thú cưng",
    "Để xe",
    "Thang máy",
    "Kệ bếp",
    "Nước nóng",
    "Tủ lạnh",
    "Giường",
    "Nệm",
    "Tủ quần áo",
)

INSUFFICIENT_VERIFIED_DATA = (
    "Dạ em chưa có dữ liệu xác minh đủ cho phần này ạ ({fields}). "
    "Anh/chị xem thêm chi tiết trên tin đăng hoặc nhắn em mã phòng khác để kiểm tra lại giúp mình nha."
)

LANDMARK_NEAR_HINT = "gần {landmarks}"
