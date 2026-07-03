"""Kịch bản tư vấn theo giọng nhân viên nhatrovn (docx + chat thực tế)."""

from __future__ import annotations

import re
from typing import Any

# Công thức 3 lớp: thấu cảm → thông tin → hướng hành động (xem phòng / đặt lịch)
STAFF_OPENERS = (
    "Dạ em chào anh/chị ạ.",
    "Dạ em hiểu mà ạ.",
    "Dạ vâng ạ.",
)

STAFF_FOLLOW_UPS = (
    "Anh/chị đang ở mấy người và dự kiến dọn vào khi nào để em tư vấn cho khớp ạ?",
    "Anh/chị cho em xin khu vực và tầm giá dễ thở nhất, em lọc ngay mấy căn phù hợp nha.",
    "Nếu rảnh anh/chị sắp xếp ghé xem phòng thực tế, em hỗ trợ đặt lịch xem phòng luôn ạ 😊",
)

STAFF_FAQ: list[dict[str, Any]] = [
    {
        "topic": "deposit",
        "keywords": ("tiền cọc", "tien coc", "đặt cọc", "dat coc", "cọc", "coc bao nhieu", "coc bao nhieu"),
        "answer": (
            "Dạ thông thường cọc **1 tháng tiền phòng** ạ, tùy từng chủ nhà và từng căn. "
            "Em chưa thấy mức cọc chính thức trong dữ liệu căn này nên em sẽ xác nhận lại giúp anh/chị trước khi chốt nha. "
            "Anh/chị cho em biết căn nào đang quan tâm, em check nhanh rồi báo chính xác ạ."
        ),
    },
    {
        "topic": "pets",
        "keywords": ("nuôi mèo", "nuoi meo", "nuôi chó", "nuoi cho", "thú cưng", "thu cung", "nuoi pet", "cho nuoi"),
        "answer": (
            "Dạ việc nuôi mèo/chó **tùy quy định từng phòng và chủ nhà** ạ, em không dám khẳng định bừa. "
            "Anh/chị cho em khu vực + ngân sách + số người, em lọc vài căn phù hợp rồi em hỏi giúp chủ nhà xác nhận nuôi pet nha."
        ),
    },
    {
        "topic": "electricity_water",
        "keywords": ("điện nước", "dien nuoc", "giá điện", "gia dien", "giá nước", "gia nuoc", "điện bao nhiêu", "nuoc may kwh"),
        "answer": (
            "Dạ điện nước thường tính theo đồng hồ hoặc khoán tháng, mỗi nhà một mức ạ. "
            "Nếu căn anh/chị quan tâm có ghi trong hồ sơ phòng, em sẽ đọc đúng số đó; không có thì em xác nhận lại với chủ nhà trước khi báo nha."
        ),
    },
    {
        "topic": "viewing",
        "keywords": ("xem phòng", "xem phong", "đặt lịch", "dat lich", "hẹn xem", "hen xem", "qua xem"),
        "answer": (
            "Dạ anh/chị cho em xin **khu vực, tầm giá và thời gian rảnh** để em gửi vài căn phù hợp và hỗ trợ hẹn xem phòng nha. "
            "Phòng xem thực tế thường đẹp và rõ hơn ảnh nhiều ạ 😊"
        ),
    },
    {
        "topic": "negotiate",
        "keywords": ("giảm giá", "giam gia", "thương lượng", "thuong luong", "cao quá", "cao qua", "đắt quá"),
        "answer": (
            "Dạ em hiểu mà ạ. Giá phòng đôi khi chủ nhà có thể linh hoạt **100–200k** nếu ở lâu dài, "
            "nhưng em không hứa trước được — quan trọng là mình xem phòng ưng ý trước đã nha. "
            "Anh/chị muốn tầm giá dễ thở hơn, em lọc thêm khu lân cận giúp ạ."
        ),
    },
    {
        "topic": "contract",
        "keywords": ("hợp đồng", "hop dong", "thời hạn thuê", "thoi han thue", "ky han"),
        "answer": (
            "Dạ thời hạn thuê và điều khoản hợp đồng **tùy từng chủ nhà** ạ. "
            "Em có thể tư vấn thông tin phòng và chi phí có trong hệ thống; phần ký kết anh/chị trao đổi trực tiếp khi xem phòng nha."
        ),
    },
    {
        "topic": "booking",
        "keywords": ("đặt phòng", "dat phong", "giữ phòng", "giu phong"),
        "answer": (
            "Dạ em hỗ trợ tư vấn và hướng dẫn **đặt lịch xem phòng** trên nhatrovn ạ. "
            "Anh/chị chọn căn ưng ý, em sẽ hướng dẫn bước tiếp theo để xem thực tế nha."
        ),
    },
    {
        "topic": "payment",
        "keywords": ("thanh toán", "thanh toan", "chuyển khoản", "chuyen khoan"),
        "answer": (
            "Dạ em **không thu tiền hay giữ cọc hộ** ạ. Anh/chị chỉ thanh toán sau khi xác minh trực tiếp với bên cho thuê "
            "hoặc qua kênh chính thức trên nhatrovn nha."
        ),
    },
    {
        "topic": "rental_tips",
        "keywords": (
            "sinh viên", "sinh vien", "lưu ý", "luu y", "nên chú ý", "nen chu y",
            "mẹo thuê", "meo thue", "kinh nghiệm thuê", "kinh nghiem thue",
            "cọc mấy tháng", "coc may thang", "hợp lý", "hop ly",
        ),
        "answer": (
            "Dạ em gợi ý nhanh cho anh/chị ạ:\n"
            "- **Cọc** thường 1–2 tháng tiền phòng, nên hỏi rõ điều kiện hoàn cọc trước khi chuyển.\n"
            "- **Xem phòng thực tế** trước khi cọc; kiểm tra điện nước, giờ giấc, nội thất.\n"
            "- **Hợp đồng** nên ghi rõ giá, cọc, thời hạn, ai chịu phí sửa chữa.\n"
            "- Sinh viên nên ưu tiên khu gần trường, an ninh, và chi phí đi lại.\n"
            "Anh/chị cho em khu vực + ngân sách, em lọc vài căn phù hợp nha."
        ),
    },
    {
        "topic": "scam_awareness",
        "keywords": (
            "lừa đảo", "lua dao", "nhận biết tin", "nhan biet tin", "tin giả", "tin gia",
            "bị lừa", "bi lua",
        ),
        "answer": (
            "Dạ anh/chị cẩn thận các dấu hiệu sau ạ:\n"
            "- Giá **rẻ bất thường**, ép cọc gấp trước khi xem phòng.\n"
            "- **Không cho xem thực tế**, chỉ gửi ảnh mạng hoặc đổi địa chỉ liên tục.\n"
            "- Yêu cầu **chuyển khoản cá nhân** không có hợp đồng rõ ràng.\n"
            "- Tin **chưa xác thực** hoặc thông tin mâu thuẫn (giá, địa chỉ, chủ nhà).\n"
            "Trên nhatrovn, anh/chị ưu tiên tin đã xác thực và đặt lịch xem phòng qua hệ thống nha."
        ),
    },
    {
        "topic": "deposit_refund",
        "keywords": (
            "hoàn cọc", "hoan coc", "lấy lại cọc", "lay lai coc",
            "chấm dứt hợp đồng", "cham dut hop dong", "gia hạn hợp đồng", "gia han hop dong",
        ),
        "answer": (
            "Dạ **hoàn cọc, gia hạn hay chấm dứt hợp đồng** tùy điều khoản từng chủ nhà ạ. "
            "Em không thay chủ nhà quyết định được, nhưng em khuyên anh/chị:\n"
            "- Giữ **biên bản bàn giao** và hóa đơn điện nước khi trả phòng.\n"
            "- **Báo trước** theo hợp đồng (thường 15–30 ngày).\n"
            "- Trao đổi **bằng văn bản** (Zalo/email) để có căn cứ.\n"
            "Nếu cần, anh/chị xem phòng và trao đổi trực tiếp với chủ nhà khi ký HĐ nha."
        ),
    },
]


def match_staff_faq(question: str, limit: int = 3) -> list[dict[str, str]]:
    q = (question or "").lower()
    matches: list[dict[str, str]] = []
    for item in STAFF_FAQ:
        if any(keyword in q for keyword in item["keywords"]):
            matches.append({"topic": item["topic"], "answer": item["answer"]})
    return matches[:limit]


def is_policy_question(question: str, current_state: dict[str, Any] | None = None) -> bool:
    """Câu hỏi chính sách/quy trình (cọc, pet, xem phòng...) không cần search phòng."""
    q = (question or "").lower()
    if not match_staff_faq(q):
        return False
    if re.search(r"\b[a-f0-9]{24}\b", q) or re.search(r"#[A-Za-z0-9][A-Za-z0-9_-]{1,40}", q):
        return False
    normalized = _norm_policy(q)
    if current_state and (
        current_state.get("current_room_id")
        or current_state.get("last_result_ids")
        or current_state.get("selected_room_ids")
    ):
        room_refs = (
            "phong nay", "phong do", "can nay", "tin nay", "can ho nay",
            "nha nay", "muc nay", "dang xem",
        )
        cost_hints = (
            "chi phi", "dien nuoc", "gia dien", "gia nuoc", "tien dien",
            "tien nuoc", "tien coc", "phi dien", "phi nuoc", "phi quan ly",
        )
        if any(ref in normalized for ref in room_refs) or any(hint in normalized for hint in cost_hints):
            return False
    search_signals = (
        "tim phong", "tim nha", "can phong", "phong tro", "nha tro",
        "muon thue", "can thue", "thue phong", "co phong", "con phong",
    )
    return not any(signal in normalized for signal in search_signals)


def _norm_policy(text: str) -> str:
    import re
    import unicodedata

    def strip_accents(value: str) -> str:
        return "".join(
            ch for ch in unicodedata.normalize("NFD", value)
            if unicodedata.category(ch) != "Mn"
        )

    return re.sub(r"\s+", " ", strip_accents(text).lower().replace("đ", "d")).strip()


def staff_cta_line() -> str:
    return STAFF_FOLLOW_UPS[-1]
