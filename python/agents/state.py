"""State dùng chung cho Nhatrovn Agent pipeline.

Định nghĩa NhatrovnAgentState — trạng thái luân chuyển qua các node
trong luồng xử lý mỗi lượt hội thoại.
"""
from typing import Any
from typing_extensions import TypedDict


class NhatrovnAgentState(TypedDict, total=False):
    """Trạng thái chung cho Nhatrovn Agent pipeline."""

    # --- Input người dùng ---
    session_id: str
    question: str           # Câu hỏi của người dùng
    history: list[dict]     # Lịch sử hội thoại [{role, content}, ...]

    # --- Phân tích cảm xúc người dùng ---
    user_mood: str          # "frustrated" | "urgent" | "normal"
    user_mood_score: float  # Độ tin cậy 0.0 – 1.0

    # --- Kết quả tìm kiếm phòng ---
    rooms: list[dict]       # Danh sách phòng tìm được
    room_context: str       # Context đầy đủ của phòng đang xem

    # --- Viết lại query ---
    rewritten_query: str    # Query sau khi được rewrite để tìm lại
    rewrite_count: int      # Số lần đã rewrite

    # --- Đánh giá chất lượng kết quả ---
    grade_result: str       # "GOOD" | "POOR" — chất lượng kết quả tìm kiếm

    # --- Sinh câu trả lời ---
    answer: str             # Câu trả lời cuối cùng gửi cho người dùng

    # --- Kiểm duyệt câu trả lời ---
    reviewer_approved: bool
    reviewer_issues: list[str]

    # --- Metadata ---
    agent_trace: dict
    processing_time_ms: int
    stream_callback: Any
