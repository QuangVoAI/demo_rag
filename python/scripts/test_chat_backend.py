import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.graph import run_streaming
from utils.console import console

async def main():
    import uuid
    session_id = f"test_backend_session_{uuid.uuid4().hex[:8]}"
    history = []
    
    questions = [
        "Tìm phòng trọ quận 8 giá dưới 5 triệu.",
        "À mình muốn có chỗ để xe máy nữa.",
        "Tổng chi phí dự kiến cho phòng đầu tiên là bao nhiêu nếu thuê 6 tháng?",
        "Có phòng nào tương tự ở khu vực đó không? So sánh 3 phòng giúp mình.",
        "Tóm tắt lại giúp mình điểm phù hợp nhất.",
        "Vậy nhờ bạn đặt lịch xem phòng giúp mình."
    ]
    
    for idx, q in enumerate(questions):
        console.print(f"\n[bold cyan]--- Lượt {idx + 1} ---[/]")
        console.print(f"[bold yellow]USER:[/] {q}")
        
        history.append({"role": "user", "content": q})
        
        # Gọi backend
        result = await run_streaming(question=q, history=history, session_id=session_id)
        
        answer = result.get("answer", "No answer generated")
        console.print(f"[bold green]BOT:[/] {answer}\n")
        
        history.append({"role": "assistant", "content": answer})
        
        # Nghỉ 2s giữa các request để tránh rate limit nếu có
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
