import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.graph import run_streaming
from utils.console import console

async def main():
    session_id = "test_backend_v2_session_1"
    history = []
    
    questions = [
        "Tìm giúp mình studio có gác quận 1 giá dưới 5 tr thang 3.",
        "Phòng có sạc xe điện không?",
        "À mình muốn thêm giờ tự do nữa.",
        "Có CHDV 1 phòng ngủ nào ở đó giá rẻ hơn xíu không?",
        "So sánh các phòng giúp mình nha."
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
        
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
