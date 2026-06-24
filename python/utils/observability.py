"""
Langfuse Observability — Khởi tạo và cấu hình Langfuse client.

Langfuse chạy 100% trên Cloud (non-blocking, async HTTP).
Không tiêu thụ VRAM, không tiêu thụ GPU.

Graceful degradation: Nếu chưa cấu hình LANGFUSE_SECRET_KEY,
hệ thống vẫn chạy bình thường mà không crash.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
from utils.console import console

_langfuse_client = None
_langfuse_available = False


def get_langfuse():
    """
    Lazy init Langfuse client.
    Trả None nếu chưa cấu hình (graceful degradation).
    """
    global _langfuse_client, _langfuse_available

    if _langfuse_client is not None:
        return _langfuse_client if _langfuse_available else None

    if not LANGFUSE_SECRET_KEY:
        console.print("[dim]  Langfuse: Not configured, tracing disabled[/]")
        _langfuse_available = False
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
        _langfuse_available = True
        console.print("[green]  Langfuse: Connected ✓[/]")
        return _langfuse_client
    except Exception as e:
        console.print(f"[yellow]  Langfuse: Init failed: {e}[/]")
        _langfuse_available = False
        return None


def flush_langfuse():
    """Flush pending events cho cả manual client (nếu có) và decorator context (v4)."""
    # Flush global observability queue (Langfuse v3/v4)
    try:
        from langfuse import get_client
        client = get_client()
        if client:
            client.flush()
    except Exception:
        pass
