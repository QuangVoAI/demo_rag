"""
Shared Rich Console — Fix Windows Unicode encoding.

Tất cả module import console từ đây thay vì tạo Console() riêng.
Giải quyết lỗi UnicodeEncodeError khi in emoji trên Windows (CP1252).
"""
import sys
import os

# Fix Windows console encoding for Unicode emoji
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rich.console import Console
    console = Console(force_terminal=True, safe_box=True)
except ImportError:
    class _PlainConsole:
        def print(self, *args, **kwargs):
            print(*args)

    console = _PlainConsole()
