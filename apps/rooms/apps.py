import os
import sys
from django.apps import AppConfig
from django.conf import settings

class RoomsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.rooms'

    def ready(self):
        # Đảm bảo thư mục 'python' nằm trong sys.path để import code AI
        python_path = os.path.join(settings.BASE_DIR, 'python')
        if python_path not in sys.path:
            sys.path.insert(0, python_path)
            
        # Chỉ tự động tải model khi chạy server (không tải khi chạy makemigrations, shell, v.v.)
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv or 'daphne' in sys.argv:
            # Ngăn chặn việc tải model 2 lần khi dùng runserver (do cơ chế autoreload của Django)
            if os.environ.get('RUN_MAIN', None) == 'true' or 'runserver' not in sys.argv:
                import threading
                try:
                    from agents.model_registry import warmup
                    # Chạy quá trình tải model ở một luồng (thread) nền để không làm chậm lúc khởi động server
                    t = threading.Thread(target=warmup, daemon=True)
                    t.start()
                except Exception as e:
                    print(f"Không thể pre-load AI models: {e}")
