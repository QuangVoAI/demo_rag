from django.apps import AppConfig
import sys
import os
from pathlib import Path

class RoomsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.rooms'

    def ready(self):
        if os.environ.get('RUN_MAIN'):
            project_root = Path(__file__).resolve().parent.parent.parent
            python_dir = project_root / 'python'
            if str(python_dir) not in sys.path:
                sys.path.append(str(python_dir))
                
            try:
                from agents.model_registry import warmup
                from agents.sentiment_analyzer import _ensure_centroids
                import threading
                
                def init_models():
                    print("\n[INFO] Bắt đầu load AI models dưới nền...")
                    warmup()
                    _ensure_centroids()
                    print("[INFO] Đã load xong tất cả models!\n")
                    
                threading.Thread(target=init_models, daemon=True).start()
            except Exception as e:
                print(f"[ERROR] Không thể load model: {e}")
