import sys, os
import json
import django

sys.path.append(os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.rooms.services import get_rooms_collection
rooms_col = get_rooms_collection()
doc = rooms_col.find_one({"metadata.status_code": "0"})
print(json.dumps(doc, default=str, indent=2))
