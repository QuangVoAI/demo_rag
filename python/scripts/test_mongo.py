import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from room_assistant.repository import create_listing_repository

repo = create_listing_repository()
constraints = {
    "location": {"districts": ["quận 8"]},
    "budget": {"max": 5000000}
}
res = repo.search_by_constraints(constraints)
print(f"Found {len(res)} rooms")
if res:
    print(res[0]["listing_id"], res[0]["title"], res[0]["district"], res[0]["rent_price"])
