import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))
from room_assistant.workflow import _get_room_repository, startup
import asyncio

async def main():
    await startup()
    repo = _get_room_repository()
    rooms = repo.search_by_constraints({"budget": {"max": 999999999}}, limit=1000)
    districts = {}
    for r in rooms:
        d = r.get("district", "Unknown")
        districts[d] = districts.get(d, 0) + 1
    
    print("Available districts:")
    for d, c in sorted(districts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {d}: {c}")

asyncio.run(main())
