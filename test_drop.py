import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))
from room_assistant.workflow import _get_room_repository, startup
from room_assistant.repository import room_matches_constraints
import asyncio

async def main():
    await startup()
    repo = _get_room_repository()
    
    constraints = {"location": {"districts": ["Quận 5"]}}
    base_candidates = repo.search_by_constraints(constraints=constraints, limit=10)
    print("Found in Mongo search_by_constraints:", len(base_candidates))
    
    if not base_candidates:
        return
        
    candidate_ids = [str(c["room_id"]) for c in base_candidates[:3]]
    print("Candidate IDs:", candidate_ids)
    
    auth_rooms = repo.get_many_by_ids(candidate_ids)
    print("auth_rooms length:", len(auth_rooms))
    
    for r in auth_rooms:
        matches = room_matches_constraints(r, constraints)
        print("Room", r.get("room_id"), "matches:", matches, "District:", r.get("district"))

asyncio.run(main())
