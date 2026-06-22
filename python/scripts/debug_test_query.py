import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import create_listing_repository
from room_assistant.schemas import ALLOWED_OPERATION_PATHS

async def main():
    repo = create_listing_repository()
    
    # Text to test
    texts = [
        "Tìm giúp mình studio có gác quận 1 giá dưới 5 tr thang 3.",
        "Phòng trọ quận 8 giá dưới 5 triệu",
        "Có phòng giường tầng nào nhận xe điện và giờ tự do không?"
    ]
    
    for text in texts:
        print(f"\n--- USER: {text}")
        res = parse_intent_and_constraint_patch(text)
        print("Intent:", res["intent"])
        
        # Build constraints
        constraints = {}
        for op in res["operations"]:
            path = op["path"]
            val = op.get("value")
            
            if path not in ALLOWED_OPERATION_PATHS:
                continue
                
            if op["op"] == "set":
                if "." in path:
                    parent, child = path.split(".")
                    if parent not in constraints:
                        constraints[parent] = {}
                    constraints[parent][child] = val
                else:
                    constraints[path] = val
            elif op["op"] == "append":
                if "." in path:
                    parent, child = path.split(".")
                    if parent not in constraints:
                        constraints[parent] = {}
                    if child not in constraints[parent]:
                        constraints[parent][child] = []
                    constraints[parent][child].append(val)
                else:
                    if path not in constraints:
                        constraints[path] = []
                    constraints[path].append(val)

        print("Constraints:", constraints)
        
        # Test search constraints
        results = repo.search_by_constraints(constraints, limit=5)
        print(f"Found {len(results)} from constraints.")

        # Test search metadata
        from retrieval.metadata_search import extract_metadata_signals
        from room_assistant.repository import listing_matches_constraints
        
        signals = extract_metadata_signals(text)
        print("Metadata signals:", signals)
        meta_results = repo.search_by_metadata(text, limit=10)
        print(f"Found {len(meta_results)} from metadata.")
        
        meta_results = [item for item in meta_results if listing_matches_constraints(item, constraints)]
        print(f"Kept {len(meta_results)} metadata items after constraint filter.")
        
        final_results = results + meta_results
        
        if final_results:
            for r in final_results[:2]:
                print(f" - {r.get('id')}: {r.get('category')} / {r.get('rent_price')} / {r.get('district')} / Amenities: {r.get('amenities')}")

if __name__ == "__main__":
    asyncio.run(main())
