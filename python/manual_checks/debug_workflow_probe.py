import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))
from room_assistant.workflow import _get_room_repository, _get_semantic_index, startup
import asyncio

async def main():
    await startup()
    repo = _get_room_repository()
    idx = _get_semantic_index()
    
    constraints = {"location": {"districts": ["Quận 5"]}}
    base_candidates = repo.search_by_constraints(constraints=constraints, limit=10)
    print("Found in Mongo:", len(base_candidates))
    
    if not base_candidates:
        return
        
    candidate_ids = [str(c["room_id"]) for c in base_candidates]
    print("Candidate IDs:", candidate_ids[:3])
    
    from retrieval.qdrant_client import QdrantWrapper
    qdrant = QdrantWrapper()
    q_filter = qdrant._room_filter(candidate_ids, {})
    
    # Try empty query_vector
    import numpy as np
    dummy_vec = np.random.rand(1024).astype(np.float32)
    
    res = qdrant.search_dense(dummy_vec, top_k=5, query_filter=q_filter)
    print("Qdrant results count:", len(res))
    if res:
        print("First result:", res[0])

asyncio.run(main())
