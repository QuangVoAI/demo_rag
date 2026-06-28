import asyncio
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'python'))

from room_assistant.workflow import _get_room_repository, _get_semantic_index, startup
from room_assistant.tools import search_rooms, ToolExecutionContext

async def main():
    await startup()
    q = "tìm cho tôi nhà quận 5"
    constraints = {"location": {"districts": ["Quận 5"]}}
    
    repo = _get_room_repository()
    idx = _get_semantic_index()
    ctx = ToolExecutionContext(repository=repo, semantic_index=idx)
    
    rooms = search_rooms({"query_text": q, "constraints": constraints, "top_k": 5}, ctx)
    print("Found rooms:", len(rooms))
    if rooms:
        print("First room title:", rooms[0].get("title"))
    else:
        print("Trace:", ctx.retrieval_trace)

asyncio.run(main())
