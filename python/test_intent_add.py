import asyncio
from room_assistant.intent import parse_intent_and_constraint_patch, _extract_budget, _norm

query = "nới ngân sách thêm 1 triệu"
norm = _norm(query)
ops = []
_extract_budget(query, norm, ops)
print("Extracted budget ops:", ops)
