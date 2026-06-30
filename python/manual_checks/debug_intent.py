import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))
from room_assistant.intent import parse_intent_async
import asyncio

async def main():
    q = "Đổi sang khu vực gần trường hơn"
    state_before = {"constraints": {"location": {"districts": ["Quận 5"]}}}
    parsed = await parse_intent_async(q, state_before)
    print("Parsed intent:", parsed)

asyncio.run(main())
