import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from room_assistant.intent import parse_intent_and_constraint_patch, _norm

text = "chỗ để xe điện"
print("Norm:", _norm(text))
res = parse_intent_and_constraint_patch(text)
print("Ops:", res["operations"])
