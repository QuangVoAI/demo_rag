$env:PYTHONIOENCODING = "utf-8"
& "C:\cygwin64\bin\python3.9.exe" -c @"
import sys
sys.path.insert(0, 'python')
from retrieval.metadata_search import extract_metadata_signals, metadata_lookup, score_metadata_hit
from room_assistant.repository import InMemoryRoomRepository

s = extract_metadata_signals('So sanh #A101 va #B202 o quan Binh Thanh')
assert 'A101' in s['room_id'], s
assert 'B202' in s['room_id'], s
assert any('binh thanh' in d for d in s['district']), s
print('signals', s)

s = extract_metadata_signals('Phong A101 co gi?')
assert 'A101' in s['room_id'], s
print('bare room id', s)

payload = {'room_id': 'A101', 'district': 'Binh Thanh', 'title': 'Studio Binh Thanh', 'amenities': ['air_conditioner']}
hit = score_metadata_hit(payload, {'room_id': ['A101'], 'district': ['binh thanh']})
assert hit == 1.0, hit
print('score', hit)

repo = InMemoryRoomRepository([
    {'room_id': 'A101', 'house_id': 'H1', 'title': 'Studio Binh Thanh', 'district': 'Binh Thanh', 'available': True, 'status': 'active', 'rent_price': 4500000},
    {'room_id': 'B202', 'house_id': 'H2', 'title': 'Phong Q7', 'district': 'Quan 7', 'available': True, 'status': 'active', 'rent_price': 6000000},
])
hits = metadata_lookup({'room_id': ['A101', 'X999'], 'district': []}, repo)
assert 'A101' in hits and 'X999' not in hits, hits
print('lookup OK', list(hits.keys()))
"@
