$env:PYTHONIOENCODING = "utf-8"
& "C:\cygwin64\bin\python3.9.exe" -c @"
import sys
sys.path.insert(0, 'python')
from retrieval.metadata_search import extract_metadata_signals, metadata_lookup, score_metadata_hit

# Test 1: listing_id
s = extract_metadata_signals('So sánh #A101 và #B202 ở quận Bình Thạnh')
assert 'A101' in s['listing_id'], s
assert 'B202' in s['listing_id'], s
assert any('binh thanh' in d for d in s['district']), s
print('test1', s)

# Test 2: arxiv-like (mapped to listing domain via title)
s = extract_metadata_signals('2604.08423v1 DPG reward là dataset-level hay individual synthetic texts?')
assert '2604.08423v1' in s['arxiv_like'], s
print('test2', s)

# Test 3: bare listing id
s = extract_metadata_signals('Phòng A101 có gì?')
assert 'A101' in s['listing_id'], s
print('test3', s)

# Test 4: score
payload = {'listing_id': 'A101', 'district': 'Bình Thạnh', 'title': 'Studio Bình Thạnh', 'amenities': ['air_conditioner']}
hit = score_metadata_hit(payload, {'listing_id': ['A101'], 'district': ['binh thanh']})
assert hit == 1.0, hit
print('test4', hit)

# Test 5: lookup against InMemoryListingRepository
from room_assistant.repository import InMemoryListingRepository
repo = InMemoryListingRepository([
    {'listing_id': 'A101', 'title': 'Studio Bình Thạnh', 'district': 'Bình Thạnh', 'available': True, 'status': 'active', 'rent_price': 4500000},
    {'listing_id': 'B202', 'title': 'Phòng Q7', 'district': 'Quận 7', 'available': True, 'status': 'active', 'rent_price': 6000000},
])
hits = metadata_lookup({'listing_id': ['A101', 'X999'], 'arxiv_like': [], 'district': []}, repo)
assert 'A101' in hits and 'X999' not in hits, hits
print('test5 OK', list(hits.keys()))
print('ALL OK')
"@
