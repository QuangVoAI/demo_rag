import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from room_assistant.intent import parse_intent_and_constraint_patch

def test_extract_room_type():
    res = parse_intent_and_constraint_patch("tìm studio có gác")
    ops = res["operations"]
    
    categories = [op["value"] for op in ops if op["path"] == "categories"]
    assert "studio" in categories

    amenities = [op["value"] for op in ops if op["path"] == "amenities_required"]
    assert "mezzanine" in amenities

def test_extract_ev_charging_vs_parking():
    # ev_charging (tiện ích phòng)
    res1 = parse_intent_and_constraint_patch("phòng có sạc xe điện không")
    ops1 = res1["operations"]
    am1 = [op["value"] for op in ops1 if op["path"] == "amenities_required"]
    assert "ev_charging" in am1

    # electric_bike (phương tiện cá nhân để gửi)
    res2 = parse_intent_and_constraint_patch("chỗ để xe điện")
    ops2 = res2["operations"]
    veh2 = [op["value"] for op in ops2 if op["path"] == "vehicles"]
    assert "electric_bike" in veh2
    
    # "nhận xe điện" (tiện ích phòng)
    res3 = parse_intent_and_constraint_patch("phòng có nhận xe điện không")
    ops3 = res3["operations"]
    am3 = [op["value"] for op in ops3 if op["path"] == "amenities_required"]
    assert "ev_charging" in am3

def test_extract_free_hours():
    res = parse_intent_and_constraint_patch("tìm phòng giờ tự do")
    ops = res["operations"]
    am = [op["value"] for op in ops if op["path"] == "amenities_required"]
    assert "free_hours" in am

def test_compact_million():
    res = parse_intent_and_constraint_patch("phòng dưới 3tr5 ở quận 1")
    ops = res["operations"]
    budget = next((op["value"] for op in ops if op["path"] == "budget.max"), None)
    assert budget == 3500000

    # Negative lookahead for "thang"
    res2 = parse_intent_and_constraint_patch("phòng dưới 1 tr 3 thang")
    ops2 = res2["operations"]
    budget2 = next((op["value"] for op in ops2 if op["path"] == "budget.max"), None)
    # Sẽ bắt 1tr, bỏ qua thang 3 vì có negative lookahead trong compact_million
    assert budget2 == 1000000

def test_chdv_2pn():
    res = parse_intent_and_constraint_patch("muốn thuê CHDV 2 phòng ngủ")
    ops = res["operations"]
    cats = [op["value"] for op in ops if op["path"] == "categories"]
    assert "chdv" in cats
    assert "2pn" in cats
