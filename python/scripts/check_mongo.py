import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pymongo import MongoClient
from config import MONGODB_URI, MONGODB_DATABASE, MONGODB_LISTINGS_COLLECTION

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]
col = db[MONGODB_LISTINGS_COLLECTION]
doc = col.find_one()
if doc:
    print("Sample district:", doc.get("location", {}).get("district", "N/A"), "Price:", doc.get("rent_price"))
else:
    print("No docs found.")
