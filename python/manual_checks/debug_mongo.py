import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))
from config import MONGODB_URI, MONGODB_DATABASE, MONGODB_ROOMS_COLLECTION
from pymongo import MongoClient
from room_assistant.repository import build_mongo_query

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]
col = db[MONGODB_ROOMS_COLLECTION]

constraints = {"location": {"districts": ["Quận 5"]}}
query = build_mongo_query(constraints)

docs = list(col.find(query))
print("Total matches for Quận 5 query:", len(docs))

q5_actual = [d for d in docs if "quận 5" in str(d.get("metadata", {}).get("district_name")).lower() or "quận 5" in str(d.get("district")).lower()]
print("Actual Quận 5 rooms in matches:", len(q5_actual))

