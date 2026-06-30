import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))
from config import MONGODB_URI, MONGODB_DATABASE, MONGODB_ROOMS_COLLECTION
from pymongo import MongoClient

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]
col = db[MONGODB_ROOMS_COLLECTION]

docs = list(col.find({"$or": [
    {"metadata.district_name": {"$regex": "5", "$options": "i"}},
    {"metadata.district_name": {"$regex": "quận 5", "$options": "i"}},
    {"metadata.district_name": {"$regex": "q5", "$options": "i"}}
]}))

print("Total Q5 docs found:", len(docs))
if docs:
    for d in docs[:5]:
        print("Status:", d.get("metadata", {}).get("status_code"), "| District:", d.get("metadata", {}).get("district_name"), "| Available:", d.get("available"))

