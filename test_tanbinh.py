import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))
from config import MONGODB_URI, MONGODB_DATABASE, MONGODB_ROOMS_COLLECTION
from pymongo import MongoClient

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]
col = db[MONGODB_ROOMS_COLLECTION]

docs = list(col.find({"metadata.district_name": "Quận Tân Bình"}).limit(5))

for d in docs:
    print("Status:", repr(d.get("metadata", {}).get("status_code")), "| District:", d.get("metadata", {}).get("district_name"))
