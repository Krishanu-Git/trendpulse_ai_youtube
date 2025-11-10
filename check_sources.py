# check_sources.py
from src.database.db_handler import MongoHandler
import pandas as pd

dbh = MongoHandler()  # use your usual constructor
try:
    coll = getattr(dbh, "collection", None)
    if coll is None:
        db = getattr(dbh, "db", None)
        # try to find a reddit collection heuristically
        if db is not None:
            for name in db.list_collection_names():
                c = db[name]
                try:
                    if c.count_documents({"subreddit": {"$exists": True}}) > 0:
                        coll = c
                        break
                except Exception:
                    continue
    if coll is None:
        print("Could not find a collection to check. Inspect MongoHandler.")
    else:
        print("Sample reddit documents count (with subreddit field):", coll.count_documents({"subreddit": {"$exists": True}}))
        sample = coll.find_one({"subreddit": {"$exists": True}})
        print("Sample doc keys:", list(sample.keys())[:20])
except Exception as e:
    print("Error while checking:", e)
