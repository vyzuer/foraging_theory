from pymongo import MongoClient
import numpy as np
import sys

client = None
db_foraging = None
db_yfcc = None

def load_globals():
    global client
    global db_foraging
    global db_yfcc

    client = MongoClient()

    db_foraging = client.ysr_foraging_db
    db_yfcc = client.ysr_db

def get_geo_coords(poi):
    col = db_foraging.pois_geo

    doc = col.find_one({"_id":poi})

    lat0 = doc['lat0']
    lat1 = doc['lat1']
    lon0 = doc['lon0']
    lon1 = doc['lon1']

    return lat0, lon0, lat1, lon1

def  create_db(poi):
    # get the geo coordinates from database
    lat0, lon0, lat1, lon1 = get_geo_coords(poi)

    # load the collection yfcc100m
    col = db_yfcc.yfcc100m

    # new subset
    sub_col = db_foraging[poi]

    # search for poi images
    cursor = col.find({"Latitude": {"$gte": lat0, "$lte": lat1}, "Longitude": {"$gte": lon0, "$lte": lon1}})

    for doc in cursor:
        sub_col.insert(doc)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python create_poi_db.py location_name"

    poi = str(sys.argv[1])

    load_globals()

    create_db(poi)

