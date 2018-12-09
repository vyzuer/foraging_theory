from pymongo import MongoClient
import numpy as np
import sys
import os
import urllib
import logging

client = None
db_foraging = None

def load_globals():
    global client
    global db_foraging

    client = MongoClient()

    db_foraging = client.ysr_foraging_db

def download_images(poi, base_dir):

    # load the dataset
    col = db_foraging[poi]

    print col.count()

    cursor = col.find(no_cursor_timeout=True)

    r_path = poi + '/images_500/'

    for doc in cursor:
        img_name = str(doc['Photo_id']) + '.jpg'
        img_r_path = r_path + img_name
        img_path = base_dir + img_r_path

        print img_path

        if os.path.exists(img_path):
            continue

        url = doc['Photo_download_URL']
        # prefix, ext = os.path.splitext(url0)

        # url = prefix + '_n' + ext

        print url

        # download image
        try:
            urllib.urlretrieve(url, filename=img_path)
        except Exception, e:
            logging.warn("error downloading %s: %s" % (url, e))


    cursor.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python create_poi_db.py location_name base_directory"

    poi = str(sys.argv[1])
    base_dir = str(sys.argv[2])

    load_globals()

    poi_dir = base_dir + poi
    if not os.path.exists(poi_dir):
        os.makedirs(poi_dir)

    media_path = poi_dir + '/images_500/'
    if not os.path.exists(media_path):
        os.makedirs(media_path)

    download_images(poi, base_dir)

