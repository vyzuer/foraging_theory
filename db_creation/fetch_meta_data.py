from pymongo import MongoClient
import numpy as np
import sys
import os
import urllib
import logging
import requests
import flickrapi

# to suppress insecureplatform warning
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

# logging.captureWarnings(True)

client = None
db_foraging = None
flickr = None

api_key = u'8404454a9d4516f167c59adf23ee4833'
api_secret = u'badcb7275642e3b5'

# api_key = u'7e541a9005ff2c4c88d7853e7a5cdeae'
# api_secret = u'a09d692332d8b72d'

# api_key = u'58aa3d2630e98953beda061c3f8fd358'
# api_secret = u'4d8d91795f879131'

def load_globals():
    global client
    global db_foraging
    global flickr

    client = MongoClient()

    db_foraging = client.ysr_foraging_db

    flickr = flickrapi.FlickrAPI(api_key, api_secret)

def fetch_metadata(poi):

    # load the dataset
    col = db_foraging[poi]

    print col.count()

    cursor = col.find(no_cursor_timeout=True)

    for doc in cursor:
        photo_id = doc['Photo_id']
        secret = doc['Photo_secret']
        print photo_id, secret

        response_format = 'parsed-json'
        # response_format = 'etree'

        photo_info = None
        try:
            photo_info = flickr.photos.getInfo(photo_id=photo_id, secret=secret, format=response_format)
            photo_favorites = flickr.photos.getFavorites(photo_id=photo_id, format=response_format)
            # photo_comments = flickr.photos.comments.getList(photo_id=photo_id, format=response_format)

            # update the document for info and favorites
            doc['photo_info'] = photo_info['photo']
            doc['favorites_info'] = photo_favorites['photo']

            col.save(doc)

            # col.update_one({'_id': doc['_id']}, {'$inc': {'photo_info': photo_info['photo'], 'favorites_info': photo_favorites['photo']}})

            # print 'number of views: ', photo_info['photo']['views']
            # print 'number of comments: ', photo_info['photo']['comments']['_content']
            # print 'number of likes: ', photo_favorites['photo']['total']

        except Exception, e:
            logging.warn("error downloading %d: %s" % (photo_id, e))

    cursor.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_name"

    poi = str(sys.argv[1])

    load_globals()

    fetch_metadata(poi)

