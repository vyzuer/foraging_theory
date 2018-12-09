from pymongo import MongoClient
import numpy as np
import sys
import pandas as pd
import os
from pprint import pprint
import datetime
from dateutil import parser
import subprocess
import time
import cv2
import collections as coll
import numpy as np
from scipy import ndimage as ndi
import time
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import argparse
import PIL
from scipy import ndimage
from skimage import morphology, measure
from skimage.segmentation import relabel_sequential
import scipy.misc
import skimage
from skimage.transform import resize
import threading
import bson.binary as bbin
import pickle
import socket

# this is required for avoiding plot issues
import matplotlib.pyplot as plt

_DEBUG = False
__DEBUG = True

caffe_root = '/home/vyzuer/work/caffe/'
sys.path.insert(0, caffe_root + 'python')

client = None
col_seg_fv = None
dump_base = None
base_dir = None
net = None
transformer = None
# batch size
batch_size = 64
layer_name = 'fc7'

# slic parameters
img_w = 320
img_h = 240
num_segments = 64
gslic_object = None
min_area = 0.0

# add the code package to the path
sys.path.append('/home/vyzuer/work/code/foraging_theory/')
sys.path.append('/home/vyzuer/work/tools/lib/')

import caffe
import libgslic as lgs

import common.globals as gv

class dump_thread(threading.Thread):
    def __init__(self, photo_id, image, objects, segments, seg_dump_base):
        threading.Thread.__init__(self)
        self.pid = photo_id
        self.image = image
        self.objects = objects
        self.segments = segments
        self.seg_dump_base = seg_dump_base
    def run(self):
        dump_image_segments(self.pid, self.image, self.objects, self.segments, self.seg_dump_base)

class myThread(threading.Thread):
    def __init__(self, tid, image):
        threading.Thread.__init__(self)
        self.tid = tid
        self.image = image
    def run(self):
        transform_image(self.tid, self.image)


def dump_image_segments(pid, image, objects, segments, seg_dump_base):

    t0 = 0
    if _DEBUG:
        t0 = time.time()

    sub_dir = str(pid)[0:3]

    spath = seg_dump_base + str(sub_dir) + '/' + str(pid) 
    if not os.path.exists(spath):
        os.makedirs(spath)

    """ pyplot is not thread safe
    # show the output of SLIC
    if _DEBUG:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        f_name = spath + '/image.jpg'
        plt.savefig(f_name)

    """

    # pick only top 64 segments
    i = 0
    for obj in objects:
        _id = obj.label

        if obj.area <= min_area:
            continue

        min_row, min_col, max_row, max_col = obj.bbox

        segment_img = image[min_row:max_row, min_col:max_col,:]
        segment_map = segments[min_row:max_row, min_col:max_col]
        segment_copy = np.copy(segment_img)
        mask = segment_map != _id
        segment_copy[mask,:] = [255,255,255]

        fname = spath + '/' + str(i) + '.jpg'
        scipy.misc.imsave(fname, cv2.cvtColor(segment_img, cv2.COLOR_BGR2RGB))
        
        i += 1


def load_globals(poi):
    global client
    global dump_base
    global base_dir
    global col_seg_fv

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')
        
    db = client.foraging_imgseg_fv
    col_seg_fv = db[poi]

    base_dir = gv.__dataset_path + poi
    dump_base = gv.__base_dir + poi


def _init_slic():
    global gslic_object

    size_seg = 20
    num_iter = 5
    weight = 1.5
    connectivity = True
    dump_seg = False
    
    gslic_object = lgs.slic_gpu(img_w, img_h, num_segments, size_seg, num_iter, weight, lgs.color.CIELAB, lgs.smethod.GIVEN_NUM, connectivity, dump_seg)


def _init_caffe():
    global net
    global transformer

    # caffe.set_mode_cpu()
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    model_def = caffe_root + 'models/places365/deploy_alexnet_places365.prototxt'
    model_weights = caffe_root + 'models/places365/alexnet_places365.caffemodel'
    
    mean_file = caffe_root + 'models/places365/places365CNN_mean.binaryproto'
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file , 'rb').read()
    blob.ParseFromString(data)
    mu = np.array(caffe.io.blobproto_to_array(blob))[0]
    
    # mu = np.load(caffe_root + 'models/places365/places365CNN_mean.binaryproto')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)

    t0 = time.time()
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    
    
    print 'model loaded'
    print 'load time : ', time.time() - t0
    
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    # mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    
    t0 = time.time()
    
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    
    # transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    # transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    
    
    print 'preprocessing time : ', time.time() - t0

def _init(poi):
    # load global variables
    load_globals(poi)

    # initialize slic object
    _init_slic()

    # initialize caffe network
    _init_caffe()

def _load_image(img_src):
    t0 = 0
    if _DEBUG:
        t0 = time.time()
    # image = caffe.io.load_image(img_src)
    image = cv2.imread(img_src)
    if _DEBUG:
        print 'image load time: ', time.time() - t0
        print image.shape

    # h, w = image.shape[:2]

    # if _DEBUG:
    #     t0 = time.time()

    # rotate image if portrait mode
    # if w < h:
        # print 'rotating image'
        # image = scipy.misc.imrotate(image, 90.0)
        # M = cv2.getRotationMatrix2D((w/2,h/2),90,3)
        # image = cv2.warpAffine(image,M,(w,h))

    # mage = scipy.misc.imresize(image, (img_h, img_w))
    # image = cv2.resize(image, (img_w, img_h))
    # image = resize(image, (img_h, img_w, 3))

    # if _DEBUG:
    #     print 'image resize time: ', time.time() - t0

    return image
 

def _perform_segmentation(image):
    t0 = 0
    h, w = image.shape[:2]

    if _DEBUG:
        t0 = time.time()
    labels = gslic_object.gslic(image)
    if _DEBUG:
        print 'slic run time: ', time.time() - t0

    if _DEBUG:
        t0 = time.time()
    labels = np.array(labels)
    segments = np.reshape(labels, (img_h, img_w))

    # resize to original image size
    # segments = resize(segments, (h, w), preserve_range=True)
    segments = ndi.interpolation.zoom(segments, (1.0*h/img_h, 1.0*w/img_w), order=0, mode='nearest')
    if _DEBUG:
        print segments.shape
        print 'reshape run time: ', time.time() - t0

    segments += 1

    if _DEBUG:
        print "Number of SLIC segments = ", len(np.unique(segments))
        print np.unique(segments)
    
    # if _DEBUG:
    #     t0 = time.time()
    # # Find connected components of the segmentation
    # # the segments may not be connected afte slic
    # dummy_map = measure.label(segments, connectivity=1)
    # num_of_segments = len(np.unique(dummy_map))
    # if _DEBUG:
    #     print "Number of connected components = ", num_of_segments
    #     print 'connected components run time: ', time.time() - t0
    # # print np.unique(dummy_map)

    # if _DEBUG:
    #     t0 = time.time()
    # # remove small segments
    # morphology.remove_small_objects(dummy_map, min_size=100, connectivity=1, in_place=True)
    # if _DEBUG:
    #     print 'remove small objects run time: ', time.time() - t0
    #     print np.unique(dummy_map)

    # if _DEBUG:
    #     t0 = time.time()
    # segments, fwd, inv = relabel_sequential(dummy_map, offset=1)
    # if _DEBUG:
    #     print 'relabel run time: ', time.time() - t0
    #     print np.unique(segments)

    return segments

def get_data():

    # the database for poi
    dataset_path = base_dir + '/poi_info.pkl'

    print dataset_path
    # first check if data is present 
    assert os.path.exists(dataset_path)

    data = pd.read_pickle(dataset_path)

    return data



def _extract_patches(image, segments):
    global min_area

    t0 = 0

    if _DEBUG:
        t0 = time.time()
    segs = measure.regionprops(segments)

    if _DEBUG:
        print 'regionprops run time: ', time.time() - t0
        print 'number of objects: ', len(segs)

    if _DEBUG:
        t0 = time.time()

    n_segs = len(segs)
    # if there are more the 64 segments sort them and keep 
    # only top 64
    if n_segs > num_segments:
        # keep only top 64 segments
        area_list = [s.area for s in segs]
        area_list.sort(reverse=True)
        # print area_list
        min_area = area_list[num_segments]
    if _DEBUG:
        print 'minimum area: ', min_area
        print 'sorting run time: ', time.time() - t0

    return segs

def transform_image(idx, img):
    t_img = transformer.preprocess('data', img)

    net.blobs['data'].data[idx, ...] = t_img


def _extract_features(image, segments):
    # extract features of patches in batch
    # and store them in mongodb
    t0 = 0
    if _DEBUG:
        t0 = time.time()

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(len(segments),        # batch size
                              3,         # 3-channel (BGR) images
                              227, 227)  # image size is 227x227
    img_seg_list = []
    for obj in segments:

        if obj.area <= min_area:
            continue

        min_row, min_col, max_row, max_col = obj.bbox

        seg_img = image[min_row:max_row, min_col:max_col,:]
        # seg_img = np.copy(seg_img)
        img_seg_list.append(seg_img)

    if _DEBUG:
        print 'extract segments time: ', time.time() - t0
    
    if _DEBUG:
        t0 = time.time()

    threads = []
    i = 0
    for img in img_seg_list:
        thread = myThread(i, img)
        threads.append(thread)
        thread.start()
        i += 1

    if _DEBUG:
        print 'thread creation time: ', time.time() - t0

    if _DEBUG:
        t0 = time.time()
    # wait for all the threads to finish
    for t in threads:
        t.join()
    if _DEBUG:
        print 'thread wait time: ', time.time() - t0

    # net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', x), img_seg_list)
    
    if _DEBUG:
        t0 = time.time()
    # perform forward pass on the network
    output = net.forward()

    if _DEBUG:
        print 'net forward time: ', time.time() - t0


    if _DEBUG:
        t0 = time.time()
    fv = net.blobs[layer_name].data
    if _DEBUG:
        print 'feature extraction time: ', time.time() - t0

    return fv


def _store_features(fv, photo_id):
    # store the extracted features

    t0 = 0
    if _DEBUG:
        t0 = time.time()
    num_segments = fv.shape[0]
    bin_fv = bbin.Binary(pickle.dumps(fv, protocol=2), subtype=128)
    col_seg_fv.insert_one({'Photo_id': photo_id,
                           'n_segments': num_segments,
                           'features': bin_fv})
    if _DEBUG:
        print 'feature storing time: ', time.time() - t0

    """ loading the binary dump
        fv_1 = pickle.loads(bin_fv)
        fv_2 = pickle.loads(col_seg_fv.find_one()['features'])

        np.savetxt('fv_0.txt', fv, fmt='%.6f')
        np.savetxt('fv_1.txt', fv_1, fmt='%.6f')
        np.savetxt('fv_2.txt', fv_2, fmt='%.6f')
    """


def _dump_segments(pid, image, objects, segments):
    seg_dump_base = dump_base + '/image_segments/'

    if not os.path.exists(seg_dump_base):
        os.makedirs(seg_dump_base)

    thread = dump_thread(pid, image, objects, segments, seg_dump_base)
    thread.start()

def master():
    # load the data
    data = get_data()

    photo_ids = data.iloc[:,6].values

    # base path of image source
    image_base = base_dir + '/images_500/'

    # check if features already extracted

    feature_flag = dump_base + '/.seg_features'
    if os.path.exists(feature_flag):
        print 'Feature extraction already done...'
        exit(0)

    # drop this collection before populating
    col_seg_fv.drop()

    # iterate over each of the images
    for photo_id in photo_ids:
        t0 = 0

        if __DEBUG:
            t0 = time.time()

        img_src = image_base + str(photo_id) + '.jpg'

        if __DEBUG:
            print img_src

        # assert os.path.exists(img_src)

        image = None
        # load image
        if os.path.exists(img_src):
            image = _load_image(img_src)

        # if this is a video
        if image is None:
            col_seg_fv.insert_one({'Photo_id': photo_id,
                           'n_segments': 0,
                           'features': None})
            continue

        # Perform segmentation
        segments = _perform_segmentation(image)

        # extract patches
        objects = _extract_patches(image, segments)

        # extract features for all the patches
        fv = _extract_features(image, objects)

        # store the features in mongodb
        _store_features(fv, photo_id)

        # dump image patches
        _dump_segments(photo_id, image, objects, segments)
        
        if __DEBUG:
            print 'total processing time per image : ', time.time() - t0

    open(feature_flag, 'a').close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_name"

    poi = str(sys.argv[1])

    _init(poi)

    master()

