from __future__ import absolute_import, division

import os, sys, time
from glob import glob

import numpy as np
from scipy import signal
from scipy import ndimage

import tensorflow as tf
from osgeo import gdal, ogr, osr, gdal_array
import datetime
import cPickle as pickle
sys.setrecursionlimit(50000)


def read_landsat_metadata(fn):
    with open(fn) as infile:
        d = read_metadata_group(infile)
    return d

def read_metadata_group(infile):
    d = dict()
    for line in infile:
        x = [x.strip() for x in line.split('=')]
        if len(x)==1: continue
        k = x[0]
        v = x[1]
        if k == 'GROUP':
            d[v] = read_metadata_group(infile)
        if k == 'END_GROUP':
            return d
        else:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v.strip('"')
    return d


def toa_correction(data, fn):
    # Valid for OLI-TIRS only

    base = os.path.basename(fn)
    metafn = "{0}/{1}_MTL.txt".format(fn,os.path.basename(fn))
    M = read_landsat_metadata(metafn)['L1_METADATA_FILE']

    cos_z = np.sin(np.radians(M['IMAGE_ATTRIBUTES']['SUN_ELEVATION']))
    for b in range(8):
        gain = M['RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_{}'.format(b+1)]
        off = M['RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_{}'.format(b+1)]
        data[b,:,:] = (data[b,:,:] * gain + off) / cos_z


    k1 = M['TIRS_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_10']
    k2 = M['TIRS_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_10']
    data[8,:,:] = k2 / np.log(1+ k1/(data[8,:,:])) / 1e5

    k1 = M['TIRS_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_11']
    k2 = M['TIRS_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_11']
    data[9,:,:] = k2 / np.log(1+ k1/(data[9,:,:])) / 1e5
    data[np.isnan(data)]=0
    return data


def read_data(fn):

    data = []
    for i in [1,2,3,4,5,6,7,9,10,11]:
        ds = gdal.Open("{0}/{1}_B{2}.TIF".format(fn, os.path.basename(fn),i))
        data.append( ds.ReadAsArray().astype(np.float32))
    data = np.array(data)

    b,x,y = data.shape
    x0 = int((x-4000)/2)
    y0 = int((y-4000)/2)
    data = data[:,x0:x0+4000, y0:y0+4000]

    P = data[[5, 4, 3],:,:]
    P = 255*(P*2e-5-0.075)
    P[0,:] *= 1.2
    P[1,:] *= 1.1
    P[2,:] *= 1.2
    P[P<0] = 0;
    P[P>255] = 255;
    gdal_array.SaveArray(P.astype('uint8'), "{0}_photo.png".format(fn.replace('d/', 'd/photos/')), format = "PNG", prototype = ds)


    data = toa_correction(data, fn)
    data = np.transpose(data, [1, 2, 0])
    data[:,:,8] += data[:,:,9]
    data = data[:,:,:9]

    label_file = "{0}/{1}_fixedmask.img".format(fn, os.path.basename(fn))
    labels = gdal.Open(label_file).ReadAsArray().astype(np.uint8)

    labels[labels==64] = 1
    labels[labels==128] = 4;
    labels[labels==192] = 5;
    labels[labels==255] = 5;

    onehot = (np.arange(6) == labels[...,None]).astype(np.uint8)

    onehot = onehot[x0:x0+4000, y0:y0+4000,:]


    return onehot, data


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def main():

    writer = None
    files = glob('ccavd/LC*')

    for i, fn in zip(range(len(files)),files):
        print fn


        labels, data = read_data(fn)


        if i % 4 == 0:
            if writer: writer.close()
            filename = './tfrecords/oli/{0}-{1:02d}.tfrecord'.format('ccavd',int(i/4))
            writer = tf.python_io.TFRecordWriter(filename)

            example = tf.train.Example(features=tf.train.Features(feature={
                'size': _int64_feature(data.shape[1]),
                'data': _bytes_feature(data.tostring()),
                'data_type':_bytes_feature(str(data.dtype)),
                'labels': _bytes_feature(labels.tostring()),
                'labels_type':_bytes_feature(str(labels.dtype))
            }))

        writer.write(example.SerializeToString())

main()
