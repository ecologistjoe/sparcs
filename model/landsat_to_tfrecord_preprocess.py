from __future__ import absolute_import, division

import os, sys, time
from glob import glob

import numpy as np
from scipy import signal
from scipy import ndimage

import tensorflow as tf
from osgeo import gdal, ogr, osr

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
    metafn = fn.replace(base, base.split('_')[0] + '_mtl.txt')
    M = read_landsat_metadata(metafn)['L1_METADATA_FILE']

    cos_z = np.sin(np.radians(M['IMAGE_ATTRIBUTES']['SUN_ELEVATION']))
    for b in range(8):
        gain = M['RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_{}'.format(b+1)]
        off = M['RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_{}'.format(b+1)]
        data[b,:,:] = (data[b,:,:] * gain + off) / cos_z


    k1 = M['TIRS_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_10']
    k2 = M['TIRS_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_10']
    data[8,:,:] = k2 / np.log(1+ k1/data[8,:,:]) / 1e5

    k1 = M['TIRS_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_11']
    k2 = M['TIRS_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_11']
    data[9,:,:] = k2 / np.log(1+ k1/data[9,:,:]) / 1e5
    return data


def read_data(fn):

    data = gdal.Open(fn).ReadAsArray().astype(np.float32)
    data = toa_correction(data, fn)
    data = np.transpose(data, [1, 2, 0])
    data[:,:,8] += data[:,:,9]
    data = data[:,:,:9]

    label_file = fn.replace('data.tif', 'mask.png')
    labels = gdal.Open(label_file).ReadAsArray().astype(np.uint8)
    shadow_water = labels==1
    labels[labels==0] = 1

    onehot = (np.arange(7) == labels[...,None]).astype(np.uint8)
    onehot[shadow_water,2]=1

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
    samples = {#'training': glob('oli/LC*data.tif') #+
                # [ 'oli/LC81370452014073LGN00_23_data.tif',
                #   'oli/LC81370452014073LGN00_23_data.tif',
                #   'oli/LC81370452014073LGN00_23_data.tif',
                #   'oli/LC81370452014073LGN00_23_data.tif',
                #   'oli/LC81140272013261LGN00_32_data.tif',
                #   'oli/LC81140272013261LGN00_32_data.tif',
                #   'oli/LC81550232014135LGN00_26_data.tif',
                #   'oli/LC80340412013132LGN01_20_data.tif' ],

                'testing': glob('oli/testing/LC*data.tif')
                }


    for s in samples:
        files = samples[s]
        labels, data = ([],[])

        #Read all of the files
        for i, fn in zip(range(len(files)),files):
            #print fn
            l, d = read_data(fn)
            l = np.pad(l, ((12,12), (12,12), (0,0)), 'constant')
            d = np.pad(d, ((12,12), (12,12), (0,0)), 'constant')
            labels.append(l)
            data.append(d)

        window_size = 256

        i=0
        filename = './tfrecords_pre/oli/{0}-{1:02d}.tfrecord'.format(s,int(i))
        writer = tf.python_io.TFRecordWriter(filename)


        #for i in range(16,100):
        for x0 in range(0,1024,256):
          for y0 in range(0,1024,256):
            i+=1
            print i, x0,y0

            for j in range(len(labels)):

                sz = labels[j].shape[0]
                #x0 = np.random.randint(sz-window_size)
                #y0 = np.random.randint(sz-window_size)

                X = data[j][x0:x0+window_size, y0:y0+window_size,:]
                L = labels[j][x0:x0+window_size, y0:y0+window_size,:]

                example = tf.train.Example(features=tf.train.Features(feature={
                    'size': _int64_feature(X.shape[1]),
                    'data': _bytes_feature(X.tostring()),
                    'data_type':_bytes_feature(str(X.dtype)),
                    'labels': _bytes_feature(L.tostring()),
                    'labels_type':_bytes_feature(str(L.dtype))
                }))

                writer.write(example.SerializeToString())
        writer.close()

main()
