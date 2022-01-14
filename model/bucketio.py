import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.lib.io import file_io
import numpy as np
import os, sys


def copy_to_bucket(local, bucket):
    with file_io.FileIO(local, mode='rb') as infile:
        with file_io.FileIO(bucket, mode='wb+') as outfile:
            outfile.write(infile.read())


# Have to make a local copy and then move it to the bucket
def save_model_to_bucket(model, fn):
    model_fn = os.path.basename(fn)
    model.save(model_fn)
    copy_to_bucket(model_fn, fn)

def save_weights_to_bucket(model, fn):
    model_fn = os.path.basename(fn)
    model.save_weights(model_fn, save_format='h5')
    copy_to_bucket(model_fn, fn)

def load_model_from_bucket(fn):
    #load model
    bucket_fn = fn+'.h5'
    local_fn = os.path.basename(bucket_fn)
    if not os.path.exists(local_fn):
        with file_io.FileIO(bucket_fn, mode='rb') as infile:
            with file_io.FileIO(local_fn, mode='wb+') as outfile:
                outfile.write(infile.read())
    model = K.models.load_model(local_fn)

    #model = cnn_models.compile_tpu_model(model)
    #model = load_model_weights_from_bucket(model,fn)
    return model


def load_model_weights_from_bucket(model, bucket_fn):
    # load weights
    local_fn = os.path.basename(bucket_fn)
    if not os.path.exists(local_fn):
        with file_io.FileIO(bucket_fn, mode='rb') as infile:
            with file_io.FileIO(local_fn, mode='wb+') as outfile:
                outfile.write(infile.read())

    print local_fn

    model.load_weights(local_fn, by_name=True)
    return model
