from __future__ import division

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.lib.io import file_io
import numpy as np
import cnn_models
import os
import sys
from subprocess import Popen
import socket
import png as PNG
from bucketio import *

K.backend.set_image_data_format('channels_last')
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BUCKET_PATH = 'gs://sparcsdata/oli'
IMAGES_PER_TFRECORD = 8
NUM_BANDS = 9
NUM_CLASSES = 6
PREDICT_BATCH_SIZE = 32
EXAMPLES_PER_IMAGE = 8
BATCH_SIZE = IMAGES_PER_TFRECORD * EXAMPLES_PER_IMAGE
CLASS_WEIGHTS = np.array([[1, 1, 1, 1, 0.5, 1]])


FEATURE_DESCRIPTION = {
    'size'       : tf.io.FixedLenFeature([], tf.int64),
    'data'       : tf.io.FixedLenFeature([], tf.string),
    'data_type'  : tf.io.FixedLenFeature([], tf.string),
    'labels'     : tf.io.FixedLenFeature([], tf.string),
    'labels_type': tf.io.FixedLenFeature([], tf.string),
}

SPARCS_PALETTE = np.zeros(256, 'int32')
SPARCS_PALETTE[0:8] = [0xff0000, 0x000000, 0x0000ff, 0x00ffff, 0x888888, 0xffffff, 0x888800, 0xeeffdd]
#legend = ['none', 'shadow', 'water', 'snow/ice', 'clear-sky', 'cloud', 'flood','thin']



def keras_data_generator(ds):
    G = ds.make_one_shot_iterator().get_next()
    while True:
        try:
            yield K.backend.get_session().run(G)
        except tf.errors.OutOfRangeError:
            return



def unpack_tfrecord(X):
    P = tf.parse_single_example(X, FEATURE_DESCRIPTION)

    D = tf.reshape( tf.io.decode_raw(P['data'], tf.float32),
            (P['size'], P['size'],-1))
    L = tf.reshape( tf.io.decode_raw(P['labels'], tf.uint8),
            (P['size'], P['size'],-1))


    D = tf.math.subtract(D, [[0.133, 0.128, 0.126, 0.142, 0.186, 0.144, 0.118, 0.00665, 0.466]])
    D = tf.math.multiply(D, [[5.11,  4.98,  5.32,  5.31,  5.21,  6.68,  8.67,  53.4,    2.07]])

    return D,L



def parse_example_pre(X, L,window_size, target_size):
    L_start = tf.cast(np.floor((window_size-target_size)/2), tf.int32)
    L_end = tf.cast(L_start+target_size, tf.int32)

    OUT = L[L_start:L_end, L_start:L_end,:]
    return (X,(OUT,OUT))


def parse_example(D, L, window_size, target_size, examples_per_image=20):

    window_size = tf.cast(window_size, tf.int32)
    pad = tf.cast(window_size/4, tf.int32)
    D = tf.pad(D,  ((pad,pad), (pad,pad), (0,0)))
    L = tf.pad(L,  ((pad,pad), (pad,pad), (0,0)))

    max_offset = tf.cast(tf.shape(D)[0]-window_size, tf.int32)

    offset  = tf.random.uniform((examples_per_image,2), maxval=max_offset, dtype=tf.int32)
    L_start = tf.cast((window_size-target_size)/2, tf.int32)
    L_end   = tf.cast(L_start+target_size, tf.int32)

    IN = tf.map_fn(lambda x: D[x[0]:x[0]+window_size, x[1]:x[1]+window_size,:], offset, dtype=tf.float32)
    OUT = tf.map_fn(lambda x: L[x[0]+L_start:x[0]+L_end, x[1]+L_start:x[1]+L_end,:], offset, dtype=tf.uint8)

    #IN = tf.cast(IN, tf.bfloat16)

    return (IN, (OUT, OUT))



def make_training_dataset(data_path, window_size, target_size):
    files = tf.data.Dataset.list_files(data_path)
    ds = files.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    ds = ds.apply(tf.data.experimental.parallel_interleave(
        lambda x: tf.data.Dataset.from_tensors(unpack_tfrecord(x))
        , cycle_length=4))
    #ds = ds.cache()

    ds = ds.apply(tf.data.experimental.parallel_interleave(
        lambda D,L: tf.data.Dataset.from_tensor_slices(
            parse_example(D,L, window_size, target_size, EXAMPLES_PER_IMAGE))
        , cycle_length=4))

    ds = ds.shuffle(buffer_size=100)
    ds = ds.repeat()

    ds = ds.batch(batch_size=BATCH_SIZE)
    ds = ds.prefetch(1)

    return ds


def make_prediction_dataset(data_path):
    files = tf.data.Dataset.list_files(data_path)
    ts = files.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=2))
    ts = ts.flat_map(lambda x: tf.data.Dataset.from_tensors(unpack_tfrecord(x)))

    return ts

def confusion_matrix(true, pred, N):
    cells = N*N
    C, bins = np.histogram(pred*N+true, cells, (0,cells))
    C = C.reshape((N,N))

    # Accuracy and Recall stats, as percents
    d = 100.0 * np.diag(C)
    O = np.zeros((C.shape[0]+1, C.shape[1]+1), np.uint32)
    O[:-1,:-1] = C
    O[-1,:-1] = np.round(d / (1e-6+np.sum(C, axis=0)))
    O[:-1,-1] = np.round(d / (1e-6+np.sum(C, axis=1)))
    O[-1,-1] = np.round(np.sum(d) / (np.sum(C)))

    return O


def log_predictions(run_name, predicted, labels, i):
    outpath = "{0}/output/{1}".format(BUCKET_PATH, run_name)

    P = np.argmax(predicted,-1)
    L = np.argmax(labels,-1)

    fn = "{0}/confusion_{1}r.txt".format(outpath,i)
    confusion = confusion_matrix(L.ravel(), P.ravel(), NUM_CLASSES)
    print "\n",confusion
    with file_io.FileIO(fn, mode='wb+') as out:
        np.savetxt(out, confusion)

    fn = "{0}/predicted_{1}r.png".format(outpath,i)
    with file_io.FileIO(fn, mode='wb+') as out:
        out.write(PNG.build(P, SPARCS_PALETTE))

    fn = "{0}/true_{1}.png".format(outpath,i)
    with file_io.FileIO(fn, mode='wb+') as out:
        out.write(PNG.build(L, SPARCS_PALETTE))



def prediction_windows(A, window_size, hop, stride):

    shp = A.shape
    gap = int((window_size-hop-stride)/2)
    A = np.pad(A, ((gap, gap), (gap,gap), (0,0)), 'constant')

    W = np.zeros((PREDICT_BATCH_SIZE, window_size, window_size, A.shape[-1]), 'float32')

    i=0
    for y in range(0,shp[0]-hop,stride):
        for x in range(0,shp[1]-hop,stride):
            W[i,:,:,:] = A[y:y+window_size, x:x+window_size,:]
            i+=1
            if i==W.shape[0]:
                i=0
                yield W
                W = np.zeros(W.shape)


    if i<W.shape[0]: yield W


def predict_image(X, model, window_size, target_size):
    shp = X.shape

    stride = target_size//2
    hop = target_size-stride
    X = np.pad(X, ((hop, hop), (hop,hop), (0,0)), 'constant')

    # Ensure that our Data is a multiple of the target_size;
    # we'll crop it back down later
    pady = int(np.ceil(1.0*X.shape[0]/target_size)*target_size - X.shape[0])
    padx = int(np.ceil(1.0*X.shape[1]/target_size)*target_size - X.shape[1])
    padding = ((0,pady), (0,padx), (0,0))
    X = np.pad(X, padding, 'constant')

    num_tiles = int(np.ceil(X.shape[0]/stride) * np.ceil(X.shape[1]/stride))
    num_steps = int(np.ceil(num_tiles/PREDICT_BATCH_SIZE))

    [R,C] = model.predict_generator(
                prediction_windows(X, window_size, hop, stride),
                workers=0,
                verbose=1,
                steps=num_steps
            )

    i=0
    center0 = int(0.25*target_size)
    center1 = int(0.75*target_size)
    P = np.zeros((X.shape[0], X.shape[1], NUM_CLASSES))
    for y in range(0,X.shape[0]-hop, stride):
        for x in range(0, X.shape[1]-hop, stride):
            t = R[i,:,:,:]
            t[center0:center1, center0:center1,:] *= 2
            P[y:y+target_size, x:x+target_size,:] += t
            i+=1
    P = P[hop:hop+shp[0],hop:hop+shp[1],:] #crop to original

    return P


def predict_model(run_name, tpu_model, data_path, epoch, window_size, target_size):
    tpu_model = load_model_weights_from_bucket(tpu_model,"{0}/output/{1}/{1}-best_weights.h5".format(BUCKET_PATH, run_name))

    ts = make_prediction_dataset(data_path)
    target_size= tpu_model.output_shape[0][1]

    i=0
    for X, L in keras_data_generator(ts):
        i+=1
        P = predict_image(X, tpu_model, window_size, target_size)
        log_predictions(run_name, P, L, i)


def save_checkpoint(epoch, logs, model, output_fn):
    loss = logs['val_loss']
    try:
        ml = save_checkpoint.min_loss
    except:
        ml = loss+1

    if  loss < ml:
        save_checkpoint.min_loss = loss
        save_weights_to_bucket(model.sync_to_cpu(), output_fn+'-best_weights.h5')



def train_model(run_name, tpu_model, num_epochs=10, window_size=256, target_size=16):

    output_fn = BUCKET_PATH+"/output/{0}/{0}".format(run_name)
    data_path = BUCKET_PATH + "/data/{0}*.tfrecord"

    training = make_training_dataset(data_path.format('train'), window_size=window_size, target_size=target_size)
    validating = make_training_dataset(data_path.format('test'), window_size=window_size, target_size=target_size)

    H = tpu_model.fit_generator(
        generator = keras_data_generator(training),
        steps_per_epoch=45,
        workers=0,
        verbose=1,
        validation_data = keras_data_generator(validating),
        validation_steps=5,
        callbacks=[
             K.callbacks.CSVLogger(run_name+'-log.csv')
            ,K.callbacks.LambdaCallback( on_epoch_end=lambda epoch, logs: save_checkpoint(epoch, logs, tpu_model, output_fn))
            #,K.callbacks.TensorBoard(log_dir='gs://sparcsdata/profile', write_images=True)
            ],
        epochs=num_epochs)

    save_weights_to_bucket(tpu_model, output_fn+'-end_weights.h5')
    copy_to_bucket(run_name+'-log.csv', output_fn+'-log.csv')


def main(name, mode="train", num_epochs=20, window_size=256, crop_size=28, trainable=False, clear_weight=0.5, iteration=None):

    num_epochs = int(num_epochs)
    window_size = int(window_size)
    crop_size = int(crop_size)
    trainable = int(trainable)
    cnn_models.CLASS_WEIGHTS[:,4] =  float(clear_weight)
    run_name = name

    print "Run Name: ", run_name

    #tpu_model = cnn_models.fullyconnected(window_size, NUM_BANDS, NUM_CLASSES, crop_size)
    tpu_model = cnn_models.vgg16_like(window_size, NUM_BANDS, NUM_CLASSES, crop_size, trainable)

    test_path = BUCKET_PATH+"/data/test*.tfrecord"
    target_size = tpu_model.output_shape[0][1]
    print "Target Size: ", target_size

    #with tf.contrib.tpu.bfloat16_scope():
    if mode.lower() == 'continue':
        tpu_model = load_model_weights_from_bucket(tpu_model,
            BUCKET_PATH+"/output/{0}/{0}-best_weights.h5".format(run_name))
        if iteration:
            p = run_name.split('-')
            p[-1] = str(int(p[-1])+1)
            run_name = '-'.join(p)
        mode='train'

    if mode.lower() == 'staged':
        tpu_model = cnn_models.vgg16_like(window_size, NUM_BANDS, NUM_CLASSES, crop_size, False)
        run_name_mod = run_name + '-0'
        train_model(run_name_mod, tpu_model, 100, window_size, target_size)
        predict_model(run_name_mod, tpu_model, test_path, num_epochs, window_size, target_size)

        tpu_model = cnn_models.vgg16_like(window_size, NUM_BANDS, NUM_CLASSES, crop_size, True)

        for i in range(1,5):
            tpu_model = load_model_weights_from_bucket(tpu_model,
                BUCKET_PATH+"/output/{0}/{0}-best_weights.h5".format(run_name_mod))

            run_name_mod = "{0}-{1}".format(run_name,i)
            train_model(run_name_mod, tpu_model, 25, window_size, target_size)
            predict_model(run_name_mod, tpu_model, test_path, num_epochs, window_size, target_size)


    elif mode.lower() == "train":
        train_model(run_name, tpu_model, num_epochs, window_size, target_size)
        predict_model(run_name, tpu_model, test_path, num_epochs, window_size, target_size)

    elif mode.lower() == "predict":
        #test_path = BUCKET_PATH+"/data/training*.tfrecord"
        predict_model(run_name, tpu_model, test_path, num_epochs, window_size, target_size)
    else:
        print "Mode should be either 'train', 'continue', or 'predict'"



main(*sys.argv[1:])
