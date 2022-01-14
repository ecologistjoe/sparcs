import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import *
import numpy as np
import socket
from bucketio import *

#from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dense
K.backend.set_image_data_format('channels_last')
CLASS_WEIGHTS = np.array([[1, 1, 1, 1, 0.5, 1]])


#feature_shape = (10, 256,256))
def fullyconnected(window_size, num_bands, num_classes, crop_size):

    IN = K.Input(shape=(window_size, window_size, num_bands))
    N = IN
    #N = BatchNormalization()(IN)
    #N = GaussianNoise(stddev=0.01)(N)

    L0 = Conv2D(32, (1,1), padding='same', activation='relu')(N)

    L1 = Conv2D(64, (3,3), padding='same', activation='relu')(L0)
    L1 = Conv2D(64, (3,3), padding='same', activation='relu')(L1)
    P1 = MaxPool2D((2,2))(L1)
    #P1 = BatchNormalization()(P1)

    L2 = Conv2D(128, (3,3), padding='same', activation='relu')(P1)
    L2 = Conv2D(128, (3,3), padding='same', activation='relu')(L2)
    P2 = MaxPool2D((2,2))(L2)
    #P2 = BatchNormalization()(P2)

    L3 = Conv2D(256, (3,3), padding='same', activation='relu')(P2)
    L3 = Conv2D(256, (3,3), padding='same', activation='relu')(L3)
    P3 = MaxPool2D((2,2))(L3)
    #P3 = BatchNormalization()(P3)

    L4 = Conv2D(256, (3,3), padding='same', activation='relu')(P3)
    L4 = Conv2D(256, (3,3), padding='same', activation='relu')(L4)
    P4 = MaxPool2D((2,2))(L4)
    #P4 = BatchNormalization()(P4)

    L5 = Conv2D(256, (3,3), padding='same', activation='relu')(P4)
    L5 = Conv2D(256, (3,3), padding='same', activation='relu')(L5)
    P5 = MaxPool2D((2,2))(L5)
    #P5 = BatchNormalization()(P5)

    DENSE = Conv2D(512, (3,3), padding='same', activation='relu')(P5)
    #DENSE = BatchNormalization()(DENSE)

    D4 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', activation='relu')(DENSE)
    #D4 = add([P4,D4])

    D3 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', activation='relu')(D4)
    D3 = add([P3,D3])

    D2 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu')(D3)
    D2 = add([P2,D2])
    #D2 = SpatialDropout2D(rate=0.25)(D2)

    D1 = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu')(D2)
    #D1 = add([P1,D1])
    #D1 = SpatialDropout2D(rate=0.25)(D1)

    D0 = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu')(D1)

    #Make a first convolutional-based output
    C = SpatialDropout2D(rate=0.25)(D0)
    C = Conv2D(64, (3,3), padding='same', activation='relu')(C)
    OUT1 = Cropping2D(crop_size-1)(C)
    OUT1 = Conv2D(num_classes, (3,3), padding='valid', activation='softmax',  name="conv")(OUT1)

    # Make a second output that includes near-pixelwise features
    R = concatenate([L0, D0])
    R = SpatialDropout2D(rate=0.25)(R)
    R = Conv2D(64, (3,3), padding='same', activation='relu')(R)
    OUT2 = Cropping2D(crop_size-1)(R)
    OUT2 = Conv2D(num_classes, (3,3), padding='valid', activation='softmax', name="rslv")(OUT2)


    model = K.Model(IN, [OUT2, OUT1])
    tpu_model = compile_tpu_model(model)

    return tpu_model



def vgg16_like(window_size, num_bands, num_classes, crop_size, trainable=False):

    print CLASS_WEIGHTS

    IN = K.Input(shape=(window_size, window_size, num_bands), name='input')
    N = IN
    #N = BatchNormalization()(IN)
    #N = GaussianNoise(stddev=0.01)(N)

    L0 = Conv2D(64, (3,3), padding='same', activation='relu', name='conv1')(N)
    L1 = Conv2D(64, (3,3), padding='same', activation='relu', name='block1_conv2', trainable=trainable)(L0)
    P1 = MaxPool2D((2,2), name='block1_pool')(L1)

    L2 = Conv2D(128, (3,3), padding='same', activation='relu', name='block2_conv1', trainable=trainable)(P1)
    L2 = Conv2D(128, (3,3), padding='same', activation='relu', name='block2_conv2', trainable=trainable)(L2)
    P2 = MaxPool2D((2,2), name='block2_pool')(L2)

    L3 = Conv2D(256, (3,3), padding='same', activation='relu', name='block3_conv1', trainable=trainable)(P2)
    L3 = Conv2D(256, (3,3), padding='same', activation='relu', name='block3_conv2', trainable=trainable)(L3)
    P3 = MaxPool2D((2,2), name='block3_pool')(L3)

    L4 = Conv2D(512, (3,3), padding='same', activation='relu', name='block4_conv1', trainable=trainable)(P3)
    L4 = Conv2D(512, (3,3), padding='same', activation='relu', name='block4_conv2', trainable=trainable)(L4)
    L4 = Conv2D(512, (3,3), padding='same', activation='relu', name='block4_conv3', trainable=trainable)(L4)
    P4 = MaxPool2D((2,2), name='block4_pool')(L4)

    L5 = Conv2D(512, (3,3), padding='same', activation='relu', name='block5_conv1', trainable=trainable)(P4)
    L5 = Conv2D(512, (3,3), padding='same', activation='relu', name='block5_conv2', trainable=trainable)(L5)
    L5 = Conv2D(512, (3,3), padding='same', activation='relu', name='block5_conv3', trainable=trainable)(L5)
    P5 = MaxPool2D((2,2), name='block5_pool')(L5)

    DENSE = Conv2D(512, (3,3), padding='same', activation='relu', name='dense')(P5)
    #DENSE = BatchNormalization()(P5)

    D4 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', activation='relu', name='dconv1')(DENSE)

    D3 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', activation='relu', name='dconv2')(D4)
    D3 = add([P3,D3], name='add_p3')

    D2 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu', name='dconv3')(D3)
    D2 = add([P2,D2], name='add_p2')

    D1 = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu', name='dconv4')(D2)

    D0 = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu', name='dconv5')(D1)

    #Make a first convolutional-based output
    C = SpatialDropout2D(rate=0.25, name='convdrop')(D0)
    C = Conv2D(64, (3,3), padding='same', activation='relu', name='convmix')(C)
    OUT1 = Cropping2D(crop_size-1, name='convcrop')(C)
    OUT1 = Conv2D(num_classes, (3,3), padding='valid', activation='softmax', name="conv")(OUT1)

    # Make a second output that includes near-pixelwise features
    R = concatenate([L0, D0], name='add_l0')
    R = SpatialDropout2D(rate=0.25, name='rslvdrop')(R)
    R = Conv2D(64, (3,3), padding='same', activation='relu', name='rslvmix')(R)
    OUT2 = Cropping2D(crop_size-1, name='rslvcrop')(R)
    OUT2 = Conv2D(num_classes, (3,3), padding='valid', activation='softmax', name="rslv")(OUT2)

    model = K.Model(IN, [OUT2, OUT1])
    tpu_model = compile_tpu_model(model)
    load_model_weights_from_bucket(tpu_model, 'gs://sparcsdata/vgg16weights.h5')

    return tpu_model



def weighted_kld(y_true, y_pred):
      W = tf.math.reduce_sum(CLASS_WEIGHTS * y_true, axis=-1)
      y_pred = K.backend.clip(y_pred, K.backend.epsilon(), 1)
      kld = -tf.math.log(tf.math.reduce_sum(y_true*y_pred+K.backend.epsilon(), axis=-1))

      return kld*W



def compile_tpu_model(model):
    tpu_model = tf.contrib.tpu.keras_to_tpu_model( model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(
                    tpu=socket.gethostname())
            ))

    tpu_model.compile(
        optimizer=tf.keras.optimizers.Adam(amsgrad=True),
        #optimizer=tf.train.AdamOptimizer(),
        #loss={'conv':tf.keras.losses.CategoricalCrossentropy(),'rslv':tf.keras.losses.CategoricalCrossentropy()},
        loss=weighted_kld,
        #loss=tf.nn.softmax_cross_entropy_with_logits_v2,
        #loss=tf.keras.losses.CategoricalCrossentropy(),
        loss_weights=[0.5, 1],
        #metrics={'conv':tf.keras.metrics.CategoricalAccuracy(),'rslv':tf.keras.metrics.CategoricalAccuracy()}

    )

    return tpu_model
