import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam,Nadam

import cfg
from network import East,PixelLink
from losses import quad_loss
from data_generator import gen
import tensorflow as tf
import keras.backend as K

# num_cores = 1
# GPU = 2
# CPU = 1
# if GPU:
#     num_GPU = 1
#     num_CPU = 1
# if CPU:
#     num_CPU = 1
#     num_GPU = 0
#
# config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
#         inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
#         device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
# session = tf.Session(config=config)
# K.set_session(session)

def gpu_config(gpu_num):
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    import os
    if isinstance(gpu_num, (list, tuple)):
        gpu_num = ','.join(str(i) for i in gpu_num)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print('GPU config done!')

if cfg.gpu != None:
    gpu_config(cfg.gpu)


def iou(y_true,y_pred):
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    predicts = tf.nn.sigmoid(logits)

    return tf.metrics.mean_iou(labels,predicts,num_classes=2)

def iou_loss_core(y_true, y_pred, smooth=1):
    y_pred = y_pred[:, :, :, :1]
    y_true = y_true[:, :, :, :1]
    y_pred = tf.nn.sigmoid(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

east = East()
east_network = east.east_network()
east_network.summary()
east_network.compile(loss=quad_loss, optimizer=Nadam(lr=cfg.lr),
                                                    # clipvalue=cfg.clipvalue,
                                                    # decay=cfg.decay),
                     metrics=[iou_loss_core])

if cfg.load_weights and os.path.exists(cfg.load_weight):
    east_network.load_weights('_downmodel/east_model_weights_3T736.h5')

east_network.fit_generator(generator=gen(),
                           steps_per_epoch=cfg.steps_per_epoch,
                           epochs=cfg.epoch_num,
                           validation_data=gen(is_val=True),
                           validation_steps=cfg.validation_steps,
                           verbose=1,
                           initial_epoch=cfg.initial_epoch,
                           callbacks=[
                               EarlyStopping(patience=cfg.patience, verbose=1),
                               ModelCheckpoint(filepath=cfg.checkpoint_path,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)])

# east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)
