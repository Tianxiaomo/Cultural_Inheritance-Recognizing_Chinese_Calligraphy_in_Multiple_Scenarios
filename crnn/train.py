#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: train.py
@time: 2019/1/7 21:40
@desc:
'''
import keras.backend as K
from keras.models import Model,load_model
from keras.layers import Lambda,Input
from keras.optimizers import adam,Nadam,Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

import crnn.cfg as cfg
from crnn.model import CRNN_slim
from crnn.dataset import data_gen,DATASequence
from crnn.util.util import gpuConfig

gpuConfig(cfg.gpu)

# tensorflow keras cpu gpu set
# import tensorflow as tf
# num_cores = 1
# GPU = 0
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

def train(net):
    opt = Adam(lr=0.001,decay=0.0001)

    train_data_seq = DATASequence()
    val_data_seq = DATASequence(train=False)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    y_pred = net.output
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[net.input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt,metrics=['accuracy'])

    earlystopping = EarlyStopping(monitor='val_loss',mode='min',patience=cfg.patience)
    checkpoint = ModelCheckpoint(cfg.model_weights_path,verbose=1,save_best_only=True,save_weights_only=True)
    tensorbord = TensorBoard(cfg.tensorboard_dir)
    callbacks = [earlystopping,checkpoint,tensorbord]

    model.fit_generator(generator=train_data_seq,
                        steps_per_epoch=cfg.train_steps,
                        epochs = cfg.epoch_num,
                        validation_data = val_data_seq,
                        validation_steps = cfg.val_steps,
                        callbacks=callbacks,
                        workers = 32,
                        # use_multiprocessing = True
                        max_queue_size=10,
                        )


if __name__ == '__main__':

    for i in dir(cfg):
        print(i, eval('cfg.' + i))

    model = CRNN_slim().model
    train(model)