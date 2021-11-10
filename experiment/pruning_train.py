# -*- coding: utf-8 -*-

"""
Created on 03/18/2021
pruning.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import numpy as np
import math
import tensorflow as tf
import models.lenet5 as lenet5
import models.vgg as vgg
import models.vgg16 as vgg16
import models.resnet as resnet
import utility.loaddata as ld
from utility.log_helper import *
import penalty
import myparser
import datetime
import torch
import torchvision
import torchvision.transforms as transforms


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


# --- set args ---
args = myparser.parse_args()
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# --- create log ---
logging_config(folder=args.save_dir, name='running', no_console=False)
# log.logging.info(args)

# --- Load  dataset ---
train_images, train_labels, test_images, test_labels = None, None, None, None
db_train, db_test = None, None
if args.data_name == 'mnist':
    train_images, train_labels, test_images, test_labels = ld.get_mnist_dataset()
    db_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    db_train = db_train.shuffle(1000).batch(args.batch_size)
    db_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    db_test = db_test.batch(args.batch_size*4)
elif args.data_name == 'cifar10' or args.data_name == 'cifar100':
    raw_data = ld.Cifar(args.data_dir)
    train_images, train_labels, test_images, test_labels = raw_data.prepare_data()
    train_images, test_images = ld.data_preprocessing(train_images, test_images)
    db_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    db_train = db_train.shuffle(6666).map(ld.preprocess).batch(args.batch_size)
    db_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    db_test = db_test.batch(args.batch_size*4)
train_set_size = train_labels.shape[0]
test_set_size = test_labels.shape[0]

# --- Define the model architecture ---
model = None
if args.model == 'lenet':
    model = lenet5.LeNet5()
    model.build(input_shape=(None, 28, 28, 1))
elif args.model == 'vgg':
    if args.data_name == 'cifar100':
        model = vgg16.VGG16(100)
    else:
        model = vgg16.VGG16()
        # model = vgg.VGG('VGG16')
    model.build(input_shape=(None, 32, 32, 3))
elif args.model == 'resnet':
    if args.data_name == 'cifar100':
        model = resnet.resnet34(100)
    else:
        model = resnet.resnet18()
    model.build(input_shape=(None, 32, 32, 3))
else:
    model = lenet5.LeNet5()
    model.build(input_shape=(None, 28, 28, 1))

# --- learning rate ---
# initial_learning_rate = args.init_lr
# decay_steps = math.ceil(train_images.shape[0] / args.batch_size) * 30
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=decay_steps, decay_rate=0.96, staircase=True
# )
cnt_step = math.ceil(train_set_size / args.batch_size)
boundaries = [cnt_step * 20, cnt_step * 30, cnt_step * 40, cnt_step * 150]
values = [0.01, 0.001, 0.0002, 0.0001, 0.0001]
# boundaries = [cnt_step * 10, cnt_step * 20, cnt_step * 40, cnt_step * 150]
# values = [0.001, 0.001, 0.0003, 0.0001, 0.0001]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

# --- optimizer ---
# optimizer = 'adam'
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)


# --- loss ---
go_loss = penalty.GoGoGoLoss(model, args)

# --- reverse ---
if args.is_restore:
    model.load_weights(args.restore_path)

# --- trian ---
if args.train:
    for epoch in range(args.epochs):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        total_correct = 0
        total_num = 0
        step = 0
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = go_loss(y, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss += loss
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            correct = tf.reduce_sum(tf.cast(correct, tf.float32))
            total_num += x.shape[0]
            total_correct += int(correct)
        print('train - Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (step + 1), 100. * total_correct / total_num, total_correct, total_num))

        test_loss = 0
        total_correct = 0
        total_num = 0
        step = 0
        for step, (x, y) in enumerate(db_test):
            logits = model(x)
            loss = go_loss(y, logits)
            test_loss += loss
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            correct = tf.reduce_sum(tf.cast(correct, tf.float32))
            total_num += x.shape[0]
            total_correct += int(correct)
        print('test - Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (step + 1), 100. * total_correct / total_num, total_correct, total_num))


# --- save model to h5 ---
if args.store_weight:
    keras_file = args.baseline_keras + '.h5'
    model.save_weights(keras_file)
    # print('Saved baseline model to:', keras_file)
    logging.info('Saved baseline model to: {}'.format(keras_file))

# --- Evaluate  model ---
if args.test:
    pass

# --- Pruning ---
if args.prune:
    pass
