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
from utility.cosine_lr import *
import penalty
import myparser

# --- set args ---
args = myparser.parse_args()
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# --- use tpu ---
print("Tensorflow version " + tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    raise BaseException(
        'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
print('tpu_strategy.num_replicas_in_sync:', tpu_strategy.num_replicas_in_sync)
args.batch_size = args.batch_size * tpu_strategy.num_replicas_in_sync

# --- create log ---
logging_config(folder=args.save_dir, name='running', no_console=False)
# log.logging.info(args)

# --- Load  dataset ---
db_train, db_test, train_set_size, test_set_size = ld.ld(args)

# --- creating the model in TPU ---
with tpu_strategy.scope():  # creating the model in the TPUStrategy scope means we will train the model on the TPU
    # --- Define the model architecture ---
    model = None
    if args.model == 'lenet':
        model = lenet5.LeNet5(args.class_num)
    elif args.model == 'vgg':
        model = vgg16.VGG16(args.class_num)
        # model = vgg.VGG('VGG16', args.class_num)
    elif args.model == 'resnet':
        model = resnet.resnet18(args.class_num)
    else:
        model = lenet5.LeNet5(args.class_num)
    model.build(input_shape=(None, args.image_size, args.image_size, args.image_channel))

    # --- learning rate ---
    cnt_step = math.ceil(train_set_size / args.batch_size)
    total_steps = int(args.epochs * train_set_size / args.batch_size)

    # --- optimizer ---
    # optimizer = tf.keras.optimizers.Adam(learning_rate=args.init_lr)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.init_lr, momentum=0.9)

    # --- loss ---
    go_loss = penalty.GoGoGoLoss(model, args)

    # --- callbacks ---
    callbacks = [
        # Create the Learning rate scheduler.
        WarmUpCosineDecayScheduler(learning_rate_base=args.init_lr,
                                total_steps=total_steps,
                                warmup_learning_rate=0.001,
                                warmup_steps=10,
                                # log_dir=log_dir,
                                ),
        tf.keras.callbacks.CSVLogger(args.save_dir + 'training.log'),
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     min_delta=1e-4,
        #     patience=25,
        #     verbose=1,
        # ),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=args.save_dir+'anoi',
        #     save_best_only=True,  # Only save a model if `val_loss` has improved.
        #     save_weights_only=True,
        #     monitor="val_accuracy",
        #     verbose=1,
        # ),
    ]

    # --- compile  model ---
    model.compile(optimizer=optimizer,
                  loss=go_loss,
                  metrics=['accuracy'])

# --- reverse ---
if args.is_restore:
    model.load_weights(args.restore_path)

# --- Train ---
if args.train:
    model.fit(
        db_train,
        validation_data=db_test,
        epochs=args.epochs,
        verbose=2,
        callbacks=callbacks,
    )
    # model.load_weights(args.save_dir+'anoi')  # get best weights
    _, baseline_model_accuracy = model.evaluate(db_test, verbose=0)
    # print('Baseline test accuracy:', baseline_model_accuracy)
    logging.info('Baseline test accuracy: {:.5f}'.format(baseline_model_accuracy))

# --- save model to h5 ---
if args.store_weight:
    keras_file = args.baseline_keras + '.h5'
    model.save_weights(keras_file)
    # print('Saved baseline model to:', keras_file)
    logging.info('Saved baseline model to: {}'.format(keras_file))

# --- Evaluate  model ---
if args.test:
    _, baseline_model_accuracy = model.evaluate(db_test, verbose=0)
    # print('Baseline test accuracy:', baseline_model_accuracy)
    logging.info('Baseline test accuracy: {:.5f}'.format(baseline_model_accuracy))

# --- Pruning ---
if args.prune:
    pass
