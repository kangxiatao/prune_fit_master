# -*- coding: utf-8 -*-

"""
Created on 03/18/2021
main.
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
import models.mobilenet as mobilenet
import models.mobilenet_v2 as mobilenetv2
import models.mobilenet_v3_small as mobilenetv3s
import models.mobilenet_v3_large as mobilenetv3l
import utility.loaddata as ld
from utility.log_helper import *
from utility.cosine_lr import *
import myparser
import penalty
import prune
import mycallback
import datetime

# --- set args ---
args = myparser.parse_args()
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# --- create log ---
logging_config(folder=args.save_dir, name='running', no_console=False)

# --- Load  dataset ---
db_train, db_test, train_images, train_labels, test_images, test_labels = ld.ld(args)
train_set_size = train_labels.shape[0]
test_set_size = test_labels.shape[0]

# --- Define the model architecture ---
model = None
if args.model == 'lenet':
    model = lenet5.LeNet5(args.class_num)
elif args.model == 'vgg':
    model = vgg16.VGG16(args.class_num)
    # model = vgg.VGG('VGG16', args.class_num)
elif args.model == 'resnet':
    model = resnet.resnet18(args.class_num)
elif args.model == 'resnet34':
    model = resnet.resnet34(args.class_num)
elif args.model == 'mobilenet':
    model = mobilenet.MobileNet(args.class_num)
elif args.model == 'mobilenetv2':
    model = mobilenetv2.MobileNetV2(args.class_num)
elif args.model == 'mobilenetv3s':
    model = mobilenetv3s.MobileNetV3Small(args.class_num)
elif args.model == 'mobilenetv3l':
    model = mobilenetv3l.MobileNetV3Large(args.class_num)
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
_cost = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# _l1_reg = penalty.L1Loss(model, args)
_l2_reg = penalty.L2Loss(model, args)
# _var_reg = penalty.SeparateAngleLoss(model, args)
# _grouplasso = penalty.GroupLassoLoss(model, args)
# go_loss = [_cost, _l1_reg, _l2_reg, _var_reg, _grouplasso]

# --- metrics ---
go_metrics = ['accuracy']
go_metrics.append(_cost)
if args.l2_value != 0.0:
    go_metrics.append(_l2_reg)
# if args.gl_1 != 0.0 or args.gl_2 != 0.0:
#     go_metrics.append(_grouplasso)
# if args.var_1 != 0.0 or args.var_2 != 0.0:
#     go_metrics.append(_var_reg)

# --- callbacks ---
log_dir = args.save_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print("tensorboard --logdir {}".format(args.save_dir))
callbacks = [
    # Create the Learning rate scheduler.
    WarmUpCosineDecayScheduler(learning_rate_base=args.init_lr,
                               total_steps=total_steps,
                               warmup_learning_rate=0.001,
                               warmup_steps=10,
                               # log_dir=log_dir,
                               ),
    tf.keras.callbacks.CSVLogger(args.save_dir + 'training.log'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    # tf.keras.callbacks.ModelCheckpoint(
    #     filepath=args.save_dir+'anoi.h5',
    #     save_best_only=True,  # Only save a model if `val_loss` has improved.
    #     save_weights_only=True,
    #     monitor="val_accuracy",
    #     verbose=1,
    # ),
    # mycallback.ModelSaveToH5(
    #     filepath=args.save_dir + 'anoi.h5',
    #     monitor="val_accuracy",
    #     verbose=1,
    # )
]
if args.stop_acc > 0:
    callbacks.append(
        mycallback.EarlyStopping(
            monitor="val_accuracy",
            patience=1,
            baseline=args.stop_acc,
            verbose=1,
        ),
    )

# --- compile  model ---
model.compile(optimizer=optimizer,
              loss=go_loss,
              metrics=go_metrics)
# for i, weight in enumerate(model.trainable_variables):
#     print(weight.name, '---', weight.get_shape())

# --- reverse ---
if args.is_restore:
    model.load_weights(args.restore_path + 'anoi.h5')

# --- prior pruning ---
args.prior_prune_bool_list = None
if args.prior_prune and args.is_restore:
    _accuracy, _prune_rate, args.prior_prune_bool_list = prune.prior_pruning(model, db_test, args, 'auto')
    logging.info(
        '--- prior pruning --- threshold: auto => accuracy: {:.5f} | prune rate: {:.5f}'.format(_accuracy, _prune_rate))

# --- Train ---
if args.train:
    model.fit(
        db_train,
        validation_data=db_test,
        epochs=args.epochs,
        verbose=2,
        callbacks=callbacks,
    )
    model.load_weights(args.save_dir + 'anoi.h5')  # get best weights
    baseline_model_accuracy = model.evaluate(db_test, verbose=0)[1]
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
    baseline_model_accuracy = model.evaluate(db_test, verbose=0)[1]
    # print('Baseline test accuracy:', baseline_model_accuracy)
    logging.info('Baseline test accuracy: {:.5f}'.format(baseline_model_accuracy))

# --- Pruning ---
if args.prune:
    _accuracy, _prune_rate = prune.start_pruning(model, db_test, args, 'auto')
    logging.info('threshold: auto => accuracy: {:.5f} | prune rate: {:.5f}'.format(_accuracy, _prune_rate))
    # model.load_weights(args.save_dir + 'anoi.h5')  # get best weights
    # _accuracy, _prune_rate = prune.start_pruning(model, db_test, args)
    # logging.info('threshold: {} => accuracy: {:.5f} | prune rate: {:.5f}'.format(args.threshold, _accuracy, _prune_rate))

# --- Observe output ---
# if args.model == 'vgg':
#     model.observe_output(test_images[:32], 9, 0)
