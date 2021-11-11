# -*- coding: utf-8 -*-

"""
Created on 03/18/2021
parser.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import os
import re
import argparse
import utility.myhelp as myhelp


def parse_args():

    # --- default args ---
    # data_name = 'mnist'
    # model = 'lenet'
    data_name = 'cifar10'
    model = 'vgg'
    # data_name = 'cifar100'
    # model = 'resnet'
    # data_name = 'caltech101'
    # model = 'mobilenetv3l'
    # model = 'mobilenetv3s'

    # restore_path = 'trained_model/vgg/cifar10/_l1_0.0001_/'
    restore_path = 'trained_model/vgg/cifar10/_gl1_0.0012_gl2_0.0012_/'
    # restore_path = 'trained_model/vgg/cifar10/_gl1_0.0012_gl2_0.0012_/restore_gl1_0.0012_gl2_0.0012_var2_0.0008_/'

    train = 1
    test = 1
    prune = 1
    prior_prune = 0
    store_weight = 0
    is_restore = 0

    init_lr = 0.1
    epochs = 200
    batch_size = 250

    threshold = 0.001
    penalty_ratio = 1.0
    stop_acc = 0.992

    l1_value = 0.000
    l2_value = 0.00
    gl_1 = 0.000
    gl_2 = 0.000
    gl_a = 0.000
    var_1 = 0.0
    var_2 = 0.0
    prop_1 = 0.000
    prop_2 = 0.000
    prop_a = 0.0

    parser = argparse.ArgumentParser(description="Run pruning.")

    # --- seed ---
    parser.add_argument('--seed', type=int, default=2021,
                        help='Random seed.')

    # --- dataset ---
    parser.add_argument('--data_name', nargs='?', default=data_name,
                        help='Choose a dataset')

    # --- model ---
    parser.add_argument('--model', nargs='?', default=model,
                        help='Choose a model from {lenet, vgg, resnet}')

    # --- reverse ---
    parser.add_argument('--is_restore', type=int, default=is_restore)
    parser.add_argument('--restore_path', nargs='?', default=restore_path)

    # --- switch ---
    parser.add_argument('--train', type=int, default=train)
    parser.add_argument('--test', type=int, default=test)
    parser.add_argument('--prune', type=int, default=prune)
    parser.add_argument('--prior_prune', type=int, default=prior_prune)
    parser.add_argument('--store_weight', type=int, default=store_weight)

    # --- initial learning rate ---
    parser.add_argument('--init_lr', type=float, default=init_lr,
                        help='initial learning rate')

    # --- prune threshold ---
    parser.add_argument('--threshold', type=float, default=threshold,
                        help='prune threshold')

    # --- penalty ratio---
    parser.add_argument('--penalty_ratio', type=float, default=penalty_ratio,
                        help='penalty_ratio')

    # --- train args ---
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=epochs,
                        help='epochs')
    parser.add_argument('--stop_acc', type=float, default=stop_acc,
                        help='stop_acc')

    parser.add_argument('--l1_value', type=float, default=l1_value)
    parser.add_argument('--l2_value', type=float, default=l2_value)
    parser.add_argument('--gl_1', type=float, default=gl_1)
    parser.add_argument('--gl_2', type=float, default=gl_2)
    parser.add_argument('--gl_a', type=float, default=gl_a)
    parser.add_argument('--var_1', type=float, default=var_1)
    parser.add_argument('--var_2', type=float, default=var_2)
    parser.add_argument('--prop_1', type=float, default=prop_1)
    parser.add_argument('--prop_2', type=float, default=prop_2)
    parser.add_argument('--prop_a', type=float, default=prop_a)

    args = parser.parse_args()

    # --- data dir ---
    if args.data_name == 'cifar10':
        args.data_dir = "../Data/cifar-10-python/cifar-10-batches-py"
    elif args.data_name == 'cifar100':
        args.data_dir = "../Data/cifar-100-python"
    elif args.data_name == 'caltech101':
        args.data_dir = "../Data/caltech101"

    # --- image info ---
    if args.data_name == 'cifar10':
        args.image_size = 32
        args.image_channel = 3
        args.class_num = 10
    elif args.data_name == 'cifar100':
        args.image_size = 32
        args.image_channel = 3
        args.class_num = 100
    elif args.data_name == 'mnist':
        args.image_size = 28
        args.image_channel = 1
        args.class_num = 10
    elif args.data_name == 'mnist2':
        args.image_size = 28
        args.image_channel = 1
        args.class_num = 2
    elif args.data_name == 'caltech101':
        args.image_size = 200
        args.image_channel = 3
        args.class_num = 102

    # --- save model dir ---
    arg_sta = ''
    if args.l1_value != 0:
        arg_sta = arg_sta + '_l1_' + str(args.l1_value)
    if args.l2_value != 0:
        arg_sta = arg_sta + '_l2_' + str(args.l2_value)
    if args.gl_1 != 0:
        arg_sta = arg_sta + '_gl1_' + str(args.gl_1)
    if args.gl_2 != 0:
        arg_sta = arg_sta + '_gl2_' + str(args.gl_2)
    if args.gl_a != 0:
        arg_sta = arg_sta + '_gla_' + str(args.gl_a)
    if args.var_1 != 0:
        arg_sta = arg_sta + '_var1_' + str(args.var_1)
    if args.var_2 != 0:
        arg_sta = arg_sta + '_var2_' + str(args.var_2)
    if args.prop_1 != 0:
        arg_sta = arg_sta + '_prop1_' + str(args.prop_1)
    if args.prop_2 != 0:
        arg_sta = arg_sta + '_prop2_' + str(args.prop_2)
    if args.prop_a != 0:
        arg_sta = arg_sta + '_propa_' + str(args.prop_a)
    save_dir = 'trained_model/{}/{}/{}_'.format(args.model, args.data_name, arg_sta)
    if args.is_restore:
        save_dir = args.restore_path + 'restore{}_'.format(arg_sta)
    save_dir = myhelp.get_fit_path(save_dir)
    os.makedirs(save_dir)
    args.save_dir = save_dir + '/'

    args.baseline_keras = myhelp.get_fit_path(args.save_dir + 'baseline_keras_')
    args.pruned_keras = myhelp.get_fit_path(args.save_dir + 'pruned_keras_')
    args.pruned_hzero = myhelp.get_fit_path(args.save_dir + 'pruned_hzero_')

    # -- save args --
    argsDict = args.__dict__
    with open(args.save_dir+'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    return args

