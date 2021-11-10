# -*- coding: utf-8 -*-

"""
Created on 04/13/2021
callback.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.io_utils import path_to_string


class ModelSaveToH5(keras.callbacks.Callback):
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 mode='auto',
                 verbose=0):
        super(ModelSaveToH5, self).__init__()
        self._supports_tf_logs = True
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = path_to_string(filepath)
        # self.epochs_since_last_save = 0
        # self._batches_seen_since_last_saving = 0
        # self._last_batch_seen = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # print(logs)
        self._save_model(epoch=epoch, logs=logs)

    def _save_model(self, epoch, logs):
        """Saves the model.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        logs = tf_utils.to_numpy_or_python_type(logs)
        filepath = self.filepath

        current = logs.get(self.monitor)

        if self.monitor_op(current, self.best):
            if self.verbose > 0:
                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
            self.best = current
            self.model.save_weights(filepath)
        else:
            if self.verbose > 0:
                print('Epoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))


class EarlyStopping(keras.callbacks.Callback):
    def __init__(self,
                 monitor='val_loss',
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current, self.best):
            # patience=1，Stop at a certain accuracy
            if self.patience == 1:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            else:
                self.best = current
                self.wait = 0
        else:
            if self.patience == 1:
                pass
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        return monitor_value
