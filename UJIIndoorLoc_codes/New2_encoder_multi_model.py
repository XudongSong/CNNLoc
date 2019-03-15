import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
from keras.layers import Dense, Input, Dropout, Conv1D, Flatten, Reshape
from keras.constraints import max_norm
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam,Adagrad,Adadelta,Nadam,Adamax,Nadam,Optimizer

from keras.layers.advanced_activations import PReLU
import data_helper
import numpy as np
import keras
base_dir=os.getcwd()
# AE_model_dir=os.path.join(base_dir,'AE_model')
# Building_model_dir=os.path.join(base_dir,'Building_model')

Floor_model_dir=os.path.join(base_dir,'New_Auto_Floor_model')


if not os.path.isdir(Floor_model_dir):
    os.mkdir(Floor_model_dir)
#rewrite class Earlystopping()

from keras import backend as K
from keras.legacy import interfaces
class AMSgrad(Optimizer):
    """AMSGrad optimizer.
    Default parameters follow those provided in the Adam paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(AMSgrad, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            vhat_t = K.maximum(vhat, v_t)
            p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(vhat, vhat_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(AMSgrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class myEarlyStopping(EarlyStopping):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None
                 ):
        super(myEarlyStopping,self).__init__(

            monitor=monitor,
            baseline=baseline,
            patience=patience,
            verbose=verbose,
            min_delta=min_delta,
            mode=mode

        )

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        elif self.best-current>0.02 or current<self.baseline:
            self.wait=0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

        # if current>self.baseline:
        #     self.wait+=1
        #     if self.wait>=self.patience:
        #         self.stopped_epoch = epoch
        #         self.model.stop_training = True
        # else:
        #     self.wait=0

class EncoderDNN(object):
    normY = data_helper.NormY()
    def __init__(self):
        self.epoch_AE=6
        self.epoch_floor=6
        self.epoch_position=6
        self.adam_lr=0.0001
        self.dropout=0.7
        self.patience=3
        self.regularlization=0.0001
        self.b=2.71828
        self.input = Input((520,))

    def fnBuildAEModelAuto(self,input,layerList_2D):
        encode_layer_nums=layerList_2D[0]
        decode_layer_nums=layerList_2D[1]
        encode_layer=input
        for en in encode_layer_nums:

            if self.regularlization<1:
                encode_layer=Dense(en, activation='elu',kernel_regularizer=keras.regularizers.l2(self.regularlization))(encode_layer)
            else:
                encode_layer = Dense(en, activation='elu')(encode_layer)
        decode_layer=encode_layer
        for de in decode_layer_nums:
            if self.regularlization<1:
                decode_layer=Dense(de, activation='elu',kernel_regularizer=keras.regularizers.l2(self.regularlization))(decode_layer)#
            else:
                decode_layer = Dense(de, activation='elu')(decode_layer)
        SAE_model=Model(inputs=input,output=decode_layer)
        bottleneck_model=Model(inputs=input,outputs=encode_layer)
        return SAE_model,bottleneck_model

    def fnBuildFloorModelAuto(self,bottleneck_model,CNN_layerList_2D):
        for layer in bottleneck_model.layers:
            layer.trainable=True
        CnnLayer = Reshape((bottleneck_model.output_shape[1], 1))(bottleneck_model.output)
        if self.dropout<1:
            CnnLayer=Dropout(self.dropout)(CnnLayer)
        for cnn in CNN_layerList_2D:

            if self.regularlization<1:
                CnnLayer= Conv1D(cnn[0], cnn[1], activation='elu',kernel_regularizer=keras.regularizers.l2(self.regularlization))(CnnLayer)
            else:
                CnnLayer = Conv1D(cnn[0], cnn[1], activation='elu')(CnnLayer)
        CnnLayer = Flatten()(CnnLayer)
        output = Dense(5, activation='softmax')(CnnLayer)
        Floor_model = Model(inputs=bottleneck_model.input, outputs=output)
        return Floor_model

    def _preprocess(self,x, y, valid_x, valid_y):
        self.normalize_x = data_helper.normalizeX(x, self.b)
        self.normalize_valid_x = data_helper.normalizeX(valid_x, self.b)
        self.floorID_y = y[:, 2]

        self.floorID_valid_y = valid_y[:, 2]
    def fitAuto(self,x, y, valid_x, valid_y,SAE_list,CNN_list):

        early_stopping = EarlyStopping(monitor='val_acc', patience=self.patience)
        save_path=os.path.join(Floor_model_dir,('Floor_model.h5'))
        save_best=keras.callbacks.ModelCheckpoint(filepath=save_path,monitor='val_acc',save_best_only=True)
        self._preprocess(x, y, valid_x, valid_y)
        adamm = Adam(lr=self.adam_lr)
        encoder_model,bottleneck_model=self.fnBuildAEModelAuto(input=self.input,layerList_2D=SAE_list)
        encoder_model.compile(
            loss='mse',
            optimizer=adamm
        )
        encoder_model.fit(self.normalize_x,
                          self.normalize_x,
                          validation_data=(self.normalize_valid_x,self.normalize_valid_x),
                          # shuffle=True,
                          # validation_split=0.2,
                          epochs=self.epoch_AE,
                          batch_size=66,
                          callbacks=[early_stopping])
        # bottleneck_model.save(os.path.join(AE_model_dir,(AE_bottleneck.h5')))

        floor_model=self.fnBuildFloorModelAuto(bottleneck_model=bottleneck_model,CNN_layerList_2D=CNN_list)
        floor_model.compile(
            loss='mse',
            optimizer=adamm,
            metrics=['accuracy']
        )
        h_floor=floor_model.fit(self.normalize_x,
                                data_helper.oneHotEncode(self.floorID_y),
                                epochs=self.epoch_floor,
                                batch_size=66,
                                # shuffle=True,
                                # validation_split=0.2,
                             validation_data=(
                             self.normalize_valid_x, data_helper.oneHotEncode(self.floorID_valid_y)),
                             callbacks=[save_best,early_stopping])
        # floor_model.save(
        #     os.path.join(Floor_model_dir,('Floor_model.h5')))

        return(h_floor)

    def predict(self, x):
        x = data_helper.normalizeX(x,self.b)
        self.floor_model = load_model(os.path.join(Floor_model_dir,('Floor_model.h5')))
        predict_floorID = self.floor_model.predict(x)
        predict_floorID = data_helper.oneHotDecode(predict_floorID)

        return predict_floorID

    def error(self, x, y):
        _y = self.predict(x)
        floor_right=np.sum(np.equal(np.round(_y),y[:,2]))
        return floor_right



