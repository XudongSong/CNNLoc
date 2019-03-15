import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
from keras.layers import Dense, Input, Dropout, Conv1D, Flatten, Reshape
from keras.constraints import max_norm
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam,Adagrad,Adadelta,Nadam,Adamax,Nadam,Optimizer
from keras.utils import multi_gpu_model
from keras.layers.advanced_activations import PReLU
import data_helper
import numpy as np
import keras
base_dir=os.getcwd()
# AE_model_dir=os.path.join(base_dir,'AE_model')
# Building_model_dir=os.path.join(base_dir,'Building_model')

Location_model_dir=os.path.join(base_dir,'New_Auto_Location_model')

rng=np.random.RandomState(666)

if not os.path.isdir(Location_model_dir):
    os.mkdir(Location_model_dir)
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
        self.pre_train=0
        self.regularlization_l1=0.0000008
        self.regularlization_l2=0.00004
        self.patience=3
        self.b=2.71828
        self.input = Input((520,))

    def fnBuildInputFullConnectAuto(self,input,layerList):
        hidden_layer=input
        # hidden_layer=Dropout(0.7)(hidden_layer)
        if len(layerList)>0:
            for en in layerList:
                # hidden_layer=Dense(en, activation='elu',kernel_initializer='zeros',kernel_regularizer=keras.regularizers.l1_l2(self.regularlization_l1,self.regularlization_l2))(hidden_layer)
                hidden_layer = Dense(en, activation='elu')(hidden_layer)
        model= Model(inputs=input, output=hidden_layer)
        return model
    def fnBuildCnnAuto(self,input_model,CNN_layerList_2D):

        for layer in input_model.layers:
            layer.trainable=True
        CnnLayer = Reshape((input_model.output_shape[1], 1))(input_model.output)
        if self.dropout<1:
            CnnLayer=Dropout(self.dropout)(CnnLayer)
        for cnn in CNN_layerList_2D:
            CnnLayer = Conv1D(cnn[0], cnn[1], activation='elu')(CnnLayer)
            # CnnLayer= Conv1D(cnn[0], cnn[1], activation='elu',kernel_initializer='zeros',kernel_regularizer=keras.regularizers.l1_l2(self.regularlization_l1,self.regularlization_l2))(CnnLayer)
        CnnLayer = Flatten()(CnnLayer)
        output = Dense(2, activation='elu')(CnnLayer)
        Floor_model = Model(inputs=input_model.input, outputs=output)
        return Floor_model


    def _preprocess(self,x, y, valid_x, valid_y):
        self.normalize_x = data_helper.normalizeX(x, self.b)
        self.normalize_valid_x = data_helper.normalizeX(valid_x, self.b)
        # self.floorID_y = y[:, 2]
        #
        # self.floorID_valid_y = valid_y[:, 2]
        self.normY.fit(y[:, 0], y[:, 1])
        self.longitude_normalize_y, self.latitude_normalize_y = self.normY.normalizeY(y[:, 0], y[:, 1])
        self.longitude_normalize_valid_y, self.latitude_normalize_valid_y = self.normY.normalizeY(valid_y[:, 0],
                                                                                                  valid_y[:, 1])
        # self.longitude_normalize_y+=(2*(np.random.random(self.longitude_normalize_y.shape)-0.5))
        # self.longitude_normalize_valid_y += (2 * (np.random.random(self.longitude_normalize_valid_y.shape) - 0.5))
        #
        # self.latitude_normalize_y+=(2*(np.random.random(self.latitude_normalize_y.shape)-0.5))
        # self.latitude_normalize_valid_y+=(2*(np.random.random(self.latitude_normalize_valid_y.shape)-0.5))
    def fitLocationAuto(self,x, y,full_list,CNN_list,valid_x,valid_y):
        save_path = os.path.join(Location_model_dir, ('Location_model.h5'))
        save_best = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        self._preprocess(x, y,valid_x,valid_y)
        adamm = Adam(lr=self.adam_lr)
        self.fullConnect_model= self.fnBuildInputFullConnectAuto(input=self.input, layerList=full_list)
        model=self.fnBuildCnnAuto(input_model=self.fullConnect_model,CNN_layerList_2D=CNN_list)

        self.position = np.hstack([self.longitude_normalize_y, self.latitude_normalize_y])
        self.position_valid = np.hstack([self.longitude_normalize_valid_y, self.latitude_normalize_valid_y])







        if self.pre_train:
            model = load_model(save_path)

        model.compile(
            loss='mse',
            optimizer=adamm,
            # metrics=['accuracy']
        )

        #
        # multi_model=multi_gpu_model(model,gpus=2)
        # multi_model.compile(
        #     loss='mse',
        #     optimizer=adamm,
        #     # metrics=['accuracy']
        # )


        h_floor = model.fit(self.normalize_x,
                            self.position,
                            epochs=self.epoch_floor,
                            batch_size=66,
                            # shuffle=True,
                            validation_data=(
                                self.normalize_valid_x, self.position_valid),
                            # validation_split=0.2,
                            callbacks=[save_best,early_stopping])


        # model.save(
        #     os.path.join(Location_model_dir, ('Location_model.h5')))
        return (h_floor)


    def predict(self, x):
        x = data_helper.normalizeX(x,self.b)

        self.position_model = load_model(os.path.join(Location_model_dir, ('Location_model.h5')))

        predict_Location = self.position_model.predict(x)
        predict_longitude, predict_latitude = self.normY.reverse_normalizeY(predict_Location[:, 0],
                                                                            predict_Location[:, 1])

        return predict_longitude,predict_latitude

    def error(self, x, y):
        _y = self.predict(x)
        print(x)
        print(_y)
        print(np.shape(x),np.shape(_y))
        # print(_y[:2])
        predict_long=np.reshape(_y[0],(-1,1))
        predict_lati=np.reshape(_y[1],(-1,1))
        # print(predict_lati[:2])
        longitude_error = np.mean(np.sqrt(np.square(predict_long - y[:, 0])))
        latitude_error = np.mean(np.sqrt(np.square(predict_lati - y[:, 1])))
        mean_error=np.mean(np.sqrt(np.square(predict_long - y[:, 0])+np.square(predict_lati - y[:, 1])))
        return longitude_error,latitude_error,mean_error




