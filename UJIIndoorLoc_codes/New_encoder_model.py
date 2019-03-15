import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from keras.layers import Dense, Input, Dropout, Conv1D, Flatten, Reshape,MaxPool1D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam,Adagrad,Adadelta,Nadam,Adamax
from keras.layers.advanced_activations import PReLU
import data_helper
import numpy as np
import keras
base_dir=os.getcwd()
AE_model_dir=os.path.join(base_dir,'AE_model')
Building_model_dir=os.path.join(base_dir,'Building_model')
Floor_model_dir=os.path.join(base_dir,'Floor_model')

#Train_AE为True则训练自编码模型，如果为False则不训练
#Train_Building为True则训练模型，如果为False则采用保存的模型预测
#Train_Floor为True则训练模型，如果为False则采用保存的模型预测
#Train_Location为True则训练模型，如果为False则采用保存的模型预测
#NO_AE为True则不用自编码

Train_AE=False
NO_AE=False
Train_Building=False
Run_Building=True
Train_Floor=True
Run_Floor=True

Train_Location=False
Run_Location=True

class EncoderDNN(object):
    normY = data_helper.NormY()
    def __init__(self):
        self.input = Input((520,))
#################AE_model
        self.encode_layer = Dense(128, activation='elu', name='en1')(self.input)
        self.encode_layer = Dense(64, activation='elu', name='en2')(self.encode_layer)
        # self.encode_layer=Dense(64,activation='elu',name='en-1')(self.encode_layer)
        # decode_layer=Dense(128,activation='elu',name='de1')(self.encode_layer)
        decode_layer = Dense(128, activation='elu', name='de2')(self.encode_layer)
        decode_layer = Dense(520, activation='elu', name='de-1')(decode_layer)
        self.encoder_model_256_128 = Model(inputs=self.input, outputs=decode_layer)
        self.bottleneck_model_256_128 = Model(inputs=self.input, outputs=self.encode_layer)

####################floor_model
        self.AE_floor_bottleneck = '256_128'
        if Train_Floor or Run_Floor:
            self.floor_layers = '99-22,66-22,33-22'
            if (not Train_AE)&(not os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5')))):
                self.floor_base_model=self.bottleneck_model_256_128
            elif NO_AE:
                self.floor_base_model=self.bottleneck_model_256_128
            else:
                self.floor_base_model =load_model(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5')))

            for layer in self.floor_base_model.layers:
                layer.trainable = True
                #正则化
                # layer.kernet_regularizer = keras.regularizers.l2(l=0.0001)

            floor_net_input = Reshape((self.floor_base_model.output_shape[1], 1))(self.floor_base_model.output)
            floor_net = Conv1D(99, 22, activation='elu')(floor_net_input)
            floor_net = Conv1D(66, 22, activation='elu')(floor_net)
            floor_net = Conv1D(33, 22, activation='elu')(floor_net)
            floor_net = Flatten()(floor_net)
            output1 = Dense(5, activation='softmax')(floor_net)
            floor_net = Conv1D(99, 11, activation='elu')(floor_net_input)
            floor_net = Conv1D(66, 11, activation='elu')(floor_net)
            floor_net = Conv1D(33, 11, activation='elu')(floor_net)
            floor_net = Flatten()(floor_net)
            output2 = Dense(5, activation='softmax')(floor_net)
            floor_net = Conv1D(99, 3, activation='elu')(floor_net_input)
            floor_net = Conv1D(66, 3, activation='elu')(floor_net)
            floor_net = Conv1D(33, 3, activation='elu')(floor_net)
            floor_net = Flatten()(floor_net)
            output3 = Dense(5, activation='softmax')(floor_net)


            self.floor_model = Model(inputs=self.floor_base_model.input, outputs=[output1,output2,output3])



####################position_model
        if Train_Location or Run_Location:
            self.floor_layers = '99-22,66-22,33-22'
            if (not Train_AE) & (
            not os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5')))):
                self.position_base_model = self.bottleneck_model_256_128
            elif NO_AE:
                self.position_base_model = self.bottleneck_model_256_128
            else:
                self.position_base_model = load_model(
                    os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5')))

            for layer in self.position_base_model.layers:
                layer.trainable = True
                # 正则化
                # layer.kernet_regularizer = keras.regularizers.l2(l=0.0001)

            position_net_input = Reshape((self.position_base_model.output_shape[1], 1))(self.position_base_model.output)
            position_net = Conv1D(99, 22, activation='elu')(position_net_input)
            position_net = Conv1D(66, 22, activation='elu')(position_net)
            position_net = Conv1D(33,22, activation='elu')(position_net)
            position_net = Flatten()(position_net)
            self.position_predict_output = Dense(2, activation='softmax')(position_net)
            self.position_model = Model(inputs=self.position_base_model.input, outputs=self.position_predict_output)

###################Building_model
        self.AE_building_bottleneck = '256_128'
        if Train_Building or Run_Building:
            if os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5'))):
                self.building_base_model = load_model(os.path.join(AE_model_dir,('AE_bottleneck_' + self.AE_building_bottleneck + '.h5')))
            else:
                self.building_base_model=self.bottleneck_model_256_128
            for layer in self.building_base_model.layers:
                layer.trainable = True
            building_net = Dense(33)(self.building_base_model.output)
            # building_net_input=Reshape((64,1))(self.building_base_model.output)
            # building_net = Conv1D(66,33,activation='elu')(building_net_input)
            # building_net=Flatten()(building_net)
            self.buildingID_predict_output=Dense(3,activation='softmax')(building_net)
            self.building_model= Model(inputs=self.building_base_model.input, outputs=self.buildingID_predict_output)

    def _preprocess(self, x, y, valid_x, valid_y):
        self.normalize_x = data_helper.normalizeX(x)
        self.normalize_valid_x = data_helper.normalizeX(valid_x)

        self.normY.fit(y[:, 0], y[:, 1])
        self.longitude_normalize_y, self.latitude_normalize_y = self.normY.normalizeY(y[:, 0], y[:, 1])
        self.floorID_y = y[:, 2]
        self.buildingID_y = y[:, 3]

        self.longitude_normalize_valid_y, self.latitude_normalize_valid_y = self.normY.normalizeY(valid_y[:, 0],                                                                                                  valid_y[:, 1])
        self.floorID_valid_y = valid_y[:, 2]
        self.buildingID_valid_y = valid_y[:, 3]

    def fit(self, x, y, valid_x, valid_y,tensorbd=None):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y)
#############fit_AE
        if Train_AE:
            self.encoder_model=self.encoder_model_256_128
            self.bottleneck_model=self.bottleneck_model_256_128
            self.encoder_model.compile(
                loss='mse',
                optimizer='adam'
            )
            self.encoder_model.fit(self.normalize_x, self.normalize_x,validation_data=(self.normalize_valid_x,self.normalize_valid_x), epochs=166,batch_size=66,callbacks=[early_stopping])
            self.bottleneck_model.save(os.path.join(AE_model_dir,('AE_bottleneck_'+self.AE_floor_bottleneck+'.h5')))
            self.encoder_model.save('AE_model/AE_'+self.AE_floor_bottleneck+'.h5')

#################fit_Location
        if Train_Location:
            self.position=np.hstack([self.longitude_normalize_y,self.latitude_normalize_y])
            self.position_valid=np.hstack([self.longitude_normalize_valid_y,self.latitude_normalize_valid_y])
            self.position_model.compile(
                loss='mse',
                optimizer='adam'
            )
            self.position_model.fit(self.normalize_x, self.position,validation_data=(self.normalize_valid_x,self.position_valid),epochs=100, batch_size=66,callbacks=[early_stopping])
            self.position_model.save(os.path.join(base_dir,('Location_model/Location_model.h5')))

        self.floor_layers = '99-22,66-22,33-22'
##################fit_floor
        if Train_Floor:
            adamm=Adam()
            # adamm=Adamax()
            # adamm=Adagrad(epsilon=1e-06)

            self.floor_layers = '99-22,66-22,33-22'
            self.floor_model.compile(
                # loss=keras.losses.mse,
                # loss=keras.losses.binary_crossentropy,
                # loss=keras.losses.categorical_crossentropy,
                loss='mse',
                optimizer=adamm,
                metrics=['accuracy']
            )
            self.floor_model.fit(self.normalize_x, [data_helper.oneHotEncode(self.floorID_y),data_helper.oneHotEncode(self.floorID_y),data_helper.oneHotEncode(self.floorID_y)], epochs=166, batch_size=111,
                                 validation_data=(
                                 self.normalize_valid_x, [data_helper.oneHotEncode(self.floorID_valid_y),data_helper.oneHotEncode(self.floorID_valid_y),data_helper.oneHotEncode(self.floorID_valid_y)]),
                                 callbacks=[early_stopping])  # ,tensorbd])
            self.floor_model.save(
                os.path.join(Floor_model_dir,('floor_model(AE_' + self.AE_floor_bottleneck + ')-Conv(' + self.floor_layers + ').h5')))
####################fit_Building

        if Train_Building:
            self.building_model.compile(
                loss='mse',
                optimizer='adam',
                metrics=['accuracy']
            )
            self.building_model.fit(self.normalize_x, data_helper.oneHotEncode(self.buildingID_y),validation_data=(self.normalize_valid_x,data_helper.oneHotEncode(self.buildingID_valid_y)),epochs=100, batch_size=66,callbacks=[early_stopping])
            self.building_model.save(os.path.join(Building_model_dir,('building_model(AE_' + self.AE_building_bottleneck + ')-3.h5')))

    def predict(self, x):
        predict_buildingID=[]
        predict_floorID=[]
        predict_longitude=[]
        predict_latitude=[]
        x = data_helper.normalizeX(x)

#################predict_Location
        if Run_Location:
            self.position_model = load_model('Location_model/Location_model.h5')
            predict_Location= self.position_model.predict(x)
            predict_longitude, predict_latitude = self.normY.reverse_normalizeY(predict_Location[:, 0],
                                                                             predict_Location[:, 1])

##################predict_floor
        if Run_Floor:
            self.floor_model = load_model('Floor_model/floor_model(AE_' + self.AE_floor_bottleneck + ')-Conv(' + self.floor_layers + ').h5')
            predict_floorID = self.floor_model.predict(x)


            predict_floorID=predict_floorID[0]+predict_floorID[1]+predict_floorID[2]
            predict_floorID=data_helper.oneHotDecode(predict_floorID)

            # predict_floorID = data_helper.oneHotDecode_list(predict_floorID)
            # predict_floorID=np.round(np.mean(predict_floorID,axis=0))

            # predict_floorID = data_helper.oneHotDecode_list(predict_floorID)
            # predict_floorID = np.round(np.median(predict_floorID, axis=0))
################predict_building
        if Run_Building:
            self.building_model= load_model(os.path.join(Building_model_dir,('building_model(AE_' + self.AE_building_bottleneck + ')-3.h5')))
            predict_buildingID = self.building_model.predict(x)
            predict_buildingID = data_helper.oneHotDecode(predict_buildingID)

        return predict_buildingID,predict_floorID ,predict_longitude,predict_latitude

    def error(self, x, y):
        _y = self.predict(x)
        building_right = np.sum(np.equal(np.round(_y[0]), y[:, 3]))
        floor_right=np.sum(np.equal(np.round(_y[1]),y[:,2]))
        predict_long=np.reshape(_y[2],(1,len(_y[2])))
        predict_lati=np.reshape(_y[3],(1,len(_y[3])))
        longitude_error = np.mean(np.sqrt(np.square(predict_long - y[:, 0])))
        latitude_error = np.mean(np.sqrt(np.square(predict_lati - y[:, 1])))
        mean_error=np.mean(np.sqrt(np.square(predict_long - y[:, 0])+np.square(predict_lati - y[:, 1])))
        return  building_right,floor_right,longitude_error,latitude_error,mean_error
        # return floor_right
