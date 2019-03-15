import os
from keras.layers import Dense, Input, Dropout, Conv1D, Flatten, Reshape,MaxPool1D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam,Adagrad,Adadelta,Nadam,Adamax
from keras.layers.advanced_activations import PReLU
import keras
import data_helper
import numpy as np
import time
base_dir=os.getcwd()
AE_model_dir=os.path.join(base_dir,'AE_model')
Building_model_dir=os.path.join(base_dir,'Building_model')
Floor_model_dir=os.path.join(base_dir,'Floor_model')
Location_model_for_every_floor=os.path.join(base_dir,'Location_model_for_every_floor')

# class DEEPLEARNING_MODEL(object):
#
#     def __init__(self,save_model_name):
#         self.normY = data_helper.NormY()
#         self.model_save_path=os.path.join(Location_model_for_every_floor,save_model_name)
#         self.log_path=os.path.join(base_dir,'DEEPLEARNING_MODEL_log.txt')
#     def model(self):
#         pass
#
#     def preprocess(self,x, y, valid_x, valid_y):
#
#         return x,y,valid_x,valid_y
#
#     def fit(self,x, y, valid_x, valid_y):
#         early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#         # Data pre-processing
#         x,y,valid_x,valid_y=self.preprocess(x, y, valid_x, valid_y)
#         self.Model.compile(loss='mse',optimizer='adam')
#         self.Model.fit(x,y,validation_data=(valid_x,valid_y),epochs=166,batch_size=66, callbacks=[early_stopping])
#         self.Model.save(self.model_save_path)
#
#     def predict(self,x):
#         x = data_helper.normalizeX(x)
#         model = load_model(self.model_save_path)
#         y= model.predict(x)
#         return y
#     def postprocess(self, y):
#
#         return y
#     def evaluate(self,x,y):
#         pass
#     def log(self):
#         f=open(self.log_path,'a+')
#         f.write('\n'+'#'*10+str(time.get_clock_info())+'#'*10+'\n')
#         f.write('Hello world!')

class LOCATION_MODEL(object):
    def __init__(self,save_model_name):
        self.normY = data_helper.NormY()
        self.model_save_path=os.path.join(Location_model_for_every_floor,save_model_name)
    def preprocess(self,x, y, valid_x, valid_y):
        x = data_helper.normalizeX(x)
        valid_x = data_helper.normalizeX(valid_x)
        self.normY.fit(y[:, 0], y[:, 1])
        y = self.normY.normalizeY(y[:, 0], y[:, 1])
        y= np.hstack([y[0],y[1]])
        valid_y = self.normY.normalizeY(valid_y[:, 0],valid_y[:,1])
        valid_y= np.hstack([valid_y[0],valid_y[1]])
        return x,y,valid_x,valid_y

    def fit(self,x, y, valid_x, valid_y):
        def SAE():
            input = Input((520,))
            encode_layer = Dense(256, activation='elu', name='en1')(input)
            encode_layer = Dense(128, activation='elu', name='en2')(encode_layer)
            encode_layer = Dense(64, activation='elu', name='en3')(encode_layer)
            decode_layer = Dense(128, activation='elu', name='de1')(encode_layer)
            decode_layer = Dense(256, activation='elu', name='de2')(decode_layer)
            decode_layer = Dense(520, activation='elu', name='de3')(decode_layer)
            self.SAE_Model=Model(inputs=input,outputs=decode_layer)
            self.SAE_neck=Model(inputs=input,outputs=encode_layer)
        def SAE_Location_Model():
            input=Reshape((self.SAE_neck.output_shape[1], 1))(self.SAE_neck.output)
            location_net = Conv1D(99, 22, activation='elu')(input)
            location_net = Conv1D(66, 22, activation='elu')(location_net)
            location_net = Conv1D(33, 22, activation='elu')(location_net)
            location_net = Flatten()(location_net)
            location_predict_output = Dense(2, activation='elu', name='last')(location_net)
            self.Model = Model(inputs=self.SAE_neck.input, outputs=location_predict_output)
        def model_dense():
            input = Input((520,))
            encode_layer = Dense(256, activation='elu', name='en1')(input)
            encode_layer = Dense(256, activation='elu', )(encode_layer)
            encode_layer = Dense(128, activation='elu', name='en2')(encode_layer)
            encode_layer = Dense(64, activation='elu', name='en3')(encode_layer)
            encode_layer = Dense(64, activation='elu', )(encode_layer)
            location_predict_output = Dense(2, activation='elu', name='last')(encode_layer)
            self.Model = Model(inputs=input, outputs=location_predict_output)
        def model_mix():
            input = Input((520,))
            encode_layer = Dense(256, activation='elu', name='en1')(input)
            encode_layer = Dense(128, activation='elu', name='en2')(encode_layer)
            # encode_layer = Dense(64, activation='elu', name='en3')(encode_layer)
            encoder_model= Model(inputs=input, outputs=encode_layer)
            location_net_input = Reshape((encoder_model.output_shape[1], 1))(encoder_model.output)
            location_net = Conv1D(99, 22, activation='elu')(location_net_input)
            location_net = Conv1D(66, 22, activation='elu')(location_net)
            location_net = Conv1D(33, 22, activation='elu')(location_net)
            location_net = Flatten()(location_net)
            location_predict_output = Dense(2, activation='elu',name='last')(location_net)
            self.Model=Model(inputs=encoder_model.input, outputs=location_predict_output)
        # model_mix()
        # model_dense()
        SAE()
        SAE_Location_Model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        # Data pre-processing
        x,y,valid_x,valid_y=self.preprocess(x, y, valid_x, valid_y)
        # adam = Adam(lr=0.0002)

        def train_sae(x,valid_x):
            # for layer in self.SAE_Model.layers:
            #     # layer.trainable = True
            #     #正则化
            #     layer.kernet_regularizer = keras.regularizers.l2(l=0.0001)

            self.SAE_Model.compile(loss='mse', optimizer='adam')
            self.SAE_Model.fit(x, x, validation_data=(valid_x, valid_x), epochs=166, batch_size=66,
                           callbacks=[early_stopping])
            # self.SAE_Model.save(self.model_save_path)

        def train_model(x,y,valid_x,valid_y):
            # for layer in self.Model.layers:
            #     # layer.trainable = True
            #     #正则化
            #     layer.kernet_regularizer = keras.regularizers.l2(l=0.0001)

            self.Model.compile(loss='mse',optimizer='adam')
            self.Model.fit(x,y,validation_data=(valid_x,valid_y),epochs=166,batch_size=66, callbacks=[early_stopping])
            self.Model.save(self.model_save_path)
        train_sae(x,valid_x,)
        train_model(x,y,valid_x,valid_y)

    def predict(self,x):
        x = data_helper.normalizeX(x)
        model = load_model(self.model_save_path)
        y= model.predict(x)
        return y

    def postprocess(self, y):
        y = self.normY.reverse_normalizeY(y[:, 0], y[:, 1])
        return y

    def evaluate(self, x, y):
        _y = self.predict(x)
        _y=self.postprocess(_y)

        predict_long = np.reshape(_y[0], (1, len(_y[0])))
        predict_lati = np.reshape(_y[1], (1, len(_y[1])))
        longitude_error = np.mean(np.sqrt(np.square(predict_long - y[:, 0])))
        latitude_error = np.mean(np.sqrt(np.square(predict_lati - y[:, 1])))
        mean_error = np.mean(np.sqrt(np.square(predict_long - y[:, 0]) + np.square(predict_lati - y[:, 1])))
        return longitude_error, latitude_error, mean_error

train_csv_path = os.path.join(base_dir,'trainingData.csv')
test_csv_path=os.path.join(base_dir,'validationData.csv')
Data_groupby_floor_path=R'C:\Users\Administrator\PycharmProjects\Indoor_locate\Data_groupby_floor'

if __name__ == '__main__':
    # Load data
    data_by_floor_name = ['b0f0', 'b0f1', 'b0f2', 'b0f3', 'b1f0', 'b1f1', 'b1f2', 'b1f3', 'b2f0', 'b2f1', 'b2f2',
                          'b2f3', 'b2f4']
    datas=[]
    if len(os.listdir(Data_groupby_floor_path))==0:
        data_helper.save_data_perspective_by_floor(train_csv_path,test_csv_path)

    for name in data_by_floor_name:
        train_file_path=os.path.join(Data_groupby_floor_path,'chazhi')
        train_file_path = os.path.join(train_file_path, ('train_' + name + '.csv'))

        # train_file_path = os.path.join(Data_groupby_floor_path, ('train_' + name + '.csv'))
        test_file_path=os.path.join(Data_groupby_floor_path,('text_'+name+'.csv'))
        # train_x, train_y, valid_x, valid_y, test_x, test_y =\
        #     data_helper.load(train_file_path, test_file_path)
        datas.append((name,data_helper.load(train_file_path,test_file_path)))
    # Training
    errors_long=[]
    errors_lati=[]
    errors=[]
    for data in datas:
        floor_name, (train_x, train_y, valid_x, valid_y, test_x, test_y)=data
        strat = time.time()
        Location_Model = LOCATION_MODEL(str(floor_name)+'_model.h5')
        Location_Model.fit(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        longitude_error, latitude_error, mean_error=Location_Model.evaluate(test_x, test_y)
        errors_long.append(longitude_error)
        errors_lati.append(latitude_error)
        errors.append(mean_error)
        print(floor_name+'_position_error'+str(mean_error)+'m')
        print(floor_name+'_long_err:'+str(longitude_error)+'m')
        print(floor_name+'_lati_err:'+str(latitude_error)+'m')
        end = time.time()
        print(floor_name+'_run_time:'+str(end-strat))
    print('mean_error_long:',np.mean(errors_long))
    print('mean_error_lati:',np.mean(errors_lati))
    print('mean_error:',np.mean(errors))

