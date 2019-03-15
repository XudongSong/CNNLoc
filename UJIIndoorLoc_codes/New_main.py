from New_encoder_model import EncoderDNN
import numpy as np
import data_helper
import time
import keras
import csv
import matplotlib.pyplot as plt
import os

# np.random.seed(423453)
base_dir= os.getcwd()
train_csv_path = os.path.join(base_dir,'trainingData.csv')
test_csv_path=os.path.join(base_dir,'validationData.csv')

if __name__ == '__main__':

    results=[]
    for i in range(10):
        # Load data
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            data_helper.load(train_csv_path, test_csv_path)
        # Training
        encode_dnn_model = EncoderDNN()
        strat = time.time()
        # tbCallBack=keras.callbacks.TensorBoard(log_dir='./Graph',
        #                                        histogram_freq=1,
        #                                        write_graph=True,
        #                                        write_images=True)

        encode_dnn_model.fit(train_x, train_y, valid_x=valid_x, valid_y=valid_y)#,tensorbd=tbCallBack)
        end=time.time()

        floor_right=encode_dnn_model.error(test_x, test_y)
        # r=encode_dnn_model.predict(test_x[:4])
        # print(r)
        del encode_dnn_model

        print('floor_accuracy',str((floor_right/1111.0)*100)+'%')

        print('time:',end-strat)
        results.append((floor_right/1111.0)*100)

    print(results)
    print(np.mean(results))