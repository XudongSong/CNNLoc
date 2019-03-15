from New2_Location_encoder_multi_model import EncoderDNN
import numpy as np
import data_helper
import time
import keras
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
# np.random.seed(423453)

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'
from keras.backend.tensorflow_backend import set_session
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3
set_session(tf.Session(config=config))


rng=np.random.RandomState(888)

base_dir= os.getcwd()
# all_train_csv_path = os.path.join(base_dir,'trainingData.csv')
test_csv_path=os.path.join(base_dir,'validationData.csv')
valid_csv_path=os.path.join(base_dir,'AllValuationData.csv')
train_csv_path=os.path.join(base_dir,'arrAllTrainingData.csv')
# train_csv_path=os.path.join(base_dir,'all_training_data.csv')
log_dir='New_Location_Auto_DEEPLEARNING_MODEL_log.txt'

compare_list=[]
if __name__ == '__main__':
    # Load data
    (train_x,train_y),(valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path,valid_csv_path,test_csv_path)


    # (test_x,test_y)=data_helper.load_data_perspective(test_csv_path)
    # (valid_x,valid_y)=data_helper.load_grouped_data(valid_csv_path)
    # (train_x, train_y)=data_helper.load_grouped_data(train_csv_path)


    # # train_x, train_y, valid_x, valid_y, test_x, test_y=data_helper.load(train_csv_path, test_csv_path)

    # (train_x,train_y)=data_helper.load_data_perspective(all_train_csv_path)
    # (test_x,test_y)=data_helper.load_data_perspective(test_csv_path)

    encode_dnn_model = EncoderDNN()

    SAE=[
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
        [[128, 64], [128, 520]],
         [[128, 64], [128, 520]],

         # [[128,64],[128,520]],
         # [[256,64],[256,520]],
         # [[256,128,64],[128,256,520]],
         # [[512,128],[128,520]],
         # [[512,256,128],[256,512,520]],
         # [[512,256,64],[256,512,520]],
         # [[512,64],[512,520]],
         # [[512,256,128,64],[128,256,512,520]],
         # [[256,128],[256,520]],

         ]
    CNN=[
        [[99,22],[66,22],[33,22]],
         [[129,22],[96,22],[63,22]],
         [[99,33],[66,32]],
         [[66,33],[99,32]],
         [[99,53],[66,12]],
         [[119,22],[86,22],[53,22]],
         [[81,33],[81,32]],
         [[333,33],[66,32]],
         [[139,22],[106,22],[73,22]],
         [[99,13],[66,52]],
         [[66,22],[44,22],[22,22]],
         [[333,33],[33,32]],
         [[66,33],],
         [[66,6],],
         [[33,6],],
         [[149,22],[116,22],[83,22]],
        #
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],



         ]

    SAE = [
        [128,64],
    ]
    CNN = [
        [[99,22],[66,22],[33,22]],
    ]

    for i in range(len(CNN)):
        sae, cnn=SAE[i],CNN[i]
        with open(log_dir, 'a') as file:
            file.write('\n' + "**" * 10 + 'SAE:'+str(sae)+'CNN:'+str(cnn)+' Start' + "**" * 10)


        #just for one model!!!
        pre_train=1
        lr=0.0001
        b=2.7
        p=3
        epoch_sae=40
        epoch_floor=88
        epoch_position=88
        dp=1
        regular_l1=0
        regular_l2=0
        results = []
        echo = 1
        while echo>0:
            echo-=1
        # for regular in regulars:
            # Load data
            #********fixed verification data**************
            # type_data="fixed"
            # (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path,valid_csv_path,test_csv_path)
            #********random verification data**********
            # type_data="random"
            # train_x, train_y, valid_x, valid_y, test_x, test_y=data_helper.load(train_csv_path, test_csv_path)


            name="SAE:"+str(sae)+"CNN:"+str(cnn)+"No."+str(regular_l1)
            # Training

            encode_dnn_model.patience=int(p)
            encode_dnn_model.b=b
            encode_dnn_model.epoch_AE=epoch_sae
            encode_dnn_model.epoch_floor=epoch_floor
            encode_dnn_model.epoch_position=epoch_position
            encode_dnn_model.dropout=dp
            encode_dnn_model.regularlization_l1=regular_l1
            encode_dnn_model.regularlization_l2 = regular_l2
            encode_dnn_model.adam_lr=lr
            encode_dnn_model.pre_train=pre_train
            strat = time.time()


            h=encode_dnn_model.fitLocationAuto(train_x,
                                               train_y,
                                               valid_x=valid_x,
                                               valid_y=valid_y,
                                               full_list=sae,
                                               CNN_list=cnn)#,tensorbd=tbCallBack)

            end=time.time()

            long,lati,mean=encode_dnn_model.error(test_x, test_y)

            # print('floor_accuracy',str((floor_right/1111.0)*100)+'%')

            print('time:',end-strat)

            print('position_error', str(mean) + 'm')
            print('long_err:', long)
            print('lati_err:', lati)
            results.append(mean)


        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
        plt.title('Location Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('LocationLoss.png')
        plt.clf()
        

        with open(log_dir,'a') as file:

            file.write('\n'+str(results)+','+str(np.mean(results)))

        print(results)
        print(np.mean(results))
        # compare_list.append([str(sae),str(cnn),str(results[0]),str(results[1]),str(results[2]),str(np.mean(results))])
        with open(log_dir, 'a') as file:
            file.write('\n' + "**" * 10 + 'SAE:'+str(sae)+'CNN:'+str(cnn)+' end' + "**" * 10)
    # compare = pd.DataFrame(data=compare_list,columns=['sae', 'cnn', 'th1', 'th2', 'th3', 'mean'])
    # compare.to_csv('compare.csv')